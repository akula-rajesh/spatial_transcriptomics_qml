"""
Data pipeline component for extracting and caching fixed-size image
patches centred on each spatial transcriptomics spot.
Replaces: patch_extraction.py
"""

import shutil
import logging
import collections
import numpy as np
import pickle
import cv2
from typing import Dict, Any
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)


# ======================================================================
# Internal PyTorch Dataset helper
# (previously PatchGenerator in patch_extraction.py)
# ======================================================================

class _PatchDataset(Dataset):
    """
    Internal Dataset that extracts patches and caches them to disk.

    Returns dummy zero-tensors so DataLoader can be used purely for
    its worker/batching infrastructure (same pattern as original).
    """

    def __init__(
            self,
            patient_list: list,
            test_mode: bool,
            window: int,
            count_root: Path,
            img_root: Path,
            count_cached: Path,
            img_cached: Path,
            subtype: Dict[str, str],
    ):
        self.patient_list  = patient_list
        self.test_mode     = test_mode
        self.window        = window
        self.count_root    = count_root
        self.img_root      = img_root
        self.count_cached  = count_cached
        self.img_cached    = img_cached
        self.subtype       = subtype

        self.dataset = sorted(count_root.rglob("*.npz"))
        if patient_list:
            self.dataset = [
                d for d in self.dataset if d.parts[-2] in patient_list
            ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Extract one patch and save it to disk.

        Replicates __getitem__ from the original PatchGenerator, including:
        - White-ratio filter for training (≥ 0.5 tissue)
        - All patches kept for test mode
        """
        npz     = np.load(self.dataset[index])
        pixel   = npz["pixel"]
        patient = str(npz["patient"][0])
        section = str(npz["section"][0])
        coord   = npz["index"]

        st = self.subtype.get(patient, "unknown")

        # Load full slide image + tissue mask
        img_path  = self.img_root / st / patient / f"{patient}_{section}.jpg"
        mask_path = img_path.parent / (img_path.stem + "_mask.jpg")

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")

        # Crop patch centred on spot pixel
        half  = self.window // 2
        box   = (
            pixel[0] - half, pixel[1] - half,
            pixel[0] + half, pixel[1] + half,
        )
        patch      = img.crop(box)
        patch_mask = mask.crop(box)

        # Destination paths
        cached_count = (
                self.count_cached / st / patient /
                f"{section}_{coord[0]}_{coord[1]}.npz"
        )
        cached_image = (
                self.img_cached / st / patient /
                str(self.window) /
                f"{section}_{coord[0]}_{coord[1]}.jpg"
        )

        cached_count.parent.mkdir(parents=True, exist_ok=True)
        cached_image.parent.mkdir(parents=True, exist_ok=True)

        # --- filter rule (original patch_extraction.py logic) ---
        if not self.test_mode:
            white_ratio = (
                    np.array(patch_mask).sum()
                    / float(self.window * self.window)
            )
            if white_ratio >= 0.5:
                shutil.copy(self.dataset[index], cached_count)
                patch.save(cached_image)
        else:
            shutil.copy(self.dataset[index], cached_count)
            patch.save(cached_image)

        return torch.zeros(1), torch.zeros(1)


# ======================================================================
# Factory-compatible pipeline component
# ======================================================================

class SpatialPatchExtractor(BaseDataPipeline):
    """
    Extracts fixed-size image patches centred on each sequenced spot.

    Replaces the standalone extract_patches() function in
    patch_extraction.py with a factory-compatible pipeline component.

    For training patches a white-ratio tissue filter (≥ 0.5) is applied;
    all test patches are kept regardless of tissue content.

    Output layout::

        data/train/
        ├── counts/<subtype>/<patient>/<section>_<x>_<y>.npz
        └── images/<subtype>/<patient>/<window>/<section>_<x>_<y>.jpg

        data/test/
        ├── counts/…
        └── images/…
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the patch extractor.

        Args:
            config: Configuration dictionary containing:
                - data.count_filtered_dir : Filtered NPZ source
                - data.stained_dir        : Stain-normalised image source
                - data.train_counts_dir   : Train NPZ output
                - data.train_images_dir   : Train image output
                - data.test_counts_dir    : Test  NPZ output
                - data.test_images_dir    : Test  image output
                - preprocessing.window_size : Patch side length (default: 224)
                - data.test_patient         : Patient ID held out for test
                - training.num_workers      : DataLoader workers (default: 0)
                - training.batch_size       : DataLoader batch size (default: 32)
        """
        super().__init__(config)
        self.count_filtered_dir = self.get_config_value(
            'data.count_filtered_dir', 'data/processed/count_filtered/'
        )
        self.stained_dir        = self.get_config_value('data.stained_dir',      'data/stained/')
        self.train_counts_dir   = self.get_config_value('data.train_counts_dir', 'data/train/counts/')
        self.train_images_dir   = self.get_config_value('data.train_images_dir', 'data/train/images/')
        self.test_counts_dir    = self.get_config_value('data.test_counts_dir',  'data/test/counts/')
        self.test_images_dir    = self.get_config_value('data.test_images_dir',  'data/test/images/')
        self.window             = self.get_config_value('preprocessing.window_size', 224)
        self.test_patient       = self.get_config_value('data.test_patient')
        self.num_workers        = self.get_config_value('training.num_workers', 0)
        self.batch_size         = self.get_config_value('training.batch_size',  32)

    # ------------------------------------------------------------------
    # BaseDataPipeline interface
    # ------------------------------------------------------------------

    def execute(self) -> bool:
        """
        Execute the patch extraction stage.

        Returns:
            True if extraction succeeded, False otherwise.
        """
        self.log_info("Starting patch extraction")

        if not self.test_patient:
            self.log_error("data.test_patient must be set in configuration")
            return False

        try:
            # Create output dirs
            for d in [
                self.train_counts_dir, self.train_images_dir,
                self.test_counts_dir,  self.test_images_dir,
            ]:
                self._ensure_directory_exists(d)

            # Discover patients + subtype map
            train_patients, test_patients, subtype_map = self._discover_patients()

            # Load subtype map from metadata
            count_filtered_path = self._resolve_path(self.count_filtered_dir)
            subtype_pkl         = count_filtered_path / "subtype.pkl"
            if subtype_pkl.exists():
                with open(subtype_pkl, "rb") as f:
                    subtype_map = pickle.load(f)
                self.log_info("Loaded subtype map from count_filtered metadata")

            # Extract train patches
            self._run_extraction(
                patient_list  = train_patients,
                test_mode     = False,
                count_cached  = self._resolve_path(self.train_counts_dir),
                img_cached    = self._resolve_path(self.train_images_dir),
                subtype_map   = subtype_map,
                desc          = "Train patches",
            )

            # Extract test patches
            self._run_extraction(
                patient_list  = test_patients,
                test_mode     = True,
                count_cached  = self._resolve_path(self.test_counts_dir),
                img_cached    = self._resolve_path(self.test_images_dir),
                subtype_map   = subtype_map,
                desc          = "Test patches",
            )

            self.log_info("Patch extraction completed successfully")
            return True

        except Exception as e:
            self.log_error(f"Patch extraction failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_patients(self):
        """
        Discover all patients from stained images and split train/test.

        Replicates the patient-discovery logic in the original
        patch_extraction.py extract_patches() function.

        Returns:
            Tuple (train_patients, test_patients, subtype_map).
        """
        stained_path     = self._resolve_path(self.stained_dir)
        patient_sections = collections.defaultdict(list)
        subtype_map: Dict[str, str] = {}

        for img_path in stained_path.rglob("*_*.jpg"):
            if "_mask" in img_path.name:
                continue
            parts   = img_path.parts
            subtype = parts[-3]
            p_id    = parts[-2]
            subtype_map[p_id] = subtype
            section = img_path.stem.split("_", 1)[-1]
            patient_sections[p_id].append(section)

        all_patients   = sorted(patient_sections.keys())
        test_patients  = [self.test_patient]
        train_patients = [p for p in all_patients if p not in test_patients]

        self.log_info(f"Train patients: {train_patients}")
        self.log_info(f"Test  patients: {test_patients}")

        return train_patients, test_patients, subtype_map

    def _run_extraction(
            self,
            patient_list: list,
            test_mode: bool,
            count_cached: Path,
            img_cached: Path,
            subtype_map: Dict[str, str],
            desc: str,
    ) -> None:
        """
        Build a _PatchDataset + DataLoader and drain it to trigger caching.

        Replicates the DataLoader-drain pattern in the original
        patch_extraction.py extract_patches() function.

        Args:
            patient_list : Patients to process.
            test_mode    : False → apply tissue filter; True → keep all.
            count_cached : Destination directory for NPZ files.
            img_cached   : Destination directory for JPEG patches.
            subtype_map  : patient → subtype mapping.
            desc         : tqdm description string.
        """
        count_filtered_path = self._resolve_path(self.count_filtered_dir)
        stained_path        = self._resolve_path(self.stained_dir)

        dataset = _PatchDataset(
            patient_list  = patient_list,
            test_mode     = test_mode,
            window        = self.window,
            count_root    = count_filtered_path,
            img_root      = stained_path,
            count_cached  = count_cached,
            img_cached    = img_cached,
            subtype       = subtype_map,
        )

        loader = DataLoader(
            dataset,
            batch_size  = self.batch_size,
            num_workers = self.num_workers,
            shuffle     = False,
        )

        self.log_info(
            f"{desc}: {len(dataset)} spots across {len(patient_list)} patient(s)"
        )

        for _ in tqdm(loader, desc=desc):
            pass

    # ------------------------------------------------------------------
    # Public alias
    # ------------------------------------------------------------------

    def extract_patches(self) -> bool:
        """Public alias for execute()."""
        return self.execute()


__all__ = ['SpatialPatchExtractor']
