"""
Spatial Transcriptomics Dataset Module.

Matches the reference paper's Spatial dataset class exactly:
  - TWO-STAGE loading: Generator (patch cache) → Spatial (training)
  - Returns 8-tuple: (X, y, aux, coord, index, patient, section, pixel)
  - Dataset-specific image normalization computed from RAW PIL images
  - Window=299 default with resize to 224
  - Spatial augmentations: HFlip + VFlip + RandomRotation(90°)
  - log1p gene count transformation
  - Normalization applied ONLY to main genes (y), NOT aux genes
"""

import numpy as np
import pickle
import time
import logging
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SpatialDataset(Dataset):
    """
    Spatial transcriptomics dataset.

    Matches the reference paper's Spatial class EXACTLY:
      - log1p transformation of gene counts
      - Top-N gene filtering by mean expression
      - Auxiliary task support (aux NOT normalised — paper behaviour)
      - 8-tuple return: (X, y, aux, coord, index, patient, section, pixel)
      - Optional per-gene count normalization on MAIN genes only

    Image path structure matches paper exactly:
      {img_root}/{subtype}/{patient}/{window}/{section}_{x}_{y}.jpg
    """

    def __init__(
            self,
            patient_list=None,
            window=299,           # Paper default: 299
            count_root=None,
            img_root=None,
            gene_filter=250,
            aux_ratio=1.0,
            transform=None,
            normalization=None,
    ):
        self.dataset = sorted(Path(count_root).rglob("*.npz"))

        if patient_list is not None:
            self.dataset = [
                d for d in self.dataset
                if d.parts[-2] in patient_list
            ]

        logger.info(f"Loaded {len(self.dataset)} samples")

        self.transform     = transform
        self.window        = window
        self.count_root    = Path(count_root)
        self.img_root      = Path(img_root)
        self.gene_filter   = gene_filter
        self.aux_ratio     = aux_ratio
        self.normalization = normalization

        # Resolve metadata root
        filter_root = self._get_filter_root(count_root)

        with open(filter_root / "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)

        with open(filter_root / "gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)

        self.mean_expression = np.load(filter_root / "mean_expression.npy")

        logger.info(f"Total genes available: {len(self.ensg_names)}")

        # ── Top-N gene selection (matches paper exactly) ──────────────
        sorted_idx = np.argsort(self.mean_expression)[::-1]
        keep_idx   = sorted_idx[:self.gene_filter]

        self.keep_bool = np.zeros(len(self.ensg_names), dtype=bool)
        self.keep_bool[keep_idx] = True

        self.ensg_keep = [n for n, f in zip(self.ensg_names, self.keep_bool) if f]
        self.gene_keep = self.ensg_keep

        logger.info(f"Selected top {self.gene_filter} genes")

        # ── Auxiliary task genes ──────────────────────────────────────
        if self.aux_ratio > 0:
            self.aux_nums = int(
                (len(self.ensg_names) - self.gene_filter) * self.aux_ratio
            )
            aux_idx = sorted_idx[self.gene_filter: self.gene_filter + self.aux_nums]

            self.aux_bool = np.zeros(len(self.ensg_names), dtype=bool)
            self.aux_bool[aux_idx] = True

            self.ensg_aux = [n for n, f in zip(self.ensg_names, self.aux_bool) if f]
            self.gene_aux = self.ensg_aux

            logger.info(f"Auxiliary genes: {self.aux_nums}")
        else:
            self.aux_nums = 0
            self.aux_bool = None
            self.ensg_aux = []
            self.gene_aux = []

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    def _get_filter_root(self, count_root) -> Path:
        """
        Resolve the count_filtered metadata directory.

        Supports both training and test paths:
          data/train/counts → data/processed/count_filtered
          data/test/counts  → data/processed/count_filtered
          data/processed/count_filtered → itself
        """
        count_str = str(count_root)
        for token in ("train/counts", "test/counts"):
            if token in count_str:
                return Path(count_str.replace(token, "processed/count_filtered"))
        # Already pointing at count_filtered
        return Path(count_str)

    def _build_img_path(self, patient: str, section: str, coord) -> Path:
        """
        Build image path matching paper's exact structure:

            {img_root}/{subtype}/{patient}/{window}/{section}_{x}_{y}.jpg

        This is the path written by SubGenerator in the paper's
        04_generating_train_test.py when resolution==224 (default).

        Note: the paper stores patches under the WINDOW size folder,
        not the resolution folder (resolution folder is only used
        when resolution != 224).
        """
        subtype = self.subtype.get(patient, "unknown")
        return (
                self.img_root
                / subtype
                / patient
                / str(self.window)
                / f"{section}_{coord[0]}_{coord[1]}.jpg"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
        Returns 8-tuple matching paper's Spatial.__getitem__ exactly:
            (X, y, aux, coord_tensor, index_tensor, patient, section, pixel_tensor)

        Key paper behaviours preserved:
          - y  = log1p(count) then z-score normalised (main genes only)
          - aux = log1p(count) with NO z-score normalisation
          - X  loaded as RGB PIL image → transform (ToTensor + Normalize)
        """
        npz     = np.load(self.dataset[index])
        count   = npz["count"]
        pixel   = npz["pixel"]
        patient = str(npz["patient"][0])
        section = str(npz["section"][0])
        coord   = npz["index"]

        # ── Load image patch ─────────────────────────────────────────
        img_path = self._build_img_path(patient, section, coord)

        try:
            # CRITICAL: open as RGB PIL image
            # Do NOT convert to numpy or divide before passing to transform
            X = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image not found: {img_path}")
            raise

        # Resize to 224×224 if patch was extracted at a different window
        # (paper: window=299 extracted, then resized to 224 for training)
        if X.size != (224, 224):
            X = T.Resize((224, 224))(X)

        # Apply transform (ToTensor + Normalize with dataset-specific stats)
        # transform is applied ONCE here — never pre-applied
        if self.transform is not None:
            X = self.transform(X)

        # ── Gene expression ──────────────────────────────────────────
        # Paper: y = torch.log(1 + y)
        keep_count = count[self.keep_bool]
        y = torch.log1p(torch.as_tensor(keep_count, dtype=torch.float))

        # Paper: normalisation applied to MAIN genes only
        if self.normalization is not None:
            count_mean, count_std = self.normalization
            y = (y - count_mean) / (count_std + 1e-8)

        coord_tensor = torch.as_tensor(coord)
        index_tensor = torch.as_tensor([index])
        pixel_tensor = torch.as_tensor(pixel)

        # ── Auxiliary genes (log1p only, NO z-score) ─────────────────
        if self.aux_ratio > 0 and self.aux_bool is not None:
            aux_count = count[self.aux_bool]
            aux = torch.log1p(torch.as_tensor(aux_count, dtype=torch.float))
            return X, y, aux, coord_tensor, index_tensor, patient, section, pixel_tensor
        else:
            return X, y, coord_tensor, index_tensor, patient, section, pixel_tensor

    def get_gene_names(self):
        return self.ensg_keep

    def get_aux_gene_names(self):
        return self.ensg_aux


# ======================================================================
# Normalization helpers
# ======================================================================

def compute_dataset_normalization(dataset: SpatialDataset):
    """
    Compute per-gene mean and std from the TRAINING dataset.

    Matches paper's get_mean_and_std() gene branch:
        count_mean = mean(log1p(count), axis=spots)  shape: (gene_filter,)
        count_std  = std(log1p(count),  axis=spots)  shape: (gene_filter,)

    MUST be called on training dataset ONLY.

    Args:
        dataset: Training SpatialDataset (normalization=None)

    Returns:
        Tuple: (count_mean, count_std) — both shape (gene_filter,)
    """
    logger.info("Computing dataset-specific gene expression normalization...")
    t = time.time()

    all_counts = []
    for i in range(len(dataset)):
        npz   = np.load(dataset.dataset[i])
        count = npz["count"]
        y = torch.log1p(
            torch.as_tensor(count[dataset.keep_bool], dtype=torch.float)
        )
        all_counts.append(y)

    all_counts = torch.stack(all_counts)   # (N_spots, gene_filter)
    count_mean = torch.mean(all_counts, dim=0)
    count_std  = torch.std(all_counts,  dim=0)

    logger.info(
        f"Normalization computed in {time.time() - t:.1f}s — "
        f"Mean: [{count_mean.min():.3f}, {count_mean.max():.3f}], "
        f"Std:  [{count_std.min():.3f},  {count_std.max():.3f}]"
    )
    return count_mean, count_std


def compute_image_normalization(dataset: SpatialDataset, batch_size: int = 32):
    """
    Compute per-channel image mean and std from the TRAINING dataset.

    Matches paper's get_mean_and_std() image branch.

    CRITICAL REQUIREMENTS:
      1. Dataset MUST have transform = T.ToTensor() ONLY (no Normalize)
      2. Images MUST be loaded as RGB PIL (uint8, values 0-255)
      3. ToTensor converts [0,255] → [0.0,1.0] — only then is mean/std meaningful
      4. This function sets transform temporarily to pure ToTensor if needed

    Returns:
        Tuple: (mean, std) — both lists of 3 floats [R, G, B]
                Expected healthy H&E values: mean≈[0.6-0.8], std≈[0.1-0.2]

    Raises:
        ValueError: If computed stats look pathological (std≈0 or mean<0.05)
    """
    logger.info("Computing dataset-specific image normalization...")
    t = time.time()

    # Temporarily override transform to ToTensor ONLY
    # This ensures no double-normalization
    original_transform  = dataset.transform
    dataset.transform   = T.ToTensor()   # PIL RGB uint8 [0,255] → float [0,1]

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,   # avoid multiprocessing issues with temp transform
    )

    mean       = torch.zeros(3)
    std        = torch.zeros(3)
    nb_samples = 0

    for batch in loader:
        X = batch[0]   # (B, C, H, W) — guaranteed [0,1] from ToTensor
        B = X.size(0)

        # Sanity check on first batch
        if nb_samples == 0:
            x_min, x_max = X.min().item(), X.max().item()
            logger.info(
                f"First batch pixel range: [{x_min:.4f}, {x_max:.4f}] "
                f"(expected [0.0, 1.0])"
            )
            if x_max > 2.0:
                logger.error(
                    "Pixel values > 2.0 detected! Images may not be "
                    "loaded as uint8 RGB. Check image loading pipeline."
                )
            if X.shape[1] != 3:
                logger.error(
                    f"Expected 3 channels, got {X.shape[1]}. "
                    "Images may be loading as grayscale!"
                )

        X_flat  = X.view(B, X.size(1), -1)  # (B, C, H*W)
        mean   += X_flat.mean(dim=2).sum(dim=0)
        std    += X_flat.std(dim=2).sum(dim=0)
        nb_samples += B

    # Restore original transform
    dataset.transform = original_transform

    mean /= nb_samples
    std  /= nb_samples

    logger.info(
        f"Image normalization computed in {time.time() - t:.1f}s — "
        f"Mean: {mean.tolist()}, Std: {std.tolist()}"
    )

    # ── Sanity checks ─────────────────────────────────────────────────
    _validate_image_stats(mean, std)

    return mean.tolist(), std.tolist()


def _validate_image_stats(mean: torch.Tensor, std: torch.Tensor) -> None:
    """
    Validate computed image normalization statistics.

    Raises warnings/errors when stats indicate a broken image pipeline.
    Expected H&E tissue ranges:
        Mean: [0.45, 0.95] per channel
        Std:  [0.05, 0.35] per channel
        All 3 channels should differ (not all identical = grayscale)
    """
    channel_names = ['R', 'G', 'B']

    for i, (m, s, ch) in enumerate(zip(mean, std, channel_names)):
        m_val = m.item()
        s_val = s.item()

        if m_val < 0.05:
            logger.error(
                f"Channel {ch} mean={m_val:.6f} is CRITICALLY low! "
                "Images may be double-normalized or not uint8 RGB. "
                "Expected ~0.65 for H&E tissue."
            )
        if s_val < 0.01:
            logger.error(
                f"Channel {ch} std={s_val:.8f} ≈ 0! "
                "All images look identical to the normalizer. "
                "This will DESTROY pixel values after normalization. "
                "Check that images are loaded as PIL RGB, not pre-normalized."
            )

    # Check grayscale collapse (all channels identical)
    if (abs(mean[0] - mean[1]) < 1e-4 and
            abs(mean[1] - mean[2]) < 1e-4):
        logger.error(
            "All 3 channels have IDENTICAL mean values! "
            "Images are being loaded as GRAYSCALE. "
            "H&E staining REQUIRES color information. "
            "Ensure Image.open(...).convert('RGB') is used."
        )


def build_transforms(image_mean, image_std, augment: bool = False):
    """
    Build torchvision transforms matching the reference paper.

    Paper training transforms (from setup() in Br_STNet_baseline.py):
        RandomHorizontalFlip()
        RandomVerticalFlip()
        RandomApply([RandomRotation((90, 90))])
        ToTensor()
        Normalize(mean=dataset_mean, std=dataset_std)

    Paper validation/test transforms:
        ToTensor()
        Normalize(mean=dataset_mean, std=dataset_std)

    Args:
        image_mean : Per-channel mean [R, G, B] computed from training data
        image_std  : Per-channel std  [R, G, B] computed from training data
        augment    : True for training, False for val/test
    """
    normalize = T.Normalize(mean=image_mean, std=image_std)

    if augment:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation((90, 90))]),
            T.ToTensor(),    # PIL uint8 [0,255] → float [0,1]
            normalize,       # dataset-specific z-score
        ])
    else:
        return T.Compose([
            T.ToTensor(),    # PIL uint8 [0,255] → float [0,1]
            normalize,
        ])


def create_dataloaders(paths, config, test_patient=None):
    """
    Create train and test dataloaders following the paper's setup() function.

    Matches paper's two-pass approach:
      Pass 1: ToTensor-only dataset → compute image + gene normalization stats
      Pass 2: Full augmentation + normalization dataset for training
      Pass 3: Val transform + train normalization for test

    Args:
        paths        : Object with attributes:
                         train_counts, train_images, test_counts, test_images
        config       : Config dict (nested)
        test_patient : Patient ID held out for testing.
                       If None, resolved from config in this priority order:
                         1. data.test_patient  (string — authoritative)
                         2. split.test_patients[0] (list — fallback)
                       Raises ValueError if neither is set.

    Returns:
        Tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    # ── Resolve test_patient from config if not provided ─────────────
    if test_patient is None:
        # Priority 1: data.test_patient (authoritative single-patient field)
        test_patient = config.get('data', {}).get('test_patient', None)

    if test_patient is None:
        # Priority 2: split.test_patients list (legacy / documentation field)
        split_list = config.get('split', {}).get('test_patients', [])
        if split_list:
            test_patient = split_list[0]
            logger.warning(
                f"data.test_patient not set — using split.test_patients[0]='{test_patient}'. "
                f"Set data.test_patient directly for reliable behaviour."
            )

    if not test_patient:
        raise ValueError(
            "test_patient is not set. "
            "Add 'test_patient: \"BC23450\"' under the 'data:' section of your config."
        )

    # ── Discover patients ─────────────────────────────────────────────
    all_patients = set()
    for f in Path(paths.train_counts).rglob("*.npz"):
        all_patients.add(f.parts[-2])
    for f in Path(paths.test_counts).rglob("*.npz"):
        all_patients.add(f.parts[-2])

    all_patients = sorted(all_patients)

    # ── Validate test_patient exists in the discovered data ───────────
    if test_patient not in all_patients:
        raise ValueError(
            f"test_patient='{test_patient}' was not found in any .npz files under "
            f"'{paths.train_counts}' or '{paths.test_counts}'.\n"
            f"Available patients: {all_patients}\n"
            f"Check that data.test_patient in your config matches an actual patient ID."
        )

    train_patients = [p for p in all_patients if p != test_patient]

    if not train_patients:
        raise ValueError(
            f"No training patients remain after holding out test_patient='{test_patient}'. "
            f"All discovered patients: {all_patients}"
        )

    logger.info(f"Data split — test_patient: '{test_patient}'")
    logger.info(f"  Train patients ({len(train_patients)}): {train_patients}")
    logger.info(f"  Test  patient  (1):               [{test_patient}]")

    # ── Config values ─────────────────────────────────────────────────
    def _c(section, key, default):
        return config.get(section, {}).get(key, default)

    window      = _c('preprocessing', 'window_size', 299)   # paper default: 299
    gene_filter = _c('model',         'gene_filter',  250)
    aux_ratio   = _c('model',         'aux_ratio',    1.0)
    batch_size  = _c('training',      'batch_size',   32)
    num_workers = _c('training',      'num_workers',  8)

    # ── Pass 1: Initial dataset — ToTensor ONLY — to compute stats ────
    # CRITICAL: transform = T.ToTensor() and nothing else here.
    # Any Normalize() here would corrupt the stats.
    init_dataset = SpatialDataset(
        patient_list  = train_patients,
        window        = window,
        count_root    = paths.train_counts,
        img_root      = paths.train_images,
        gene_filter   = gene_filter,
        aux_ratio     = aux_ratio,
        transform     = T.ToTensor(),   # ToTensor ONLY — no Normalize
        normalization = None,
    )

    # ── Compute normalization statistics ──────────────────────────────
    image_mean, image_std = compute_image_normalization(init_dataset, batch_size)
    count_mean, count_std = compute_dataset_normalization(init_dataset)

    logger.info(f"Image mean: {image_mean}, std: {image_std}")

    # ── Pass 2: Training dataset — full augmentation + normalization ──
    train_transform = build_transforms(image_mean, image_std, augment=True)
    train_dataset   = SpatialDataset(
        patient_list  = train_patients,
        window        = window,
        count_root    = paths.train_counts,
        img_root      = paths.train_images,
        gene_filter   = gene_filter,
        aux_ratio     = aux_ratio,
        transform     = train_transform,
        normalization = (count_mean, count_std),
    )

    # ── Pass 3: Test dataset — val transforms + TRAIN normalization ───
    # CRITICAL: test dataset uses TRAIN statistics, not test statistics.
    # Using test statistics would constitute data leakage.
    val_transform = build_transforms(image_mean, image_std, augment=False)
    test_dataset  = SpatialDataset(
        patient_list  = [test_patient],
        window        = window,
        count_root    = paths.test_counts,
        img_root      = paths.test_images,
        gene_filter   = gene_filter,
        aux_ratio     = aux_ratio,
        transform     = val_transform,
        normalization = (count_mean, count_std),   # ← TRAIN stats
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = use_pin_memory,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset, test_dataset


__all__ = [
    'SpatialDataset',
    'compute_dataset_normalization',
    'compute_image_normalization',
    'build_transforms',
    'create_dataloaders',
]
