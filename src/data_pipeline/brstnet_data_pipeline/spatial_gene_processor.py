"""
Data pipeline component for processing raw gene count files into
filtered, compressed NPZ spot archives.
Replaces: gene_processing.py
"""

import gc
import gzip
import logging
import pickle
import collections
import numpy as np
import pandas as pd
import cv2
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)


class SpatialGeneProcessor(BaseDataPipeline):
    """
    Processes raw spatial transcriptomics gene count data into filtered NPZ
    spot archives with per-spot quality control.

    Replaces the standalone spatial_gene_processing() function in
    gene_processing.py with a factory-compatible pipeline component.

    Processing stages::

        Raw TSV + SPOTS
          1. Global gene list collection
          2. Per-spot quality filters
               a) Boundary  — patch must fit within image
               b) Spot-ID   — spot must appear in count matrix
               c) Quality   — total reads ≥ quality_threshold
          3. Sparsity filter  — gene expressed in ≥ sparsity_threshold % spots
          4. Save filtered NPZ archives + metadata (gene.pkl, mean_expression.npy)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gene processor.

        Args:
            config: Configuration dictionary containing:
                - data.stbc_dir          : Input root (raw images + counts)
                - data.stained_dir       : Stain-normalised images root
                - data.count_raw_dir     : Intermediate NPZ output
                - data.count_filtered_dir: Final filtered NPZ output
                - preprocessing.window_size       : Patch side length  (default: 224)
                - preprocessing.quality_threshold : Min total reads    (default: 1000)
                - preprocessing.sparsity_threshold: Min spot fraction  (default: 0.10)
        """
        super().__init__(config)
        self.stbc_dir           = self.get_config_value('data.stbc_dir',           'data/stbc/')
        self.stained_dir        = self.get_config_value('data.stained_dir',        'data/stained/')
        self.count_raw_dir      = self.get_config_value('data.count_raw_dir',      'data/processed/count_raw/')
        self.count_filtered_dir = self.get_config_value('data.count_filtered_dir', 'data/processed/count_filtered/')
        self.window             = self.get_config_value('preprocessing.window_size',        224)
        self.quality_threshold  = self.get_config_value('preprocessing.quality_threshold',  1000)
        self.sparsity_threshold = self.get_config_value('preprocessing.sparsity_threshold', 0.10)

    # ------------------------------------------------------------------
    # BaseDataPipeline interface
    # ------------------------------------------------------------------

    def execute(self) -> Dict[str, Any]:
        """
        Execute the gene processing pipeline stage.

        Returns:
            Dictionary with keys:
              - 'num_filtered_genes' (int)
              - 'total_spots'        (int)
              - 'kept_spots'         (int)
        """
        self.log_info("Starting spatial gene processing")

        try:
            self._ensure_directory_exists(self.count_raw_dir)
            self._ensure_directory_exists(self.count_filtered_dir)

            # Stage 1 — discovery + global gene list
            patient_sections, subtype_map = self._discover_sections()
            gene_names = self._collect_global_gene_list(patient_sections, subtype_map)
            self._save_raw_metadata(gene_names, subtype_map)

            # Stage 2 — extract quality-filtered spots
            total_spots, kept_spots = self._extract_spots(
                patient_sections, subtype_map, gene_names
            )

            # Stage 3 — sparsity filter + save filtered archives
            num_filtered_genes = self._apply_sparsity_filter(gene_names, subtype_map)

            results = {
                'num_filtered_genes': num_filtered_genes,
                'total_spots':        total_spots,
                'kept_spots':         kept_spots,
            }

            self.log_info(
                f"Gene processing complete — "
                f"genes: {num_filtered_genes}, "
                f"spots: {kept_spots}/{total_spots}"
            )
            return results

        except Exception as e:
            self.log_error(f"Gene processing failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Stage 1 — Discovery
    # ------------------------------------------------------------------

    def _discover_sections(self):
        """
        Walk stbc_dir to build patient→sections and patient→subtype maps.

        Replicates the discovery phase in the original
        gene_processing.py spatial_gene_processing() function.

        Returns:
            Tuple of (patient_sections defaultdict, subtype_map dict).
        """
        stbc_path = self._resolve_path(self.stbc_dir)
        images    = [
            p for p in stbc_path.rglob("*_*.jpg")
            if "_mask" not in p.name
        ]

        patient_sections: Dict[str, list] = collections.defaultdict(list)
        subtype_map: Dict[str, str]        = {}

        for img_path in images:
            parts   = img_path.parts
            subtype = parts[-3]
            p_id    = parts[-2]
            section = img_path.stem.replace(f"{p_id}_", "")
            patient_sections[p_id].append(section)
            subtype_map[p_id] = subtype

        self.log_info(
            f"Discovered {len(patient_sections)} patients, "
            f"subtypes: {sorted(set(subtype_map.values()))}"
        )
        return patient_sections, subtype_map

    # ------------------------------------------------------------------
    # Stage 1 — Global gene list
    # ------------------------------------------------------------------

    def _collect_global_gene_list(
            self,
            patient_sections: Dict[str, list],
            subtype_map: Dict[str, str],
    ) -> list:
        """
        Collect the union of gene names across all sections.

        Replicates the global gene-collection loop in the original
        gene_processing.py spatial_gene_processing() function.

        Args:
            patient_sections: patient → [section, …] mapping.
            subtype_map      : patient → subtype mapping.

        Returns:
            Sorted list of all gene names.
        """
        self.log_info("Collecting global gene list …")
        stbc_path  = self._resolve_path(self.stbc_dir)
        gene_names = set()

        for p_id, sections in tqdm(patient_sections.items(), desc="Gene list"):
            for sec in sections:
                data = self._load_section_data(stbc_path, subtype_map[p_id], p_id, sec)
                if data is not None:
                    gene_names.update(data["count"].columns[1:])

        gene_names = sorted(gene_names)
        self.log_info(f"Found {len(gene_names)} total genes")
        return gene_names

    def _load_section_data(
            self,
            root: Path,
            subtype: str,
            patient: str,
            section: str,
    ) -> Optional[Dict]:
        """
        Load count DataFrame and spot DataFrame for one section.

        Replicates the load_section_data() helper in the original
        gene_processing.py.

        Args:
            root   : stbc root directory.
            subtype: Tissue subtype folder name.
            patient: Patient ID.
            section: Section ID.

        Returns:
            Dict with 'count' and 'spot' DataFrames, or None on failure.
        """
        file_root  = root / subtype / patient / f"{patient}_{section}"

        count_path = file_root.with_suffix('.tsv.gz')
        if not count_path.exists():
            return None

        with gzip.open(count_path, "rb") as f:
            count_df = pd.read_csv(f, sep="\t")

        # spots file — try both plain and gzipped
        spot_path = file_root.with_suffix('.spots')
        if not spot_path.exists():
            spot_path = spot_path.with_suffix('.spots.gz')
        if not spot_path.exists():
            return None

        opener = gzip.open if str(spot_path).endswith('.gz') else open
        mode   = "rb"   if str(spot_path).endswith('.gz') else "r"
        with opener(spot_path, mode) as f:
            spot_df = pd.read_csv(f, sep="\t")

        return {"count": count_df, "spot": spot_df}

    # ------------------------------------------------------------------
    # Stage 1 — Save raw metadata
    # ------------------------------------------------------------------

    def _save_raw_metadata(
            self, gene_names: list, subtype_map: Dict[str, str]
    ) -> None:
        """
        Persist gene.pkl and subtype.pkl to count_raw_dir.

        Replicates the metadata-save block in the original
        gene_processing.py spatial_gene_processing() function.
        """
        raw_path = self._resolve_path(self.count_raw_dir)

        with open(raw_path / "gene.pkl", "wb") as f:
            pickle.dump(gene_names, f)

        with open(raw_path / "subtype.pkl", "wb") as f:
            pickle.dump(subtype_map, f)

        self.log_info(f"Raw metadata saved to {raw_path}")

    # ------------------------------------------------------------------
    # Stage 2 — Spot extraction with QC filters
    # ------------------------------------------------------------------

    def _extract_spots(
            self,
            patient_sections: Dict[str, list],
            subtype_map: Dict[str, str],
            gene_names: list,
    ):
        """
        Extract per-spot NPZ archives, applying three quality filters.

        Replicates the spot-extraction loop in the original
        gene_processing.py spatial_gene_processing() function, including:
          - Boundary filter
          - Spot-ID presence filter
          - Quality (total reads) filter

        Args:
            patient_sections: patient → [section, …] mapping.
            subtype_map      : patient → subtype mapping.
            gene_names       : Global sorted gene list.

        Returns:
            Tuple (total_spots, kept_spots).
        """
        stbc_path    = self._resolve_path(self.stbc_dir)
        stained_path = self._resolve_path(self.stained_dir)
        raw_path     = self._resolve_path(self.count_raw_dir)
        half         = self.window // 2

        total_spots = kept_spots = 0

        for p_id, sections in patient_sections.items():
            st = subtype_map[p_id]
            (raw_path / st / p_id).mkdir(parents=True, exist_ok=True)

            for sec in sections:
                data = self._load_section_data(stbc_path, st, p_id, sec)
                if data is None:
                    continue

                count_map = self._build_count_map(data["count"], gene_names)

                # Use stain-normalised image for boundary check
                img_path = stained_path / st / p_id / f"{p_id}_{sec}.jpg"
                if not img_path.exists():
                    # fall back to original
                    img_path = stbc_path / st / p_id / f"{p_id}_{sec}.jpg"
                if not img_path.exists():
                    self.log_warning(f"No image for {p_id}/{sec} — skipping")
                    continue

                img       = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]

                for _, row in data["spot"].iterrows():
                    total_spots += 1

                    # parse spot info  (original gene_processing.py logic)
                    spot_info = str(row.iloc[0]).split(',')
                    if len(spot_info) < 3:
                        continue

                    spot_id = spot_info[0]
                    try:
                        x_px = round(float(spot_info[1]))
                        y_px = round(float(spot_info[2]))
                    except ValueError:
                        continue

                    # Filter 1 — boundary
                    if not (
                            x_px - half >= 0 and x_px + half <= img_w and
                            y_px - half >= 0 and y_px + half <= img_h
                    ):
                        continue

                    # Filter 2 — spot ID in count matrix
                    if spot_id not in count_map:
                        continue

                    # Filter 3 — quality threshold
                    if np.sum(count_map[spot_id]) < self.quality_threshold:
                        continue

                    # Save spot NPZ
                    spot_x, spot_y = spot_id.split('x')
                    out_file = raw_path / st / p_id / f"{sec}_{spot_x}_{spot_y}.npz"

                    np.savez_compressed(
                        out_file,
                        count   = count_map[spot_id],
                        pixel   = np.array([x_px, y_px]),
                        patient = np.array([p_id]),
                        section = np.array([sec]),
                        index   = np.array([int(spot_x), int(spot_y)]),
                    )
                    kept_spots += 1

        self.log_info(
            f"Spot extraction — kept {kept_spots}/{total_spots} spots"
        )
        return total_spots, kept_spots

    def _build_count_map(
            self, count_df: pd.DataFrame, gene_names: list
    ) -> Dict[str, np.ndarray]:
        """
        Align a section's count matrix to the global gene list.

        Replicates the gene-alignment logic from the original
        gene_processing.py spatial_gene_processing() function.

        Args:
            count_df  : Raw count DataFrame (first col = spot IDs).
            gene_names: Global sorted gene list.

        Returns:
            Dict mapping spot_id → aligned expression vector.
        """
        existing_genes  = list(count_df.columns[1:])
        missing_genes   = list(set(gene_names) - set(existing_genes))

        c_matrix = count_df.iloc[:, 1:].values.astype(float)
        if missing_genes:
            pad      = np.zeros((c_matrix.shape[0], len(missing_genes)))
            c_matrix = np.concatenate((c_matrix, pad), axis=1)

        all_cols = np.array(existing_genes + missing_genes)
        sort_idx = np.argsort(all_cols)
        c_matrix = c_matrix[:, sort_idx]

        return {
            str(count_df.iloc[i, 0]): c_matrix[i, :]
            for i in range(len(count_df))
        }

    # ------------------------------------------------------------------
    # Stage 3 — Sparsity filter
    # ------------------------------------------------------------------

    def _apply_sparsity_filter(
            self, gene_names: list, subtype_map: Dict[str, str]
    ) -> int:
        """
        Remove genes expressed in fewer than sparsity_threshold of spots.

        Replicates the sparsity-filter block in the original
        gene_processing.py spatial_gene_processing() function.

        Args:
            gene_names : Full global gene list (pre-filter).
            subtype_map: patient → subtype mapping (for output paths).

        Returns:
            Number of genes that passed the sparsity filter.
        """
        raw_path      = self._resolve_path(self.count_raw_dir)
        filtered_path = self._resolve_path(self.count_filtered_dir)

        self.log_info("Applying sparsity filter …")
        all_npz   = list(raw_path.rglob("*.npz"))

        # Load full gene matrix into memory
        gene_data = [
            np.load(f)["count"][:, np.newaxis]
            for f in tqdm(all_npz, desc="Loading for sparsity")
        ]
        gene_matrix = np.concatenate(gene_data, axis=1)
        del gene_data
        gc.collect()

        presence_ratio = np.sum(gene_matrix > 0, axis=1) / gene_matrix.shape[1]
        keep_filter    = presence_ratio >= self.sparsity_threshold

        filtered_genes = [g for g, k in zip(gene_names, keep_filter) if k]
        mean_expr      = np.mean(gene_matrix[keep_filter], axis=1)

        self.log_info(
            f"Sparsity filter — kept {len(filtered_genes)}/{len(gene_names)} genes"
        )

        # Persist filtered metadata
        np.save(filtered_path / "mean_expression.npy", mean_expr)
        with open(filtered_path / "gene.pkl", "wb") as f:
            pickle.dump(filtered_genes, f)
        with open(filtered_path / "subtype.pkl", "wb") as f:
            pickle.dump(subtype_map, f)

        # Re-save every NPZ with only the kept genes
        for npz_file in tqdm(all_npz, desc="Saving filtered NPZ"):
            data  = np.load(npz_file)
            p_id  = str(data["patient"][0])
            st    = subtype_map[p_id]

            target_dir = filtered_path / st / p_id
            target_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                target_dir / npz_file.name,
                count   = data["count"][keep_filter],
                pixel   = data["pixel"],
                patient = data["patient"],
                section = data["section"],
                index   = data["index"],
                )

        del gene_matrix
        gc.collect()

        return len(filtered_genes)

    # ------------------------------------------------------------------
    # Public alias
    # ------------------------------------------------------------------

    def process_genes(self) -> Dict[str, Any]:
        """Public alias for execute()."""
        return self.execute()


__all__ = ['SpatialGeneProcessor']
