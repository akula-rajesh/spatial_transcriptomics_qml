"""
Data pipeline component for organizing raw spatial transcriptomics files
into a canonical subtype/patient directory hierarchy.
Replaces: organize.py
"""

import shutil
import logging
import pandas as pd
from typing import Dict, Any
from pathlib import Path

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)


class SpatialFileOrganizer(BaseDataPipeline):
    """
    Organizes raw input files into a structured subtype/patient hierarchy.

    Replaces the standalone organize_files() function in organize.py with a
    fully factory-compatible pipeline component.

    Expected directory layout after execution::

        data/stbc/
        └── <subtype>/
            └── <patient>/
                ├── <patient>_<section>.jpg
                ├── <patient>_<section>.tsv.gz
                └── <patient>_<section>.spots
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file organizer.

        Args:
            config: Configuration dictionary containing:
                - data.input_dir   : Raw downloaded files (default: data/input/)
                - data.stbc_dir    : Organised output root  (default: data/stbc/)
        """
        super().__init__(config)
        self.input_dir = self.get_config_value('data.input_dir', 'data/input/')
        self.stbc_dir  = self.get_config_value('data.stbc_dir',  'data/stbc/')

    # ------------------------------------------------------------------
    # BaseDataPipeline interface
    # ------------------------------------------------------------------

    def execute(self) -> bool:
        """
        Execute the file organisation stage.

        Returns:
            True if organisation succeeded, False otherwise.
        """
        self.log_info("Starting file organisation")

        try:
            input_path = self._resolve_path(self.input_dir)

            if not input_path.exists():
                self.log_error(f"Input directory does not exist: {input_path}")
                return False

            self._rename_bt_to_bc(input_path)
            metadata = self._load_metadata(input_path)
            self._create_patient_dirs(metadata)
            self._move_files(input_path, metadata)

            self.log_info("File organisation completed successfully")
            return True

        except Exception as e:
            self.log_error(f"File organisation failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Private helpers  (mirrors original organize.py logic)
    # ------------------------------------------------------------------

    def _rename_bt_to_bc(self, input_path: Path) -> None:
        """
        Rename all files/dirs that contain 'BT' to use 'BC'.

        Replicates the BT→BC renaming step in the original organize.py
        organize_files() function.

        Args:
            input_path: Root directory containing raw files.
        """
        renamed = 0
        for item in list(input_path.glob("*")):
            if "BT" in item.name:
                new_path = input_path / item.name.replace("BT", "BC")
                item.rename(new_path)
                renamed += 1

        if renamed:
            self.log_info(f"Renamed {renamed} file(s): BT → BC")

    def _load_metadata(self, input_path: Path) -> pd.DataFrame:
        """
        Load and validate the metadata CSV.

        Replicates the metadata loading and validation from the original
        organize.py organize_files() function.

        Args:
            input_path: Directory that contains metadata.csv.

        Returns:
            Validated metadata DataFrame.

        Raises:
            FileNotFoundError: When metadata.csv is absent.
            ValueError       : When required columns are missing.
        """
        metadata_path = input_path / "metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found in {input_path}. "
                "Run the download stage first."
            )

        metadata = pd.read_csv(metadata_path)

        required_cols = {'patient', 'type'}
        missing = required_cols - set(metadata.columns)
        if missing:
            raise ValueError(
                f"metadata.csv is missing required columns: {missing}"
            )

        self.log_info(
            f"Loaded metadata: {len(metadata)} patients, "
            f"subtypes: {sorted(metadata['type'].unique().tolist())}"
        )
        return metadata

    def _create_patient_dirs(self, metadata: pd.DataFrame) -> None:
        """
        Pre-create the subtype/patient directory tree.

        Replicates the directory creation loop in the original
        organize.py organize_files() function.

        Args:
            metadata: DataFrame with 'patient' and 'type' columns.
        """
        stbc_path = self._ensure_directory_exists(self.stbc_dir)

        for _, row in metadata.iterrows():
            patient_dir = stbc_path / row['type'] / row['patient']
            patient_dir.mkdir(parents=True, exist_ok=True)

        self.log_info(f"Created patient directory tree under {stbc_path}")

    def _move_files(self, input_path: Path, metadata: pd.DataFrame) -> None:
        """
        Copy and rename raw files into the subtype/patient hierarchy.

        Replicates the filename cleaning and copy logic in the original
        organize.py organize_files() function:
          - Strips 'HE_' and '_stdata' from filenames
          - Converts spots_*.csv  →  *.spots
          - Skips metadata.csv itself

        Args:
            input_path: Directory containing raw files.
            metadata  : DataFrame used to resolve patient → subtype.
        """
        stbc_path = self._resolve_path(self.stbc_dir)
        patient_to_subtype: Dict[str, str] = dict(
            zip(metadata['patient'], metadata['type'])
        )

        copied = skipped = 0

        for file in input_path.glob("*"):
            if file.name == "metadata.csv" or not file.is_file():
                continue

            # --- clean filename (original organize.py logic) ---
            new_name = (
                file.name
                .replace("HE_", "")
                .replace("_stdata", "")
            )
            if "spots_" in new_name:
                new_name = new_name.replace("spots_", "").replace(".csv", ".spots")

            # --- match to patient ---
            matched_patient = next(
                (p for p in patient_to_subtype if p in new_name), None
            )
            if not matched_patient:
                self.log_warning(f"No patient match for file: {file.name} — skipping")
                skipped += 1
                continue

            subtype = patient_to_subtype[matched_patient]
            dst = stbc_path / subtype / matched_patient / new_name

            if not dst.exists():
                shutil.copy(str(file), str(dst))
                copied += 1

        self.log_info(
            f"Organisation complete — copied: {copied}, skipped: {skipped}"
        )

    # ------------------------------------------------------------------
    # Public alias
    # ------------------------------------------------------------------

    def organize_files(self) -> bool:
        """Public alias for execute()."""
        return self.execute()


__all__ = ['SpatialFileOrganizer']
