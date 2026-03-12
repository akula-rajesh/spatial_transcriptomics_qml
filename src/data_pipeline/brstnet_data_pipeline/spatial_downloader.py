"""
Data pipeline component for downloading and extracting spatial
transcriptomics datasets via HTTP streaming.
Replaces: download.py
"""

import shutil
import zipfile
import logging
import requests
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)


class SpatialDownloader(BaseDataPipeline):
    """
    Downloads and extracts spatial transcriptomics datasets via HTTP streaming.

    Replaces the standalone download_data() function in download.py with a
    fully factory-compatible pipeline component.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the spatial downloader.

        Args:
            config: Configuration dictionary containing:
                - download.url          : HTTP URL to the dataset ZIP
                - data.input_dir        : Destination directory (default: data/input/)
                - download.remove_zip   : Delete ZIP after extraction (default: True)
                - download.chunk_size   : Streaming chunk size in bytes (default: 1024)
        """
        super().__init__(config)
        self.url        = self.get_config_value('download.url')
        self.input_dir  = self.get_config_value('data.input_dir', 'data/input/')
        self.remove_zip = self.get_config_value('download.remove_zip', True)
        self.chunk_size = self.get_config_value('download.chunk_size', 1024)

    # ------------------------------------------------------------------
    # BaseDataPipeline interface
    # ------------------------------------------------------------------

    def execute(self) -> bool:
        """
        Execute the download pipeline stage.

        Returns:
            True if download and extraction succeeded, False otherwise.
        """
        self.log_info("Starting dataset download")

        try:
            input_path = self._ensure_directory_exists(self.input_dir)

            if self._data_already_exists(input_path):
                self.log_info("Dataset already present — skipping download")
                return True

            if not self.url:
                self.log_error("No download URL provided in configuration")
                return False

            zip_path = input_path / "dataset.zip"
            self._stream_download(self.url, zip_path)
            self._extract_and_flatten(zip_path, input_path)

            self.log_info("Dataset download completed successfully")
            return True

        except Exception as e:
            self.log_error(f"Download failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Private helpers  (mirrors original download.py logic)
    # ------------------------------------------------------------------

    def _data_already_exists(self, input_path: Path) -> bool:
        """
        Check whether the dataset has already been downloaded.

        Mirrors the idempotency guard in the original download.py that
        checked for metadata.csv before proceeding.

        Args:
            input_path: Root input directory.

        Returns:
            True if metadata.csv exists (dataset present).
        """
        sentinel = input_path / "metadata.csv"
        if sentinel.exists():
            self.log_info(f"Found sentinel file: {sentinel}")
            return True
        return False

    def _stream_download(self, url: str, zip_path: Path) -> None:
        """
        Stream-download a URL to disk with a tqdm progress bar.

        Replicates the requests.get(stream=True) logic from the original
        download.py download_data() function.

        Args:
            url     : Remote URL to fetch.
            zip_path: Local path to write the ZIP file.

        Raises:
            requests.HTTPError: If the server returns a non-2xx status.
        """
        self.log_info(f"Downloading dataset from {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f, tqdm(
                total=total,
                unit='iB',
                unit_scale=True,
                desc="Downloading"
        ) as bar:
            for chunk in response.iter_content(self.chunk_size):
                bar.update(len(chunk))
                f.write(chunk)

        self.log_info(f"Download saved to {zip_path} "
                      f"({zip_path.stat().st_size / 1e6:.1f} MB)")

    def _extract_and_flatten(self, zip_path: Path, dest: Path) -> None:
        """
        Extract a ZIP file and flatten any single-level nested folder.

        Replicates the flatten logic from the original download.py that
        moved inner-folder contents up one level when the ZIP contained
        a single root directory.

        Args:
            zip_path: Path to the downloaded ZIP file.
            dest    : Directory to extract into.
        """
        self.log_info(f"Extracting {zip_path.name} to {dest}")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest)

        # --- flatten single nested folder (original download.py logic) ---
        items = [p for p in dest.iterdir() if p != zip_path]
        if len(items) == 1 and items[0].is_dir():
            inner = items[0]
            self.log_info(f"Flattening nested folder: {inner.name}")
            for item in list(inner.iterdir()):
                shutil.move(str(item), str(dest / item.name))
            inner.rmdir()

        # --- optionally remove ZIP ---
        if self.remove_zip and zip_path.exists():
            zip_path.unlink()
            self.log_info("Removed ZIP file after extraction")

    # ------------------------------------------------------------------
    # Public alias (backward compatibility)
    # ------------------------------------------------------------------

    def download(self) -> bool:
        """Public alias for execute()."""
        return self.execute()


__all__ = ['SpatialDownloader']
