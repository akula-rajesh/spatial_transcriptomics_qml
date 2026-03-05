"""
Data pipeline component for downloading data from Mendeley repository.
"""

import os
import urllib.request
import zipfile
import logging
from typing import Dict, Any
from pathlib import Path

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)

class MendeleyDownloader(BaseDataPipeline):
    """Downloads spatial transcriptomics data from Mendeley repository."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Mendeley downloader.
        
        Args:
            config: Configuration dictionary containing download settings
        """
        super().__init__(config)
        self.url = self.get_config_value('mendeley_url')
        self.input_dir = self.get_config_value('input_dir', 'data/input/')
        self.max_retries = self.get_config_value('download_max_retries', 3)
        self.timeout = self.get_config_value('download_timeout', 300)  # 5 minutes
        
    def execute(self) -> bool:
        """
        Execute the download process.
        
        Returns:
            True if download successful, False otherwise
        """
        self.log_info("Starting Mendeley data download")
        
        try:
            # Ensure input directory exists
            input_path = self._ensure_directory_exists(self.input_dir)
            
            # Check if data already exists
            if self._data_already_downloaded(input_path):
                self.log_info("Data already downloaded, skipping download")
                return True
                
            # Download the data
            success = self._download_data(input_path)
            
            if success:
                self.log_info("Mendeley data download completed successfully")
                return True
            else:
                self.log_error("Mendeley data download failed")
                return False
                
        except Exception as e:
            self.log_error(f"Error during download: {str(e)}")
            return False
    
    def _data_already_downloaded(self, input_path: Path) -> bool:
        """
        Check if data has already been downloaded.
        
        Args:
            input_path: Path to input directory
            
        Returns:
            True if data exists, False otherwise
        """
        # Check for common files that would indicate data is already downloaded
        expected_files = [
            "images/",
            "metadata.csv",
            "spot_coordinates.csv"
        ]
        
        for file_pattern in expected_files:
            if (input_path / file_pattern).exists():
                return True
                
        # Check for zip files
        zip_files = list(input_path.glob("*.zip"))
        return len(zip_files) > 0
        
    def _download_data(self, input_path: Path) -> bool:
        """
        Download data from Mendeley URL.
        
        Args:
            input_path: Path to input directory
            
        Returns:
            True if successful, False otherwise
        """
        if not self.url:
            self.log_error("No Mendeley URL provided in configuration")
            return False
            
        zip_filename = "spatial_transcriptomics_data.zip"
        zip_filepath = input_path / zip_filename
        
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                self.log_info(f"Downloading data from {self.url} (attempt {retry_count + 1})")
                
                # Download with progress
                self._download_with_progress(self.url, str(zip_filepath))
                
                # Verify download
                if zip_filepath.exists() and zip_filepath.stat().st_size > 0:
                    self.log_info(f"Download completed: {zip_filepath}")
                    
                    # Extract the zip file
                    if self._extract_zip(zip_filepath, input_path):
                        # Optionally remove zip file after extraction
                        if self.get_config_value('remove_zip_after_extract', True):
                            zip_filepath.unlink()
                            self.log_info("Removed zip file after extraction")
                            
                        return True
                    else:
                        self.log_error("Failed to extract downloaded zip file")
                        return False
                else:
                    self.log_error("Downloaded file is empty or missing")
                    
            except Exception as e:
                self.log_warning(f"Download attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                
                if retry_count > self.max_retries:
                    self.log_error(f"Max retries exceeded. Download failed.")
                    return False
                    
        return False
        
    def _download_with_progress(self, url: str, filepath: str) -> None:
        """
        Download file with progress reporting.
        
        Args:
            url: URL to download from
            filepath: Local path to save file
        """
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:  # Report every 10%
                self.log_info(f"Download progress: {percent}%")
                
        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        
    def _extract_zip(self, zip_filepath: Path, extract_to: Path) -> bool:
        """
        Extract ZIP file to specified directory.
        
        Args:
            zip_filepath: Path to ZIP file
            extract_to: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.log_info(f"Extracting {zip_filepath.name} to {extract_to}")
            
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
            self.log_info("Extraction completed successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Error extracting ZIP file: {str(e)}")
            return False
            
    def download(self) -> bool:
        """
        Public method to download data.
        
        Returns:
            True if successful, False otherwise
        """
        return self.execute()

# For backward compatibility
MendeleyDataDownloader = MendeleyDownloader

__all__ = ['MendeleyDownloader', 'MendeleyDataDownloader']
