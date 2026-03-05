"""
Data pipeline component for organizing downloaded files into proper directory structure.
"""

import os
import shutil
import logging
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)

class FileOrganizer(BaseDataPipeline):
    """Organizes downloaded spatial transcriptomics data into structured directories."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file organizer.
        
        Args:
            config: Configuration dictionary containing organization settings
        """
        super().__init__(config)
        self.input_dir = self.get_config_value('input_dir', 'data/input/')
        self.processed_dir = self.get_config_value('processed_dir', 'data/processed/')
        self.train_dir = self.get_config_value('train_dir', 'data/train/')
        self.test_dir = self.get_config_value('test_dir', 'data/test/')
        self.test_patients = self.get_config_value('test_patients', [])
        
    def execute(self) -> bool:
        """
        Execute the file organization process.
        
        Returns:
            True if organization successful, False otherwise
        """
        self.log_info("Starting file organization")
        
        try:
            # Ensure all required directories exist
            self._ensure_directory_exists(self.processed_dir)
            self._ensure_directory_exists(self.train_dir)
            self._ensure_directory_exists(self.test_dir)
            
            # Organize files
            success = self._organize_files()
            
            if success:
                self.log_info("File organization completed successfully")
                return True
            else:
                self.log_error("File organization failed")
                return False
                
        except Exception as e:
            self.log_error(f"Error during file organization: {str(e)}")
            return False
    
    def _organize_files(self) -> bool:
        """
        Organize files into appropriate directories.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            input_path = self._resolve_path(self.input_dir)
            
            # Check if input directory exists
            if not input_path.exists():
                self.log_error(f"Input directory does not exist: {input_path}")
                return False
                
            # Load metadata if available
            metadata = self._load_metadata(input_path)
            
            # Separate patient files
            patient_files = self._find_patient_files(input_path)
            
            # Organize into train/test splits
            train_files, test_files = self._split_train_test(patient_files, metadata)
            
            # Copy files to appropriate directories
            self._copy_files(train_files, self.train_dir)
            self._copy_files(test_files, self.test_dir)
            
            # Process and copy additional files
            self._copy_additional_files(input_path)
            
            return True
            
        except Exception as e:
            self.log_error(f"Error organizing files: {str(e)}")
            return False
            
    def _load_metadata(self, input_path: Path) -> pd.DataFrame:
        """
        Load metadata file if it exists.
        
        Args:
            input_path: Path to input directory
            
        Returns:
            Metadata DataFrame or empty DataFrame
        """
        metadata_file = input_path / "metadata.csv"
        if metadata_file.exists():
            try:
                return pd.read_csv(metadata_file)
            except Exception as e:
                self.log_warning(f"Could not load metadata: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
        
    def _find_patient_files(self, input_path: Path) -> Dict[str, List[Path]]:
        """
        Find all patient-related files in input directory.
        
        Args:
            input_path: Path to input directory
            
        Returns:
            Dictionary mapping patient IDs to their files
        """
        patient_files = {}
        
        # Look for common file patterns
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        for file_path in input_path.rglob('*'):
            if file_path.is_file():
                # Extract patient ID from filename (assuming pattern like BC23450_)
                filename = file_path.name
                patient_id = self._extract_patient_id(filename)
                
                if patient_id:
                    if patient_id not in patient_files:
                        patient_files[patient_id] = []
                    patient_files[patient_id].append(file_path)
                    
        return patient_files
        
    def _extract_patient_id(self, filename: str) -> str:
        """
        Extract patient ID from filename.
        
        Args:
            filename: Name of file
            
        Returns:
            Patient ID or empty string
        """
        # Common patterns for patient IDs in spatial transcriptomics
        import re
        patterns = [
            r'(BC\d+)',  # Pattern like BC23450
            r'([A-Z]{2}\d+)',  # Two letters followed by digits
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
                
        return ""
        
    def _split_train_test(self, patient_files: Dict[str, List[Path]], 
                         metadata: pd.DataFrame) -> tuple:
        """
        Split patient files into train and test sets.
        
        Args:
            patient_files: Dictionary of patient files
            metadata: Metadata DataFrame
            
        Returns:
            Tuple of (train_files, test_files)
        """
        train_files = []
        test_files = []
        
        for patient_id, files in patient_files.items():
            if patient_id in self.test_patients:
                test_files.extend(files)
                self.log_info(f"Assigned patient {patient_id} to test set ({len(files)} files)")
            else:
                train_files.extend(files)
                self.log_info(f"Assigned patient {patient_id} to train set ({len(files)} files)")
                
        self.log_info(f"Train set: {len(train_files)} files, Test set: {len(test_files)} files")
        return train_files, test_files
        
    def _copy_files(self, files: List[Path], destination_dir: str) -> None:
        """
        Copy files to destination directory.
        
        Args:
            files: List of files to copy
            destination_dir: Destination directory
        """
        dest_path = self._ensure_directory_exists(destination_dir)
        
        for file_path in files:
            try:
                # Maintain relative directory structure
                relative_path = file_path.relative_to(self._resolve_path(self.input_dir))
                dest_file_path = dest_path / relative_path
                
                # Create subdirectories if needed
                dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, dest_file_path)
                
            except Exception as e:
                self.log_warning(f"Could not copy {file_path}: {str(e)}")
                
    def _copy_additional_files(self, input_path: Path) -> None:
        """
        Copy additional important files like metadata, coordinates, etc.
        
        Args:
            input_path: Path to input directory
        """
        important_files = [
            "metadata.csv",
            "spot_coordinates.csv",
            "gene_expression.csv",
            "histology.tif",
            "alignment.json"
        ]
        
        # Copy to both train and processed directories
        destinations = [
            self._ensure_directory_exists(self.train_dir),
            self._ensure_directory_exists(self.processed_dir)
        ]
        
        for filename in important_files:
            source_file = input_path / filename
            if source_file.exists():
                for dest_path in destinations:
                    try:
                        shutil.copy2(source_file, dest_path / filename)
                    except Exception as e:
                        self.log_warning(f"Could not copy {filename}: {str(e)}")
                        
    def organize_files(self) -> bool:
        """
        Public method to organize files.
        
        Returns:
            True if successful, False otherwise
        """
        return self.execute()

__all__ = ['FileOrganizer']
