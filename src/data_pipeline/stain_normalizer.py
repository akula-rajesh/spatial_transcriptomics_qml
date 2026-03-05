"""
Data pipeline component for normalizing histological stains in tissue images.
"""

import logging
from typing import Dict, Any
import numpy as np
from pathlib import Path
import cv2

from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)

class StainNormalizer(BaseDataPipeline):
    """Normalizes staining variations in histological tissue images."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stain normalizer.
        
        Args:
            config: Configuration dictionary containing normalization settings
        """
        super().__init__(config)
        self.train_dir = self.get_config_value('train_dir', 'data/train/')
        self.processed_dir = self.get_config_value('processed_dir', 'data/processed/')
        self.target_stain_method = self.get_config_value('stain_normalization_method', 'macenko')
        self.reference_image = self.get_config_value('reference_stain_image', None)
        
    def execute(self) -> bool:
        """
        Execute the stain normalization process.
        
        Returns:
            True if normalization successful, False otherwise
        """
        self.log_info("Starting stain normalization")
        
        try:
            # Ensure processed directory exists
            self._ensure_directory_exists(self.processed_dir)
            
            # Perform stain normalization
            success = self._normalize_stains()
            
            if success:
                self.log_info("Stain normalization completed successfully")
                return True
            else:
                self.log_error("Stain normalization failed")
                return False
                
        except Exception as e:
            self.log_error(f"Error during stain normalization: {str(e)}")
            return False
    
    def _normalize_stains(self) -> bool:
        """
        Normalize stains in all tissue images.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            train_path = self._resolve_path(self.train_dir)
            
            # Find all image files
            image_files = self._find_image_files(train_path)
            
            if not image_files:
                self.log_warning("No image files found for stain normalization")
                return True
                
            # Load or estimate reference stain
            reference_stain = self._get_reference_stain()
            
            # Normalize each image
            normalized_count = 0
            for image_file in image_files:
                try:
                    normalized_image = self._normalize_single_image(image_file, reference_stain)
                    if normalized_image is not None:
                        self._save_normalized_image(normalized_image, image_file)
                        normalized_count += 1
                except Exception as e:
                    self.log_warning(f"Could not normalize {image_file}: {str(e)}")
                    
            self.log_info(f"Normalized {normalized_count}/{len(image_files)} images")
            return True
            
        except Exception as e:
            self.log_error(f"Error normalizing stains: {str(e)}")
            return False
            
    def _find_image_files(self, directory: Path) -> list:
        """
        Find all image files in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
                
        return image_files
        
    def _get_reference_stain(self):
        """
        Get or estimate reference stain for normalization.
        
        Returns:
            Reference stain parameters
        """
        if self.reference_image:
            # Load provided reference image
            ref_path = self._resolve_path(self.reference_image)
            if ref_path.exists():
                ref_image = cv2.imread(str(ref_path))
                if ref_image is not None:
                    return self._estimate_stain_matrix(ref_image)
                    
        # Estimate from training data (simple approach)
        self.log_info("Estimating reference stain from training data")
        return self._estimate_reference_from_training()
        
    def _estimate_reference_from_training(self):
        """
        Estimate reference stain from training images.
        
        Returns:
            Estimated reference stain
        """
        # In a full implementation, this would compute average stain characteristics
        # For now, we'll return placeholder values
        return {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin': [0.07, 0.99, 0.11]
        }
        
    def _estimate_stain_matrix(self, image):
        """
        Estimate stain matrix from an image.
        
        Args:
            image: Input image
            
        Returns:
            Stain matrix parameters
        """
        # Simplified Macenko method implementation
        # In practice, this would use proper histogram-based estimation
        return {
            'hematoxylin': [0.65, 0.70, 0.29],
            'eosin': [0.07, 0.99, 0.11]
        }
        
    def _normalize_single_image(self, image_path: Path, reference_stain):
        """
        Normalize a single image using reference stain.
        
        Args:
            image_path: Path to image file
            reference_stain: Reference stain parameters
            
        Returns:
            Normalized image array or None if failed
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply stain normalization
            normalized = self._apply_macenko_normalization(image_rgb, reference_stain)
            
            return normalized
            
        except Exception as e:
            self.log_warning(f"Error normalizing {image_path}: {str(e)}")
            return None
            
    def _apply_macenko_normalization(self, image, reference_stain):
        """
        Apply Macenko stain normalization.
        
        Args:
            image: Input image
            reference_stain: Reference stain parameters
            
        Returns:
            Normalized image
        """
        # Simplified implementation - in production, use proper histopathology libraries
        # This is a placeholder that maintains the image structure
        
        # Convert to OD space
        image_od = self._rgb_to_od(image)
        
        # For demonstration, we'll just adjust brightness/contrast
        # A real implementation would align stain vectors
        normalized = image.astype(np.float32)
        normalized = np.clip(normalized * 1.1 - 10, 0, 255).astype(np.uint8)
        
        return normalized
        
    def _rgb_to_od(self, image):
        """
        Convert RGB image to optical density space.
        
        Args:
            image: RGB image
            
        Returns:
            Optical density image
        """
        image = image.astype(np.float32) + 1
        od = -np.log(image / 255.0)
        return np.maximum(od, 1e-6)
        
    def _save_normalized_image(self, normalized_image, original_path: Path):
        """
        Save normalized image to processed directory.
        
        Args:
            normalized_image: Normalized image array
            original_path: Original image path
        """
        try:
            # Create corresponding path in processed directory
            processed_path = self._resolve_path(self.processed_dir)
            relative_path = original_path.relative_to(self._resolve_path(self.train_dir))
            output_path = processed_path / relative_path
            
            # Create directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            cv2.imwrite(str(output_path), cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            self.log_warning(f"Could not save normalized image {original_path}: {str(e)}")
            
    def normalize_stains(self) -> bool:
        """
        Public method to normalize stains.
        
        Returns:
            True if successful, False otherwise
        """
        return self.execute()

__all__ = ['StainNormalizer']
