"""
Template for creating a custom data processor.

Copy this file and modify to create your own data processing step.
Replace "CustomDataProcessor" with your actual processor name.
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from src.data_pipeline.base_pipeline import BaseDataPipeline

class CustomDataProcessor(BaseDataPipeline):
    """
    Custom data processor template.

    Replace this docstring with a description of what your processor does.
    For example: "Applies custom augmentation to spatial transcriptomics images"
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the custom data processor.

        Args:
            config: Configuration dictionary containing all settings
        """
        super().__init__(config)

        # Get configuration values specific to your processor

        # Input/output directories
        self.input_dir = self.get_config_value('input_dir', 'data/input/')
        self.output_dir = self.get_config_value('processed_dir', 'data/processed/')

        # Processing parameters
        self.parameter1 = self.get_config_value('custom_parameter1', 'default_value')
        self.parameter2 = self.get_config_value('custom_parameter2', 42)

        # Optional: Get nested configuration
        self.advanced_setting = self.get_config_value('processing.custom_advanced', True)

    def execute(self) -> bool:
        """
        Execute the data processing.

        This is the main method that will be called by the pipeline orchestrator.

        Returns:
            True if successful, False otherwise
        """
        self.log_info("Starting custom data processing")

        try:
            # Step 1: Validate input
            if not self._validate_input():
                self.log_error("Input validation failed")
                return False

            # Step 2: Create output directory
            output_path = self._ensure_directory_exists(self.output_dir)

            # Step 3: Load input data
            input_data = self._load_input_data()
            if input_data is None:
                self.log_error("Failed to load input data")
                return False

            # Step 4: Process the data
            processed_data = self._process_data(input_data)

            # Step 5: Save processed data
            success = self._save_output(processed_data, output_path)

            if success:
                self.log_info("Custom data processing completed successfully")
                return True
            else:
                self.log_error("Failed to save processed data")
                return False

        except Exception as e:
            self.log_error(f"Error during processing: {str(e)}")
            return False

    def _validate_input(self) -> bool:
        """
        Validate that input data exists and is in correct format.

        Returns:
            True if valid, False otherwise
        """
        input_path = Path(self.input_dir)

        if not input_path.exists():
            self.log_error(f"Input directory not found: {input_path}")
            return False

        # Add specific validation for your data
        # Example: Check for required files
        # required_files = ['images/', 'metadata.csv']
        # for file_name in required_files:
        #     if not (input_path / file_name).exists():
        #         self.log_error(f"Required file not found: {file_name}")
        #         return False

        return True

    def _load_input_data(self) -> Any:
        """
        Load input data for processing.

        Returns:
            Loaded data or None if failed
        """
        self.log_info(f"Loading data from {self.input_dir}")

        try:
            # Implement your data loading logic here
            # Example: Load images
            # image_files = list(Path(self.input_dir).glob('*.png'))
            # images = [cv2.imread(str(f)) for f in image_files]

            # Example: Load CSV
            # import pandas as pd
            # data = pd.read_csv(Path(self.input_dir) / 'data.csv')

            # For template, return a placeholder
            data = {"placeholder": "Replace with actual data loading"}

            self.log_info("Input data loaded successfully")
            return data

        except Exception as e:
            self.log_error(f"Failed to load input data: {str(e)}")
            return None

    def _process_data(self, input_data: Any) -> Any:
        """
        Main data processing logic.

        Args:
            input_data: Input data to process

        Returns:
            Processed data
        """
        self.log_info("Processing data")

        # Implement your processing logic here
        # Examples:

        # Example 1: Image augmentation
        # processed_images = []
        # for img in input_data:
        #     augmented = self._augment_image(img)
        #     processed_images.append(augmented)

        # Example 2: Feature extraction
        # features = self._extract_features(input_data)

        # Example 3: Data normalization
        # normalized = (input_data - mean) / std

        # For template, just pass through
        processed_data = input_data

        self.log_info("Data processing completed")
        return processed_data

    def _save_output(self, processed_data: Any, output_path: Path) -> bool:
        """
        Save processed data.

        Args:
            processed_data: Data to save
            output_path: Path to output directory

        Returns:
            True if successful, False otherwise
        """
        self.log_info(f"Saving processed data to {output_path}")

        try:
            # Implement your saving logic here
            # Example: Save images
            # for i, img in enumerate(processed_data):
            #     cv2.imwrite(str(output_path / f'processed_{i}.png'), img)

            # Example: Save CSV
            # import pandas as pd
            # df = pd.DataFrame(processed_data)
            # df.to_csv(output_path / 'processed_data.csv', index=False)

            # Example: Save numpy array
            # np.save(output_path / 'processed_data.npy', processed_data)

            # For template, create a marker file
            marker_file = output_path / 'PROCESSING_COMPLETE.txt'
            with open(marker_file, 'w') as f:
                f.write("Processing completed successfully\n")

            self.log_info("Processed data saved successfully")
            return True

        except Exception as e:
            self.log_error(f"Failed to save output: {str(e)}")
            return False


class ImageAugmentationProcessor(BaseDataPipeline):
    """Example: Processor that applies image augmentation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dir = self.get_config_value('input_dir', 'data/input/images/')
        self.output_dir = self.get_config_value('augmented_dir', 'data/augmented/')
        self.augmentation_factor = self.get_config_value('augmentation_factor', 2)
        self.apply_rotation = self.get_config_value('apply_rotation', True)
        self.apply_flip = self.get_config_value('apply_flip', True)

    def execute(self) -> bool:
        """Execute image augmentation."""
        self.log_info(f"Starting image augmentation (factor={self.augmentation_factor})")

        try:
            import cv2
            import numpy as np

            input_path = Path(self.input_dir)
            output_path = self._ensure_directory_exists(self.output_dir)

            # Get all image files
            image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
            self.log_info(f"Found {len(image_files)} images to augment")

            # Augment each image
            for img_file in image_files:
                img = cv2.imread(str(img_file))

                # Original image
                cv2.imwrite(str(output_path / img_file.name), img)

                # Apply augmentations
                for i in range(self.augmentation_factor - 1):
                    augmented = img.copy()

                    if self.apply_rotation:
                        # Random rotation
                        angle = np.random.randint(-30, 30)
                        center = (img.shape[1] // 2, img.shape[0] // 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        augmented = cv2.warpAffine(augmented, matrix, (img.shape[1], img.shape[0]))

                    if self.apply_flip:
                        # Random flip
                        if np.random.rand() > 0.5:
                            augmented = cv2.flip(augmented, 1)  # Horizontal flip

                    # Save augmented image
                    aug_filename = f"{img_file.stem}_aug_{i}{img_file.suffix}"
                    cv2.imwrite(str(output_path / aug_filename), augmented)

            total_images = len(image_files) * self.augmentation_factor
            self.log_info(f"Augmentation completed: {total_images} total images")
            return True

        except Exception as e:
            self.log_error(f"Image augmentation failed: {str(e)}")
            return False


class FeatureExtractionProcessor(BaseDataPipeline):
    """Example: Processor that extracts features from images."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.input_dir = self.get_config_value('input_dir', 'data/input/images/')
        self.output_dir = self.get_config_value('features_dir', 'data/features/')
        self.feature_type = self.get_config_value('feature_type', 'histogram')

    def execute(self) -> bool:
        """Execute feature extraction."""
        self.log_info(f"Extracting features (type={self.feature_type})")

        try:
            import cv2
            import numpy as np
            import pandas as pd

            input_path = Path(self.input_dir)
            output_path = self._ensure_directory_exists(self.output_dir)

            # Get all images
            image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))

            features_list = []
            for img_file in image_files:
                img = cv2.imread(str(img_file))

                # Extract features based on type
                if self.feature_type == 'histogram':
                    # Color histogram
                    features = []
                    for channel in range(3):
                        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
                        features.extend(hist.flatten())
                    features = np.array(features)

                elif self.feature_type == 'texture':
                    # Simple texture features (variance, mean)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    features = [
                        np.mean(gray),
                        np.std(gray),
                        np.median(gray)
                    ]

                else:
                    self.log_warning(f"Unknown feature type: {self.feature_type}")
                    features = []

                features_list.append({
                    'filename': img_file.name,
                    'features': features
                })

            # Save features
            df = pd.DataFrame(features_list)
            df.to_csv(output_path / 'features.csv', index=False)

            self.log_info(f"Extracted features from {len(image_files)} images")
            return True

        except Exception as e:
            self.log_error(f"Feature extraction failed: {str(e)}")
            return False


class DataMerger(BaseDataPipeline):
    """Example: Processor that merges data from multiple sources."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.source_dirs = self.get_config_value('merge_sources', ['data/source1/', 'data/source2/'])
        self.output_dir = self.get_config_value('merged_dir', 'data/merged/')

    def execute(self) -> bool:
        """Execute data merging."""
        self.log_info(f"Merging data from {len(self.source_dirs)} sources")

        try:
            output_path = self._ensure_directory_exists(self.output_dir)

            # Merge files from all sources
            import shutil

            merged_count = 0
            for source_dir in self.source_dirs:
                source_path = Path(source_dir)
                if not source_path.exists():
                    self.log_warning(f"Source directory not found: {source_dir}")
                    continue

                # Copy all files
                for file in source_path.rglob('*'):
                    if file.is_file():
                        relative_path = file.relative_to(source_path)
                        dest_file = output_path / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, dest_file)
                        merged_count += 1

            self.log_info(f"Merged {merged_count} files from {len(self.source_dirs)} sources")
            return True

        except Exception as e:
            self.log_error(f"Data merging failed: {str(e)}")
            return False


# Export your processor classes
__all__ = [
    'CustomDataProcessor',
    'ImageAugmentationProcessor',
    'FeatureExtractionProcessor',
    'DataMerger'
]
