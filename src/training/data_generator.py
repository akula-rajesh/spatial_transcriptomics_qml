"""
Data generator for training spatial transcriptomics models.
"""

import logging
from typing import Dict, Any, Tuple, Generator, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from skimage import io, transform
import json
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SpatialTranscriptomicsDataset(Dataset):
    """Dataset class for spatial transcriptomics data."""
    
    def __init__(self, hdf5_file: str, gene_list: Optional[list] = None, 
                 transform=None, augmentations=None):
        """
        Initialize the dataset.
        
        Args:
            hdf5_file: Path to HDF5 file containing the data
            gene_list: List of genes to include (None for all)
            transform: Transform to apply to images
            augmentations: Augmentation transforms for training
        """
        self.hdf5_file = hdf5_file
        self.gene_list = gene_list
        self.transform = transform
        self.augmentations = augmentations
        
        # Open HDF5 file
        self.h5_file = h5py.File(hdf5_file, 'r')
        self.image_keys = list(self.h5_file['images'].keys())
        
        # Get available genes
        if gene_list is None:
            self.genes = list(self.h5_file['gene_names'][:])
        else:
            self.genes = gene_list
            
        # Create gene index mapping
        all_genes = list(self.h5_file['gene_names'][:])
        self.gene_indices = [all_genes.index(gene) for gene in self.genes if gene in all_genes]
        
        logger.info(f"Loaded dataset with {len(self.image_keys)} samples and {len(self.genes)} genes")
        
    def __len__(self) -> int:
        return len(self.image_keys)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        key = self.image_keys[idx]
        
        # Load image
        image = self.h5_file['images'][key][:]
        
        # Load gene expression
        gene_expr = self.h5_file['gene_expressions'][key][:]
        
        # Select specific genes if specified
        if self.gene_indices:
            gene_expr = gene_expr[self.gene_indices]
            
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        gene_tensor = torch.from_numpy(gene_expr).float()
        
        # Apply augmentations for training
        if self.augmentations:
            image_tensor = self.augmentations(image_tensor)
            
        # Apply standard transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, gene_tensor
        
    def __del__(self):
        """Clean up HDF5 file handle."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

class DataGenerator:
    """Generator for creating training and validation data loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config.get('data.dir', 'data/'))
        self.batch_size = config.get('training.batch_size', 32)
        self.num_workers = config.get('data.num_workers', 4)
        self.validation_split = config.get('training.validation_split', 0.2)
        self.shuffle = config.get('data.shuffle', True)
        
        # Image preprocessing parameters
        self.image_size = config.get('data.image_size', (224, 224))
        self.normalize = config.get('data.normalize', True)
        
        # Gene expression parameters
        self.gene_list_file = config.get('data.gene_list_file', None)
        self.scale_genes = config.get('data.scale_genes', True)
        
        logger.info("Initialized DataGenerator")
        
    def create_data_loaders(self, data_file: str) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation data loaders.
        
        Args:
            data_file: Path to HDF5 data file
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load gene list if specified
        gene_list = self._load_gene_list() if self.gene_list_file else None
        
        # Create full dataset
        full_dataset = SpatialTranscriptomicsDataset(
            data_file, 
            gene_list=gene_list,
            transform=self._get_transforms()
        )
        
        # Split into train and validation
        train_dataset, val_dataset = self._split_dataset(full_dataset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and True,  # Always shuffle training data
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader
        
    def _load_gene_list(self) -> list:
        """Load gene list from file."""
        gene_list_path = Path(self.gene_list_file)
        
        if not gene_list_path.exists():
            logger.warning(f"Gene list file not found: {gene_list_path}")
            return None
            
        try:
            if gene_list_path.suffix == '.json':
                with open(gene_list_path, 'r') as f:
                    gene_data = json.load(f)
                # Handle different JSON structures
                if isinstance(gene_data, dict):
                    if 'genes' in gene_data:
                        return gene_data['genes']
                    elif 'gene_list' in gene_data:
                        return gene_data['gene_list']
                    else:
                        return list(gene_data.values())[0]  # Assume first value is gene list
                else:
                    return gene_data
            elif gene_list_path.suffix == '.csv':
                df = pd.read_csv(gene_list_path)
                return df.iloc[:, 0].tolist()  # First column
            else:
                # Plain text file with one gene per line
                with open(gene_list_path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading gene list: {str(e)}")
            return None
            
    def _get_transforms(self):
        """Get image transforms."""
        # In a real implementation, this would use torchvision transforms
        # For now, we'll return a simple normalization if requested
        if self.normalize:
            # Simple min-max normalization to [0, 1]
            return lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        else:
            return None
            
    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into training and validation sets.
        
        Args:
            dataset: Full dataset to split
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        dataset_size = len(dataset)
        val_size = int(dataset_size * self.validation_split)
        train_size = dataset_size - val_size
        
        # Simple random split
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # For simplicity, we'll create two new datasets with the same HDF5 file
        # In a more sophisticated implementation, we'd create subset datasets
        gene_list = dataset.gene_list if hasattr(dataset, 'gene_list') else None
        
        train_dataset = SpatialTranscriptomicsDataset(
            dataset.hdf5_file,
            gene_list=gene_list,
            transform=dataset.transform,
            augmentations=self._get_augmentations()
        )
        
        val_dataset = SpatialTranscriptomicsDataset(
            dataset.hdf5_file,
            gene_list=gene_list,
            transform=dataset.transform,
            augmentations=None  # No augmentations for validation
        )
        
        return train_dataset, val_dataset
        
    def _get_augmentations(self):
        """Get data augmentation transforms."""
        # Simplified augmentations
        def augment(image_tensor):
            # Random horizontal flip
            if np.random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                
            # Random vertical flip
            if np.random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-2])
                
            # Random rotation (multiples of 90 degrees)
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                image_tensor = torch.rot90(image_tensor, k=rot_k, dims=[-2, -1])
                
            # Add small amount of noise
            if np.random.random() > 0.7:
                noise = torch.randn_like(image_tensor) * 0.01
                image_tensor = image_tensor + noise
                image_tensor = torch.clamp(image_tensor, 0, 1)
                
            return image_tensor
            
        return augment
        
    def create_test_loader(self, data_file: str) -> DataLoader:
        """
        Create test data loader.
        
        Args:
            data_file: Path to HDF5 test data file
            
        Returns:
            Test data loader
        """
        gene_list = self._load_gene_list() if self.gene_list_file else None
        
        test_dataset = SpatialTranscriptomicsDataset(
            data_file,
            gene_list=gene_list,
            transform=self._get_transforms(),
            augmentations=None
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created test loader with {len(test_dataset)} samples")
        
        return test_loader
        
    def preprocess_and_save_hdf5(self, image_dir: str, coordinate_file: str, 
                                gene_expression_file: str, output_file: str,
                                spot_radius: int = 50) -> None:
        """
        Preprocess raw data and save to HDF5 format.
        
        Args:
            image_dir: Directory containing histology images
            coordinate_file: CSV file with spot coordinates
            gene_expression_file: CSV file with gene expression data
            output_file: Output HDF5 file path
            spot_radius: Radius for extracting spot images
        """
        logger.info("Preprocessing data and saving to HDF5")
        
        # Load spot coordinates
        coords_df = pd.read_csv(coordinate_file, index_col=0)
        logger.info(f"Loaded coordinates for {len(coords_df)} spots")
        
        # Load gene expression data
        gene_df = pd.read_csv(gene_expression_file, index_col=0)
        logger.info(f"Loaded gene expression for {len(gene_df)} spots, {len(gene_df.columns)} genes")
        
        # Scale gene expression if requested
        if self.scale_genes:
            scaler = StandardScaler()
            gene_scaled = scaler.fit_transform(gene_df.T).T
            gene_df_scaled = pd.DataFrame(gene_scaled, index=gene_df.index, columns=gene_df.columns)
        else:
            gene_df_scaled = gene_df
            
        # Open output HDF5 file
        with h5py.File(output_file, 'w') as h5f:
            # Create groups
            img_group = h5f.create_group('images')
            expr_group = h5f.create_group('gene_expressions')
            
            # Save gene names
            gene_names = gene_df_scaled.columns.tolist()
            h5f.create_dataset('gene_names', data=[g.encode('utf-8') for g in gene_names])
            
            # Save spot IDs
            spot_ids = gene_df_scaled.index.tolist()
            h5f.create_dataset('spot_ids', data=[s.encode('utf-8') for s in spot_ids])
            
            # Process each spot
            processed_count = 0
            for spot_id in spot_ids:
                if spot_id not in coords_df.index:
                    logger.warning(f"Spot {spot_id} not found in coordinates")
                    continue
                    
                # Get coordinates
                row, col = coords_df.loc[spot_id, ['row', 'col']]
                
                # Load corresponding image (assuming one image per dataset for simplicity)
                # In practice, there might be multiple images or different naming conventions
                image_files = list(Path(image_dir).glob("*.png")) or list(Path(image_dir).glob("*.jpg"))
                if not image_files:
                    logger.error("No image files found")
                    break
                    
                image_path = image_files[0]
                image = io.imread(str(image_path))
                
                # Extract spot image
                row_int, col_int = int(row), int(col)
                radius = spot_radius
                
                # Ensure coordinates are within bounds
                row_start = max(0, row_int - radius)
                row_end = min(image.shape[0], row_int + radius)
                col_start = max(0, col_int - radius)
                col_end = min(image.shape[1], col_int + radius)
                
                spot_image = image[row_start:row_end, col_start:col_end]
                
                # Convert to CHW format if needed
                if spot_image.ndim == 3:
                    spot_image = np.transpose(spot_image, (2, 0, 1))  # HWC to CHW
                elif spot_image.ndim == 2:
                    spot_image = spot_image[np.newaxis, :, :]  # Add channel dimension
                    
                # Save to HDF5
                img_group.create_dataset(spot_id, data=spot_image, compression='gzip')
                expr_group.create_dataset(spot_id, data=gene_df_scaled.loc[spot_id].values, compression='gzip')
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count} spots")
                    
        logger.info(f"Saved {processed_count} spots to {output_file}")

# Factory function for creating data generators
def create_data_generator(config: Dict[str, Any]) -> DataGenerator:
    """Factory function to create a data generator."""
    return DataGenerator(config)

__all__ = ['SpatialTranscriptomicsDataset', 'DataGenerator', 'create_data_generator']
