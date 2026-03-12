"""
Supervised trainer for training models on spatial transcriptomics data.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from src.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class SpatialTranscriptomicsDataset(Dataset):
    """Dataset class for spatial transcriptomics data."""
    
    def __init__(self, images: np.ndarray, gene_expressions: np.ndarray, 
                 transforms=None):
        """
        Initialize the dataset.
        
        Args:
            images: Array of histology images
            gene_expressions: Array of gene expression values
            transforms: Optional transforms to apply to images
        """
        self.images = images
        self.gene_expressions = gene_expressions
        self.transforms = transforms
        
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        gene_expr = self.gene_expressions[idx]
        
        # Convert to tensors
        image_tensor = torch.FloatTensor(image)
        gene_tensor = torch.FloatTensor(gene_expr)
        
        # Apply transforms if specified
        if self.transforms:
            image_tensor = self.transforms(image_tensor)
            
        return image_tensor, gene_tensor

class SupervisedTrainer(BaseTrainer):
    """Supervised trainer for spatial transcriptomics models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize the supervised trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary for the trainer
        """
        super().__init__(model, config)
        
        # Training parameters specific to supervised training
        self.validation_split = self.get_config_value('training.validation_split', 0.2)
        self.early_stopping_patience = self.get_config_value('training.early_stopping_patience', 20)
        self.validation_metric = self.get_config_value('training.validation_metric', 'loss')
        self.save_best_only = self.get_config_value('models.save_best_only', True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Early stopping
        self.early_stopping_counter = 0
        
        # Move model to device
        self.model.to(self.device)
        
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training results
        """
        self.log_info("Starting supervised training")
        
        # Initialize training components
        optimizer = self._initialize_optimizer()
        scheduler = self._initialize_scheduler(optimizer)
        criterion = self._initialize_criterion()
        
        # Prepare data loaders (in a real implementation, this would load actual data)
        train_loader, val_loader = self._prepare_data_loaders()
        
        # Training loop
        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Update metrics tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Logging
            self.log_info(
                f"Epoch [{epoch+1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Train MAE: {train_metrics.get('mae', 0):.6f} - "
                f"Val MAE: {val_metrics.get('mae', 0):.6f}"
            )
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if self.validation_metric == 'loss' else val_metrics.get(self.validation_metric, 0))
                else:
                    scheduler.step()
                    
            # Checkpoint saving
            current_metric = val_loss if self.validation_metric == 'loss' else val_metrics.get(self.validation_metric, 0)
            is_best = self._is_best_model(current_metric)
            
            if is_best:
                self.best_loss = val_loss
                self.best_metric = current_metric
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            # Save checkpoint
            if not self.save_best_only or is_best:
                self._save_checkpoint(epoch, val_loss, current_metric, is_best)
                
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.log_info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        training_time = time.time() - start_time
        self.log_info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'final_epoch': self.current_epoch,
            'best_val_loss': self.best_loss,
            'best_val_metric': self.best_metric,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary containing evaluation results
        """
        self.log_info("Starting model evaluation")
        
        # Prepare test data loader (in a real implementation, this would load test data)
        test_loader = self._prepare_test_loader()
        
        # Initialize criterion
        criterion = self._initialize_criterion()
        
        # Evaluation
        start_time = time.time()
        eval_loss, eval_metrics = self._validate_epoch(test_loader, criterion)
        eval_time = time.time() - start_time
        
        self.log_info(
            f"Evaluation completed - "
            f"Loss: {eval_loss:.6f} - "
            f"MAE: {eval_metrics.get('mae', 0):.6f} - "
            f"Time: {eval_time:.2f}s"
        )
        
        return {
            'loss': eval_loss,
            'metrics': eval_metrics,
            'evaluation_time': eval_time
        }
        
    def _prepare_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # In a real implementation, this would load actual data
        # For demonstration, we'll create synthetic data
        
        # Create synthetic training data
        train_size = 800
        input_channels = self.model.input_channels
        input_height = self.model.input_height
        input_width = self.model.input_width
        output_genes = self.model.output_genes
        
        # Generate random data
        train_images = np.random.rand(train_size, input_channels, input_height, input_width).astype(np.float32)
        train_genes = np.random.rand(train_size, output_genes).astype(np.float32)
        
        # Split into train and validation
        val_size = int(train_size * self.validation_split)
        train_size = train_size - val_size
        
        val_images = train_images[train_size:]
        val_genes = train_genes[train_size:]
        train_images = train_images[:train_size]
        train_genes = train_genes[:train_size]
        
        # Create datasets
        train_dataset = SpatialTranscriptomicsDataset(train_images, train_genes)
        val_dataset = SpatialTranscriptomicsDataset(val_images, val_genes)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        
        self.log_info(f"Prepared data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_loader, val_loader
        
    def _prepare_test_loader(self) -> DataLoader:
        """
        Prepare test data loader.
        
        Returns:
            Test data loader
        """
        # In a real implementation, this would load actual test data
        # For demonstration, we'll create synthetic data
        
        test_size = 200
        input_channels = self.model.input_channels
        input_height = self.model.input_height
        input_width = self.model.input_width
        output_genes = self.model.output_genes
        
        # Generate random test data
        test_images = np.random.rand(test_size, input_channels, input_height, input_width).astype(np.float32)
        test_genes = np.random.rand(test_size, output_genes).astype(np.float32)
        
        # Create dataset and loader
        test_dataset = SpatialTranscriptomicsDataset(test_images, test_genes)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        
        self.log_info(f"Prepared test loader - Test: {len(test_dataset)}")
        
        return test_loader
        
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                     criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss criterion
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}", leave=False)
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, aux_outputs = self.model(images)
            
            # Compute loss
            main_loss = criterion(outputs, targets)
            
            # Add auxiliary loss if available
            # Check if aux_outputs is not None and is a list/tuple with elements
            if aux_outputs is not None and isinstance(aux_outputs, (list, tuple)) and len(aux_outputs) > 0:
                aux_weight = self.get_config_value('training.auxiliary_weight', 0.3)
                aux_loss = 0.0
                for aux_output in aux_outputs:
                    # Ensure aux output matches target size
                    if aux_output.shape[1] != targets.shape[1]:
                        # Select subset of targets for auxiliary output
                        aux_targets = targets[:, :aux_output.shape[1]]
                    else:
                        aux_targets = targets
                    aux_loss += criterion(aux_output, aux_targets)
                aux_loss /= len(aux_outputs)
                loss = main_loss + aux_weight * aux_loss
            else:
                loss = main_loss
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            self._clip_gradients(optimizer)
            
            # Optimization step
            optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += torch.abs(outputs - targets).mean().item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / total_samples:.4f}"
            })

            # Debug mode logging
            if self.debug_mode and batch_idx % 10 == 0:
                self.log_info(
                    f"  Batch [{batch_idx}/{len(train_loader)}] - "
                    f"Loss: {loss.item():.6f} - "
                    f"MAE: {torch.abs(outputs - targets).mean().item():.6f}"
                )

        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        
        return avg_loss, {'mae': avg_mae}
        
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss criterion
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        total_cc = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(images)
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Compute metrics
                batch_size = images.size(0)
                mae = torch.abs(outputs - targets).mean().item()
                rmse = torch.sqrt(((outputs - targets) ** 2).mean()).item()
                
                # Compute correlation coefficient
                cc = self._compute_correlation_coefficient(outputs, targets)
                
                # Update totals
                total_loss += loss.item() * batch_size
                total_mae += mae * batch_size
                total_rmse += rmse * batch_size
                total_cc += cc * batch_size
                total_samples += batch_size
                
        # Calculate average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0
        avg_rmse = total_rmse / total_samples if total_samples > 0 else 0
        avg_cc = total_cc / total_samples if total_samples > 0 else 0
        
        return avg_loss, {
            'mae': avg_mae,
            'rmse': avg_rmse,
            'correlation_coefficient': avg_cc
        }
        
    def _compute_correlation_coefficient(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            predictions: Predicted values
            targets: Target values
            
        Returns:
            Correlation coefficient
        """
        try:
            # Flatten tensors
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            
            # Compute correlation
            pred_mean = pred_flat.mean()
            target_mean = target_flat.mean()
            
            # Numerator
            numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
            
            # Denominator
            pred_var = ((pred_flat - pred_mean) ** 2).sum()
            target_var = ((target_flat - target_mean) ** 2).sum()
            denominator = torch.sqrt(pred_var * target_var)
            
            if denominator == 0:
                return 0.0
                
            cc = (numerator / denominator).item()
            return max(-1.0, min(1.0, cc))  # Clamp to [-1, 1]
        except:
            return 0.0
            
    def _is_best_model(self, current_metric: float) -> bool:
        """
        Check if current model is the best so far.
        
        Args:
            current_metric: Current validation metric
            
        Returns:
            True if best model, False otherwise
        """
        if self.validation_metric == 'loss':
            return current_metric < self.best_loss
        else:
            # For metrics where higher is better (like correlation coefficient)
            return current_metric > self.best_metric
            
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on images.
        
        Args:
            images: Input images tensor
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            predictions, _ = self.model(images)
            return predictions.cpu()

__all__ = ['SupervisedTrainer', 'SpatialTranscriptomicsDataset']
