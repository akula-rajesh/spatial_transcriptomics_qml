"""
Custom metrics for evaluating spatial transcriptomics models.
"""

import logging
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class SpatialMetrics:
    """Collection of metrics for spatial transcriptomics prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.gene_correlation_per_gene = config.get('metrics.gene_correlation_per_gene', True)
        self.spatial_distance_metrics = config.get('metrics.spatial_distance_metrics', False)
        self.top_k_accuracy = config.get('metrics.top_k_accuracy', [5, 10, 20])
        
    def compute_all_metrics(self, predictions: torch.Tensor, 
                          targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute all relevant metrics.
        
        Args:
            predictions: Predicted gene expressions (batch_size, num_genes)
            targets: Target gene expressions (batch_size, num_genes)
            
        Returns:
            Dictionary of metric names and values
        """
        # Convert to numpy for easier computation
        if isinstance(predictions, torch.Tensor):
            preds_np = predictions.detach().cpu().numpy()
        else:
            preds_np = np.array(predictions)
            
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = np.array(targets)
            
        # Basic regression metrics
        metrics = {}
        
        # Mean Squared Error
        metrics['mse'] = float(mean_squared_error(targets_np.flatten(), preds_np.flatten()))
        
        # Root Mean Squared Error
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # Mean Absolute Error
        metrics['mae'] = float(mean_absolute_error(targets_np.flatten(), preds_np.flatten()))
        
        # Mean Absolute Percentage Error
        non_zero_mask = targets_np != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((targets_np[non_zero_mask] - preds_np[non_zero_mask]) / targets_np[non_zero_mask]))
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = 0.0
            
        # Correlation coefficients
        corr_metrics = self._compute_correlations(preds_np, targets_np)
        metrics.update(corr_metrics)
        
        # Gene-wise metrics if requested
        if self.gene_correlation_per_gene:
            gene_metrics = self._compute_gene_wise_metrics(preds_np, targets_np)
            metrics.update(gene_metrics)
            
        # Top-K accuracy metrics
        for k in self.top_k_accuracy:
            topk_metrics = self._compute_topk_accuracy(preds_np, targets_np, k)
            metrics.update(topk_metrics)
            
        # Expression level accuracy
        expr_level_metrics = self._compute_expression_level_accuracy(preds_np, targets_np)
        metrics.update(expr_level_metrics)
        
        return metrics
        
    def _compute_correlations(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute correlation metrics."""
        metrics = {}
        
        # Global Pearson correlation
        valid_mask = np.isfinite(preds) & np.isfinite(targets)
        if np.sum(valid_mask) > 1:
            try:
                pearson_global, _ = pearsonr(targets[valid_mask].flatten(), preds[valid_mask].flatten())
                metrics['pearson_global'] = float(pearson_global)
            except:
                metrics['pearson_global'] = 0.0
                
        # Global Spearman correlation
        if np.sum(valid_mask) > 1:
            try:
                spearman_global, _ = spearmanr(targets[valid_mask].flatten(), preds[valid_mask].flatten())
                metrics['spearman_global'] = float(spearman_global)
            except:
                metrics['spearman_global'] = 0.0
                
        return metrics
        
    def _compute_gene_wise_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute gene-wise metrics."""
        metrics = {}
        
        if preds.shape[1] == 0:
            return metrics
            
        # Per-gene correlations
        pearson_per_gene = []
        spearman_per_gene = []
        
        for i in range(preds.shape[1]):
            gene_preds = preds[:, i]
            gene_targets = targets[:, i]
            
            # Remove NaN/infinite values
            valid_idx = np.isfinite(gene_preds) & np.isfinite(gene_targets)
            
            if np.sum(valid_idx) > 1:
                try:
                    pearson_r, _ = pearsonr(gene_targets[valid_idx], gene_preds[valid_idx])
                    pearson_per_gene.append(pearson_r)
                except:
                    pass
                    
                try:
                    spearman_r, _ = spearmanr(gene_targets[valid_idx], gene_preds[valid_idx])
                    spearman_per_gene.append(spearman_r)
                except:
                    pass
                    
        if pearson_per_gene:
            metrics['pearson_per_gene_mean'] = float(np.mean(pearson_per_gene))
            metrics['pearson_per_gene_median'] = float(np.median(pearson_per_gene))
            
        if spearman_per_gene:
            metrics['spearman_per_gene_mean'] = float(np.mean(spearman_per_gene))
            metrics['spearman_per_gene_median'] = float(np.median(spearman_per_gene))
            
        return metrics
        
    def _compute_topk_accuracy(self, preds: np.ndarray, targets: np.ndarray, k: int) -> Dict[str, float]:
        """Compute top-K accuracy metrics."""
        metrics = {}
        
        # For each sample, check if true top-K genes are in predicted top-K
        correct_topk = 0
        total_samples = preds.shape[0]
        
        for i in range(total_samples):
            pred_indices = np.argsort(preds[i])[::-1][:k]  # Top K predicted
            target_indices = np.argsort(targets[i])[::-1][:k]  # Top K actual
            
            # Jaccard similarity between top-K sets
            intersection = len(set(pred_indices) & set(target_indices))
            union = len(set(pred_indices) | set(target_indices))
            jaccard = intersection / union if union > 0 else 0
            
            correct_topk += jaccard
            
        metrics[f'top{k}_jaccard'] = float(correct_topk / total_samples) if total_samples > 0 else 0.0
        
        return metrics
        
    def _compute_expression_level_accuracy(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute expression level classification accuracy."""
        metrics = {}
        
        # Define expression levels: 0-low, 1-medium, 2-high
        # Using percentiles to define thresholds
        target_flat = targets.flatten()
        low_thresh = np.percentile(target_flat[target_flat > 0], 33) if np.sum(target_flat > 0) > 0 else 0
        high_thresh = np.percentile(target_flat[target_flat > 0], 67) if np.sum(target_flat > 0) > 0 else 0
        
        def classify_expression(values):
            classes = np.zeros_like(values, dtype=int)
            classes[values > low_thresh] = 1
            classes[values > high_thresh] = 2
            return classes
            
        target_classes = classify_expression(targets)
        pred_classes = classify_expression(preds)
        
        # Accuracy
        accuracy = np.mean(target_classes == pred_classes)
        metrics['expression_level_accuracy'] = float(accuracy)
        
        # Per-class accuracy
        for class_label in [0, 1, 2]:
            class_mask = target_classes == class_label
            if np.sum(class_mask) > 0:
                class_acc = np.mean(pred_classes[class_mask] == target_classes[class_mask])
                metrics[f'expression_level_accuracy_class_{class_label}'] = float(class_acc)
                
        return metrics

class SpatialLoss(nn.Module):
    """Custom loss functions for spatial transcriptomics."""
    
    def __init__(self, loss_type: str = 'mse', **kwargs):
        """
        Initialize custom loss.
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'huber', 'correlation', 'combined')
            **kwargs: Additional arguments for specific losses
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.kwargs = kwargs
        
        if self.loss_type == 'huber':
            self.delta = kwargs.get('delta', 1.0)
        elif self.loss_type == 'combined':
            self.mse_weight = kwargs.get('mse_weight', 0.7)
            self.corr_weight = kwargs.get('corr_weight', 0.3)
            
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Predicted values
            targets: Target values
            
        Returns:
            Computed loss
        """
        if self.loss_type == 'mse':
            return nn.functional.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            return nn.functional.l1_loss(predictions, targets)
        elif self.loss_type == 'huber':
            return nn.functional.huber_loss(predictions, targets, delta=self.delta)
        elif self.loss_type == 'correlation':
            return self._correlation_loss(predictions, targets)
        elif self.loss_type == 'combined':
            mse_loss = nn.functional.mse_loss(predictions, targets)
            corr_loss = self._correlation_loss(predictions, targets)
            return self.mse_weight * mse_loss + self.corr_weight * corr_loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
    def _correlation_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute negative correlation loss."""
        # Compute correlation coefficient
        pred_mean = torch.mean(predictions, dim=1, keepdim=True)
        target_mean = torch.mean(targets, dim=1, keepdim=True)
        
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        # Numerator
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        
        # Denominator
        pred_var = torch.sum(pred_centered ** 2, dim=1)
        target_var = torch.sum(target_centered ** 2, dim=1)
        denominator = torch.sqrt(pred_var * target_var)
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)
        
        # Correlation coefficient (negated because we want to maximize correlation)
        correlation = numerator / denominator
        loss = -torch.mean(correlation)
        
        # Clamp to reasonable range
        return torch.clamp(loss, min=-1.0, max=1.0)

# Factory functions
def create_metrics_calculator(config: Dict[str, Any]) -> SpatialMetrics:
    """Create metrics calculator."""
    return SpatialMetrics(config)
    
def create_custom_loss(config: Dict[str, Any]) -> SpatialLoss:
    """Create custom loss function."""
    loss_type = config.get('training.loss_function', 'mse')
    loss_kwargs = {}
    
    if loss_type == 'huber':
        loss_kwargs['delta'] = config.get('training.huber_delta', 1.0)
    elif loss_type == 'combined':
        loss_kwargs['mse_weight'] = config.get('training.combined_mse_weight', 0.7)
        loss_kwargs['corr_weight'] = config.get('training.combined_corr_weight', 0.3)
        
    return SpatialLoss(loss_type=loss_type, **loss_kwargs)

__all__ = ['SpatialMetrics', 'SpatialLoss', 'create_metrics_calculator', 'create_custom_loss']
