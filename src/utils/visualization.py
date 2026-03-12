"""
Visualization utilities for the spatial transcriptomics ML pipeline.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class SpatialTranscriptomicsVisualizer:
    """Visualizer for spatial transcriptomics data and model results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), 
                 dpi: int = 150, save_dir: str = 'results/visualizations'):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for plots
            dpi: DPI for plots
            save_dir: Directory to save visualizations
        """
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized SpatialTranscriptomicsVisualizer")
        
    def plot_gene_expression_heatmap(self, gene_expression: np.ndarray, 
                                   gene_names: Optional[List[str]] = None,
                                   spot_positions: Optional[np.ndarray] = None,
                                   title: str = "Gene Expression Heatmap",
                                   save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot gene expression heatmap.
        
        Args:
            gene_expression: Gene expression matrix (spots x genes)
            gene_names: Optional list of gene names
            spot_positions: Optional spot positions (spots x 2)
            title: Plot title
            save_name: Optional filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # If spatial positions provided, create spatial heatmap
        if spot_positions is not None and len(spot_positions) == len(gene_expression):
            self._plot_spatial_heatmap(ax, gene_expression, spot_positions, gene_names)
        else:
            # Standard heatmap
            im = ax.imshow(gene_expression.T, aspect='auto', cmap='viridis')
            ax.set_xlabel('Spots')
            ax.set_ylabel('Genes')
            plt.colorbar(im, ax=ax, label='Expression Level')
            
        ax.set_title(title)
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
            
        return fig
        
    def _plot_spatial_heatmap(self, ax, gene_expression: np.ndarray, 
                            spot_positions: np.ndarray, 
                            gene_names: Optional[List[str]] = None):
        """Plot spatial gene expression heatmap."""
        # For visualization, show expression of first few genes
        n_genes_show = min(6, gene_expression.shape[1])
        
        for i in range(n_genes_show):
            gene_expr = gene_expression[:, i]
            scatter = ax.scatter(spot_positions[:, 0], spot_positions[:, 1], 
                               c=gene_expr, cmap='viridis', s=20, alpha=0.7)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            plt.colorbar(scatter, ax=ax, label='Expression Level')
            
            gene_label = gene_names[i] if gene_names and i < len(gene_names) else f'Gene {i+1}'
            ax.set_title(f'Spatial Expression: {gene_label}')
            
    def plot_prediction_scatter(self, true_values: np.ndarray, 
                              predicted_values: np.ndarray,
                              title: str = "Predicted vs True Gene Expression",
                              xlabel: str = "True Expression",
                              ylabel: str = "Predicted Expression",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot predicted vs true values scatter plot.
        
        Args:
            true_values: True gene expression values
            predicted_values: Predicted gene expression values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_name: Optional filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Flatten arrays for scatter plot
        true_flat = true_values.flatten()
        pred_flat = predicted_values.flatten()
        
        # Scatter plot
        ax.scatter(true_flat, pred_flat, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Statistics
        correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
        mae = np.mean(np.abs(true_flat - pred_flat))
        rmse = np.sqrt(np.mean((true_flat - pred_flat) ** 2))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nCorrelation: {correlation:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved scatter plot to {save_path}")
            
        return fig
        
    def plot_training_history(self, train_losses: List[float], 
                            val_losses: List[float],
                            train_metrics: Optional[List[float]] = None,
                            val_metrics: Optional[List[float]] = None,
                            metric_name: str = "MAE",
                            title: str = "Training History",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            train_metrics: Training metrics per epoch
            val_metrics: Validation metrics per epoch
            metric_name: Name of the metric being plotted
            title: Plot title
            save_name: Optional filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=3)
        ax1.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot metrics if provided
        if train_metrics is not None and val_metrics is not None:
            ax2.plot(epochs, train_metrics, label=f'Training {metric_name}', marker='o', markersize=3)
            ax2.plot(epochs, val_metrics, label=f'Validation {metric_name}', marker='s', markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(metric_name)
            ax2.set_title(f'Model {metric_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.axis('off')
            
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved training history to {save_path}")
            
        return fig
        
    def plot_gene_similarity_matrix(self, gene_expression: np.ndarray,
                                  gene_names: Optional[List[str]] = None,
                                  title: str = "Gene Similarity Matrix",
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot gene similarity matrix.
        
        Args:
            gene_expression: Gene expression matrix (spots x genes)
            gene_names: Optional list of gene names
            title: Plot title
            save_name: Optional filename to save plot
            
        Returns:
            Matplotlib figure
        """
        # Transpose to get genes x spots
        gene_expr_t = gene_expression.T
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(gene_expr_t)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(title)
        
        # Labels
        if gene_names and len(gene_names) == correlation_matrix.shape[0]:
            ax.set_xticks(range(len(gene_names)))
            ax.set_yticks(range(len(gene_names)))
            ax.set_xticklabels(gene_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(gene_names, fontsize=8)
        else:
            ax.set_xlabel('Genes')
            ax.set_ylabel('Genes')
            
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved similarity matrix to {save_path}")
            
        return fig
        
    def plot_model_architecture(self, model: torch.nn.Module,
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot model architecture overview.
        
        Args:
            model: PyTorch model
            save_name: Optional filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Get model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Layer information (simplified)
        layers_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                params = sum(p.numel() for p in module.parameters())
                layers_info.append((name, type(module).__name__, params))
                
        # Plot layers as boxes
        y_positions = np.arange(len(layers_info))
        widths = [np.log10(info[2] + 1) for info in layers_info]  # Log scale for better visualization
        
        bars = ax.barh(y_positions, widths, height=0.6, alpha=0.7)
        
        # Labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{info[1]}\n({info[0]})" for info in layers_info])
        ax.set_xlabel('Log(Number of Parameters)')
        ax.set_title(f'Model Architecture\nTotal Params: {total_params:,}, Trainable: {trainable_params:,}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add parameter counts
        for i, (bar, info) in enumerate(zip(bars, layers_info)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{info[2]:,}', va='center', ha='left', fontsize=8)
                   
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved model architecture to {save_path}")
            
        return fig
        
    def plot_spatial_clusters(self, spot_positions: np.ndarray,
                            cluster_labels: np.ndarray,
                            title: str = "Spatial Clusters",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot spatial clustering results.
        
        Args:
            spot_positions: Spot positions (spots x 2)
            cluster_labels: Cluster labels for each spot
            title: Plot title
            save_name: Optional filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Scatter plot with cluster colors
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            ax.scatter(spot_positions[mask, 0], spot_positions[mask, 1],
                      c=[colors[i]], label=f'Cluster {label}', s=20, alpha=0.7)
                      
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved clusters plot to {save_path}")
            
        return fig

# Convenience functions
def create_visualizer(figsize: Tuple[int, int] = (12, 8), 
                     dpi: int = 150, 
                     save_dir: str = 'results/visualizations') -> SpatialTranscriptomicsVisualizer:
    """
    Create visualizer instance.
    
    Args:
        figsize: Figure size for plots
        dpi: DPI for plots
        save_dir: Directory to save visualizations
        
    Returns:
        Visualizer instance
    """
    return SpatialTranscriptomicsVisualizer(figsize, dpi, save_dir)

def quick_plot_gene_expression(gene_expression: np.ndarray, 
                              save_dir: str = 'results/quick_plots',
                              prefix: str = 'gene_expression') -> str:
    """
    Quick plot of gene expression data.
    
    Args:
        gene_expression: Gene expression matrix
        save_dir: Directory to save plots
        prefix: Filename prefix
        
    Returns:
        Path to saved plot
    """
    vis = SpatialTranscriptomicsVisualizer(save_dir=save_dir)
    fig = vis.plot_gene_expression_heatmap(gene_expression, 
                                         title="Quick Gene Expression Plot",
                                         save_name=f"{prefix}_heatmap")
    save_path = str(vis.save_dir / f"{prefix}_heatmap.png")
    plt.close(fig)
    return save_path

__all__ = [
    'SpatialTranscriptomicsVisualizer',
    'create_visualizer',
    'quick_plot_gene_expression'
]
