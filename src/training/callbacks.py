"""
Callbacks for training monitoring and control.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    def __init__(self, name: str):
        """
        Initialize the callback.
        
        Args:
            name: Name of the callback
        """
        self.name = name
        
    @abstractmethod
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]) -> None:
        """
        Called at the end of each epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            logs: Dictionary containing training logs
        """
        pass
        
    @abstractmethod
    def on_train_begin(self, trainer: Any) -> None:
        """
        Called at the beginning of training.
        
        Args:
            trainer: Trainer instance
        """
        pass
        
    @abstractmethod
    def on_train_end(self, trainer: Any) -> None:
        """
        Called at the end of training.
        
        Args:
            trainer: Trainer instance
        """
        pass

class EarlyStoppingCallback(Callback):
    """Early stopping callback to halt training when validation metric stops improving."""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, 
                 min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored quantity to qualify as improvement
            mode: 'min' if lower is better, 'max' if higher is better
        """
        super().__init__('early_stopping')
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
        
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        self.best_value = None
        self.wait_count = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
            
        if self.best_value is None:
            self.best_value = current_value
            return
            
        # Check if improvement is significant
        improved = False
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        # Stop training if patience exceeded
        if self.wait_count >= self.patience:
            trainer.log_info(f"Early stopping at epoch {epoch+1}")
            trainer.early_stop = True
            self.stopped_epoch = epoch
            
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        if self.stopped_epoch > 0:
            trainer.log_info(f"Early stopping triggered after epoch {self.stopped_epoch+1}")

class ModelCheckpointCallback(Callback):
    """Model checkpoint callback to save best models during training."""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 save_best_only: bool = True, mode: str = 'min', 
                 save_weights_only: bool = False):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: If True, only save the best model
            mode: 'min' if lower is better, 'max' if higher is better
            save_weights_only: If True, only save model weights
        """
        super().__init__('model_checkpoint')
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.best_value = None
        self.saved_paths = []
        
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        self.best_value = None
        self.saved_paths = []
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
            
        # Save model
        should_save = False
        if self.save_best_only:
            if self.best_value is None:
                self.best_value = current_value
                should_save = True
            else:
                if (self.mode == 'min' and current_value < self.best_value) or \
                   (self.mode == 'max' and current_value > self.best_value):
                    self.best_value = current_value
                    should_save = True
        else:
            should_save = True
            
        if should_save:
            save_path = self.filepath
            if not self.save_best_only:
                # Include epoch number in filename
                save_path = self.filepath.parent / f"{self.filepath.stem}_epoch_{epoch+1}{self.filepath.suffix}"
                
            try:
                if self.save_weights_only:
                    torch.save(trainer.model.state_dict(), save_path)
                else:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': getattr(trainer, 'optimizer', None),
                        'loss': logs.get('val_loss', logs.get('loss')),
                        'config': getattr(trainer, 'config', {}),
                    }
                    torch.save(checkpoint, save_path)
                    
                self.saved_paths.append(save_path)
                logger.info(f"Saved model checkpoint to {save_path}")
                
                if self.save_best_only:
                    logger.info(f"Best {self.monitor}: {self.best_value:.6f}")
                    
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {str(e)}")
                
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        if self.saved_paths:
            logger.info(f"Saved {len(self.saved_paths)} checkpoint(s)")

class LRReductionCallback(Callback):
    """Learning rate reduction callback for plateau detection."""
    
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.1, 
                 patience: int = 10, min_lr: float = 1e-7, mode: str = 'min'):
        """
        Initialize learning rate reduction callback.
        
        Args:
            monitor: Metric to monitor
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which LR will be reduced
            min_lr: Lower bound on the learning rate
            mode: 'min' if lower is better, 'max' if higher is better
        """
        super().__init__('lr_reduction')
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.best_value = None
        self.wait_count = 0
        
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        self.best_value = None
        self.wait_count = 0
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
            
        if self.best_value is None:
            self.best_value = current_value
            return
            
        # Check if improvement is significant
        improved = False
        if self.mode == 'min':
            improved = current_value < self.best_value
        else:
            improved = current_value > self.best_value
            
        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        # Reduce learning rate if patience exceeded
        if self.wait_count >= self.patience:
            if hasattr(trainer, 'optimizer'):
                old_lr = trainer.optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                if new_lr < old_lr:
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = new_lr
                        
                    logger.info(f"Reduced learning rate from {old_lr:.2e} to {new_lr:.2e}")
                    self.wait_count = 0  # Reset counter
                    
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

class PlottingCallback(Callback):
    """Callback to plot training history during and after training."""
    
    def __init__(self, save_dir: str = 'results/plots/', figsize: tuple = (12, 4)):
        """
        Initialize plotting callback.
        
        Args:
            save_dir: Directory to save plots
            figsize: Figure size for plots
        """
        super().__init__('plotting')
        self.save_dir = Path(save_dir)
        self.figsize = figsize
        self.history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        self.history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        # Store metrics
        self.history['train_loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['train_metric'].append(logs.get('mae', 0))
        self.history['val_metric'].append(logs.get('val_mae', 0))
        
        # Plot every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == trainer.epochs - 1:
            self._plot_history(epoch + 1)
            
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        # Final plot
        if self.history['train_loss']:
            self._plot_history(len(self.history['train_loss']))
            
    def _plot_history(self, current_epoch: int) -> None:
        """Plot training history."""
        try:
            epochs = range(1, current_epoch + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
            # Plot losses
            ax1.plot(epochs, self.history['train_loss'], label='Training Loss')
            ax1.plot(epochs, self.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot metrics
            ax2.plot(epochs, self.history['train_metric'], label='Training MAE')
            ax2.plot(epochs, self.history['val_metric'], label='Validation MAE')
            ax2.set_title('Model Metrics')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            save_path = self.save_dir / f'training_history_epoch_{current_epoch}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved training history plot to {save_path}")
            
        except Exception as e:
            logger.warning(f"Failed to plot training history: {str(e)}")

# Predefined callback configurations
DEFAULT_CALLBACKS = {
    'early_stopping': {
        'class': EarlyStoppingCallback,
        'params': {
            'monitor': 'val_loss',
            'patience': 15,
            'min_delta': 1e-4,
            'mode': 'min'
        }
    },
    'model_checkpoint': {
        'class': ModelCheckpointCallback,
        'params': {
            'filepath': 'results/checkpoints/best_model.pth',
            'monitor': 'val_loss',
            'save_best_only': True,
            'mode': 'min',
            'save_weights_only': False
        }
    },
    'lr_reduction': {
        'class': LRReductionCallback,
        'params': {
            'monitor': 'val_loss',
            'factor': 0.2,
            'patience': 10,
            'min_lr': 1e-8,
            'mode': 'min'
        }
    },
    'plotting': {
        'class': PlottingCallback,
        'params': {
            'save_dir': 'results/plots/',
            'figsize': (12, 4)
        }
    }
}

def create_callbacks(callback_configs: Dict[str, Dict]) -> list:
    """
    Create callbacks from configuration.
    
    Args:
        callback_configs: Dictionary of callback configurations
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    for name, config in callback_configs.items():
        if name in DEFAULT_CALLBACKS:
            callback_class = DEFAULT_CALLBACKS[name]['class']
            default_params = DEFAULT_CALLBACKS[name]['params'].copy()
            default_params.update(config.get('params', {}))
            callbacks.append(callback_class(**default_params))
        else:
            logger.warning(f"Unknown callback: {name}")
            
    return callbacks

__all__ = [
    'Callback',
    'EarlyStoppingCallback',
    'ModelCheckpointCallback',
    'LRReductionCallback',
    'PlottingCallback',
    'create_callbacks',
    'DEFAULT_CALLBACKS'
]
