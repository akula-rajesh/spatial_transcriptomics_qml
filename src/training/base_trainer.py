"""
Base class for all trainer components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for all trainer components."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """
        Initialize the trainer component.
        
        Args:
            model: Model to train
            config: Configuration dictionary for the trainer
        """
        self.model = model
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")
        
        # Training parameters
        self.epochs = self.get_config_value('training.epochs', 100)
        self.batch_size = self.get_config_value('training.batch_size', 32)
        self.learning_rate = self.get_config_value('training.learning_rate', 1e-4)
        self.weight_decay = self.get_config_value('training.weight_decay', 1e-5)
        self.gradient_clip = self.get_config_value('training.gradient_clip', 1.0)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  self.get_config_value('execution.cuda_enabled', True) else 'cpu')
        
        # Results directory
        self.results_dir = Path(self.get_config_value('results.base_dir', 'results/'))
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_metric = float('-inf')
        
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training results
        """
        pass
        
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary containing evaluation results
        """
        pass
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.
        
        Args:
            key: Configuration key (supports dot notation for nested values)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def _validate_config(self, required_keys: list) -> None:
        """
        Validate that required configuration keys are present.
        
        Args:
            required_keys: List of required configuration keys
            
        Raises:
            ValueError: If any required key is missing
        """
        missing_keys = [key for key in required_keys if self.get_config_value(key) is None]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys for {self.name}: {missing_keys}")
            
    def log_info(self, message: str) -> None:
        """Log an info message with component context."""
        logger.info(f"[{self.name}] {message}")
        
    def log_warning(self, message: str) -> None:
        """Log a warning message with component context."""
        logger.warning(f"[{self.name}] {message}")
        
    def log_error(self, message: str) -> None:
        """Log an error message with component context."""
        logger.error(f"[{self.name}] {message}")
        
    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """
        Initialize optimizer based on configuration.
        
        Returns:
            Initialized optimizer
        """
        optimizer_name = self.get_config_value('training.optimizer', 'adam').lower()
        weight_decay = self.get_config_value('training.weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.get_config_value('training.momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        self.log_info(f"Initialized {optimizer_name.upper()} optimizer with lr={self.learning_rate}")
        return optimizer
        
    def _initialize_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """
        Initialize learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Initialized scheduler or None
        """
        scheduler_name = self.get_config_value('training.scheduler', None)
        if not scheduler_name:
            return None
            
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name == 'step':
            step_size = self.get_config_value('training.scheduler_step_size', 30)
            gamma = self.get_config_value('training.scheduler_gamma', 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = self.get_config_value('training.scheduler_T_max', self.epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_name == 'plateau':
            mode = self.get_config_value('training.scheduler_mode', 'min')
            factor = self.get_config_value('training.scheduler_factor', 0.1)
            patience = self.get_config_value('training.scheduler_patience', 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
        else:
            self.log_warning(f"Unsupported scheduler: {scheduler_name}")
            return None
            
        self.log_info(f"Initialized {scheduler_name.upper()} scheduler")
        return scheduler
        
    def _initialize_criterion(self) -> torch.nn.Module:
        """
        Initialize loss criterion based on configuration.
        
        Returns:
            Initialized loss criterion
        """
        criterion_name = self.get_config_value('training.loss_function', 'mse').lower()
        
        if criterion_name == 'mse':
            criterion = torch.nn.MSELoss()
        elif criterion_name == 'mae':
            criterion = torch.nn.L1Loss()
        elif criterion_name == 'smooth_l1':
            beta = self.get_config_value('training.smooth_l1_beta', 1.0)
            criterion = torch.nn.SmoothL1Loss(beta=beta)
        elif criterion_name == 'huber':
            delta = self.get_config_value('training.huber_delta', 1.0)
            criterion = torch.nn.HuberLoss(delta=delta)
        else:
            raise ValueError(f"Unsupported loss function: {criterion_name}")
            
        self.log_info(f"Initialized {criterion_name.upper()} loss criterion")
        return criterion
        
    def _save_checkpoint(self, epoch: int, loss: float, metric: float, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            metric: Current validation metric
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'metric': metric,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.log_info(f"Saved best model checkpoint with loss={loss:.6f}, metric={metric:.6f}")
            
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.log_info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
        
    def _clip_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Clip gradients if gradient clipping is enabled.
        
        Args:
            optimizer: Optimizer with parameters to clip
        """
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get information about the trainer.
        
        Returns:
            Dictionary containing trainer information
        """
        return {
            'name': self.name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'best_metric': self.best_metric
        }

__all__ = ['BaseTrainer']
