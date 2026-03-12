"""
Base class for all model components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import torch.nn as nn
import torch

logger = logging.getLogger(__name__)

class BaseModel(ABC, nn.Module):
    """Abstract base class for all model components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model component.
        
        Args:
            config: Configuration dictionary for the model
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")
        
        # Model attributes
        self.input_channels = self.get_config_value('input.channels', 3)
        self.input_height = self.get_config_value('input.height', 224)
        self.input_width = self.get_config_value('input.width', 224)
        self.output_genes = self.get_config_value('output.main_genes', 250)
        
        # Dropout and regularization
        self.dropout_rate = self.get_config_value('architecture.dropout_rate', 0.2)
        
        # Device configuration with Mac GPU (MPS), CUDA, and CPU support
        self.device = self._get_device()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor(s)
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
            
    def _get_device(self) -> torch.device:
        """
        Get the appropriate device (MPS for Mac GPU, CUDA for NVIDIA GPU, or CPU).

        Returns:
            torch.device: The selected device
        """
        # Check if GPU acceleration is enabled in config
        gpu_enabled = self.get_config_value('execution.gpu_enabled', True)
        cuda_enabled = self.get_config_value('execution.cuda_enabled', True)
        mps_enabled = self.get_config_value('execution.mps_enabled', True)

        if not gpu_enabled:
            self.log_info("GPU acceleration disabled in config, using CPU")
            return torch.device('cpu')

        # Prefer CUDA if available and enabled
        if torch.cuda.is_available() and cuda_enabled:
            device = torch.device('cuda')
            self.log_info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device

        # Use MPS (Metal Performance Shaders) for Mac if available and enabled
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and mps_enabled:
            device = torch.device('mps')
            self.log_info("Using Apple Metal Performance Shaders (MPS) for GPU acceleration")
            return device

        # Fallback to CPU
        self.log_info("No GPU available, using CPU")
        return torch.device('cpu')

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
        
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to model's device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor moved to device
        """
        return tensor.to(self.device)
        
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'name': self.name,
            'trainable_parameters': self.count_parameters(),
            'input_shape': [self.input_channels, self.input_height, self.input_width],
            'output_genes': self.output_genes,
            'device': str(self.device),
            'dropout_rate': self.dropout_rate
        }
        
    def save_model(self, filepath: str) -> None:
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save model weights
        """
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'model_info': self.get_model_info()
            }, filepath)
            self.log_info(f"Model saved to {filepath}")
        except Exception as e:
            self.log_error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: str) -> None:
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load model weights from
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.log_info(f"Model loaded from {filepath}")
        except Exception as e:
            self.log_error(f"Error loading model: {str(e)}")
            raise
            
    @property
    def is_cuda(self) -> bool:
        """Check if model is on CUDA."""
        return next(self.parameters()).is_cuda

__all__ = ['BaseModel']
