"""
Base class for all data pipeline components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseDataPipeline(ABC):
    """Abstract base class for all data pipeline components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data pipeline component.
        
        Args:
            config: Configuration dictionary for the component
        """
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")
        
    @abstractmethod
    def execute(self) -> Any:
        """
        Execute the data pipeline component.
        
        Returns:
            Result of the pipeline execution
        """
        pass
        
    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path relative to the project root or as absolute path.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved Path object
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            # Assume relative to project root
            return Path.cwd() / path
            
    def _ensure_directory_exists(self, path: str) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Path to directory
            
        Returns:
            Path object for the directory
        """
        dir_path = self._resolve_path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
        
    def _validate_config(self, required_keys: list) -> None:
        """
        Validate that required configuration keys are present.
        
        Args:
            required_keys: List of required configuration keys
            
        Raises:
            ValueError: If any required key is missing
        """
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys for {self.name}: {missing_keys}")
            
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
            
    def log_info(self, message: str) -> None:
        """Log an info message with component context."""
        logger.info(f"[{self.name}] {message}")
        
    def log_warning(self, message: str) -> None:
        """Log a warning message with component context."""
        logger.warning(f"[{self.name}] {message}")
        
    def log_error(self, message: str) -> None:
        """Log an error message with component context."""
        logger.error(f"[{self.name}] {message}")

# Convenience alias for execute method
BaseDataPipeline.run = BaseDataPipeline.execute

__all__ = ['BaseDataPipeline']
