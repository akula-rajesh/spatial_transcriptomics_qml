"""
Configuration manager for loading and validating YAML configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages loading and validation of configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration data
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        # Check cache first
        if config_path in self.config_cache:
            return self.config_cache[config_path]
            
        # Resolve full path
        if not os.path.isabs(config_path):
            config_path = self.config_dir / config_path
            
        config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        # Load YAML file
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.config_cache[config_path] = config
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading configuration file {config_path}: {e}")
            raise
            
    def load_pipeline_config(self) -> Dict[str, Any]:
        """
        Load the main pipeline configuration.
        
        Returns:
            Dictionary containing pipeline configuration
        """
        return self.load_config("pipeline_config.yaml")
        
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Load a model-specific configuration.
        
        Args:
            model_name: Name of the model configuration to load
            
        Returns:
            Dictionary containing model configuration
        """
        config_file = f"model_configs/{model_name}.yaml"
        return self.load_config(config_file)
        
    def load_hyperparameters(self, hyperparam_set: str = "default") -> Dict[str, Any]:
        """
        Load hyperparameters configuration.
        
        Args:
            hyperparam_set: Name of hyperparameter set (default, optimized, etc.)
            
        Returns:
            Dictionary containing hyperparameters
        """
        config_file = f"hyperparameters/{hyperparam_set}_params.yaml"
        return self.load_config(config_file)
        
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration to override base values
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged, override_config)
        return merged
        
    def get_nested_value(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Get a nested value from a configuration dictionary using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path to the value (e.g., "training.batch_size")
            default: Default value if path not found
            
        Returns:
            Value at the specified path or default value
        """
        keys = path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
            
    def validate_required_fields(self, config: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate that required fields are present in configuration.
        
        Args:
            config: Configuration dictionary
            required_fields: List of required field paths
            
        Returns:
            True if all required fields are present, False otherwise
        """
        missing_fields = []
        for field in required_fields:
            if self.get_nested_value(config, field) is None:
                missing_fields.append(field)
                
        if missing_fields:
            logger.error(f"Missing required configuration fields: {missing_fields}")
            return False
            
        return True
        
    def resolve_paths(self, config: Dict[str, Any], base_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve relative paths in configuration to absolute paths.
        
        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
            
        Returns:
            Configuration with resolved paths
        """
        if base_path is None:
            base_path = os.getcwd()
            
        resolved_config = config.copy()
        
        def resolve_dict_paths(d, parent_key=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    resolve_dict_paths(value, f"{parent_key}.{key}" if parent_key else key)
                elif isinstance(value, str) and ('dir' in key.lower() or 'path' in key.lower() or 'file' in key.lower()):
                    # If it looks like a path and isn't already absolute, resolve it
                    if not os.path.isabs(value):
                        d[key] = os.path.abspath(os.path.join(base_path, value))
                        
        resolve_dict_paths(resolved_config)
        return resolved_config

# Global configuration manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager instance
    """
    return config_manager

def load_main_config() -> Dict[str, Any]:
    """
    Load the main pipeline configuration.
    
    Returns:
        Dictionary containing main configuration
    """
    return config_manager.load_pipeline_config()

def load_model_configuration(model_name: str) -> Dict[str, Any]:
    """
    Load a specific model configuration.
    
    Args:
        model_name: Name of the model to load configuration for
        
    Returns:
        Dictionary containing model configuration
    """
    return config_manager.load_model_config(model_name)

def load_hyperparameter_set(hyperparam_set: str = "default") -> Dict[str, Any]:
    """
    Load a hyperparameter set.
    
    Args:
        hyperparam_set: Name of the hyperparameter set to load
        
    Returns:
        Dictionary containing hyperparameters
    """
    return config_manager.load_hyperparameters(hyperparam_set)
