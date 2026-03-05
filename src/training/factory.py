"""
Factory for creating trainer components.
"""

import logging
from typing import Dict, Callable, Any

from src.core.factory_registry import register_factory, ComponentType
from src.training.base_trainer import BaseTrainer
from src.training.supervised_trainer import SupervisedTrainer

logger = logging.getLogger(__name__)

class TrainerFactory:
    """Factory for creating trainer components."""
    
    # Registry of available trainer components
    _components: Dict[str, Callable] = {
        'supervised_trainer': lambda model, config: SupervisedTrainer(model, config),
    }
    
    @classmethod
    def register_component(cls, name: str, constructor: Callable) -> None:
        """
        Register a new trainer component.
        
        Args:
            name: Name of the component
            constructor: Constructor function for the component
        """
        cls._components[name] = constructor
        logger.info(f"Registered trainer component: {name}")
        
    @classmethod
    def create_trainer(cls, name: str, model: Any, config: Dict[str, Any]) -> BaseTrainer:
        """
        Create a trainer component.
        
        Args:
            name: Name of the component to create
            model: Model to train
            config: Configuration dictionary for the component
            
        Returns:
            Created trainer component
            
        Raises:
            ValueError: If component is not registered
        """
        if name not in cls._components:
            available = list(cls._components.keys())
            raise ValueError(f"Trainer component '{name}' not registered. Available: {available}")
            
        logger.info(f"Creating trainer component: {name}")
        return cls._components[name](model, config)

def _create_supervised_trainer(model: Any, config: Dict[str, Any]) -> SupervisedTrainer:
    """Create a supervised trainer instance."""
    return SupervisedTrainer(model, config)

# Register all trainer components with the global factory registry
def register_trainer_factories() -> None:
    """Register all trainer factories with the global registry."""
    register_factory(ComponentType.TRAINER, 'supervised_trainer', _create_supervised_trainer)
    logger.info("Registered all trainer factories")

# Automatically register factories when module is imported
register_trainer_factories()

# Make factory available at module level
factory = TrainerFactory()

__all__ = ['TrainerFactory', 'factory', 'register_trainer_factories']
