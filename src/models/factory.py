"""
Factory for creating model components.
"""

import logging
from typing import Dict, Callable, Any

from src.core.factory_registry import register_factory, ComponentType
from src.models.base_model import BaseModel
from src.models.classical.efficientnet_model import EfficientNetModel
from src.models.classical.auxnet_model import AuxNetModel
from src.models.quantum.amplitude_embedding_qml import QuantumAmplitudeEmbeddingModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating model components."""
    
    # Registry of available model components
    _components: Dict[str, Callable] = {
        'classical_efficientnet': lambda config: EfficientNetModel(config),
        'auxnet_model': lambda config: AuxNetModel(config),
        'quantum_amplitude_embedding': lambda config: QuantumAmplitudeEmbeddingModel(config),
    }
    
    @classmethod
    def register_component(cls, name: str, constructor: Callable) -> None:
        """
        Register a new model component.
        
        Args:
            name: Name of the component
            constructor: Constructor function for the component
        """
        cls._components[name] = constructor
        logger.info(f"Registered model component: {name}")
        
    @classmethod
    def create_model(cls, name: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model component.
        
        Args:
            name: Name of the component to create
            config: Configuration dictionary for the component
            
        Returns:
            Created model component
            
        Raises:
            ValueError: If component is not registered
        """
        if name not in cls._components:
            available = list(cls._components.keys())
            raise ValueError(f"Model component '{name}' not registered. Available: {available}")
            
        logger.info(f"Creating model component: {name}")
        return cls._components[name](config)

def _create_efficientnet_model(config: Dict[str, Any]) -> EfficientNetModel:
    """Create an EfficientNet model instance."""
    return EfficientNetModel(config)

def _create_auxnet_model(config: Dict[str, Any]) -> AuxNetModel:
    """Create an AuxNet model instance."""
    return AuxNetModel(config)

def _create_quantum_amplitude_embedding_model(config: Dict[str, Any]) -> QuantumAmplitudeEmbeddingModel:
    """Create a Quantum Amplitude Embedding model instance."""
    return QuantumAmplitudeEmbeddingModel(config)

# Register all model components with the global factory registry
def register_model_factories() -> None:
    """Register all model factories with the global registry."""
    register_factory(ComponentType.MODEL, 'classical_efficientnet', _create_efficientnet_model)
    register_factory(ComponentType.MODEL, 'auxnet_model', _create_auxnet_model)
    register_factory(ComponentType.MODEL, 'quantum_amplitude_embedding', _create_quantum_amplitude_embedding_model)
    logger.info("Registered all model factories")

# Automatically register factories when module is imported
register_model_factories()

# Make factory available at module level
factory = ModelFactory()

# Try to import quantum components (may fail if dependencies not installed)
try:
    from src.models.quantum.amplitude_embedding_qml import QuantumAmplitudeEmbeddingModel
    ModelFactory._components['quantum_amplitude_embedding'] = lambda config: QuantumAmplitudeEmbeddingModel(config)
    register_factory(ComponentType.MODEL, 'quantum_amplitude_embedding', _create_quantum_amplitude_embedding_model)
except ImportError:
    logger.warning("Quantum components not available - quantum models will not be usable")

__all__ = ['ModelFactory', 'factory', 'register_model_factories']
