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
from src.models.quantum.efficientnet_quantum_head import EfficientNetQuantumHead
from src.models.quantum.qnn_gene_predictor import QNNGenePredictor
from src.models.quantum.qnn_gene_predictor_v2 import QNNGenePredictorV2
from src.models.quantum.qnn_gene_predictor_v4 import QNNGenePredictorV4

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating model components."""
    
    _components: Dict[str, Callable] = {
        'classical_efficientnet':       lambda config: EfficientNetModel(config),
        'auxnet_model':                 lambda config: AuxNetModel(config),
        'quantum_amplitude_embedding':  lambda config: QuantumAmplitudeEmbeddingModel(config),
        'efficientnet_quantum_head':    lambda config: EfficientNetQuantumHead(config),
        'qnn_gene_predictor':           lambda config: QNNGenePredictor(config),
        'qnn_gene_predictor_v2':        lambda config: QNNGenePredictorV2(config),
        'qnn_gene_predictor_v4':        lambda config: QNNGenePredictorV4(config),
    }

    # ...existing code...

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

def _create_efficientnet_quantum_head_model(config: Dict[str, Any]) -> EfficientNetQuantumHead:
    """Create an EfficientNet + Quantum Head hybrid model instance."""
    return EfficientNetQuantumHead(config)

def _create_qnn_gene_predictor(config: Dict[str, Any]) -> QNNGenePredictor:
    """Create a QNN Gene Predictor model instance."""
    return QNNGenePredictor(config)

def _create_qnn_gene_predictor_v2(config: Dict[str, Any]) -> QNNGenePredictorV2:
    """Create a QNN Gene Predictor V2 model instance (improved gradient flow + anti-barren-plateau)."""
    return QNNGenePredictorV2(config)

def _create_qnn_gene_predictor_v4(config: Dict[str, Any]) -> QNNGenePredictorV4:
    """Create a QNN Gene Predictor V4 model instance (quantum kernel alignment + KRR head)."""
    return QNNGenePredictorV4(config)

# Register all model components with the global factory registry
def register_model_factories() -> None:
    """Register all model factories with the global registry."""
    register_factory(ComponentType.MODEL, 'classical_efficientnet',      _create_efficientnet_model)
    register_factory(ComponentType.MODEL, 'auxnet_model',                _create_auxnet_model)
    register_factory(ComponentType.MODEL, 'quantum_amplitude_embedding', _create_quantum_amplitude_embedding_model)
    register_factory(ComponentType.MODEL, 'efficientnet_quantum_head',   _create_efficientnet_quantum_head_model)
    register_factory(ComponentType.MODEL, 'qnn_gene_predictor',          _create_qnn_gene_predictor)
    register_factory(ComponentType.MODEL, 'qnn_gene_predictor_v2',        _create_qnn_gene_predictor_v2)
    register_factory(ComponentType.MODEL, 'qnn_gene_predictor_v4',        _create_qnn_gene_predictor_v4)
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

    from src.models.quantum.efficientnet_quantum_head import EfficientNetQuantumHead
    ModelFactory._components['efficientnet_quantum_head'] = lambda config: EfficientNetQuantumHead(config)
    register_factory(ComponentType.MODEL, 'efficientnet_quantum_head', _create_efficientnet_quantum_head_model)

    from src.models.quantum.qnn_gene_predictor import QNNGenePredictor
    ModelFactory._components['qnn_gene_predictor'] = lambda config: QNNGenePredictor(config)
    register_factory(ComponentType.MODEL, 'qnn_gene_predictor', _create_qnn_gene_predictor)

    from src.models.quantum.qnn_gene_predictor_v2 import QNNGenePredictorV2
    ModelFactory._components['qnn_gene_predictor_v2'] = lambda config: QNNGenePredictorV2(config)
    register_factory(ComponentType.MODEL, 'qnn_gene_predictor_v2', _create_qnn_gene_predictor_v2)

    from src.models.quantum.qnn_gene_predictor_v4 import QNNGenePredictorV4 as _V4
    ModelFactory._components['qnn_gene_predictor_v4'] = lambda config: _V4(config)
    register_factory(ComponentType.MODEL, 'qnn_gene_predictor_v4', _create_qnn_gene_predictor_v4)

    logger.info("Quantum model factories registered: quantum_amplitude_embedding, efficientnet_quantum_head, qnn_gene_predictor, qnn_gene_predictor_v2, qnn_gene_predictor_v4")
except ImportError:
    logger.warning("Quantum components not available - quantum models will not be usable")

__all__ = ['ModelFactory', 'factory', 'register_model_factories', 'QNNGenePredictorV2', 'QNNGenePredictorV4']
