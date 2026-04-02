"""
Quantum models package for the spatial transcriptomics ML pipeline.

This package contains quantum machine learning models for predicting
spatial gene expression from histology images.
"""

# Try to import quantum components
try:
    from .amplitude_embedding_qml import QuantumAmplitudeEmbeddingModel
    from .efficientnet_quantum_head import EfficientNetQuantumHead
    from .qnn_gene_predictor import QNNGenePredictor, QNNLayer, FeatureReducer, ClassicalDecoder
    from .quantum_layers import (
        QuantumMeasurementLayer,
        QuantumFeatureEncoder,
        HybridQuantumClassicalLayer,
        QuantumVariationalLayer,
        amplitude_encode,
        compute_expectation_values
    )
    QUANTUM_AVAILABLE = True
except ImportError as e:
    from typing import Any

    class QuantumAmplitudeEmbeddingModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")

    class EfficientNetQuantumHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")

    class QNNGenePredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")

    class QuantumMeasurementLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")
    
    class QuantumFeatureEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")
    
    class HybridQuantumClassicalLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")
    
    class QuantumVariationalLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Quantum models require PennyLane installation: pip install pennylane")
    
    def amplitude_encode(*args, **kwargs):
        raise ImportError("Quantum models require PennyLane installation: pip install pennylane")
        
    def compute_expectation_values(*args, **kwargs):
        raise ImportError("Quantum models require PennyLane installation: pip install pennylane")
        
    QUANTUM_AVAILABLE = False

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.models.quantum import *"
__all__ = [
    'QuantumAmplitudeEmbeddingModel',
    'EfficientNetQuantumHead',
    'QNNGenePredictor',
    'QNNLayer',
    'FeatureReducer',
    'ClassicalDecoder',
    'QuantumMeasurementLayer',
    'QuantumFeatureEncoder',
    'HybridQuantumClassicalLayer',
    'QuantumVariationalLayer',
    'amplitude_encode',
    'compute_expectation_values'
]

# Initialize package-level logging
import logging

# Create a logger for the quantum models package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Log package initialization
if QUANTUM_AVAILABLE:
    logger.info("Quantum models package initialized")
else:
    logger.warning("Quantum models package initialized without quantum dependencies")

# Cleanup imports
del logging
