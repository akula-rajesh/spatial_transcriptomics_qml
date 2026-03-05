"""
Models package for the spatial transcriptomics ML pipeline.

This package contains implementations of various models for predicting
spatial gene expression from histology images.
"""

# Import key components for easy access
from .base_model import BaseModel
from .factory import ModelFactory, factory, register_model_factories

# Import classical models
from .classical.efficientnet_model import EfficientNetModel
from .classical.auxnet_model import AuxNetModel

# Try to import quantum models (may fail if dependencies not installed)
try:
    from .quantum.amplitude_embedding_qml import QuantumAmplitudeEmbeddingModel
    QUANTUM_MODELS_AVAILABLE = True
except ImportError:
    QUANTUM_MODELS_AVAILABLE = False
    from typing import Any as QuantumAmplitudeEmbeddingModel

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.models import *"
__all__ = [
    'BaseModel',
    'ModelFactory',
    'factory',
    'register_model_factories',
    'EfficientNetModel',
    'AuxNetModel',
    'QuantumAmplitudeEmbeddingModel'
]

# Initialize package-level logging
import logging

# Create a logger for the models package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Register model factories
register_model_factories()

# Log package initialization
if QUANTUM_MODELS_AVAILABLE:
    logger.info("Models package initialized with quantum support")
else:
    logger.info("Models package initialized (quantum models not available)")

# Cleanup
del logging
