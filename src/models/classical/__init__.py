"""
Classical models package for the spatial transcriptomics ML pipeline.

This package contains classical deep learning models for predicting
spatial gene expression from histology images.
"""

# Import classical model components
from .efficientnet_model import EfficientNetModel
from .auxnet_model import AuxNetModel

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.models.classical import *"
__all__ = [
    'EfficientNetModel',
    'EfficientNet',
    'AuxNet',
    'AuxNetModel'
]

# Initialize package-level logging
import logging

# Create a logger for the classical models package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Log package initialization
logger.info("Classical models package initialized")

# Cleanup imports
del logging
