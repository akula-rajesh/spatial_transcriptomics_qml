"""
Training package for the spatial transcriptomics ML pipeline.

This package contains components for training and evaluating models
for predicting spatial gene expression from histology images.
"""

# Import key components for easy access
from .base_trainer import BaseTrainer
from .supervised_trainer import SupervisedTrainer, SpatialTranscriptomicsDataset
from .factory import TrainerFactory, factory, register_trainer_factories
from .callbacks import (
    Callback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LRReductionCallback,
    PlottingCallback,
    create_callbacks,
    DEFAULT_CALLBACKS
)
from .cross_validator import CrossValidator, run_cross_validation

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.training import *"
__all__ = [
    'BaseTrainer',
    'SupervisedTrainer',
    'SpatialTranscriptomicsDataset',
    'TrainerFactory',
    'factory',
    'register_trainer_factories',
    'Callback',
    'EarlyStoppingCallback',
    'ModelCheckpointCallback',
    'LRReductionCallback',
    'PlottingCallback',
    'create_callbacks',
    'DEFAULT_CALLBACKS',
    'CrossValidator',
    'run_cross_validation'
]

# Initialize package-level logging
import logging

# Create a logger for the training package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Register trainer factories
register_trainer_factories()

# Log package initialization
logger.info("Training package initialized")

# Cleanup
del logging
