"""
Training package for the spatial transcriptomics ML pipeline.

This package contains components for training and evaluating models
for predicting spatial gene expression from histology images.
"""

# Import key components for easy access
from .base_trainer import BaseTrainer
from .supervised_trainer import SupervisedTrainer, SpatialTrainer, EarlyStopping
from .data_generator import (
    SpatialDataset,
    compute_dataset_normalization,
    compute_image_normalization,
    build_transforms,
    create_dataloaders,
)
# Backward-compat alias — old code that imports SpatialTranscriptomicsDataset still works
SpatialTranscriptomicsDataset = SpatialDataset

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
from .metrics import (
    average_correlation_coefficient,
    average_mae,
    average_rmse,
    compute_all_metrics,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.training import *"
__all__ = [
    # Trainers
    'BaseTrainer',
    'SupervisedTrainer',
    'SpatialTrainer',
    'EarlyStopping',
    # Dataset
    'SpatialDataset',
    'SpatialTranscriptomicsDataset',   # backward compat alias
    'compute_dataset_normalization',
    'compute_image_normalization',
    'build_transforms',
    'create_dataloaders',
    # Metrics
    'average_correlation_coefficient',
    'average_mae',
    'average_rmse',
    'compute_all_metrics',
    # Factory / callbacks
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
    'run_cross_validation',
]

# Initialize package-level logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Register trainer factories
register_trainer_factories()

# Log package initialization
logger.info("Training package initialized")

# Cleanup
del logging
