"""
Data pipeline package for the spatial transcriptomics ML pipeline.

This package contains components for downloading, organizing, and preprocessing
spatial transcriptomics data.
"""

# Import key components for easy access
from .base_pipeline import BaseDataPipeline
from .factory import DataPipelineFactory, factory, register_data_pipeline_factories

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.data_pipeline import *"
__all__ = [
    'BaseDataPipeline',
    'DataPipelineFactory',
    'factory',
    'register_data_pipeline_factories'
]

# Initialize package-level logging
import logging

# Create a logger for the data_pipeline package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Register data pipeline factories
register_data_pipeline_factories()

# Log package initialization
logger.info("Data pipeline package initialized")
