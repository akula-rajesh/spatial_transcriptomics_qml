"""
Utility package for the spatial transcriptomics ML pipeline.

This package contains various utility functions and classes for
experiment tracking, data handling, visualization, and other helper functions.
"""

# Import key components for easy access
from .result_tracker import (
    ResultTracker,
    initialize_tracker,
    get_tracker,
    log_metric,
    log_metrics
)

# Additional utility modules (would be implemented separately)
# from .data_utils import *
# from .visualization import *
# from .config_utils import *
# from .logging_utils import *

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.utils import *"
__all__ = [
    'ResultTracker',
    'initialize_tracker',
    'get_tracker',
    'log_metric',
    'log_metrics'
]

# Initialize package-level logging
import logging

# Create a logger for the utils package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Log package initialization
logger.info("Utils package initialized")

# Cleanup imports
del logging
