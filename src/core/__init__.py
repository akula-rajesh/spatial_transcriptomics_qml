"""
Core package for the spatial transcriptomics ML pipeline.

This package contains the fundamental components that orchestrate
the entire machine learning pipeline.
"""

# Import key components for easy access
from .pipeline_orchestrator import (
    PipelineOrchestrator,
    run_pipeline_from_config,
    main as run_pipeline_main
)
from .factory_registry import (
    get_factory_registry,
    register_factory,
    create_component,
    list_available_components,
    ComponentType
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Spatial Transcriptomics Research Team"
__email__ = "research@spatial-transcriptomics.org"

# Define what should be imported with "from src.core import *"
__all__ = [
    'PipelineOrchestrator',
    'run_pipeline_from_config',
    'run_pipeline_main',
    'get_factory_registry',
    'register_factory',
    'create_component',
    'list_available_components',
    'ComponentType'
]

# Initialize package-level logging
import logging

# Create a logger for the core package
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Log package initialization
logger.info("Core package initialized")
