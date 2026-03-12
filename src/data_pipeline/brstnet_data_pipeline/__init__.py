"""
Data pipeline package for the spatial transcriptomics ML pipeline.
"""

from .spatial_downloader import SpatialDownloader
from .spatial_file_organizer import SpatialFileOrganizer
from .spatial_stain_normalizer import SpatialStainNormalizer
from .spatial_gene_processor import SpatialGeneProcessor
from .spatial_patch_extractor import SpatialPatchExtractor

__version__ = "1.1.0"
__author__  = "Spatial Transcriptomics Research Team"

__all__ = [
    'SpatialDownloader',
    'SpatialFileOrganizer',
    'SpatialStainNormalizer',
    'SpatialGeneProcessor',
    'SpatialPatchExtractor',
]

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

register_data_pipeline_factories()
logger.info("Data pipeline package v%s initialized", __version__)
