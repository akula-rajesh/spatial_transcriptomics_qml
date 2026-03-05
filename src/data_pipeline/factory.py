"""
Factory for creating data pipeline components.
"""

import logging
from typing import Dict, Callable, Any

from src.core.factory_registry import register_factory, ComponentType
from src.data_pipeline.base_pipeline import BaseDataPipeline
from src.data_pipeline.mendeley_downloader import MendeleyDownloader
from src.data_pipeline.file_organizer import FileOrganizer
from src.data_pipeline.stain_normalizer import StainNormalizer
from src.data_pipeline.spatial_gene_analyzer import SpatialGeneAnalyzer

logger = logging.getLogger(__name__)

class DataPipelineFactory:
    """Factory for creating data pipeline components."""
    
    # Registry of available data pipeline components
    _components: Dict[str, Callable] = {
        'mendeley_downloader': lambda config: MendeleyDownloader(config),
        'file_organizer': lambda config: FileOrganizer(config),
        'stain_normalizer': lambda config: StainNormalizer(config),
        'spatial_gene_analyzer': lambda config: SpatialGeneAnalyzer(config),
    }
    
    @classmethod
    def register_component(cls, name: str, constructor: Callable) -> None:
        """
        Register a new data pipeline component.
        
        Args:
            name: Name of the component
            constructor: Constructor function for the component
        """
        cls._components[name] = constructor
        logger.info(f"Registered data pipeline component: {name}")
        
    @classmethod
    def create_pipeline(cls, name: str, config: Dict[str, Any]) -> BaseDataPipeline:
        """
        Create a data pipeline component.
        
        Args:
            name: Name of the component to create
            config: Configuration dictionary for the component
            
        Returns:
            Created data pipeline component
            
        Raises:
            ValueError: If component is not registered
        """
        if name not in cls._components:
            available = list(cls._components.keys())
            raise ValueError(f"Data pipeline component '{name}' not registered. Available: {available}")
            
        logger.info(f"Creating data pipeline component: {name}")
        return cls._components[name](config)

def _create_mendeley_downloader(config: Dict[str, Any]) -> MendeleyDownloader:
    """Create a Mendeley downloader instance."""
    return MendeleyDownloader(config)

def _create_file_organizer(config: Dict[str, Any]) -> FileOrganizer:
    """Create a file organizer instance."""
    return FileOrganizer(config)

def _create_stain_normalizer(config: Dict[str, Any]) -> StainNormalizer:
    """Create a stain normalizer instance."""
    return StainNormalizer(config)

def _create_spatial_gene_analyzer(config: Dict[str, Any]) -> SpatialGeneAnalyzer:
    """Create a spatial gene analyzer instance."""
    return SpatialGeneAnalyzer(config)

# Register all data pipeline components with the global factory registry
def register_data_pipeline_factories() -> None:
    """Register all data pipeline factories with the global registry."""
    register_factory(ComponentType.DATA_PIPELINE, 'mendeley_downloader', _create_mendeley_downloader)
    register_factory(ComponentType.DATA_PIPELINE, 'file_organizer', _create_file_organizer)
    register_factory(ComponentType.DATA_PIPELINE, 'stain_normalizer', _create_stain_normalizer)
    register_factory(ComponentType.DATA_PIPELINE, 'spatial_gene_analyzer', _create_spatial_gene_analyzer)
    logger.info("Registered all data pipeline factories")

# Automatically register factories when module is imported
register_data_pipeline_factories()

# Make factory available at module level
factory = DataPipelineFactory()

__all__ = ['DataPipelineFactory', 'factory', 'register_data_pipeline_factories']
