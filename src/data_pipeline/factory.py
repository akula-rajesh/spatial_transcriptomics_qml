"""
Factory for creating data pipeline components with auto-discovery.
"""

import logging
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Callable, Any, Type

from src.core.factory_registry import register_factory, ComponentType
from src.data_pipeline.base_pipeline import BaseDataPipeline

logger = logging.getLogger(__name__)

class DataPipelineFactory:
    """Factory for creating data pipeline components with auto-discovery."""

    # Registry of available data pipeline components
    _components: Dict[str, Callable] = {}
    _component_classes: Dict[str, Type[BaseDataPipeline]] = {}

    @classmethod
    def register_component(cls, name: str, constructor: Callable = None, component_class: Type[BaseDataPipeline] = None) -> None:
        """
        Register a new data pipeline component.
        
        Args:
            name: Name of the component
            constructor: Constructor function for the component (optional if component_class provided)
            component_class: Component class (optional if constructor provided)
        """
        if constructor:
            cls._components[name] = constructor
        elif component_class:
            cls._components[name] = lambda config: component_class(config)
            cls._component_classes[name] = component_class
        else:
            raise ValueError("Either constructor or component_class must be provided")

        logger.info(f"Registered data pipeline component: {name}")

    @classmethod
    def auto_discover_components(cls) -> None:
        """
        Automatically discover and register all data pipeline components in this package.

        Scans all Python modules in the data_pipeline package (including subdirectories)
        and registers classes that inherit from BaseDataPipeline.
        """
        logger.info("Starting auto-discovery of data pipeline components...")

        # Get the data_pipeline package path
        import src.data_pipeline as data_pipeline_pkg
        package_path = Path(data_pipeline_pkg.__file__).parent

        discovered_count = 0

        # Scan all Python files recursively
        def scan_directory(directory: Path, package_prefix: str):
            nonlocal discovered_count

            # Scan modules in current directory
            for _, module_name, is_pkg in pkgutil.iter_modules([str(directory)]):
                # Skip certain modules
                if module_name in ['factory', 'base_pipeline', '__init__', '__pycache__']:
                    continue

                full_module_name = f'{package_prefix}.{module_name}'

                try:
                    # Import the module
                    module = importlib.import_module(full_module_name)

                    # Find all classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # Check if it's a class that inherits from BaseDataPipeline
                        if (isinstance(attr, type) and
                            issubclass(attr, BaseDataPipeline) and
                            attr is not BaseDataPipeline):

                            # Use _component_name if defined, otherwise convert class name
                            component_name = getattr(attr, '_component_name', None)
                            if component_name is None:
                                # Convert ClassName to component_name format
                                component_name = cls._class_name_to_component_name(attr.__name__)

                            # Register the component (avoid duplicates)
                            if component_name not in cls._components:
                                cls.register_component(component_name, component_class=attr)
                                discovered_count += 1

                except Exception as e:
                    logger.warning(f"Failed to auto-discover components in {full_module_name}: {str(e)}")

                # Recursively scan subdirectories (if it's a package)
                if is_pkg:
                    subdir_path = directory / module_name
                    scan_directory(subdir_path, full_module_name)

        # Start scanning from the main data_pipeline directory
        scan_directory(package_path, 'src.data_pipeline')

        logger.info(f"Auto-discovery completed: {discovered_count} components registered")

    @staticmethod
    def _class_name_to_component_name(class_name: str) -> str:
        """
        Convert a class name to component name format.

        Examples:
            MendeleyDownloader -> mendeley_downloader
            FileOrganizer -> file_organizer
            MyCustomProcessor -> my_custom_processor

        Args:
            class_name: Class name in CamelCase

        Returns:
            Component name in snake_case
        """
        import re
        # Insert underscore before uppercase letters and convert to lowercase
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()

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


# Decorator for manual component registration (optional)
def register_pipeline_component(name: str = None):
    """
    Decorator to register a data pipeline component.

    Usage:
        @register_pipeline_component('my_component')
        class MyComponent(BaseDataPipeline):
            pass

    Or with auto-naming:
        @register_pipeline_component()
        class MyCustomProcessor(BaseDataPipeline):
            pass
        # Registers as: my_custom_processor

    Args:
        name: Optional component name (auto-generated from class name if not provided)
    """
    def decorator(cls):
        component_name = name
        if component_name is None:
            component_name = DataPipelineFactory._class_name_to_component_name(cls.__name__)

        DataPipelineFactory.register_component(component_name, component_class=cls)
        return cls

    return decorator


# Register all data pipeline components with the global factory registry
def register_data_pipeline_factories() -> None:
    """
    Register all data pipeline factories with the global registry.

    Uses auto-discovery to find and register all components automatically.
    """
    logger.info("Registering data pipeline factories...")

    # Auto-discover all components in the data_pipeline package
    DataPipelineFactory.auto_discover_components()

    # Register all discovered components with the global factory registry
    for component_name, component_class in DataPipelineFactory._component_classes.items():
        def create_component(config: Dict[str, Any], cls=component_class) -> BaseDataPipeline:
            """Create component instance."""
            return cls(config)

        register_factory(ComponentType.DATA_PIPELINE, component_name, create_component)

    logger.info(f"Registered {len(DataPipelineFactory._component_classes)} data pipeline factories")


# Automatically register factories when module is imported
register_data_pipeline_factories()

# Make factory available at module level
factory = DataPipelineFactory()

__all__ = ['DataPipelineFactory', 'factory', 'register_data_pipeline_factories', 'register_pipeline_component']
