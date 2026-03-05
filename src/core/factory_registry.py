"""
Factory registry for managing all component factories in the pipeline.
"""

import logging
from typing import Dict, Type, Callable, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Enumeration of component types in the pipeline."""
    DATA_PIPELINE = "data_pipeline"
    MODEL = "model"
    TRAINER = "trainer"

class FactoryRegistry:
    """Registry for all factories in the pipeline system."""
    
    def __init__(self):
        """Initialize the factory registry."""
        self._factories: Dict[ComponentType, Dict[str, Callable]] = {
            ComponentType.DATA_PIPELINE: {},
            ComponentType.MODEL: {},
            ComponentType.TRAINER: {}
        }
        self._instances: Dict[str, Any] = {}
        
    def register_factory(self, component_type: ComponentType, name: str, factory_func: Callable) -> None:
        """
        Register a factory function for a component.
        
        Args:
            component_type: Type of component (DATA_PIPELINE, MODEL, TRAINER)
            name: Name of the factory
            factory_func: Factory function that creates instances
        """
        if component_type not in self._factories:
            raise ValueError(f"Invalid component type: {component_type}")
            
        logger.info(f"Registering {component_type.value} factory: {name}")
        self._factories[component_type][name] = factory_func
        
    def get_factory(self, component_type: ComponentType, name: str) -> Callable:
        """
        Get a registered factory function.
        
        Args:
            component_type: Type of component
            name: Name of the factory
            
        Returns:
            Factory function
            
        Raises:
            KeyError: If factory is not registered
        """
        if component_type not in self._factories:
            raise KeyError(f"Component type {component_type} not found")
            
        if name not in self._factories[component_type]:
            available = list(self._factories[component_type].keys())
            raise KeyError(f"Factory '{name}' not found for {component_type.value}. Available: {available}")
            
        return self._factories[component_type][name]
        
    def create_instance(self, component_type: ComponentType, name: str, 
                       *args, **kwargs) -> Any:
        """
        Create an instance using a registered factory.
        
        Args:
            component_type: Type of component
            name: Name of the factory
            *args: Positional arguments for factory
            **kwargs: Keyword arguments for factory
            
        Returns:
            Created instance
        """
        factory = self.get_factory(component_type, name)
        logger.info(f"Creating {component_type.value} instance with factory: {name}")
        return factory(*args, **kwargs)
        
    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a pre-created instance.
        
        Args:
            name: Name to register instance under
            instance: Instance to register
        """
        logger.info(f"Registering instance: {name}")
        self._instances[name] = instance
        
    def get_instance(self, name: str) -> Any:
        """
        Get a registered instance.
        
        Args:
            name: Name of the instance
            
        Returns:
            Registered instance
            
        Raises:
            KeyError: If instance is not registered
        """
        if name not in self._instances:
            raise KeyError(f"Instance '{name}' not registered")
        return self._instances[name]
        
    def list_factories(self, component_type: Optional[ComponentType] = None) -> Dict[str, list]:
        """
        List all registered factories.
        
        Args:
            component_type: Optional filter by component type
            
        Returns:
            Dictionary mapping component types to lists of factory names
        """
        if component_type:
            return {component_type.value: list(self._factories[component_type].keys())}
        else:
            return {
                ct.value: list(factories.keys()) 
                for ct, factories in self._factories.items()
            }
            
    def is_registered(self, component_type: ComponentType, name: str) -> bool:
        """
        Check if a factory is registered.
        
        Args:
            component_type: Type of component
            name: Name of the factory
            
        Returns:
            True if registered, False otherwise
        """
        return (
            component_type in self._factories and 
            name in self._factories[component_type]
        )
        
    def unregister_factory(self, component_type: ComponentType, name: str) -> None:
        """
        Unregister a factory.
        
        Args:
            component_type: Type of component
            name: Name of the factory
        """
        if self.is_registered(component_type, name):
            del self._factories[component_type][name]
            logger.info(f"Unregistered {component_type.value} factory: {name}")

# Global factory registry instance
factory_registry = FactoryRegistry()

def get_factory_registry() -> FactoryRegistry:
    """
    Get the global factory registry instance.
    
    Returns:
        FactoryRegistry instance
    """
    return factory_registry

def register_factory(component_type: ComponentType, name: str, factory_func: Callable) -> None:
    """
    Register a factory function globally.
    
    Args:
        component_type: Type of component
        name: Name of the factory
        factory_func: Factory function to register
    """
    factory_registry.register_factory(component_type, name, factory_func)

def create_component(component_type: ComponentType, name: str, *args, **kwargs) -> Any:
    """
    Create a component instance using a registered factory.
    
    Args:
        component_type: Type of component
        name: Name of the factory
        *args: Positional arguments for factory
        **kwargs: Keyword arguments for factory
        
    Returns:
        Created component instance
    """
    return factory_registry.create_instance(component_type, name, *args, **kwargs)

def list_available_components(component_type: Optional[ComponentType] = None) -> Dict[str, list]:
    """
    List all available components.
    
    Args:
        component_type: Optional filter by component type
        
    Returns:
        Dictionary mapping component types to lists of available components
    """
    return factory_registry.list_factories(component_type)
