"""
Example: Self-registering data loader using decorator.

This file demonstrates how to create a new data loader that
automatically registers itself without modifying factory.py.
"""

from typing import Dict, Any
from pathlib import Path
from src.data_pipeline.base_pipeline import BaseDataPipeline
from src.data_pipeline.factory import register_pipeline_component


# Option 1: Auto-registration with custom name
@register_pipeline_component('example_api_loader')
class ExampleAPILoader(BaseDataPipeline):
    """
    Example API data loader with explicit registration name.

    This component is automatically registered as 'example_api_loader'.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = self.get_config_value('example_api_url')
        self.api_key = self.get_config_value('example_api_key')
        self.output_dir = self.get_config_value('output_dir', 'data/example/')

    def execute(self) -> bool:
        """Execute API data loading."""
        self.log_info(f"Loading data from API: {self.api_url}")

        try:
            # Your API loading logic here
            output_path = self._ensure_directory_exists(self.output_dir)

            # Example: Make API request
            # import requests
            # response = requests.get(self.api_url, headers={'Authorization': f'Bearer {self.api_key}'})
            # data = response.json()

            # Save data
            # with open(output_path / 'api_data.json', 'w') as f:
            #     json.dump(data, f)

            self.log_info("API data loaded successfully")
            return True

        except Exception as e:
            self.log_error(f"API data loading failed: {str(e)}")
            return False


# Option 2: Auto-registration with auto-generated name
@register_pipeline_component()
class MyCustomDataLoader(BaseDataPipeline):
    """
    Custom data loader with auto-generated registration name.

    This component is automatically registered as 'my_custom_data_loader'
    (class name converted to snake_case).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.source_path = self.get_config_value('custom_source_path')
        self.output_dir = self.get_config_value('output_dir', 'data/custom/')

    def execute(self) -> bool:
        """Execute custom data loading."""
        self.log_info(f"Loading data from: {self.source_path}")

        try:
            # Your loading logic here
            output_path = self._ensure_directory_exists(self.output_dir)

            self.log_info("Custom data loaded successfully")
            return True

        except Exception as e:
            self.log_error(f"Custom data loading failed: {str(e)}")
            return False


# Option 3: No decorator - will be auto-discovered by scanning
class AutoDiscoveredLoader(BaseDataPipeline):
    """
    Data loader that will be auto-discovered by module scanning.

    This component is automatically registered as 'auto_discovered_loader'
    without needing a decorator.
    """

    # Optional: Specify custom registration name
    _component_name = 'my_auto_loader'  # Override auto-generated name

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_source = self.get_config_value('data_source')

    def execute(self) -> bool:
        """Execute auto-discovered loading."""
        self.log_info(f"Auto-discovered loader executing for: {self.data_source}")

        try:
            # Your logic here
            self.log_info("Auto-discovered loading completed")
            return True

        except Exception as e:
            self.log_error(f"Auto-discovered loading failed: {str(e)}")
            return False


__all__ = [
    'ExampleAPILoader',
    'MyCustomDataLoader',
    'AutoDiscoveredLoader'
]
