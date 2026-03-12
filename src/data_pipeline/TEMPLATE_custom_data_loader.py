"""
Template for creating a custom data loader.

Copy this file and modify to create your own data loader.
Replace "CustomDataLoader" with your actual loader name.
"""

from typing import Dict, Any
from pathlib import Path
from src.data_pipeline.base_pipeline import BaseDataPipeline

class CustomDataLoader(BaseDataPipeline):
    """
    Custom data loader template.

    Replace this docstring with a description of what your loader does.
    For example: "Downloads spatial transcriptomics data from Custom API v2"
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the custom data loader.

        Args:
            config: Configuration dictionary containing all settings
        """
        super().__init__(config)

        # Get configuration values specific to your loader
        # Use get_config_value() which supports nested keys with dots

        # Example: Get a required parameter (will be None if not in config)
        self.data_source_url = self.get_config_value('custom_data_source_url')

        # Example: Get optional parameter with default value
        self.output_dir = self.get_config_value('custom_output_dir', 'data/custom_input/')

        # Example: Get nested configuration value
        self.api_key = self.get_config_value('data.custom_api_key', None)

        # Example: Get integer configuration
        self.max_retries = self.get_config_value('download_max_retries', 3)

        # Example: Get boolean configuration
        self.verify_ssl = self.get_config_value('verify_ssl', True)

        # Add any other initialization here
        self.timeout = self.get_config_value('download_timeout', 300)

    def execute(self) -> bool:
        """
        Execute the data loading process.

        This is the main method that will be called by the pipeline orchestrator.

        Returns:
            True if successful, False otherwise
        """
        self.log_info("Starting custom data loading")

        try:
            # Step 1: Validate configuration
            if not self._validate_config():
                self.log_error("Configuration validation failed")
                return False

            # Step 2: Create output directory
            output_path = self._ensure_directory_exists(self.output_dir)
            self.log_info(f"Output directory: {output_path}")

            # Step 3: Check if data already exists (optional)
            if self._data_already_exists(output_path):
                self.log_info("Data already exists, skipping download")
                return True

            # Step 4: Download/load your data
            success = self._load_data(output_path)

            if success:
                self.log_info("Custom data loading completed successfully")
                return True
            else:
                self.log_error("Custom data loading failed")
                return False

        except Exception as e:
            self.log_error(f"Error during data loading: {str(e)}")
            return False

    def _validate_config(self) -> bool:
        """
        Validate that required configuration values are present.

        Returns:
            True if valid, False otherwise
        """
        # Check required parameters
        if not self.data_source_url:
            self.log_error("Missing required parameter: custom_data_source_url")
            return False

        # Add other validations as needed

        return True

    def _data_already_exists(self, output_path: Path) -> bool:
        """
        Check if data has already been downloaded/loaded.

        Args:
            output_path: Path to output directory

        Returns:
            True if data exists, False otherwise
        """
        # Example: Check for specific files or directories
        expected_files = [
            "data.csv",
            "images/",
            "metadata.json"
        ]

        for file_pattern in expected_files:
            if (output_path / file_pattern).exists():
                return True

        return False

    def _load_data(self, output_path: Path) -> bool:
        """
        Main data loading logic.

        Args:
            output_path: Path to output directory

        Returns:
            True if successful, False otherwise
        """
        self.log_info(f"Loading data from {self.data_source_url}")

        # Implement your data loading logic here
        # Examples:

        # Example 1: Download from HTTP/HTTPS
        # import requests
        # response = requests.get(self.data_source_url, verify=self.verify_ssl)
        # with open(output_path / "data.zip", 'wb') as f:
        #     f.write(response.content)

        # Example 2: Download from S3
        # import boto3
        # s3 = boto3.client('s3')
        # s3.download_file('bucket-name', 'key', str(output_path / 'data.zip'))

        # Example 3: Load from database
        # import sqlalchemy
        # engine = sqlalchemy.create_engine(self.data_source_url)
        # df = pd.read_sql_query("SELECT * FROM data", engine)
        # df.to_csv(output_path / 'data.csv')

        # Example 4: Copy from local path
        # import shutil
        # shutil.copytree(self.data_source_url, output_path)

        # For this template, we'll just create a placeholder file
        try:
            placeholder_file = output_path / "README.txt"
            with open(placeholder_file, 'w') as f:
                f.write("This is a placeholder. Replace with actual data loading logic.\n")

            self.log_info("Data loading completed")
            return True

        except Exception as e:
            self.log_error(f"Data loading failed: {str(e)}")
            return False


# Alternative: If your data loader needs to inherit from a specific class
# or implement a specific interface, you can modify the base class

class CustomAPIDataLoader(BaseDataPipeline):
    """Example: Data loader that fetches from a REST API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_endpoint = self.get_config_value('api_endpoint')
        self.api_key = self.get_config_value('api_key')
        self.output_dir = self.get_config_value('output_dir', 'data/api_input/')

    def execute(self) -> bool:
        """Execute API data fetching."""
        self.log_info(f"Fetching data from API: {self.api_endpoint}")

        try:
            # Import requests (add to requirements.txt if needed)
            import requests

            # Make API request
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(self.api_endpoint, headers=headers)
            response.raise_for_status()

            # Save response
            output_path = self._ensure_directory_exists(self.output_dir)
            output_file = output_path / 'api_data.json'

            with open(output_file, 'w') as f:
                f.write(response.text)

            self.log_info(f"API data saved to {output_file}")
            return True

        except Exception as e:
            self.log_error(f"API data fetch failed: {str(e)}")
            return False


class CustomDatabaseLoader(BaseDataPipeline):
    """Example: Data loader that queries from a database."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_connection_string = self.get_config_value('db_connection_string')
        self.query = self.get_config_value('db_query', 'SELECT * FROM spatial_data')
        self.output_dir = self.get_config_value('output_dir', 'data/db_input/')

    def execute(self) -> bool:
        """Execute database query and save results."""
        self.log_info("Loading data from database")

        try:
            # Import required libraries (add to requirements.txt if needed)
            import pandas as pd
            import sqlalchemy

            # Create database connection
            engine = sqlalchemy.create_engine(self.db_connection_string)

            # Execute query
            df = pd.read_sql_query(self.query, engine)
            self.log_info(f"Loaded {len(df)} rows from database")

            # Save to CSV
            output_path = self._ensure_directory_exists(self.output_dir)
            output_file = output_path / 'database_data.csv'
            df.to_csv(output_file, index=False)

            self.log_info(f"Database data saved to {output_file}")
            return True

        except Exception as e:
            self.log_error(f"Database loading failed: {str(e)}")
            return False


# Export your loader classes
__all__ = [
    'CustomDataLoader',
    'CustomAPIDataLoader',
    'CustomDatabaseLoader'
]
