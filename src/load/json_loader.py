"""
JSON loader for file-based data loading.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from .base_loader import BaseLoader


class JSONLoader(BaseLoader):
    """JSON loader for file-based data operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JSON loader.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # JSON-specific configuration
        self.output_dir = Path(self.config.get('output_dir', 'data/output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, data, destination: str, **kwargs) -> Dict[str, Any]:
        """Load data to JSON file.

        Args:
            data: Data to load (DataFrame, dict, or file path)
            destination: Output file path (without extension)
            **kwargs: Additional arguments:
                - orient: JSON orientation ('records', 'index', 'split', 'table', 'values')
                - indent: JSON indentation

        Returns:
            Dictionary with loading results
        """
        import time
        start_time = time.time()

        try:
            # Validate destination
            if not self.validate_destination(destination):
                raise ValueError(f"Invalid destination: {destination}")

            # Convert data to DataFrame
            df = self._validate_data(data)

            if df.empty:
                self.logger.warning("No data to load")
                return self._create_loading_summary(start_time, 0)

            self.logger.info(f"Loading {len(df)} rows to JSON file: {destination}")

            # Prepare output path
            output_path = self.output_dir / f"{destination}.json"

            # Load data to JSON
            orient = kwargs.get('orient', 'records')
            indent = kwargs.get('indent', 2)

            df.to_json(output_path, orient=orient, indent=indent, date_format='iso')

            rows_loaded = len(df)
            self.logger.info(f"Successfully saved {rows_loaded} rows to {output_path}")

            return self._create_loading_summary(start_time, rows_loaded)

        except Exception as e:
            self.logger.error(f"Failed to load data to JSON: {e}")
            return self._create_loading_summary(start_time, 0, [str(e)])

    def validate_destination(self, destination: str) -> bool:
        """Validate that the destination directory is writable.

        Args:
            destination: Destination identifier

        Returns:
            True if destination is valid, False otherwise
        """
        try:
            # Check if output directory is writable
            test_file = self.output_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
            return True

        except Exception as e:
            self.logger.error(f"Destination validation failed: {e}")
            return False

    def get_supported_formats(self) -> list:
        """Get list of supported data formats.

        Returns:
            List of supported format strings
        """
        return ['dataframe', 'dict', 'csv', 'json', 'parquet']
