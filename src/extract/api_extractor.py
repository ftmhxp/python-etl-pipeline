"""
API data extractor for downloading data from REST APIs.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json

from .base_extractor import BaseExtractor


class APIExtractor(BaseExtractor):
    """Extractor for API data sources."""

    def extract(self) -> Dict[str, Any]:
        """Extract data from API endpoint.

        Returns:
            Dictionary containing extraction results
        """
        api_endpoint = self.source_config.get("api_endpoint")
        if not api_endpoint:
            raise ValueError(f"No api_endpoint configured for {self.source_name}")

        description = self.source_config.get("description", "")
        filename = f"{self.source_name}_data.csv"

        self.logger.info(f"Downloading {self.source_name} data: {description}")

        # Download the data
        response = self._make_request(api_endpoint)
        file_path = self.raw_data_path / filename

        # Save the raw data
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Get file metadata
        file_size = file_path.stat().st_size

        # Validate the CSV file
        is_valid = self._validate_csv_file(file_path)

        file_info = {
            "name": self.source_name,
            "filename": filename,
            "description": description,
            "url": api_endpoint,
            "path": str(file_path),
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "valid": is_valid
        }

        return {
            "files": [file_info],
            "validation": [is_valid],
            "total_files": 1,
            "valid_files": 1 if is_valid else 0
        }

    def _validate_csv_file(self, file_path: Path) -> bool:
        """Validate API CSV file by attempting to read it.

        Args:
            file_path: Path to CSV file

        Returns:
            True if file is valid CSV
        """
        if not self._validate_file(file_path):
            return False

        try:
            # Try to read the CSV file to ensure it's valid
            df = pd.read_csv(file_path, nrows=5)  # Only read first 5 rows for validation

            if df.empty:
                self.logger.warning(f"API CSV file is empty: {file_path}")
                return False

            # Check if this looks like HTML instead of CSV
            first_col = str(df.columns[0]).lower()
            if '<html' in first_col or '<!doctype' in first_col:
                self.logger.warning(f"API returned HTML instead of CSV data: {file_path}")
                return False

            # Basic validation: check if we have at least some data columns
            if len(df.columns) < 3:
                self.logger.warning(f"API CSV file has too few columns: {file_path}")
                return False

            self.logger.info(f"API CSV validation passed for {file_path} (shape: {df.shape})")
            return True

        except Exception as e:
            self.logger.error(f"API CSV validation failed for {file_path}: {e}")
            return False
