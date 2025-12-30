"""
CSV data extractor for downloading and validating CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from .base_extractor import BaseExtractor


class CSVExtractor(BaseExtractor):
    """Extractor for CSV data sources."""

    def extract(self) -> Dict[str, Any]:
        """Extract CSV data from configured sources.

        Returns:
            Dictionary containing extraction results
        """
        base_url = self.source_config.get("base_url")
        if not base_url:
            raise ValueError(f"No base_url configured for {self.source_name}")

        files_config = self.source_config.get("files", [])
        if not files_config:
            raise ValueError(f"No files configured for {self.source_name}")

        downloaded_files = []
        validation_results = []

        for file_config in files_config:
            file_info = self._extract_single_file(base_url, file_config)
            downloaded_files.append(file_info)
            validation_results.append(self._validate_csv_file(file_info["path"]))

        return {
            "files": downloaded_files,
            "validation": validation_results,
            "total_files": len(downloaded_files),
            "valid_files": sum(validation_results)
        }

    def _extract_single_file(self, base_url: str, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a single CSV file.

        Args:
            base_url: Base URL for the data source
            file_config: Configuration for the specific file

        Returns:
            Dictionary with file information
        """
        filename = file_config["filename"]
        name = file_config["name"]
        description = file_config.get("description", "")

        url = self._get_file_url(base_url, filename)

        self.logger.info(f"Downloading {name}: {description}")
        file_path = self._download_file(url, filename)

        # Get file metadata
        file_size = file_path.stat().st_size
        file_info = {
            "name": name,
            "filename": filename,
            "description": description,
            "url": url,
            "path": str(file_path),
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }

        return file_info

    def _validate_csv_file(self, file_path: str) -> bool:
        """Validate CSV file by attempting to read it.

        Args:
            file_path: Path to CSV file

        Returns:
            True if file is valid CSV
        """
        path = Path(file_path)

        if not self._validate_file(path):
            return False

        try:
            # Try to read the CSV file to ensure it's valid
            df = pd.read_csv(path, nrows=5)  # Only read first 5 rows for validation

            if df.empty:
                self.logger.warning(f"CSV file is empty: {path}")
                return False

            self.logger.info(f"CSV validation passed for {path} (shape: {df.shape})")
            return True

        except Exception as e:
            self.logger.error(f"CSV validation failed for {path}: {e}")
            return False
