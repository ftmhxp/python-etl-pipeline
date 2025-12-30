"""
Base extractor class providing common functionality for data extraction.
"""

import logging
import time
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urljoin

from ..config import config


class BaseExtractor(ABC):
    """Abstract base class for data extractors."""

    def __init__(self, source_name: str, config_section: str):
        """Initialize the extractor.

        Args:
            source_name: Name of the data source
            config_section: Configuration section for this extractor
        """
        self.source_name = source_name
        self.config_section = config_section
        self.logger = logging.getLogger(f"{__name__}.{source_name}")

        # Load configuration
        self.source_config = config.get(f"data_sources.{config_section}", {})
        self.pipeline_config = config.get("pipeline.extract", {})

        # Setup directories
        self.raw_data_path = config.raw_data_path
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        # Extraction settings
        self.retry_attempts = self.pipeline_config.get("retry_attempts", 3)
        self.retry_delay = self.pipeline_config.get("retry_delay", 5)
        self.timeout = self.pipeline_config.get("timeout", 30)

    def _make_request(self, url: str, method: str = "GET", **kwargs) -> requests.Response:
        """Make HTTP request with retry logic.

        Args:
            url: URL to request
            method: HTTP method
            **kwargs: Additional request parameters

        Returns:
            Response object

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                self.logger.info(f"Making {method} request to {url} (attempt {attempt + 1}/{self.retry_attempts})")

                response = requests.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )

                response.raise_for_status()
                self.logger.info(f"Successfully retrieved data from {url}")
                return response

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")

                if attempt < self.retry_attempts - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        # All attempts failed
        error_msg = f"Failed to retrieve data from {url} after {self.retry_attempts} attempts"
        self.logger.error(error_msg)
        raise Exception(error_msg) from last_exception

    def _download_file(self, url: str, filename: str, **kwargs) -> Path:
        """Download file from URL to raw data directory.

        Args:
            url: URL to download from
            filename: Local filename to save as
            **kwargs: Additional request parameters

        Returns:
            Path to downloaded file
        """
        response = self._make_request(url, **kwargs)

        file_path = self.raw_data_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(response.content)

        self.logger.info(f"Downloaded {filename} to {file_path}")
        return file_path

    def _validate_file(self, file_path: Path) -> bool:
        """Validate downloaded file.

        Args:
            file_path: Path to file to validate

        Returns:
            True if file is valid
        """
        if not file_path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return False

        if file_path.stat().st_size == 0:
            self.logger.error(f"File is empty: {file_path}")
            return False

        self.logger.info(f"File validation passed for {file_path}")
        return True

    def _get_file_url(self, base_url: str, filename: str) -> str:
        """Construct full URL from base URL and filename.

        Args:
            base_url: Base URL
            filename: Filename

        Returns:
            Full URL
        """
        if base_url.endswith('/'):
            return f"{base_url}{filename}"
        else:
            return f"{base_url}/{filename}"

    @abstractmethod
    def extract(self) -> Dict[str, Any]:
        """Extract data from source.

        Returns:
            Dictionary containing extraction results and metadata
        """
        pass

    def run(self) -> Dict[str, Any]:
        """Run the extraction process with error handling.

        Returns:
            Dictionary containing extraction results
        """
        try:
            self.logger.info(f"Starting extraction from {self.source_name}")
            start_time = time.time()

            result = self.extract()

            end_time = time.time()
            duration = end_time - start_time

            result.update({
                "source": self.source_name,
                "status": "success",
                "duration_seconds": duration,
                "timestamp": time.time()
            })

            self.logger.info(f"Successfully extracted data from {self.source_name} in {duration:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Extraction failed for {self.source_name}: {e}")
            return {
                "source": self.source_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
