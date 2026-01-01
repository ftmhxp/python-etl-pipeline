"""
Base loader class for COVID-19 ETL pipeline.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base loader.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config or {}

        # Default batch size
        self.batch_size = self.config.get('batch_size', 1000)

        # Default if_exists behavior
        self.if_exists = self.config.get('if_exists', 'replace')

    @abstractmethod
    def load_data(self, data: Union[pd.DataFrame, Dict[str, Any], str],
                  destination: str, **kwargs) -> Dict[str, Any]:
        """Load data to the specified destination.

        Args:
            data: Data to load (DataFrame, dict, or file path)
            destination: Destination identifier (table name, file path, etc.)
            **kwargs: Additional loader-specific arguments

        Returns:
            Dictionary with loading results
        """
        pass

    @abstractmethod
    def validate_destination(self, destination: str) -> bool:
        """Validate that the destination is accessible and ready for loading.

        Args:
            destination: Destination to validate

        Returns:
            True if destination is valid, False otherwise
        """
        pass

    def _validate_data(self, data: Union[pd.DataFrame, Dict[str, Any], str]) -> pd.DataFrame:
        """Validate and convert input data to DataFrame.

        Args:
            data: Input data

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If data format is invalid
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, str):
            file_path = Path(data)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {data}")

            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.json':
                return pd.read_json(file_path)
            elif file_path.suffix.lower() == '.parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _log_loading_progress(self, current_batch: int, total_batches: int,
                             rows_loaded: int, total_rows: int) -> None:
        """Log loading progress.

        Args:
            current_batch: Current batch number
            total_batches: Total number of batches
            rows_loaded: Number of rows loaded so far
            total_rows: Total number of rows to load
        """
        progress = (rows_loaded / total_rows) * 100 if total_rows > 0 else 0
        self.logger.info(f"Loading batch {current_batch}/{total_batches} "
                        f"({rows_loaded}/{total_rows} rows, {progress:.1f}%)")

    def _create_loading_summary(self, start_time: float, rows_loaded: int,
                               errors: List[str] = None) -> Dict[str, Any]:
        """Create a summary of the loading operation.

        Args:
            start_time: Start time of the operation
            rows_loaded: Number of rows successfully loaded
            errors: List of error messages

        Returns:
            Dictionary with loading summary
        """
        duration = time.time() - start_time
        errors = errors or []

        summary = {
            "status": "success" if not errors else "partial_success" if rows_loaded > 0 else "failed",
            "rows_loaded": rows_loaded,
            "duration_seconds": duration,
            "rows_per_second": rows_loaded / duration if duration > 0 else 0,
            "errors": errors,
            "timestamp": time.time()
        }

        # Log summary
        self.logger.info(f"Loading completed: {rows_loaded} rows in {duration:.2f}s "
                        f"({summary['rows_per_second']:.1f} rows/sec)")

        if errors:
            self.logger.warning(f"Encountered {len(errors)} errors during loading")
            for error in errors[:5]:  # Log first 5 errors
                self.logger.warning(f"Error: {error}")
            if len(errors) > 5:
                self.logger.warning(f"... and {len(errors) - 5} more errors")

        return summary

    def _batch_data(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split DataFrame into batches for processing.

        Args:
            df: DataFrame to batch

        Returns:
            List of DataFrame batches
        """
        if len(df) <= self.batch_size:
            return [df]

        batches = []
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i + self.batch_size]
            batches.append(batch)

        return batches

    def get_supported_formats(self) -> List[str]:
        """Get list of supported data formats.

        Returns:
            List of supported format strings
        """
        return ['dataframe', 'dict', 'csv', 'json', 'parquet']

    def get_loader_info(self) -> Dict[str, Any]:
        """Get information about this loader.

        Returns:
            Dictionary with loader information
        """
        return {
            "loader_type": self.__class__.__name__,
            "supported_formats": self.get_supported_formats(),
            "batch_size": self.batch_size,
            "if_exists": self.if_exists
        }
