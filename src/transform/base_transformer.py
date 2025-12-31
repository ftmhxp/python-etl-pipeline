"""
Base transformer class providing common functionality for data transformation.
"""

import logging
import time
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from ..config import config


class BaseTransformer(ABC):
    """Abstract base class for data transformers."""

    def __init__(self, transform_name: str, config_section: str):
        """Initialize the transformer.

        Args:
            transform_name: Name of the transformation operation
            config_section: Configuration section for this transformer
        """
        self.transform_name = transform_name
        self.config_section = config_section
        self.logger = logging.getLogger(f"{__name__}.{transform_name}")

        # Load configuration
        self.transform_config = config.get(f"pipeline.transform.{config_section}", {})
        self.pipeline_config = config.get("pipeline.transform", {})

        # Setup directories
        self.raw_data_path = config.raw_data_path
        self.processed_data_path = config.processed_data_path
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Transformation settings
        self.missing_data_threshold = self.pipeline_config.get("missing_data_threshold", 0.5)
        self.outlier_method = self.pipeline_config.get("outlier_method", "tukey")

    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from file.

        Args:
            file_path: Path to data file

        Returns:
            DataFrame containing the data
        """
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.xml':
            # Handle XML files - convert to DataFrame
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Convert XML to list of dictionaries
            data = []
            for child in root:
                data.append({elem.tag: elem.text for elem in child})

            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _save_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to processed data directory.

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.processed_data_path / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as CSV for processed data
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved transformed data to {output_path}")

        return output_path

    def _get_data_quality_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality statistics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality statistics
        """
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': df.isnull().sum().sum(),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'columns_with_nulls': df.isnull().any().sum(),
            'columns_with_nulls_percentage': (df.isnull().any().sum() / len(df.columns)) * 100
        }

        return stats

    def _log_transformation_stats(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                                  operation: str) -> None:
        """Log statistics about the transformation.

        Args:
            df_before: DataFrame before transformation
            df_after: DataFrame after transformation
            operation: Name of the operation performed
        """
        before_stats = self._get_data_quality_stats(df_before)
        after_stats = self._get_data_quality_stats(df_after)

        self.logger.info(f"=== {operation} Statistics ===")
        self.logger.info(f"Rows: {before_stats['total_rows']} -> {after_stats['total_rows']}")
        self.logger.info(f"Columns: {before_stats['total_columns']} -> {after_stats['total_columns']}")
        self.logger.info(f"Missing data: {before_stats['missing_data']} -> {after_stats['missing_data']}")
        self.logger.info(f"Duplicate rows: {before_stats['duplicate_rows']} -> {after_stats['duplicate_rows']}")

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame.

        Args:
            df: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass

    def run(self, input_files: List[Path]) -> Dict[str, Any]:
        """Run the transformation process with error handling.

        Args:
            input_files: List of input file paths to process

        Returns:
            Dictionary containing transformation results
        """
        try:
            self.logger.info(f"Starting transformation: {self.transform_name}")
            start_time = time.time()

            transformed_files = []
            total_input_rows = 0
            total_output_rows = 0

            for input_file in input_files:
                self.logger.info(f"Processing file: {input_file}")

                # Load data
                df = self._load_data(input_file)
                total_input_rows += len(df)

                # Get original stats
                original_stats = self._get_data_quality_stats(df)

                # Apply transformation
                transformed_df = self.transform(df)
                total_output_rows += len(transformed_df)

                # Log transformation stats
                self._log_transformation_stats(df, transformed_df, self.transform_name)

                # Save transformed data
                output_filename = f"transformed_{input_file.stem}.csv"
                output_path = self._save_data(transformed_df, output_filename)

                transformed_files.append({
                    "input_file": str(input_file),
                    "output_file": str(output_path),
                    "original_rows": len(df),
                    "transformed_rows": len(transformed_df),
                    "original_stats": original_stats,
                    "transformed_stats": self._get_data_quality_stats(transformed_df)
                })

            end_time = time.time()
            duration = end_time - start_time

            result = {
                "transform_name": self.transform_name,
                "status": "success",
                "duration_seconds": duration,
                "total_input_files": len(input_files),
                "total_output_files": len(transformed_files),
                "total_input_rows": total_input_rows,
                "total_output_rows": total_output_rows,
                "files": transformed_files,
                "timestamp": time.time()
            }

            self.logger.info(f"Successfully completed transformation '{self.transform_name}' in {duration:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Transformation failed for {self.transform_name}: {e}")
            return {
                "transform_name": self.transform_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
