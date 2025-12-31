"""
Transformer orchestrator for coordinating all data transformation operations.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import config
from .data_cleaner import DataCleaner
from .feature_engineer import FeatureEngineer
from .data_validator import DataValidator


class TransformerOrchestrator:
    """Orchestrates the complete data transformation pipeline."""

    def __init__(self):
        """Initialize the transformer orchestrator."""
        self.logger = logging.getLogger(f"{__name__}.TransformerOrchestrator")

        # Load configuration
        self.transform_config = config.get("pipeline.transform", {})

        # Setup directories
        self.raw_data_path = config.raw_data_path
        self.processed_data_path = config.processed_data_path
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Initialize transformers
        self.transformers = {
            'cleaner': DataCleaner(),
            'feature_engineer': FeatureEngineer(),
            'validator': DataValidator()
        }

        # Default transformation order
        self.transformation_order = ['cleaner', 'feature_engineer', 'validator']

    def run_transformation(self, input_files: Optional[List[str]] = None,
                         output_dir: Optional[str] = None,
                         transform_order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete transformation pipeline.

        Args:
            input_files: List of input file paths (if None, uses all raw data files)
            output_dir: Output directory for processed files (if None, uses config default)
            transform_order: Order of transformations to apply (if None, uses default)

        Returns:
            Dictionary containing transformation results
        """
        try:
            self.logger.info("Starting transformation pipeline")
            start_time = time.time()

            # Determine input files
            if input_files is None:
                input_files = self._get_input_files()

            # Convert string paths to Path objects
            input_paths = [Path(f) for f in input_files]

            # Validate input files exist
            missing_files = [str(p) for p in input_paths if not p.exists()]
            if missing_files:
                raise FileNotFoundError(f"Input files not found: {missing_files}")

            # Set output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = self.processed_data_path

            # Set transformation order
            if transform_order:
                self._validate_transform_order(transform_order)
                current_order = transform_order
            else:
                current_order = self.transformation_order

            self.logger.info(f"Processing {len(input_paths)} input files")
            self.logger.info(f"Transformation order: {' -> '.join(current_order)}")

            # Run transformations
            all_results = []
            total_input_rows = 0
            total_output_rows = 0

            for input_file in input_paths:
                file_results = self._transform_single_file(input_file, output_path, current_order)
                all_results.append(file_results)
                total_input_rows += file_results['original_rows']
                total_output_rows += file_results['final_rows']

            end_time = time.time()
            duration = end_time - start_time

            # Aggregate results
            overall_result = {
                "status": "success",
                "total_duration_seconds": duration,
                "total_input_files": len(input_paths),
                "total_output_files": len(all_results),
                "total_input_rows": total_input_rows,
                "total_output_rows": total_output_rows,
                "transformations_applied": current_order,
                "file_results": all_results,
                "timestamp": time.time()
            }

            self.logger.info(f"Transformation pipeline completed successfully in {duration:.2f} seconds")
            self._log_transformation_summary(overall_result)

            return overall_result

        except Exception as e:
            self.logger.error(f"Transformation pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }

    def _get_input_files(self) -> List[str]:
        """Get list of input files from raw data directory.

        Returns:
            List of file paths
        """
        supported_extensions = ['.csv', '.json', '.xml']

        input_files = []
        for ext in supported_extensions:
            input_files.extend(self.raw_data_path.glob(f"*{ext}"))

        return [str(f) for f in input_files]

    def _validate_transform_order(self, transform_order: List[str]) -> None:
        """Validate that the transformation order contains valid transformer names.

        Args:
            transform_order: List of transformer names

        Raises:
            ValueError: If any transformer name is invalid
        """
        valid_transformers = set(self.transformers.keys())
        invalid_transformers = [t for t in transform_order if t not in valid_transformers]

        if invalid_transformers:
            raise ValueError(f"Invalid transformers: {invalid_transformers}. Valid options: {list(valid_transformers)}")

    def _transform_single_file(self, input_file: Path, output_dir: Path,
                             transform_order: List[str]) -> Dict[str, Any]:
        """Transform a single file through the pipeline.

        Args:
            input_file: Input file path
            output_dir: Output directory
            transform_order: Order of transformations

        Returns:
            Dictionary with transformation results for this file
        """
        self.logger.info(f"Transforming file: {input_file}")

        # Load initial data
        if input_file.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() == '.json':
            import pandas as pd
            df = pd.read_json(input_file)
        elif input_file.suffix.lower() == '.xml':
            # Handle XML files - convert to DataFrame
            import xml.etree.ElementTree as ET
            tree = ET.parse(input_file)
            root = tree.getroot()

            # Convert XML to list of dictionaries
            data = []
            for child in root:
                data.append({elem.tag: elem.text for elem in child})

            import pandas as pd
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")

        original_rows = len(df)
        current_df = df

        # Apply each transformation in order
        transformation_steps = []

        for transformer_name in transform_order:
            transformer = self.transformers[transformer_name]

            self.logger.info(f"Applying {transformer_name} to {input_file.name}")

            try:
                # Apply transformation
                transformed_df = transformer.transform(current_df)

                # Record transformation step
                step_result = {
                    "transformer": transformer_name,
                    "input_rows": len(current_df),
                    "output_rows": len(transformed_df),
                    "status": "success"
                }

                transformation_steps.append(step_result)
                current_df = transformed_df

            except Exception as e:
                self.logger.error(f"Transformation {transformer_name} failed for {input_file.name}: {e}")

                step_result = {
                    "transformer": transformer_name,
                    "input_rows": len(current_df),
                    "output_rows": len(current_df),  # No change on failure
                    "status": "failed",
                    "error": str(e)
                }

                transformation_steps.append(step_result)
                # Continue with original dataframe for next transformation

        # Save final transformed data
        output_filename = f"transformed_{input_file.stem}.csv"
        output_path = output_dir / output_filename

        try:
            current_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved transformed data to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save transformed data: {e}")
            output_path = None

        # Return results for this file
        file_result = {
            "input_file": str(input_file),
            "output_file": str(output_path) if output_path else None,
            "original_rows": original_rows,
            "final_rows": len(current_df),
            "transformations": transformation_steps,
            "status": "success" if all(step["status"] == "success" for step in transformation_steps) else "partial_success"
        }

        return file_result

    def _log_transformation_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of the transformation results.

        Args:
            results: Transformation results dictionary
        """
        self.logger.info("=== Transformation Pipeline Summary ===")
        self.logger.info(f"Status: {results['status']}")
        self.logger.info(f"Duration: {results['total_duration_seconds']:.2f} seconds")
        self.logger.info(f"Files processed: {results['total_input_files']}")
        self.logger.info(f"Total rows: {results['total_input_rows']} -> {results['total_output_rows']}")
        self.logger.info(f"Transformations applied: {', '.join(results['transformations_applied'])}")

        # Log per-file results
        self.logger.info("Per-file results:")
        for file_result in results['file_results']:
            status = "✓" if file_result['status'] == 'success' else "⚠"
            rows_change = f"{file_result['original_rows']} -> {file_result['final_rows']}"
            self.logger.info(f"  {status} {Path(file_result['input_file']).name}: {rows_change}")

    def get_available_transformers(self) -> List[str]:
        """Get list of available transformer names.

        Returns:
            List of transformer names
        """
        return list(self.transformers.keys())

    def add_transformer(self, name: str, transformer) -> None:
        """Add a custom transformer to the orchestrator.

        Args:
            name: Name of the transformer
            transformer: Transformer instance
        """
        if not hasattr(transformer, 'transform'):
            raise ValueError("Transformer must have a 'transform' method")

        self.transformers[name] = transformer
        self.logger.info(f"Added custom transformer: {name}")
