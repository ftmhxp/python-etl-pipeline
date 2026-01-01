"""
Loader orchestrator for coordinating all data loading operations.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd

from ..config import config
from .sql_loader import SQLLoader
from .data_loader import COVIDDataLoader
from .database_schema import (
    get_table_schema, get_all_table_names, get_table_indexes,
    get_all_indexes, get_foreign_key_constraints
)


class LoaderOrchestrator:
    """Orchestrates the complete data loading pipeline."""

    def __init__(self):
        """Initialize the loader orchestrator."""
        self.logger = logging.getLogger(f"{__name__}.LoaderOrchestrator")

        # Load configuration
        self.load_config = config.get("pipeline.load", {})

        # Initialize loaders
        self.sql_loader = SQLLoader(config.get('database', {}))
        self.data_loader = COVIDDataLoader(self.sql_loader)

        # Setup data paths
        self.processed_data_path = config.processed_data_path

        # Default loading order
        self.loading_order = ['countries', 'covid_cases', 'vaccinations', 'testing']

    def run_loading_pipeline(self, data_files: Optional[Dict[str, str]] = None,
                           create_tables: bool = True, create_indexes: bool = True,
                           create_constraints: bool = True) -> Dict[str, Any]:
        """Run the complete data loading pipeline.

        Args:
            data_files: Dictionary mapping table names to file paths (if None, auto-discover)
            create_tables: Whether to create tables if they don't exist
            create_indexes: Whether to create indexes
            create_constraints: Whether to create foreign key constraints

        Returns:
            Dictionary containing loading results
        """
        try:
            self.logger.info("Starting data loading pipeline")
            start_time = time.time()

            # Auto-discover data files if not provided
            if data_files is None:
                data_files = self._discover_data_files()

            # Create database schema
            if create_tables:
                self.logger.info("Creating database tables")
                table_results = self._create_all_tables()
            else:
                table_results = {"status": "skipped", "message": "Table creation disabled"}

            # Create indexes
            if create_indexes:
                self.logger.info("Creating database indexes")
                index_results = self._create_all_indexes()
            else:
                index_results = {"status": "skipped", "message": "Index creation disabled"}

            # Load data for each table
            loading_results = []
            total_rows_loaded = 0

            for table_name in self.loading_order:
                if table_name in data_files:
                    file_path = data_files[table_name]
                    self.logger.info(f"Loading data for table '{table_name}' from {file_path}")

                    result = self._load_table_data(table_name, file_path)
                    loading_results.append({
                        "table": table_name,
                        "file": file_path,
                        "result": result
                    })

                    if result["status"] in ["success", "partial_success"]:
                        total_rows_loaded += result.get("rows_loaded", 0)
                else:
                    self.logger.warning(f"No data file found for table '{table_name}'")
                    loading_results.append({
                        "table": table_name,
                        "file": None,
                        "result": {"status": "skipped", "message": "No data file found"}
                    })

            # Create foreign key constraints
            if create_constraints:
                self.logger.info("Creating foreign key constraints")
                constraint_results = self._create_foreign_key_constraints()
            else:
                constraint_results = {"status": "skipped", "message": "Constraint creation disabled"}

            end_time = time.time()
            duration = end_time - start_time

            # Aggregate results
            overall_status = self._determine_overall_status([
                table_results, index_results, constraint_results
            ] + [r["result"] for r in loading_results])

            overall_result = {
                "status": overall_status,
                "total_duration_seconds": duration,
                "total_rows_loaded": total_rows_loaded,
                "table_creation": table_results,
                "index_creation": index_results,
                "constraint_creation": constraint_results,
                "data_loading": loading_results,
                "timestamp": time.time()
            }

            self.logger.info(f"Data loading pipeline completed in {duration:.2f} seconds")
            self._log_loading_summary(overall_result)

            return overall_result

        except Exception as e:
            self.logger.error(f"Data loading pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }

    def _discover_data_files(self) -> Dict[str, str]:
        """Auto-discover processed data files.

        Returns:
            Dictionary mapping table names to file paths
        """
        data_files = {}

        # Look for processed data files
        if self.processed_data_path.exists():
            for file_path in self.processed_data_path.glob("*.csv"):
                filename = file_path.stem.lower()

                # Map filenames to table names
                if 'countries' in filename or 'population' in filename:
                    data_files['countries'] = str(file_path)
                elif 'covid' in filename and 'cases' in filename:
                    data_files['covid_cases'] = str(file_path)
                elif 'vaccin' in filename:
                    data_files['vaccinations'] = str(file_path)
                elif 'test' in filename:
                    data_files['testing'] = str(file_path)

        self.logger.info(f"Discovered data files: {data_files}")
        return data_files

    def _create_all_tables(self) -> Dict[str, Any]:
        """Create all database tables.

        Returns:
            Dictionary with table creation results
        """
        results = []
        errors = []

        for table_name in get_all_table_names():
            try:
                schema = get_table_schema(table_name)
                created = self.sql_loader.create_table(table_name, schema, if_exists='skip')

                if created:
                    results.append(f"Created table '{table_name}'")
                else:
                    results.append(f"Table '{table_name}' already exists")

            except Exception as e:
                error_msg = f"Failed to create table '{table_name}': {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        return {
            "status": "success" if not errors else "partial_success",
            "tables_processed": len(results),
            "results": results,
            "errors": errors
        }

    def _create_all_indexes(self) -> Dict[str, Any]:
        """Create all database indexes.

        Returns:
            Dictionary with index creation results
        """
        results = []
        errors = []

        for table_name, indexes in get_all_indexes().items():
            for index_sql in indexes:
                try:
                    self.sql_loader.execute_query(index_sql)
                    results.append(f"Created index on '{table_name}'")

                except Exception as e:
                    # Index might already exist, log but don't fail
                    if "already exists" in str(e).lower():
                        results.append(f"Index on '{table_name}' already exists")
                    else:
                        error_msg = f"Failed to create index on '{table_name}': {e}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)

        return {
            "status": "success" if not errors else "partial_success",
            "indexes_processed": len(results),
            "results": results,
            "errors": errors
        }

    def _create_foreign_key_constraints(self) -> Dict[str, Any]:
        """Create all foreign key constraints.

        Returns:
            Dictionary with constraint creation results
        """
        results = []
        errors = []

        for constraint_sql in get_foreign_key_constraints():
            try:
                self.sql_loader.execute_query(constraint_sql)
                results.append("Created foreign key constraint")

            except Exception as e:
                # Constraint might already exist, log but don't fail
                if "already exists" in str(e).lower():
                    results.append("Foreign key constraint already exists")
                else:
                    error_msg = f"Failed to create foreign key constraint: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)

        return {
            "status": "success" if not errors else "partial_success",
            "constraints_processed": len(results),
            "results": results,
            "errors": errors
        }

    def _load_table_data(self, table_name: str, file_path: str) -> Dict[str, Any]:
        """Load data for a specific table.

        Args:
            table_name: Name of the table to load
            file_path: Path to the data file

        Returns:
            Dictionary with loading results
        """
        try:
            # Determine which loader method to use
            if table_name == 'countries':
                result = self.data_loader.load_countries_data(file_path)
            elif table_name == 'covid_cases':
                result = self.data_loader.load_covid_cases_data(file_path)
            elif table_name == 'vaccinations':
                result = self.data_loader.load_vaccinations_data(file_path)
            elif table_name == 'testing':
                result = self.data_loader.load_testing_data(file_path)
            else:
                raise ValueError(f"Unknown table name: {table_name}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to load data for table '{table_name}': {e}")
            return {
                "status": "failed",
                "error": str(e),
                "rows_loaded": 0,
                "timestamp": time.time()
            }

    def _determine_overall_status(self, results: List[Dict[str, Any]]) -> str:
        """Determine overall status from multiple operation results.

        Args:
            results: List of result dictionaries

        Returns:
            Overall status string
        """
        statuses = [r.get("status", "unknown") for r in results]

        if all(status == "success" for status in statuses):
            return "success"
        elif any(status == "failed" for status in statuses):
            return "failed"
        elif any(status == "partial_success" for status in statuses):
            return "partial_success"
        else:
            return "completed_with_warnings"

    def _log_loading_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of the loading results.

        Args:
            results: Loading results dictionary
        """
        self.logger.info("=== Data Loading Pipeline Summary ===")
        self.logger.info(f"Status: {results['status']}")
        self.logger.info(f"Duration: {results['total_duration_seconds']:.2f} seconds")
        self.logger.info(f"Total rows loaded: {results['total_rows_loaded']}")

        # Log table creation
        table_result = results.get('table_creation', {})
        if table_result.get('status') != 'skipped':
            self.logger.info(f"Tables: {table_result.get('tables_processed', 0)} processed")

        # Log index creation
        index_result = results.get('index_creation', {})
        if index_result.get('status') != 'skipped':
            self.logger.info(f"Indexes: {index_result.get('indexes_processed', 0)} processed")

        # Log constraint creation
        constraint_result = results.get('constraint_creation', {})
        if constraint_result.get('status') != 'skipped':
            self.logger.info(f"Constraints: {constraint_result.get('constraints_processed', 0)} processed")

        # Log data loading results
        loading_results = results.get('data_loading', [])
        self.logger.info("Data loading results:")
        for loading_result in loading_results:
            table = loading_result['table']
            result = loading_result['result']
            status = result.get('status', 'unknown')
            rows = result.get('rows_loaded', 0)

            status_icon = "✓" if status == 'success' else "⚠" if status == 'partial_success' else "✗"
            self.logger.info(f"  {status_icon} {table}: {rows} rows ({status})")

    def get_available_tables(self) -> List[str]:
        """Get list of available table names.

        Returns:
            List of table names
        """
        return get_all_table_names()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a database table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        return self.sql_loader.get_table_info(table_name)

    def validate_database_setup(self) -> Dict[str, Any]:
        """Validate that the database is properly set up.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "database_connection": False,
            "tables_exist": [],
            "indexes_exist": [],
            "constraints_exist": [],
            "overall_ready": False
        }

        try:
            # Check database connection
            table_info = self.sql_loader.get_table_info('countries')
            validation_results["database_connection"] = True

            # Check tables exist
            for table_name in get_all_table_names():
                info = self.sql_loader.get_table_info(table_name)
                if info.get('exists', False):
                    validation_results["tables_exist"].append(table_name)

            # For a complete validation, we could check indexes and constraints
            # but that's more complex and might not be necessary for basic validation

            validation_results["overall_ready"] = (
                validation_results["database_connection"] and
                len(validation_results["tables_exist"]) == len(get_all_table_names())
            )

        except Exception as e:
            self.logger.error(f"Database validation failed: {e}")

        return validation_results
