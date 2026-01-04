"""
SQL loader for PostgreSQL database operations in COVID-19 ETL pipeline.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .base_loader import BaseLoader
from ..config import config


class SQLLoader(BaseLoader):
    """SQL loader for database operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SQL loader.

        Args:
            config: Database configuration dictionary
        """
        super().__init__(config)

        # Database configuration - use provided config or get from global config
        if config is None:
            from ..config import config as global_config
            self.db_config = global_config.get('database', {})
        else:
            self.db_config = config
        self.engine = None
        self.connection_string = self._build_connection_string()

        # Initialize engine on first use
        self._init_engine()

    def _build_connection_string(self) -> str:
        """Build database connection string from configuration.

        Returns:
            Database connection string
        """
        # Check if full connection string is provided (for transaction pooler)
        connection_string = self.db_config.get('connection_string')
        if connection_string:
            self.logger.info("Using full connection string from configuration")
            return connection_string

        # Fallback to individual parameters
        db_type = self.db_config.get('type', 'postgresql')
        username = self.db_config.get('username', '')
        password = self.db_config.get('password', '')
        host = self.db_config.get('host', 'localhost')
        port = self.db_config.get('port', 5432)
        database = self.db_config.get('database', 'postgres')

        if db_type == 'postgresql':
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'sqlite':
            return f"sqlite:///{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def _init_engine(self) -> None:
        """Initialize the SQLAlchemy engine."""
        try:
            # Create engine with connection pooling and error handling
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Test connections before use
                pool_recycle=300,    # Recycle connections after 5 minutes
                echo=False           # Set to True for SQL debugging
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.logger.info("Database connection established successfully")

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self.engine is None:
            raise RuntimeError("Database engine not initialized")

        connection = None
        try:
            connection = self.engine.connect()
            yield connection
        except SQLAlchemyError as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                connection.close()

    def load_data(self, data: Union[pd.DataFrame, Dict[str, Any], str],
                  destination: str, **kwargs) -> Dict[str, Any]:
        """Load data to database table.

        Args:
            data: Data to load (DataFrame, dict, or file path)
            destination: Table name to load data into
            **kwargs: Additional arguments:
                - if_exists: 'replace', 'append', or 'fail'
                - index: Whether to include DataFrame index
                - chunksize: Batch size for loading

        Returns:
            Dictionary with loading results
        """
        start_time = time.time()
        errors = []

        try:
            # Validate destination (table name)
            if not self.validate_destination(destination):
                raise ValueError(f"Invalid destination table: {destination}")

            # Convert data to DataFrame
            df = self._validate_data(data)

            if df.empty:
                self.logger.warning("No data to load")
                return self._create_loading_summary(start_time, 0, errors)

            self.logger.info(f"Loading {len(df)} rows to table '{destination}'")

            # Handle different loading strategies
            if_exists = kwargs.get('if_exists', self.if_exists)
            index = kwargs.get('index', False)
            chunksize = kwargs.get('chunksize', self.batch_size)

            # Load data using pandas to_sql
            try:
                df.to_sql(
                    name=destination,
                    con=self.engine,
                    if_exists=if_exists,
                    index=index,
                    chunksize=chunksize,
                    method='multi'  # Use multi-row inserts for better performance
                )

                rows_loaded = len(df)
                self.logger.info(f"Successfully loaded {rows_loaded} rows to {destination}")

            except SQLAlchemyError as e:
                self.logger.error(f"Failed to load data to table {destination}: {e}")
                errors.append(str(e))
                rows_loaded = 0

        except Exception as e:
            self.logger.error(f"Error during data loading: {e}")
            errors.append(str(e))
            rows_loaded = 0

        return self._create_loading_summary(start_time, rows_loaded, errors)

    def validate_destination(self, destination: str) -> bool:
        """Validate that the table exists and is accessible.

        Args:
            destination: Table name to validate

        Returns:
            True if table is valid, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Check if table exists
                schema = self.db_config.get('schema', 'public')
                query = text("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = :schema
                        AND table_name = :table_name
                    )
                """)

                result = conn.execute(query, {'schema': schema, 'table_name': destination})
                exists = result.fetchone()[0]

                if not exists:
                    self.logger.warning(f"Table '{destination}' does not exist in schema '{schema}'")
                    return False

                return True

        except SQLAlchemyError as e:
            self.logger.error(f"Error validating table {destination}: {e}")
            return False

    def create_table(self, table_name: str, schema: Dict[str, Any],
                     if_exists: str = 'skip') -> bool:
        """Create a database table from schema definition.

        Args:
            table_name: Name of the table to create
            schema: Dictionary defining table schema
            if_exists: Action if table exists ('skip', 'drop', 'error')

        Returns:
            True if table was created, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Check if table exists
                if self.validate_destination(table_name):
                    if if_exists == 'skip':
                        self.logger.info(f"Table '{table_name}' already exists, skipping creation")
                        return False
                    elif if_exists == 'drop':
                        self.logger.info(f"Dropping existing table '{table_name}'")
                        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                        conn.commit()
                    elif if_exists == 'error':
                        raise ValueError(f"Table '{table_name}' already exists")

                # Build CREATE TABLE statement
                columns = []
                for col_name, col_def in schema.items():
                    col_type = col_def.get('type', 'VARCHAR(255)')
                    nullable = col_def.get('nullable', True)
                    default = col_def.get('default')
                    primary_key = col_def.get('primary_key', False)

                    col_sql = f"{col_name} {col_type}"

                    if primary_key:
                        col_sql += " PRIMARY KEY"
                    elif not nullable:
                        col_sql += " NOT NULL"

                    if default is not None:
                        col_sql += f" DEFAULT {default}"

                    columns.append(col_sql)

                create_sql = f"""
                    CREATE TABLE {table_name} (
                        {', '.join(columns)}
                    )
                """

                self.logger.info(f"Creating table '{table_name}'")
                conn.execute(text(create_sql))
                conn.commit()

                self.logger.info(f"Successfully created table '{table_name}'")
                return True

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create table {table_name}: {e}")
            raise

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a raw SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query result
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a database table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        try:
            with self.get_connection() as conn:
                # Get column information
                query = text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND table_schema = :schema
                    ORDER BY ordinal_position
                """)

                result = conn.execute(query, {
                    'table_name': table_name,
                    'schema': self.db_config.get('schema', 'public')
                })

                columns = []
                for row in result:
                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2] == 'YES',
                        'default': row[3]
                    })

                # Get row count
                count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                row_count = conn.execute(count_query).fetchone()[0]

                return {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count,
                    'exists': True
                }

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get table info for {table_name}: {e}")
            return {
                'table_name': table_name,
                'exists': False,
                'error': str(e)
            }

    def truncate_table(self, table_name: str) -> None:
        """Truncate a database table.

        Args:
            table_name: Name of the table to truncate
        """
        try:
            with self.get_connection() as conn:
                self.logger.info(f"Truncating table '{table_name}'")
                conn.execute(text(f"TRUNCATE TABLE {table_name}"))
                conn.commit()
                self.logger.info(f"Successfully truncated table '{table_name}'")

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to truncate table {table_name}: {e}")
            raise

    def drop_table(self, table_name: str, if_exists: bool = True) -> None:
        """Drop a database table.

        Args:
            table_name: Name of the table to drop
            if_exists: Whether to use IF EXISTS clause
        """
        try:
            with self.get_connection() as conn:
                exists_clause = "IF EXISTS" if if_exists else ""
                self.logger.info(f"Dropping table '{table_name}'")
                conn.execute(text(f"DROP TABLE {exists_clause} {table_name}"))
                conn.commit()
                self.logger.info(f"Successfully dropped table '{table_name}'")

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to drop table {table_name}: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        """Get list of supported data formats.

        Returns:
            List of supported format strings
        """
        return ['dataframe', 'dict', 'csv', 'json', 'parquet']
