"""
Configuration management for COVID-19 ETL pipeline.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager for the ETL pipeline."""

    def __init__(self, config_file: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_file = Path(__file__).parent.parent / config_file
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._replace_env_vars(config)

        return config

    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace environment variables in configuration."""
        if isinstance(config, dict):
            return {key: self._replace_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        else:
            return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        db_config = self.get('database', {})

        # Check if full connection string is provided
        connection_string = db_config.get('connection_string')
        if connection_string:
            return connection_string

        # Fallback to individual parameters
        if db_config.get('type') == 'postgresql':
            return (f"postgresql://{db_config.get('username')}:{db_config.get('password')}@"
                   f"{db_config.get('host')}:{db_config.get('port')}/{db_config.get('database')}")
        elif db_config.get('type') == 'sqlite':
            return f"sqlite:///{db_config.get('database', 'covid_etl.db')}"
        else:
            raise ValueError(f"Unsupported database type: {db_config.get('type')}")

    @property
    def raw_data_path(self) -> Path:
        """Get path to raw data directory."""
        return Path(self.get('data_paths.raw', 'data/raw'))

    @property
    def processed_data_path(self) -> Path:
        """Get path to processed data directory."""
        return Path(self.get('data_paths.processed', 'data/processed'))

    @property
    def output_data_path(self) -> Path:
        """Get path to output data directory."""
        return Path(self.get('data_paths.output', 'data/output'))


# Global configuration instance
config = Config()
