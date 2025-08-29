"""
Configuration management for the Research MCP Server.

This module handles all configuration settings, environment variables,
and default values for the server and its components.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration manager for the Research MCP Server.

    Handles environment variables, default settings, and configuration
    validation for all server components.
    """

    def __init__(self):
        """Initialize configuration with default values and environment overrides."""
        self._config = {}

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables and defaults."""
        # Server configuration
        self._config.update({
            # Database settings
            'database_path': self._get_env('DATABASE_PATH', 'research.db'),
            'database_timeout': float(self._get_env('DATABASE_TIMEOUT', '30.0')),

            # Search engine settings
            'default_search_timeout': int(self._get_env('SEARCH_TIMEOUT', '30')),
            'max_results_per_engine': int(self._get_env('MAX_RESULTS_PER_ENGINE', '10')),
            'cache_ttl_seconds': int(self._get_env('CACHE_TTL_SECONDS', '3600')),

            # API Keys
            'serp_api_key': self._get_env('SERP_API_KEY'),
            'semantic_scholar_api_key': self._get_env('SEMANTIC_SCHOLAR_API_KEY'),
            'pubmed_api_key': self._get_env('PUBMED_API_KEY'),

            # Network settings
            'request_timeout': float(self._get_env('REQUEST_TIMEOUT', '10.0')),
            'max_retries': int(self._get_env('MAX_RETRIES', '3')),
            'retry_delay': float(self._get_env('RETRY_DELAY', '1.0')),

            # Logging settings
            'log_level': self._get_env('LOG_LEVEL', 'INFO'),
            'log_file': self._get_env('LOG_FILE', 'server.log'),
            'log_max_size': int(self._get_env('LOG_MAX_SIZE', '10485760')),  # 10MB
            'log_backup_count': int(self._get_env('LOG_BACKUP_COUNT', '5')),

            # Cache settings
            'cache_enabled': self._get_env_bool('CACHE_ENABLED', True),
            'cache_directory': self._get_env('CACHE_DIRECTORY', 'cache'),

            # Export settings
            'export_formats': self._get_env_list('EXPORT_FORMATS', ['json', 'bibtex', 'csv']),
            'max_export_size': int(self._get_env('MAX_EXPORT_SIZE', '1000')),

            # Development settings
            'debug_mode': self._get_env_bool('DEBUG_MODE', False),
            'development_mode': self._get_env_bool('DEVELOPMENT_MODE', False),
        })

        # Validate critical configuration
        self._validate_config()

        # Log API key status for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Config initialization complete:")
        logger.debug(f"  SERP_API_KEY loaded: {bool(self._config.get('serp_api_key'))}")
        logger.debug(f"  SEMANTIC_SCHOLAR_API_KEY loaded: {bool(self._config.get('semantic_scholar_api_key'))}")
        logger.debug(f"  PUBMED_API_KEY loaded: {bool(self._config.get('pubmed_api_key'))}")

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable with optional default.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)

    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """
        Get environment variable as boolean.

        Args:
            key: Environment variable name
            default: Default boolean value

        Returns:
            Boolean value of environment variable
        """
        value = self._get_env(key)
        if value is None:
            return default

        return value.lower() in ('true', '1', 'yes', 'on')

    def _get_env_list(self, key: str, default: list = None) -> list:
        """
        Get environment variable as list (comma-separated).

        Args:
            key: Environment variable name
            default: Default list value

        Returns:
            List of values from environment variable
        """
        if default is None:
            default = []

        value = self._get_env(key)
        if value is None:
            return default

        return [item.strip() for item in value.split(',') if item.strip()]

    def _validate_config(self):
        """Validate critical configuration settings."""
        # Check required directories exist or can be created
        cache_dir = Path(self._config['cache_directory'])
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create cache directory: {e}")

        # Validate numeric settings
        if self._config['max_results_per_engine'] <= 0:
            raise ValueError("MAX_RESULTS_PER_ENGINE must be positive")

        if self._config['database_timeout'] <= 0:
            raise ValueError("DATABASE_TIMEOUT must be positive")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self._config[key] = value

    def update(self, config_dict: Dict[str, Any]):
        """
        Update multiple configuration values.

        Args:
            config_dict: Dictionary of configuration key-value pairs
        """
        self._config.update(config_dict)

    def is_api_key_set(self, service_name: str) -> bool:
        """
        Check if API key is set for a service.

        Args:
            service_name: Name of the service (e.g., 'serp', 'semantic_scholar')

        Returns:
            True if API key is set, False otherwise
        """
        key_mapping = {
            'serp': 'serp_api_key',
            'google_scholar': 'serp_api_key',
            'semantic_scholar': 'semantic_scholar_api_key',
            'pubmed': 'pubmed_api_key',
        }

        config_key = key_mapping.get(service_name)
        if config_key:
            return bool(self._config.get(config_key))

        return False

    def get_api_key(self, service_name: str) -> Optional[str]:
        """
        Get API key for a service.

        Args:
            service_name: Name of the service

        Returns:
            API key if set, None otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        key_mapping = {
            'serp': 'serp_api_key',
            'google_scholar': 'serp_api_key',
            'semantic_scholar': 'semantic_scholar_api_key',
            'pubmed': 'pubmed_api_key',
        }

        config_key = key_mapping.get(service_name)
        if config_key:
            api_key = self._config.get(config_key)
            logger.debug(f"Config.get_api_key({service_name}) -> config_key: {config_key}, has_key: {bool(api_key)}")
            return api_key

        logger.debug(f"Config.get_api_key({service_name}) -> no mapping found")
        return None

    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration as dictionary.

        Returns:
            Dictionary containing all configuration values
        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary-style access."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self._config

    def __str__(self) -> str:
        """String representation of configuration (with sensitive data masked)."""
        config_copy = self._config.copy()

        # Mask sensitive information
        sensitive_keys = ['serp_api_key', 'semantic_scholar_api_key', 'pubmed_api_key']
        for key in sensitive_keys:
            if key in config_copy and config_copy[key]:
                config_copy[key] = '*' * len(config_copy[key])

        return f"Config({config_copy})"
