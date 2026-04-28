import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.connection import SnowflakeConnection

from src.exceptions.exception import ConfigError, SnowflakeConfigError
from src.logger.logger import get_logger


class Config:
    """
    Centralized configuration helper for the text-to-sql app.
    Should be instantiated in each process with a specific logger name.
    """

    def __init__(self, logger_name: str = "text-to-sql"):
        self.logger = get_logger(logger_name)
        load_dotenv()
        self.logger.info("Environment variables loaded from .env")

        # Snowflake Connection Settings
        self.sf_account = os.getenv("LOADER_SNOWFLAKE_ACCOUNT")
        self.sf_database = os.getenv("LOADER_SNOWFLAKE_DATABASE")
        self.sf_warehouse = os.getenv("LOADER_SNOWFLAKE_WAREHOUSE")
        self.sf_loader_role = os.getenv("LOADER_SNOWFLAKE_ROLE")
        self.sf_reader_role = os.getenv("READER_SNOWFLAKE_ROLE")
        self.sf_loader_schema = os.getenv("LOADER_SNOWFLAKE_SCHEMA")
        self.sf_reader_schema = os.getenv("READER_SNOWFLAKE_SCHEMA")

        # Identity Settings (Loader vs Reader)
        self.sf_loader_user = os.getenv("LOADER_SNOWFLAKE_USER")
        self.sf_loader_pass = os.getenv("LOADER_SNOWFLAKE_PASSWORD")

        self.sf_reader_user = os.getenv("READER_SNOWFLAKE_USER")
        self.sf_reader_pass = os.getenv("READER_SNOWFLAKE_PASSWORD")

        # Validation (Optional but recommended)
        self._validate_config()

    def _validate_config(self):
        """Ensure critical environment variables are present."""
        required = [
            "LOADER_SNOWFLAKE_ACCOUNT",
            "LOADER_SNOWFLAKE_USER",
            "READER_SNOWFLAKE_USER",
        ]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    def load_yaml(self, config_path: str | Path) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        path = Path(config_path).resolve()
        self.logger.info(f"Loading YAML configuration from {path}")
        try:
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            with path.open("r", encoding="utf-8") as stream:
                content = yaml.safe_load(stream) or {}
            if not isinstance(content, dict):
                raise TypeError(f"Expected mapping at root of YAML file: {path}")
            return content
        except Exception as error:
            self.logger.exception(f"Failed to load YAML configuration: {path}")
            raise ConfigError(error) from error

    def load_ingestion_config(self) -> Dict[str, Any]:
        """Load ingestion config from ingestion/config.yml."""
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "ingestion" / "config.yml"
        return self.load_yaml(config_path)

    def get_connection(self, write_access: bool = False) -> SnowflakeConnection:
        """
        Initialize and return a Snowflake connection.

        Args:
            write_access (bool): If True, uses Loader credentials.
                                 If False (default), uses Reader credentials.
        """
        try:
            access_type = "LOADER (Write)" if write_access else "READER (Read-Only)"
            self.logger.info(f"Creating Snowflake connection with {access_type} access")

            user = self.sf_loader_user if write_access else self.sf_reader_user
            password = self.sf_loader_pass if write_access else self.sf_reader_pass
            role = self.sf_loader_role if write_access else self.sf_reader_role
            schema = self.sf_loader_schema if write_access else self.sf_reader_schema

            return snowflake.connector.connect(
                user=user,
                password=password,
                account=self.sf_account,
                warehouse=self.sf_warehouse,
                database=self.sf_database,
                role=role,
                schema=schema,
            )

        except Exception as error:
            self.logger.exception(f"Snowflake connection failed for {access_type} user")
            raise SnowflakeConfigError(error) from error
