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
        # Initialize the named logger via the centralized logger utility
        self.logger = get_logger(logger_name)

        load_dotenv()
        self.logger.info("Environment variables loaded from .env")

    def load_yaml(self, config_path: str | Path) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.
        """
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
        """
        Load ingestion config from ingestion/config.yml.
        """
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "ingestion" / "config.yml"
        return self.load_yaml(config_path)

    def get_connection(self) -> SnowflakeConnection:
        """
        Initialize and return a Snowflake connection using environment variables.
        """
        try:
            self.logger.info("Creating Snowflake connection")

            return snowflake.connector.connect(
                user=os.getenv("LOADER_SNOWFLAKE_USER"),
                password=os.getenv("LOADER_SNOWFLAKE_PASSWORD"),
                account=os.getenv("LOADER_SNOWFLAKE_ACCOUNT"),
                warehouse=os.getenv("LOADER_SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("LOADER_SNOWFLAKE_DATABASE"),
                role=os.getenv("LOADER_SNOWFLAKE_ROLE"),
            )

        except Exception as error:
            self.logger.exception("Snowflake connection failed")
            raise SnowflakeConfigError(error) from error
