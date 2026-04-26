import os
from pathlib import Path
from typing import Any, Dict
from typing import TYPE_CHECKING

import yaml
from dotenv import load_dotenv

from src.exceptions.exception import ConfigError, SnowflakeConfigError
from src.logger.logger import logger

if TYPE_CHECKING:
    from snowflake.connector.connection import SnowflakeConnection


class Config:
    """
    Centralized configuration helper for the text-to-sql app.
    """

    _env_loaded = False

    @classmethod
    def _load_env_once(cls) -> None:
        """
        Load environment variables from .env once per process.
        """
        if cls._env_loaded:
            return
        load_dotenv()
        cls._env_loaded = True
        logger.info("Environment variables loaded from .env")

    @classmethod
    def get_env(
        cls, key: str, default: str | None = None, required: bool = False
    ) -> str | None:
        """
        Read one environment variable with optional required validation.
        """
        cls._load_env_once()
        value = os.getenv(key, default)

        if required and not value:
            logger.error("Required environment variable is missing: %s", key)
            raise ConfigError(
                ValueError(f"Missing required environment variable: {key}")
            )

        return value

    @staticmethod
    def load_yaml(config_path: str | Path) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.
        """
        path = Path(config_path).resolve()
        logger.info("Loading YAML configuration from %s", path)

        try:
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")

            with path.open("r", encoding="utf-8") as stream:
                content = yaml.safe_load(stream) or {}

            if not isinstance(content, dict):
                raise TypeError(f"Expected mapping at root of YAML file: {path}")

            return content

        except Exception as error:
            logger.exception("Failed to load YAML configuration: %s", path)
            raise ConfigError(error) from error

    @classmethod
    def load_ingestion_config(cls) -> Dict[str, Any]:
        """
        Load ingestion config from ingestion/config.yml.
        """
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "ingestion" / "config.yml"
        return cls.load_yaml(config_path)

    @classmethod
    def get_connection(cls) -> "SnowflakeConnection":
        """
        Initialize and return a Snowflake connection using loader env variables.
        """
        try:
            logger.info(
                "Creating Snowflake connection using loader environment variables"
            )
            cls._load_env_once()
            import snowflake.connector

            conn = snowflake.connector.connect(
                user=cls.get_env("LOADER_SNOWFLAKE_USER", required=True),
                password=cls.get_env("LOADER_SNOWFLAKE_PASSWORD", required=True),
                account=cls.get_env("LOADER_SNOWFLAKE_ACCOUNT", required=True),
                warehouse=cls.get_env("LOADER_SNOWFLAKE_WAREHOUSE", required=True),
                database=cls.get_env("LOADER_SNOWFLAKE_DATABASE", required=True),
                role=cls.get_env("LOADER_SNOWFLAKE_ROLE", required=True),
            )

            logger.info("Snowflake connection established successfully")
            return conn

        except Exception as error:
            logger.exception("Snowflake connection creation failed")
            raise SnowflakeConfigError(error) from error
