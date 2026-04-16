import os
from typing import Any, Dict
import yaml
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.connection import SnowflakeConnection


def get_connection() -> SnowflakeConnection:
    """
    Initializes and returns a connection to Snowflake using environment variables.

    Returns:
        SnowflakeConnection: An active Snowflake connection object.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Establish connection using parameters from environment
    return snowflake.connector.connect(
        user=os.getenv("LOADER_SNOWFLAKE_USER"),
        password=os.getenv("LOADER_SNOWFLAKE_PASSWORD"),
        account=os.getenv("LOADER_SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("LOADER_SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("LOADER_SNOWFLAKE_DATABASE"),
        role=os.getenv("LOADER_SNOWFLAKE_ROLE"),
    )


def load_config() -> Dict[str, Any]:
    """
    Loads the ingestion configuration from config.yml.

    Returns:
        Dict[str, Any]: The configuration parameters for the ingestion pipeline.
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(__file__)
    # Construct the absolute path to the config.yml file
    config_path = os.path.join(base_dir, "config.yml")

    # Open and parse the YAML configuration
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
