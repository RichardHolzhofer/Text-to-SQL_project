import os
import yaml
from dotenv import load_dotenv
import snowflake.connector

def get_connection():
    """Returns a connection to Snowflake using environment variables."""
    load_dotenv()
    return snowflake.connector.connect(
        user=os.getenv("LOADER_SNOWFLAKE_USER"),
        password=os.getenv("LOADER_SNOWFLAKE_PASSWORD"),
        account=os.getenv("LOADER_SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("LOADER_SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("LOADER_SNOWFLAKE_DATABASE"),
        role=os.getenv("LOADER_SNOWFLAKE_ROLE"),
    )

def load_config():
    """Loads the ingestion config.yml file."""
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
