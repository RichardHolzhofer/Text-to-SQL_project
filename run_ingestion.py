import sys

from ingestion.config import logger
from ingestion.create_schema import insert_schema
from ingestion.load_to_raw import load_raw_data
from ingestion.upload_to_stage import put_to_stage
from src.exceptions.exception import IngestionException


def run_pipeline() -> None:
    """
    Orchestrates the full data ingestion pipeline:
    1. Recreates the schema and tables.
    2. Uploads raw CSV files to the Snowflake stage.
    3. Loads data from the stage into the raw tables.

    Catches and logs any IngestionException or unexpected failures.
    """
    # Define centralized configuration constants
    RAW_SCHEMA = "RAW_DATA"
    STAGE_NAME = "olist_raw_stage"

    try:
        logger.info("--- Ingestion Pipeline Started ---")

        # Step 1: Create Schema and Tables
        # This will use the SQL definitions in data/create_ecommerce_schema.sql
        insert_schema(raw_schema_name=RAW_SCHEMA)

        # Step 2: Upload CSV files to the Snowflake stage
        # This scans the data/raw folder and PUTs files into the Snowflake stage
        put_to_stage(raw_schema_name=RAW_SCHEMA, stage_name=STAGE_NAME)

        # Step 3: Copy data from stage to raw tables
        # This populates the tables and captures metadata lineage
        load_raw_data(raw_schema_name=RAW_SCHEMA, stage_name=STAGE_NAME)

        logger.info("--- Ingestion Pipeline Completed Successfully ---")

    except IngestionException as e:
        # Handle failures that we predicted and wrapped in our custom class
        logger.error(f"Pipeline failed with IngestionException:\n{e}")
        sys.exit(1)
    except Exception as e:
        # Handle unexpected system-level or python-level failures
        logger.error(f"Pipeline failed with unexpected error:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    # Start the pipeline
    run_pipeline()
