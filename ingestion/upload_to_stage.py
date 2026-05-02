import os

from ingestion.config import config, logger
from src.exceptions.exception import SnowflakeConnectionError


def put_to_stage(raw_schema_name: str, stage_name: str) -> None:
    """
    Finds all CSV files in the data/raw directory and uploads them to a
    specified Snowflake internal stage.

    Args:
        raw_schema_name (str): The target schema where the stage resides.
        stage_name (str): The name of the Snowflake stage to upload files to.

    Raises:
        SnowflakeConnectionError: If the connection or the file transfer fails.
    """
    try:
        logger.info(f"Starting upload to stage: {stage_name}")

        # Ensure schema name is uppercase
        raw_schema_name = raw_schema_name.upper()

        # Get connection
        conn = config.get_snowflake_connection(write_access=True)
        cursor = conn.cursor()

        # Select the schema
        cursor.execute(f"USE SCHEMA {raw_schema_name}")

        # Create (or Replace) the internal stage for data loading
        logger.info(f"Ensuring stage {stage_name} exists...")
        cursor.execute(f"CREATE OR REPLACE STAGE {stage_name}")

        # Identify the directory containing raw CSV data
        base_dir = os.path.dirname(__file__)
        csv_dir = os.path.join(base_dir, "..", "data", "raw")

        # List all files in the directory
        csv_files = os.listdir(csv_dir)

        # Filter and upload only the relevant CSV files
        for csv_file in csv_files:
            # We filter for files that end in 'dataset.csv'
            if csv_file.endswith("dataset.csv"):
                # Construct the absolute path
                full_path = os.path.join(csv_dir, csv_file)
                # Snowflake PUT command works best with Unix-style slashes even on Windows
                csv_path = os.path.abspath(full_path).replace("\\", "/")

                logger.info(f"Uploading {csv_file} to @{stage_name}...")
                # PUT the file into the stage with automatic compression enabled
                cursor.execute(
                    f"PUT 'file://{csv_path}' @{stage_name} AUTO_COMPRESS=TRUE"
                )

        logger.info("All files uploaded successfully.")

    except Exception as e:
        raise SnowflakeConnectionError(e)

    finally:
        # Cleanup connection
        cursor.close()
        conn.close()


if __name__ == "__main__":
    put_to_stage(raw_schema_name="RAW_DATA", stage_name="olist_raw_stage")
