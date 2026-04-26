import os
from ingestion.utils import get_connection
from ingestion.logger import logger
from ingestion.exception import SnowflakeConnectionError


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
        conn = get_connection()
        cursor = conn.cursor()

        # Step 1: Select the schema
        cursor.execute(f"USE SCHEMA {raw_schema_name}")

        # Step 2: Create (or Replace) the internal stage for data loading
        logger.info(f"Ensuring stage {stage_name} exists...")
        cursor.execute(f"CREATE OR REPLACE STAGE {stage_name}")

        # Step 3: Identify the directory containing raw CSV data
        base_dir = os.path.dirname(__file__)
        csv_dir = os.path.join(base_dir, "..", "data", "raw")

        # Step 4: List all files in the directory
        csv_files = os.listdir(csv_dir)

        # Step 5: Filter and upload only the relevant CSV files
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

        # Step 6: Cleanup connection
        cursor.close()
        conn.close()

    except Exception as e:
        # Catch and wrap connection/upload errors
        raise SnowflakeConnectionError(e)


if __name__ == "__main__":
    # Standard entry point for isolated testing
    put_to_stage(raw_schema_name="raw", stage_name="ecommerce_raw_stage")
