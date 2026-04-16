import sys
from ingestion.utils import get_connection, load_config
from ingestion.logger import logger
from ingestion.exception import DataLoadError


def load_raw_data(raw_schema_name: str, stage_name: str) -> None:
    """
    Copies data from the Snowflake stage into the raw tables.
    Utilizes dynamic column discovery and Snowflake's METADATA$FILENAME
    to populate audit columns.

    Args:
        raw_schema_name (str): The name of the schema where tables reside.
        stage_name (str): The name of the stage containing the uploaded CSV files.

    Raises:
        DataLoadError: If a COPY INTO statement fails for any table.
    """
    try:
        logger.info(f"Starting data load from stage: {stage_name}")

        # Ensure schema name is uppercase
        raw_schema_name = raw_schema_name.upper()

        # Get connection
        conn = get_connection()
        cursor = conn.cursor()

        # Set the current schema context
        cursor.execute(f"USE SCHEMA {raw_schema_name}")

        # Load the configuration mapping files to tables
        config = load_config()

        # Iterate through each file-to-table entry in config.yml
        for item in config["files"]:
            file_name = item["file"]
            table_name = item["table"]

            logger.info(f"Loading {file_name} into {table_name}...")

            # Step 1: Dynamically get columns for the table (excluding metadata columns)
            # This ensures that we only map CSV columns to the appropriate table columns.
            cursor.execute(f"SHOW COLUMNS IN TABLE {table_name}")
            columns = [
                row[2]
                for row in cursor.fetchall()
                if row[2] not in ("_INGESTED_AT", "_FILE_NAME")
            ]

            # Step 2: Construct the column mapping for the COPY INTO command
            # We append _FILE_NAME because we will populate it using METADATA$FILENAME.
            col_list = ", ".join(columns) + ", _FILE_NAME"

            # Step 3: Construct the value list for the SELECT transformation
            # $1, $2, etc. refer to the columns in the CSV.
            val_list = (
                ", ".join([f"${i + 1}" for i in range(len(columns))])
                + ", METADATA$FILENAME"
            )

            # Step 4: Construct the final COPY INTO statement
            # This uses a transformation (SELECT) to inject metadata during the load.
            copy_sql = f"""
            COPY INTO {table_name} ({col_list})
            FROM (SELECT {val_list} FROM @{stage_name}/{file_name}.gz)
            FILE_FORMAT = (
                TYPE = 'CSV' 
                SKIP_HEADER = 1 
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                EMPTY_FIELD_AS_NULL = TRUE
                TRIM_SPACE = TRUE
            )
            """

            # Step 5: Execute the load
            cursor.execute(copy_sql)
            logger.info(f"Successfully loaded {table_name}.")

        logger.info("All tables loaded successfully.")

        # Step 6: Cleanup connection
        cursor.close()
        conn.close()

    except Exception as e:
        # Wrap table loading errors with detailed info
        raise DataLoadError(e, sys)


if __name__ == "__main__":
    # Standard entry point for isolated testing
    load_raw_data(raw_schema_name="raw", stage_name="ecommerce_raw_stage")
