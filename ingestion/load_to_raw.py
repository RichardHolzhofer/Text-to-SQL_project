import os
from ingestion.utils import get_connection, load_config


def load_raw_data(raw_schema_name, stage_name):
    raw_schema_name = raw_schema_name.upper()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(f"USE SCHEMA {raw_schema_name}")
        config = load_config()

        for item in config["files"]:
            file_name = item["file"]
            table_name = item["table"]

            print(f"Loading {file_name} into {table_name}...")

            # Dynamically get columns for the table (excluding metadata columns)
            cursor.execute(f"SHOW COLUMNS IN TABLE {table_name}")
            columns = [row[2] for row in cursor.fetchall() if row[2] not in ("_INGESTED_AT", "_FILE_NAME")]

            # Construct the COPY INTO statement with metadata
            col_list = ", ".join(columns) + ", _FILE_NAME"
            val_list = ", ".join([f"${i+1}" for i in range(len(columns))]) + ", METADATA$FILENAME"

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
            cursor.execute(copy_sql)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    load_raw_data(raw_schema_name="raw", stage_name="ecommerce_raw_stage")
