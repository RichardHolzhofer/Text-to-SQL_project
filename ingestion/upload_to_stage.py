import os
from ingestion.utils import get_connection


def put_to_stage(raw_schema_name, stage_name):
    raw_schema_name = raw_schema_name.upper()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(f"USE SCHEMA {raw_schema_name}")

        cursor.execute(f"CREATE OR REPLACE STAGE {stage_name}")

        base_dir = os.path.dirname(__file__)
        csv_dir = os.path.join(base_dir, "..", "data", "raw")
        csv_files = os.listdir(csv_dir)

        for csv_file in csv_files:
            if csv_file.endswith("dataset.csv"):
                full_path = os.path.join(csv_dir, csv_file)
                csv_path = os.path.abspath(full_path).replace("\\", "/")
                cursor.execute(
                    f"PUT 'file://{csv_path}' @{stage_name} AUTO_COMPRESS=TRUE"
                )
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    put_to_stage(
        raw_schema_name="raw",
        stage_name="ecommerce_raw_stage"
        )
