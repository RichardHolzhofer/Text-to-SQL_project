import sys
from ingestion.create_schema import insert_schema
from ingestion.upload_to_stage import put_to_stage
from ingestion.load_to_raw import load_raw_data

def run_pipeline():
    RAW_SCHEMA = "raw"
    STAGE_NAME = "ecommerce_raw_stage"

    try:
        print("--- Step 1: Creating Schema and Tables ---")
        insert_schema(raw_schema_name=RAW_SCHEMA)

        print("\n--- Step 2: Uploading Files to Snowflake Stage ---")
        put_to_stage(raw_schema_name=RAW_SCHEMA, stage_name=STAGE_NAME)

        print("\n--- Step 3: Loading Data from Stage to Raw Tables ---")
        load_raw_data(raw_schema_name=RAW_SCHEMA, stage_name=STAGE_NAME)

        print("\nPipeline execution completed successfully!")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
