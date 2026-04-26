import os
from jinja2 import Template
from ingestion.utils import get_connection
from ingestion.logger import logger
from src.exceptions.exception import IngestionException


def insert_schema(raw_schema_name: str) -> None:
    """
    Creates the Snowflake schema and tables defined in the ecommerce SQL schema file.
    Uses Jinja2 to render the schema name into the SQL statements.

    Args:
        raw_schema_name (str): The name of the target schema in Snowflake.

    Raises:
        IngestionException: If any error occurs during schema or table creation.
    """
    try:
        logger.info(f"Starting schema creation for: {raw_schema_name}")

        # Ensure schema name is uppercase for Snowflake consistency
        raw_schema_name = raw_schema_name.upper()

        # Get connection
        conn = get_connection()
        cursor = conn.cursor()

        # Step 1: Create the schema if it doesn't already exist
        logger.info(f"Ensuring schema {raw_schema_name} exists...")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {raw_schema_name}")

        # Step 2: Set the current schema context
        cursor.execute(f"USE SCHEMA {raw_schema_name}")

        # Step 3: Load the SQL file containing Data Definition
        base_dir = os.path.dirname(__file__)
        sql_path = os.path.join(base_dir, "..", "data", "create_ecommerce_schema.sql")

        with open(sql_path, "r") as f:
            sql_script = f.read()
            # Use Jinja2 to replace {{schema}} placeholders with our actual schema name
            template = Template(sql_script)
            rendered_sql = template.render(schema=raw_schema_name)

        # Step 4: Split the script into individual statements (by semicolon)
        statements = [s.strip() for s in rendered_sql.split(";") if s.strip()]

        # Step 5: Execute each statement to create tables
        logger.info(f"Executing DDL statements from {os.path.basename(sql_path)}...")
        for statement in statements:
            cursor.execute(statement)

        logger.info("Schema and tables created successfully.")

        # Step 6: Cleanup connection
        cursor.close()
        conn.close()

    except Exception as e:
        # Wrap any error into our custom IngestionException for detailed logging
        raise IngestionException(e)


if __name__ == "__main__":
    # Standard entry point for isolated testing
    insert_schema(raw_schema_name="raw")
