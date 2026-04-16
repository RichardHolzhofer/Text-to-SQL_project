import os
from jinja2 import Template
from ingestion.utils import get_connection


def insert_schema(raw_schema_name):
    raw_schema_name = raw_schema_name.upper()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # create schema
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {raw_schema_name}")
        cursor.execute(f"USE SCHEMA {raw_schema_name}")

        # Load SQL file
        base_dir = os.path.dirname(__file__)
        sql_path = os.path.join(base_dir, "..", "data", "create_ecommerce_schema.sql")

        with open(sql_path, "r") as f:
            sql_script = f.read()
            template = Template(sql_script)
            rendered_sql = template.render(schema=raw_schema_name)

        statements = [s.strip() for s in rendered_sql.split(";") if s.strip()]

        for statement in statements:
            cursor.execute(statement)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    insert_schema(raw_schema_name="raw")
