from tqdm import tqdm

from embeddings.config import config, logger
from src.exceptions.exception import EmbeddingCreationError


def create_embeddings(
    source_schema: str,
    target_schema: str,
    source_table: str,
    target_table: str,
    id_column: str,
    text_column: str,
    batch_size: int = 500,
) -> None:
    """
    Create embeddings for a text column in a Snowflake table using OpenAI
    and store them in a separate table.
    """
    conn = None
    cursor = None
    embedding_column = f"{text_column}_embedding"

    try:
        logger.info(
            f"Creating OpenAI embeddings from {source_schema}.{source_table} into "
            f"{target_schema}.{target_table}"
        )

        conn = config.get_snowflake_connection(write_access=True)
        cursor = conn.cursor()

        # Create or replace the side table with OpenAI dimensions (1536)
        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {target_schema}.{target_table} (
                {id_column} TEXT PRIMARY KEY,
                {embedding_column} VECTOR(FLOAT, 1536)
            )
            """
        )

        # Fetch data to embed
        logger.info(f"Fetching data from {source_schema}.{source_table}...")
        cursor.execute(
            f"""
            SELECT {id_column}, {text_column}
            FROM {source_schema}.{source_table}
            WHERE {text_column} IS NOT NULL
            """
        )
        rows = cursor.fetchall()
        total_rows = len(rows)
        logger.info(f"Found {total_rows} rows to embed.")

        embedding_model = config.get_embedding_model()

        # Process in batches
        for i in tqdm(range(0, total_rows, batch_size), desc="Embedding batches"):
            batch = rows[i : i + batch_size]
            ids = [row[0] for row in batch]
            texts = [row[1] for row in batch]

            try:
                # Generate embeddings
                embeddings = embedding_model.embed_documents(texts)

                # Prepare for bulk insert
                import json

                placeholders = []
                params = []
                for idx, vector in zip(ids, embeddings):
                    placeholders.append("(%s, %s)")
                    params.extend([idx, json.dumps(vector)])

                # Use single execute with flattened parameters for correct VECTOR casting
                query = (
                    f"INSERT INTO {target_schema}.{target_table} ({id_column}, {embedding_column}) SELECT column1, PARSE_JSON(column2)::VECTOR(FLOAT, 1536) FROM VALUES "
                    + ", ".join(placeholders)
                )
                cursor.execute(query, params)

            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
                # Optional: implement retry or break
                raise e

        logger.info(
            f"Embeddings successfully created for {source_schema}.{source_table}"
        )

    except Exception as e:
        raise EmbeddingCreationError(e)

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_embeddings(
        source_schema="OLIST_MART",
        target_schema="OLIST_MART",
        source_table="dim_reviews",
        target_table="dim_review_embeddings",
        id_column="ORDER_ID",
        text_column="REVIEW_COMBINED_TEXT",
        batch_size=500,
    )
