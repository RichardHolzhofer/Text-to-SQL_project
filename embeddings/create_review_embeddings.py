import json

from tqdm import tqdm

from src.config.config import Config
from src.exceptions.exception import EmbeddingCreationError

config = Config(logger_name="embeddings")
logger = config.get_logger()


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

        # Create or replace the side table with OpenAI dimensions (dynamic from config)
        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {target_schema}.{target_table} (
                {id_column} TEXT PRIMARY KEY,
                {embedding_column} VECTOR(FLOAT, {config.embedding_vector_dim})
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
        # Use configured embedding vector dimension (allow override via env vars)
        vector_dim = config.embedding_vector_dim

        # Process in batches
        # We do this to reduce the number of OpenAI API ans Snowflake
        # calls to save cost and time and to avoide rate limit
        for i in tqdm(range(0, total_rows, batch_size), desc="Embedding batches"):
            batch = rows[i : i + batch_size]
            ids = [row[0] for row in batch]
            texts = [row[1] for row in batch]

            try:
                # Generate embeddings for the current batch of texts
                embeddings = embedding_model.embed_documents(texts)

                placeholders: list[str] = []
                params: list = []

                for idx, vector in zip(ids, embeddings):
                    placeholders.append("(%s, %s)")
                    # Convert the vector (list[float]) to a JSON string so that
                    # Snowflake can parse it safely (handles special chars).
                    params.extend([idx, json.dumps(vector)])

                # Build the VALUES clause for a bulk INSERT.
                # Each row is represented by a placeholder tuple "(%s, %s)".
                # `COLUMN1` (the first placeholder) will hold the primary‑key `id`.
                # `COLUMN2` (the second placeholder) will hold the embedding vector
                # Snowflake will later parse it and cast it to VECTOR(FLOAT, {vector_dim}).

                query = f"""INSERT INTO {target_schema}.{target_table}
                ({id_column}, {embedding_column})
                SELECT COLUMN1,
                       PARSE_JSON(COLUMN2)::VECTOR(FLOAT, {vector_dim})
                FROM VALUES {", ".join(placeholders)}"""
                cursor.execute(query, params)

            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {e}")
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
