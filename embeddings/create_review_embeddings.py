from embeddings.config import config, logger
from src.exceptions.exception import EmbeddingCreationError


def create_embeddings(
    source_schema: str,
    target_schema: str,
    source_table: str,
    target_table: str,
    id_column: str,
    text_column: str,
) -> None:
    """
    Create embeddings for a text column in a Snowflake table and store them
    in a separate table.

    Args:
        source_schema (str): Schema containing the source dbt mart table.
        target_schema (str): Schema where the embeddings table should be created.
        source_table (str): Source table name, for example ``dbt_mrt_reviews``.
        target_table (str): Target embeddings table name.
        id_column (str): Primary key column from the source table.
        text_column (str): Text column to embed.
    """
    conn = None
    cursor = None
    # Keep the embedding column name tied to the source text column.
    embedding_column = f"{text_column}_embedding"

    try:
        logger.info(
            f"Creating embeddings from {source_schema}.{source_table} into "
            f"{target_schema}.{target_table}"
        )

        conn = config.get_snowflake_connection(write_access=True)
        cursor = conn.cursor()

        # Create (or recreate) the side table with new dimensions
        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {target_schema}.{target_table} (
                {id_column} TEXT PRIMARY KEY,
                {embedding_column} VECTOR(FLOAT, 768)
            )
            """
        )

        # Clear old vectors before recomputing them from the mart.
        cursor.execute(f"TRUNCATE TABLE {target_schema}.{target_table}")

        cursor.execute(
            f"""
            INSERT INTO {target_schema}.{target_table} ({id_column}, {embedding_column})
            SELECT
                {id_column},
                SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', {text_column})
            FROM {source_schema}.{source_table}
            WHERE {text_column} IS NOT NULL
            """
        )

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
    )
