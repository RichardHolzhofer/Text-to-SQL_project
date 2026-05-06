from embeddings.config import config


def test_snowflake_vector_insert():
    conn = config.get_snowflake_connection(write_access=True)
    cursor = conn.cursor()
    target_schema = "OLIST_MART"
    target_table = "test_vector_insert"
    id_column = "ID"
    embedding_column = "EMBEDDING"

    cursor.execute(
        f"""
        CREATE OR REPLACE TABLE {target_schema}.{target_table} (
            {id_column} TEXT PRIMARY KEY,
            {embedding_column} VECTOR(FLOAT, 1536)
        )
        """
    )

    vector = [0.1] * 1536
    ids = ["test1", "test2"]
    embeddings = [vector, vector]

    import json

    insert_data = []
    for idx, vec in zip(ids, embeddings):
        insert_data.append((idx, json.dumps(vec)))

    try:
        cursor.executemany(
            f"INSERT INTO {target_schema}.{target_table} ({id_column}, {embedding_column}) SELECT %s, PARSE_JSON(%s)::VECTOR(FLOAT, 1536)",
            insert_data,
        )
        print("Insert with PARSE_JSON and SELECT successful")
    except Exception as e:
        print(f"Error with PARSE_JSON and SELECT: {e}")

    # test 2: passing list directly with VALUES
    try:
        insert_data2 = [("test3", vector), ("test4", vector)]
        cursor.executemany(
            f"INSERT INTO {target_schema}.{target_table} ({id_column}, {embedding_column}) VALUES (%s, %s::VECTOR(FLOAT, 1536))",
            insert_data2,
        )
        print("Insert with list direct and VALUES successful")
    except Exception as e:
        print(f"Error with list direct and VALUES: {e}")

    # test 3: passing list directly with VALUES (no cast)
    try:
        insert_data3 = [("test5", vector), ("test6", vector)]
        cursor.executemany(
            f"INSERT INTO {target_schema}.{target_table} ({id_column}, {embedding_column}) VALUES (%s, %s)",
            insert_data3,
        )
        print("Insert with list direct and no cast successful")
    except Exception as e:
        print(f"Error with list direct and no cast: {e}")


if __name__ == "__main__":
    test_snowflake_vector_insert()
