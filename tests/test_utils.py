from decimal import Decimal

from src.utils.db_utils import is_result_empty, process_snowflake_results
from src.utils.utils import clean_sql_query, load_yaml, save_yaml


def test_clean_sql_query_removes_markdown_fence():
    raw_sql = "```sql\nSELECT * FROM users;\n```"
    assert clean_sql_query(raw_sql) == "SELECT * FROM users;"


def test_clean_sql_query_handles_literal_escaped_newlines():
    raw_sql = "SELECT\\n*\\nFROM table_name"
    assert clean_sql_query(raw_sql) == "SELECT\n*\nFROM table_name"


def test_yaml_save_and_load_roundtrip(tmp_path):
    data = {"name": "sample", "values": [1, 2, 3]}
    file_path = tmp_path / "sample.yaml"
    save_yaml(data, file_path)
    loaded = load_yaml(file_path)
    assert loaded == data


def test_process_snowflake_results_converts_decimals():
    description = [("amount",), ("count",), ("name",)]
    rows = [(Decimal("10.5"), Decimal("2"), "alpha")]
    result = process_snowflake_results(description, rows)
    assert result == [{"amount": 10.5, "count": 2, "name": "alpha"}]


def test_is_result_empty_with_none_or_zero_values():
    assert is_result_empty([{"count": 0, "sum": None}]) is True


def test_is_result_empty_with_real_data():
    assert is_result_empty([{"count": 1}]) is False
