from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from src.exceptions.exception import YAMLProcessingError
from src.utils.ui_utils import generate_chat_title, load_and_filter_schema


def test_load_and_filter_schema_missing_file():
    result = load_and_filter_schema(
        "non_existent.yaml",
        exclude_tables=[],
        exclude_columns=[],
    )

    assert result == {}


def test_load_and_filter_schema_basic(tmp_path):
    schema = {
        "models": [
            {
                "name": "customers",
                "description": " customer table ",
                "columns": [
                    {"name": "customer_id", "tests": ["unique"]},
                    {"name": "age"},
                ],
            }
        ]
    }

    file = tmp_path / "schema.yaml"
    file.write_text(yaml.dump(schema))

    result = load_and_filter_schema(
        str(file),
        exclude_tables=[],
        exclude_columns=[],
    )

    assert "customers" in result

    table = result["customers"]

    assert table["description"] == "customer table"
    assert len(table["columns"]) == 2


def test_load_and_filter_schema_exclude_table(tmp_path):
    schema = {
        "models": [
            {"name": "customers", "columns": []},
            {"name": "orders", "columns": []},
        ]
    }

    file = tmp_path / "schema.yaml"
    file.write_text(yaml.dump(schema))

    result = load_and_filter_schema(
        str(file),
        exclude_tables=["customers"],
        exclude_columns=[],
    )

    assert "customers" not in result
    assert "orders" in result


def test_load_and_filter_schema_exclude_column(tmp_path):
    schema = {
        "models": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id"},
                    {"name": "secret_column"},
                ],
            }
        ]
    }

    file = tmp_path / "schema.yaml"
    file.write_text(yaml.dump(schema))

    result = load_and_filter_schema(
        str(file),
        exclude_tables=[],
        exclude_columns=["secret_column"],
    )

    cols = result["customers"]["columns"]

    assert all(c["name"] != "secret_column" for c in cols)


def test_load_and_filter_schema_column_types(tmp_path):
    schema = {
        "models": [
            {
                "name": "test",
                "columns": [
                    {"name": "created_at"},
                    {"name": "user_id"},
                    {"name": "price_value"},
                    {"name": "total_score"},
                    {"name": "name"},
                ],
            }
        ]
    }

    file = tmp_path / "schema.yaml"
    file.write_text(yaml.dump(schema))

    result = load_and_filter_schema(
        str(file),
        exclude_tables=[],
        exclude_columns=[],
    )

    cols = {c["name"]: c["type"] for c in result["test"]["columns"]}

    assert cols["created_at"] == "DATE"
    assert cols["user_id"] == "UUID"
    assert cols["price_value"] == "FLOAT"
    assert cols["total_score"] == "INT"
    assert cols["name"] == "TEXT"


def test_load_and_filter_schema_pk_fk_tags(tmp_path):
    schema = {
        "models": [
            {
                "name": "test",
                "columns": [
                    {
                        "name": "id",
                        "tests": ["unique"],
                    },
                    {
                        "name": "user_id",
                        "tests": [{"relationships": "users"}],
                    },
                ],
            }
        ]
    }

    file = tmp_path / "schema.yaml"
    file.write_text(yaml.dump(schema))

    result = load_and_filter_schema(
        str(file),
        exclude_tables=[],
        exclude_columns=[],
    )

    cols = result["test"]["columns"]

    pk = next(c for c in cols if c["name"] == "id")
    fk = next(c for c in cols if c["name"] == "user_id")

    assert "PK" in pk["tags"]
    assert "FK" in fk["tags"]


def test_load_and_filter_schema_invalid_yaml(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text("::: invalid yaml :::")

    with pytest.raises(YAMLProcessingError):
        load_and_filter_schema(
            str(file),
            exclude_tables=[],
            exclude_columns=[],
        )


def test_generate_chat_title_success():
    llm = Mock()

    response = Mock()
    response.content = '"Sales Overview"'

    chain = MagicMock()
    chain.invoke.return_value = response

    prompt = MagicMock()
    prompt.__or__.return_value = chain

    with patch(
        "langchain_core.prompts.PromptTemplate.from_template",
        return_value=prompt,
    ):
        result = generate_chat_title(llm, "What are total sales?")

    assert result == "Sales Overview"


def test_generate_chat_title_strips_quotes():
    llm = Mock()

    response = Mock()
    response.content = '"Revenue Analysis"'

    chain = MagicMock()
    chain.invoke.return_value = response

    prompt = MagicMock()
    prompt.__or__.return_value = chain

    with patch(
        "langchain_core.prompts.PromptTemplate.from_template",
        return_value=prompt,
    ):
        result = generate_chat_title(llm, "Show revenue")

    assert result == "Revenue Analysis"


def test_generate_chat_title_truncation():
    llm = Mock()

    response = Mock()
    response.content = "This is a very long chat conversation title example"

    chain = MagicMock()
    chain.invoke.return_value = response

    prompt = MagicMock()
    prompt.__or__.return_value = chain

    with patch(
        "langchain_core.prompts.PromptTemplate.from_template",
        return_value=prompt,
    ):
        result = generate_chat_title(llm, "question")

    assert result == "This is a very long..."
    assert len(result.split()) == 5


def test_generate_chat_title_fallback():
    llm = Mock()

    with patch(
        "langchain_core.prompts.PromptTemplate.from_template",
        side_effect=Exception("fail"),
    ):
        result = generate_chat_title(
            llm,
            "This is a very long question that should be truncated",
        )

    assert result == "This is a very long question t..."
