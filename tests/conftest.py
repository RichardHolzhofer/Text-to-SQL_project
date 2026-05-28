import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config.config import Config
from src.database.db import SupabaseDB
from src.nodes.node import TextToSQLNodes
from src.states.state import Column, SQLGenerator, Table


@pytest.fixture
def minimal_required_env():
    env = {
        "SNOWFLAKE_ACCOUNT": "test_account",
        "DBT_SNOWFLAKE_DATABASE": "test_db",
        "DBT_SNOWFLAKE_WAREHOUSE": "test_wh",
        "DBT_SNOWFLAKE_ROLE": "test_dbt_role",
        "AGENT_SNOWFLAKE_ROLE": "test_agent_role",
        "DBT_SNOWFLAKE_SCHEMA": "test_dbt_schema",
        "AGENT_SNOWFLAKE_SCHEMA": "test_agent_schema",
        "DBT_SNOWFLAKE_USER": "test_dbt_user",
        "DBT_SNOWFLAKE_PASSWORD": "test_dbt_pass",
        "AGENT_SNOWFLAKE_USER": "test_agent_user",
        "AGENT_SNOWFLAKE_PASSWORD": "test_agent_pass",
        "SUPABASE_DB_URI": "postgresql://user:pass@localhost:5432/db",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_ANON_KEY": "anon_key",
        "SUPABASE_SERVICE_ROLE_KEY": "service_role_key",
        "SMART_LLM_MODEL": "openai:gpt-4.1-mini",
        "FAST_LLM_MODEL": "groq:llama3",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "LANGFUSE_PUBLIC_KEY": "pk_test",
        "LANGFUSE_SECRET_KEY": "sk_test",
        "OPENAI_API_KEY": "test_openai_key",
    }
    with patch.dict(os.environ, env, clear=True):
        yield env


@pytest.fixture
def config(minimal_required_env):
    with patch.dict(os.environ, minimal_required_env, clear=True):
        yield Config()


@pytest.fixture
def mock_supabase_factory():
    with patch("src.config.config.create_client") as mock_create_client:
        client = Mock(name="supabase_client")
        mock_create_client.return_value = client
        yield mock_create_client, client


@pytest.fixture
def mock_db(config):
    mock_auth = Mock()

    mock_supabase = Mock()
    mock_supabase.auth = mock_auth

    config.get_supabase_connection = Mock(return_value=mock_supabase)

    db = SupabaseDB(config)

    return db, mock_supabase, mock_auth


@pytest.fixture
def mock_run_prompt():
    with patch("src.nodes.node.run_prompt") as mock_run_prompt:
        yield mock_run_prompt


@pytest.fixture
def node(config):
    with patch("src.nodes.node.SupabaseDB"):
        yield TextToSQLNodes(config)


@pytest.fixture
def mock_load_yaml():
    with patch("src.nodes.node.load_yaml") as mock_load_yaml:
        yield mock_load_yaml


@pytest.fixture
def mock_dump_yaml():
    with patch("src.nodes.node.dump_yaml") as mock_dump_yaml:
        yield mock_dump_yaml


@pytest.fixture
def mock_schema_validate():
    with patch("src.nodes.node.Schema.model_validate") as mock_validate:
        yield mock_validate


@pytest.fixture
def table_factory():
    def _factory(**overrides):
        data = {
            "table_name": "customers",
            "description": "customer table",
            "grain": "one row per customer",
            "primary_key": ["customer_id"],
            "columns": [
                Column(
                    column_name="customer_id",
                    description="customer id",
                )
            ],
        }

        data.update(overrides)

        return Table(**data)

    return _factory


@pytest.fixture
def sql_generator_factory():
    def _factory(**overrides):
        base = {
            "sql_query": "SELECT 1",
            "thought_process": "thinking...",
            "fuzzy_match_warning": None,
            "search_concept": None,
            "query_vector": None,
            "unsupported_explanation": None,
        }
        base.update(overrides)
        return SQLGenerator(**base)

    return _factory


@pytest.fixture
def mock_snowflake_connection():
    mock_cursor = Mock()

    mock_cursor_cm = MagicMock()
    mock_cursor_cm.__enter__.return_value = mock_cursor
    mock_cursor_cm.__exit__.return_value = None

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor_cm

    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None

    return mock_conn, mock_cursor


@pytest.fixture
def validator_factory():
    def _factory(**overrides):
        base = {
            "is_valid_query": True,
            "error_message": None,
            "iteration_count": 0,
        }

        base.update(overrides)

        def create_validator(data):
            v = Mock()

            v.is_valid_query = data["is_valid_query"]
            v.error_message = data["error_message"]
            v.iteration_count = data["iteration_count"]

            def model_copy(update=None):
                new_data = data.copy()

                if update:
                    new_data.update(update)

                return create_validator(new_data)

            v.model_copy.side_effect = model_copy

            return v

        return create_validator(base)

    return _factory
