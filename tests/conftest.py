import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.config.config import Config
from src.database.db import SupabaseDB


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
def sample_state_factory():
    def _factory(**overrides):
        base = {
            "router": None,
            "answer": None,
            "is_general_conversation": False,
            "validator": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    return _factory
