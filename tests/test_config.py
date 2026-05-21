import os
from unittest.mock import MagicMock, patch

import pytest

# Adjust this import string based on your exact project structure
from src.config.config import Config
from src.exceptions.exception import (
    ConfigError,
    SnowflakeConfigError,
)

# =====================================================================
# FIXTURES: Setting up a clean testing environment
# =====================================================================


@pytest.fixture
def mock_env_vars():
    """
    Provides a dictionary representing a complete, valid set of
    environment variables required by the Config class validation.
    """
    return {
        "SNOWFLAKE_ACCOUNT": "test_acc",
        "DBT_SNOWFLAKE_DATABASE": "test_db",
        "DBT_SNOWFLAKE_WAREHOUSE": "test_wh",
        "DBT_SNOWFLAKE_ROLE": "dbt_role",
        "AGENT_SNOWFLAKE_ROLE": "agent_role",
        "DBT_SNOWFLAKE_SCHEMA": "dbt_schema",
        "AGENT_SNOWFLAKE_SCHEMA": "agent_schema",
        "DBT_SNOWFLAKE_USER": "dbt_user",
        "DBT_SNOWFLAKE_PASSWORD": "dbt_password",
        "AGENT_SNOWFLAKE_USER": "agent_user",
        "AGENT_SNOWFLAKE_PASSWORD": "agent_password",
        "SUPABASE_DB_URI": "postgres://supabase",
        "SUPABASE_URL": "https://supabase.co",
        "SUPABASE_ANON_KEY": "anon_key",
        "SUPABASE_SERVICE_ROLE_KEY": "service_key",
        "SMART_LLM_MODEL": "openai:gpt-4",
        "FAST_LLM_MODEL": "groq:llama3",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "LANGFUSE_PUBLIC_KEY": "lf_pub",
        "LANGFUSE_SECRET_KEY": "lf_sec",
        "OPENAI_API_KEY": "sk-mock-openai-key",
        "GROQ_API_KEY": "gsk-mock-groq-key",
        "LLM_TEMPERATURE": "0.7",
    }


@pytest.fixture
def clean_config(mock_env_vars):
    """
    Instantiates the Config class with a fully mocked environment.
    Prevents tests from bleeding into your actual machine's .env file.
    """
    with (
        patch("src.config.config.load_dotenv"),
        patch.dict(os.environ, mock_env_vars, clear=True),
    ):
        config_instance = Config(logger_name="test-logger")
        yield config_instance


# =====================================================================
# TESTS: Initialization and Variable Validation
# =====================================================================


def test_config_init_success(mock_env_vars):
    """Verifies all attributes map accurately when the environment is valid."""
    with (
        patch("src.config.config.load_dotenv"),
        patch.dict(os.environ, mock_env_vars, clear=True),
    ):
        config = Config()

        assert config.sf_account == "test_acc"
        assert config.temperature == 0.7
        assert (
            config.langgraph_url == "http://localhost:2024"
        )  # Verifies default value fallback


def test_config_missing_required_vars(mock_env_vars):
    """Ensures ConfigError is raised if a core required variable is completely missing."""
    mock_env_vars.pop("SNOWFLAKE_ACCOUNT")  # Strip out a critical key

    with (
        patch("src.config.config.load_dotenv"),
        patch.dict(os.environ, mock_env_vars, clear=True),
    ):
        with pytest.raises(ConfigError) as exc_info:
            Config()
        assert "Missing required environment variables: SNOWFLAKE_ACCOUNT" in str(
            exc_info.value
        )


def test_config_missing_conditional_provider_key(mock_env_vars):
    """Ensures dynamic provider checking throws errors if specific API keys are omitted."""
    mock_env_vars["SMART_LLM_MODEL"] = "openai:gpt-4"
    mock_env_vars.pop("OPENAI_API_KEY")  # Missing key for selected model provider

    with (
        patch("src.config.config.load_dotenv"),
        patch.dict(os.environ, mock_env_vars, clear=True),
    ):
        with pytest.raises(ConfigError) as exc_info:
            Config()
        assert "OPENAI_API_KEY" in str(exc_info.value)


# =====================================================================
# TESTS: Internal Utility Methods
# =====================================================================


@pytest.mark.parametrize(
    "env_val, default, expected",
    [
        ("1.5", 0.0, 1.5),
        ("", 0.5, 0.5),
        (None, 0.2, 0.2),
    ],
)
def test_parse_float_env_valid(clean_config, env_val, default, expected):
    """Tests variable float conversion boundaries (valid states, empty states, None states)."""
    with patch("os.getenv", return_value=env_val):
        result = clean_config._parse_float_env("TEST_VAR", default)
        assert result == expected


def test_parse_float_env_invalid(clean_config):
    """Tests that a clean descriptive ConfigError is raised if conversion fails."""
    with patch("os.getenv", return_value="not-a-float"):
        with pytest.raises(ConfigError) as exc_info:
            clean_config._parse_float_env("TEST_VAR", default=0.0)
        assert "must be a valid number" in str(exc_info.value)


# =====================================================================
# TESTS: Third-Party Connections & SDK Factories (Mocked Execution)
# =====================================================================


def test_get_snowflake_connection_read_vs_write(clean_config):
    """Verifies credential split mapping logic for Read-only vs Admin credentials."""
    with patch("snowflake.connector.connect") as mock_connect:
        # Test default read-only credentials routing
        clean_config.get_snowflake_connection(write_access=False)
        mock_connect.assert_called_with(
            user="agent_user",
            password="agent_password",
            account="test_acc",
            warehouse="test_wh",
            database="test_db",
            role="agent_role",
            schema="agent_schema",
        )

        # Test dbt write credentials routing
        clean_config.get_snowflake_connection(write_access=True)
        mock_connect.assert_called_with(
            user="dbt_user",
            password="dbt_password",
            account="test_acc",
            warehouse="test_wh",
            database="test_db",
            role="dbt_role",
            schema="dbt_schema",
        )


def test_get_snowflake_connection_failure(clean_config):
    """Ensures unexpected SDK errors are caught and re-raised as custom abstractions."""
    with patch("snowflake.connector.connect", side_effect=Exception("Network Timeout")):
        with pytest.raises(SnowflakeConfigError):
            clean_config.get_snowflake_connection()


def test_get_supabase_connection_success(clean_config):
    """Confirms correct API key routing depending on user permission level requested."""
    with patch("src.config.config.create_client") as mock_create_client:
        clean_config.get_supabase_connection(write_access=False)
        mock_create_client.assert_called_with("https://supabase.co", "anon_key")

        clean_config.get_supabase_connection(write_access=True)
        mock_create_client.assert_called_with("https://supabase.co", "service_key")


# =====================================================================
# TESTS: Caching / Singleton Mechanics
# =====================================================================


def test_llm_caching_singleton(clean_config):
    """Validates that asset initialization singletons perform caching correctly."""
    mock_llm_instance = MagicMock()
    with patch(
        "src.config.config.init_chat_model", return_value=mock_llm_instance
    ) as mock_init:
        # First call hits initialization code block
        llm1 = clean_config.get_smart_llm()
        # Second call returns internal instance attribute directly
        llm2 = clean_config.get_smart_llm()

        assert llm1 is llm2
        mock_init.assert_called_once()  # Crucial: Assures underlying setup code wasn't hit twice


# =====================================================================
# TESTS: File Loading Parsers
# =====================================================================


def test_load_yaml_success(clean_config, tmp_path):
    """Ensures file parser handles clean filesystem contexts reliably."""
    test_yaml_file = tmp_path / "test_config.yml"
    test_yaml_file.write_text("app:\n  timeout: 30\n  debug: true", encoding="utf-8")

    parsed_data = clean_config.load_yaml(test_yaml_file)
    assert parsed_data == {"app": {"timeout": 30, "debug": True}}


def test_load_yaml_not_found(clean_config):
    """Ensures missing files yield clear exceptional bubbling wrapped in ConfigError."""
    with pytest.raises(ConfigError):
        clean_config.load_yaml("non_existent_file_path.yml")
