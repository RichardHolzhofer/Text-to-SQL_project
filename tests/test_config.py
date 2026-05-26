import os
from unittest.mock import Mock, patch

import pytest

from src.config.config import Config
from src.exceptions.exception import (
    ConfigError,
    SnowflakeConfigError,
    SupabaseConnectionError
)


def _build_config(minimal_required_env):
    with patch("src.config.config.get_logger") as mock_get_logger:
        mock_get_logger.return_value = Mock()
        return Config()


def test_config_parses_valid_temperature(minimal_required_env):
    with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.35"}, clear=False):
        config = _build_config(minimal_required_env)
    assert config.temperature == 0.35


def test_config_uses_default_temperature_when_missing(
    minimal_required_env
):
    with patch.dict(os.environ, {"LLM_TEMPERATURE": ""}, clear=False):
        config = _build_config(minimal_required_env)
    assert config.temperature == 0.0


def test_config_raises_for_invalid_temperature(
    minimal_required_env
):
    with patch.dict(os.environ, {"LLM_TEMPERATURE": "not-a-number"}, clear=False):
        with pytest.raises(ConfigError):
            _build_config(minimal_required_env)


def test_config_raises_when_required_var_missing(
    minimal_required_env
):
    with patch.dict(os.environ, {"SNOWFLAKE_ACCOUNT": ""}, clear=False):
        with pytest.raises(ConfigError):
            _build_config(minimal_required_env)

def test_validate_config_raises_when_required_var_missing(
    minimal_required_env,
):
    env = minimal_required_env.copy()
    env["SNOWFLAKE_ACCOUNT"] = ""

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError):
            Config()
            
def test_validate_config_raises_when_openai_api_key_missing(
    minimal_required_env,
):
    env = minimal_required_env.copy()
    env["OPENAI_API_KEY"] = ""

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError):
            Config()
            
def test_validate_config_raises_when_groq_api_key_missing(
    minimal_required_env,
):
    env = minimal_required_env.copy()
    env["GROQ_API_KEY"] = ""

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError):
            Config()
            
def test_validate_config_requires_openai_key_when_embedding_model_is_set(
    minimal_required_env,
):
    env = minimal_required_env.copy()

    env["OPENAI_API_KEY"] = ""
    env["SMART_LLM_MODEL"] = "groq:llama3"
    env["FAST_LLM_MODEL"] = "groq:llama3"

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError) as exc:
            Config()

    assert "OPENAI_API_KEY" in str(exc.value)
    
def test_load_yaml_returns_dict(config, tmp_path):
    file_path = tmp_path / "config.yml"
    file_path.write_text("""
name: test_app
version: 1
enabled: true
""")

    result = config.load_yaml(file_path)

    assert result == {
        "name": "test_app",
        "version": 1,
        "enabled": True,
    }
    
def test_load_yaml_file_does_not_exist(config, tmp_path):
    missing_file = tmp_path / "missing.yml"

    with pytest.raises(ConfigError) as exc:
        config.load_yaml(missing_file)

    assert "Configuration file not found" in str(exc.value)
    
def test_load_yaml_invalid_syntax_raises_config_error(config, tmp_path):
    file_path = tmp_path / "invalid.yml"
    file_path.write_text("""
                        key: value:
                        broken_yaml
                        """)

    with pytest.raises(ConfigError):
        config.load_yaml(file_path)

def test_load_ingestion_config_calls_load_yaml(config):
    with patch.object(config, "load_yaml") as mock_load_yaml:
        mock_load_yaml.return_value = {"dummy": "data"}

        config.load_ingestion_config()

    assert mock_load_yaml.called

def test_get_snowflake_connection_uses_agent_credentials(
    config,
):
    fake_connection = Mock(name="snowflake_connection")

    with patch("src.config.config.snowflake.connector.connect") as mock_connect:
        mock_connect.return_value = fake_connection

        config.get_snowflake_connection()

    mock_connect.assert_called_once_with(
        user="test_agent_user",
        password="test_agent_pass",
        account="test_account",
        warehouse="test_wh",
        database="test_db",
        role="test_agent_role",
        schema="test_agent_schema",
    )
    
def test_get_snowflake_connection_uses_dbt_credentials(
    config,
):
    with patch("src.config.config.snowflake.connector.connect") as mock_connect:
        config.get_snowflake_connection(write_access=True)

    mock_connect.assert_called_once_with(
        user="test_dbt_user",
        password="test_dbt_pass",
        account="test_account",
        warehouse="test_wh",
        database="test_db",
        role="test_dbt_role",
        schema="test_dbt_schema",
    )
    
def test_get_snowflake_connection_raises_custom_error_on_failure(
    config,
):
    with patch("src.config.config.snowflake.connector.connect") as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(SnowflakeConfigError):
            config.get_snowflake_connection()
            
def test_get_supabase_connection_uses_anon_key(
    config,
    mock_supabase_factory,
):
    mock_create_client, client = mock_supabase_factory

    result = config.get_supabase_connection()

    mock_create_client.assert_called_once_with(
        "https://example.supabase.co",
        "anon_key",
    )

    assert result is client
    
def test_get_supabase_connection_uses_service_role_key(
    config,
    mock_supabase_factory,
):
    mock_create_client, client = mock_supabase_factory

    result = config.get_supabase_connection(write_access=True)

    mock_create_client.assert_called_once_with(
        "https://example.supabase.co",
        "service_role_key",
    )

    assert result is client
    
def test_get_supabase_connection_raises_when_anon_key_missing(
    minimal_required_env,
    config,
    mock_supabase_factory,
):
    env = minimal_required_env.copy()
    env["SUPABASE_ANON_KEY"] = "anon_key"

    mock_create_client, _ = mock_supabase_factory

    config.sb_anon_key = ""

    with pytest.raises(SupabaseConnectionError):
        config.get_supabase_connection()
        
def test_get_supabase_connection_raises_on_sdk_failure(
    config,
    mock_supabase_factory,
):
    mock_create_client, _ = mock_supabase_factory
    mock_create_client.side_effect = Exception("Supabase down")

    with pytest.raises(SupabaseConnectionError):
        config.get_supabase_connection()

def test_get_smart_llm_initializes_model(config):
    fake_llm = Mock(name="llm")

    with patch("src.config.config.init_chat_model") as mock_init:
        mock_init.return_value = fake_llm

        result = config.get_smart_llm()

    mock_init.assert_called_once_with(
        config.smart_model,
        temperature=config.temperature,
    )

    assert result is fake_llm
    
def test_get_fast_llm_initializes_model(config):
    fake_llm = Mock(name="fast_llm")

    with patch("src.config.config.init_chat_model") as mock_init:
        mock_init.return_value = fake_llm

        result = config.get_fast_llm()

    mock_init.assert_called_once_with(
        config.fast_model,
        temperature=config.temperature,
    )

    assert result is fake_llm
    

def test_get_smart_llm_caches_instance(config):
    fake_llm = Mock(name="llm")

    with patch("src.config.config.init_chat_model") as mock_init:
        mock_init.return_value = fake_llm

        result1 = config.get_smart_llm()
        result2 = config.get_smart_llm()

    mock_init.assert_called_once_with(
        config.smart_model,
        temperature=config.temperature,
    )

    assert result1 is result2
    
def test_get_fast_llm_caches_instance(config):
    fake_llm = Mock(name="llm")

    with patch("src.config.config.init_chat_model") as mock_init:
        mock_init.return_value = fake_llm

        result1 = config.get_fast_llm()
        result2 = config.get_fast_llm()

    mock_init.assert_called_once_with(
        config.fast_model,
        temperature=config.temperature,
    )

    assert result1 is result2

def test_get_embedding_model_initializes_once(config):
    fake_embedding = Mock(name="embedding_model")

    with patch("src.config.config.OpenAIEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = fake_embedding

        result = config.get_embedding_model()

    mock_embeddings.assert_called_once_with(
        model=config.embedding_model
    )

    assert result is fake_embedding

def test_get_embedding_model_caches_instance(config):
    fake_embedding = Mock(name="embedding_model")

    with patch("src.config.config.OpenAIEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = fake_embedding

        result1 = config.get_embedding_model()
        result2 = config.get_embedding_model()

    mock_embeddings.assert_called_once_with(
        model=config.embedding_model
    )

    assert result1 is result2
    
def test_get_langfuse_initializes_client(config):
    fake_client = Mock(name="langfuse_client")

    with patch("src.config.config.Langfuse") as mock_langfuse:
        mock_langfuse.return_value = fake_client

        result = config.get_langfuse()

    mock_langfuse.assert_called_once_with(
        public_key=config.lf_public_key,
        secret_key=config.lf_secret_key,
        host=config.lf_host,
    )

    assert result is fake_client
    
def test_get_langfuse_caches_instance(config):
    fake_client = Mock(name="langfuse_client")

    with patch("src.config.config.Langfuse") as mock_langfuse:
        mock_langfuse.return_value = fake_client

        result1 = config.get_langfuse()
        result2 = config.get_langfuse()

    mock_langfuse.assert_called_once_with(
        public_key=config.lf_public_key,
        secret_key=config.lf_secret_key,
        host=config.lf_host,
    )

    assert result1 is result2
    
def test_get_langfuse_raises_config_error_on_failure(config):
    with patch("src.config.config.Langfuse") as mock_langfuse:
        mock_langfuse.side_effect = Exception("Langfuse failed")

        with pytest.raises(ConfigError):
            config.get_langfuse()
            
def test_reload_mcp_credentials_updates_state(config):
    with patch("src.config.config.load_dotenv") as mock_load:
        # override env AFTER initial config load
        with patch.dict(os.environ, {
            "MCP_USER_ID": "user123",
            "MCP_USER_EMAIL": "test@example.com",
            "MCP_USER_PASSWORD": "secret"
        }):
            result = config.reload_mcp_credentials()

    mock_load.assert_called_once_with(override=True)

    assert config.mcp_user_id == "user123"
    assert config.mcp_user_email == "test@example.com"
    assert config.mcp_user_password == "secret"

    assert result == ("user123", "test@example.com", "secret")
    
def test_reload_mcp_credentials_raises_config_error_on_failure(config):
    with patch("src.config.config.load_dotenv") as mock_load:
        mock_load.side_effect = Exception("dotenv failure")

        with pytest.raises(ConfigError):
            config.reload_mcp_credentials()