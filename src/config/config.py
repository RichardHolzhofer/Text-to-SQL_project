import os
from pathlib import Path
from typing import Any, Dict

import snowflake.connector
import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langfuse import Langfuse
from snowflake.connector.connection import SnowflakeConnection
from supabase import Client, create_client

from src.exceptions.exception import (
    ConfigError,
    SnowflakeConfigError,
    SupabaseConnectionError,
)
from src.logger.logger import get_logger


class Config:
    """
    Centralized configuration helper for the text-to-sql app.
    Should be instantiated in each process with a specific logger name.
    """

    def __init__(self, logger_name: str = "text-to-sql"):
        self.logger = get_logger(logger_name)
        load_dotenv()
        self.logger.info("Environment variables loaded from .env")

        # Snowflake connection settings
        self.sf_account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.sf_database = os.getenv("DBT_SNOWFLAKE_DATABASE")
        self.sf_warehouse = os.getenv("DBT_SNOWFLAKE_WAREHOUSE")
        self.sf_dbt_role = os.getenv("DBT_SNOWFLAKE_ROLE")
        self.sf_agent_role = os.getenv("AGENT_SNOWFLAKE_ROLE")
        self.sf_dbt_schema = os.getenv("DBT_SNOWFLAKE_SCHEMA")
        self.sf_agent_schema = os.getenv("AGENT_SNOWFLAKE_SCHEMA")

        # Identity settings (DBT vs Agent)
        self.sf_dbt_user = os.getenv("DBT_SNOWFLAKE_USER")
        self.sf_dbt_pass = os.getenv("DBT_SNOWFLAKE_PASSWORD")

        self.sf_agent_user = os.getenv("AGENT_SNOWFLAKE_USER")
        self.sf_agent_pass = os.getenv("AGENT_SNOWFLAKE_PASSWORD")

        # Supabase settings for persistant memory handling
        self.sb_project_name = os.getenv("SUPABASE_PROJECT_NAME")
        self.sb_db_uri = os.getenv("SUPABASE_DB_URI")
        self.sb_url = os.getenv("SUPABASE_URL")
        self.sb_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.sb_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.sb_password = os.getenv("SUPABASE_PASSWORD")

        # Langfuse settings for prompt management and tracing
        self.lf_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.lf_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.lf_host = os.getenv("LANGFUSE_HOST") or os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )

        # LLM Model settings
        self.smart_model = os.getenv("SMART_LLM_MODEL")
        self.fast_model = os.getenv("FAST_LLM_MODEL")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
        self.temperature = self._parse_float_env("LLM_TEMPERATURE", default=0.0)

        # MCP Authentication settings (optional, loaded dynamically)
        self.mcp_user_id = os.getenv("MCP_USER_ID")
        self.mcp_user_email = os.getenv("MCP_USER_EMAIL")
        self.mcp_user_password = os.getenv("MCP_USER_PASSWORD")

        # Validation (Optional)
        self._validate_config()

    def _parse_float_env(self, var_name: str, default: float) -> float:
        value = os.getenv(var_name)
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError as error:
            raise ConfigError(
                f"{var_name} must be a valid number, got: {value}"
            ) from error

    def _validate_config(self):
        """Ensure critical environment variables are present."""
        required = [
            "SNOWFLAKE_ACCOUNT",
            "DBT_SNOWFLAKE_DATABASE",
            "DBT_SNOWFLAKE_WAREHOUSE",
            "DBT_SNOWFLAKE_ROLE",
            "AGENT_SNOWFLAKE_ROLE",
            "DBT_SNOWFLAKE_SCHEMA",
            "AGENT_SNOWFLAKE_SCHEMA",
            "DBT_SNOWFLAKE_USER",
            "DBT_SNOWFLAKE_PASSWORD",
            "AGENT_SNOWFLAKE_USER",
            "AGENT_SNOWFLAKE_PASSWORD",
            "SUPABASE_DB_URI",
            "SUPABASE_URL",
            "SUPABASE_ANON_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SMART_LLM_MODEL",
            "FAST_LLM_MODEL",
            "EMBEDDING_MODEL",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        ]
        missing = [var for var in required if not os.getenv(var)]

        provider_key_requirements = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        for model in [self.smart_model, self.fast_model]:
            if not model or ":" not in model:
                continue
            provider = model.split(":", 1)[0].lower()
            api_key = provider_key_requirements.get(provider)
            if api_key and not os.getenv(api_key):
                missing.append(api_key)

        if self.embedding_model and not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")

        missing = sorted(set(missing))
        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    def load_yaml(self, config_path: str | Path) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        path = Path(config_path).resolve()
        self.logger.info(f"Loading YAML configuration from {path}")
        try:
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            with path.open("r", encoding="utf-8") as stream:
                content = yaml.safe_load(stream) or {}
            if not isinstance(content, dict):
                raise TypeError(f"Expected mapping at root of YAML file: {path}")
            return content
        except Exception as error:
            self.logger.exception(f"Failed to load YAML configuration: {path}")
            raise ConfigError(error) from error

    def load_ingestion_config(self) -> Dict[str, Any]:
        """Load ingestion config from ingestion/config.yml."""
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "ingestion" / "config.yml"
        return self.load_yaml(config_path)

    def get_snowflake_connection(
        self, write_access: bool = False
    ) -> SnowflakeConnection:
        """
        Initialize and return a Snowflake connection.

        Args:
            write_access (bool): If True, uses DBT credentials.
                                 If False (default), uses AGENT credentials.
        """
        try:
            access_type = "DBT (Write)" if write_access else "AGENT (Read-Only)"
            self.logger.info(f"Creating Snowflake connection with {access_type} access")

            user = self.sf_dbt_user if write_access else self.sf_agent_user
            password = self.sf_dbt_pass if write_access else self.sf_agent_pass
            role = self.sf_dbt_role if write_access else self.sf_agent_role
            schema = self.sf_dbt_schema if write_access else self.sf_agent_schema

            return snowflake.connector.connect(
                user=user,
                password=password,
                account=self.sf_account,
                warehouse=self.sf_warehouse,
                database=self.sf_database,
                role=role,
                schema=schema,
            )

        except Exception as error:
            self.logger.exception(f"Snowflake connection failed for {access_type} user")
            raise SnowflakeConfigError(error) from error

    def get_supabase_connection(self, write_access: bool = False) -> Client:
        """
        Initialize and return a Supabase Client.

        Args:
            write_access (bool): If True, uses the SERVICE_ROLE_KEY (bypasses RLS, can manage schema).
                                 If False (default), uses the ANON_KEY (subject to RLS).
        """
        try:
            access_type = "SERVICE_ROLE (Admin)" if write_access else "ANON (User)"
            self.logger.info(f"Creating Supabase client with {access_type} access")

            key = self.sb_service_role_key if write_access else self.sb_anon_key

            if not key:
                raise ConfigError(
                    f"Supabase {access_type} key is missing from configuration."
                )
            return create_client(self.sb_url, key)

        except Exception as error:
            self.logger.exception(
                f"Supabase connection failed for {access_type} access"
            )
            raise SupabaseConnectionError(error) from error

    def get_smart_llm(self):
        """Returns a cached instance of the smart LLM."""
        if not hasattr(self, "_smart_llm") or self._smart_llm is None:
            self.logger.info(
                f"Initializing Smart LLM: {self.smart_model} with temperature={self.temperature}"
            )
            self._smart_llm = init_chat_model(
                self.smart_model, temperature=self.temperature
            )
        return self._smart_llm

    def get_fast_llm(self):
        """Returns a cached instance of the fast LLM."""
        if not hasattr(self, "_fast_llm") or self._fast_llm is None:
            self.logger.info(
                f"Initializing Fast LLM: {self.fast_model} with temperature={self.temperature}"
            )
            self._fast_llm = init_chat_model(
                self.fast_model, temperature=self.temperature
            )
        return self._fast_llm

    def get_llm(self, model_name: str):
        """
        Returns an instance of an LLM for a specific model name.
        Does NOT cache by default to allow dynamic parameter binding.
        """
        self.logger.info(f"Instantiating LLM for model: {model_name}")
        return init_chat_model(model_name)

    def get_langfuse(self):
        """Returns a cached Langfuse client singleton."""
        if not hasattr(self, "_langfuse") or self._langfuse is None:
            self.logger.info("Initializing Langfuse client singleton via Config")
            self._langfuse = Langfuse(
                public_key=self.lf_public_key,
                secret_key=self.lf_secret_key,
                host=self.lf_host,
            )
        return self._langfuse

    def get_embedding_model(self):
        """Returns a cached instance of the OpenAI embedding model."""
        if not hasattr(self, "_embedding_model") or self._embedding_model is None:
            self.logger.info(
                f"Initializing OpenAI Embedding model: {self.embedding_model}"
            )
            self._embedding_model = OpenAIEmbeddings(model=self.embedding_model)
        return self._embedding_model

    def reload_mcp_credentials(self) -> tuple[str | None, str | None, str | None]:
        """
        Reloads .env variables dynamically and returns the latest MCP credentials.
        """
        load_dotenv(override=True)
        self.mcp_user_id = os.getenv("MCP_USER_ID")
        self.mcp_user_email = os.getenv("MCP_USER_EMAIL")
        self.mcp_user_password = os.getenv("MCP_USER_PASSWORD")
        return self.mcp_user_id, self.mcp_user_email, self.mcp_user_password
