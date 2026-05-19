import sys


def error_message_detail(error: Exception) -> str:
    """
    Extract detailed error information including file name and line number
    from the active traceback.
    """
    _, _, exc_tb = sys.exc_info()

    if exc_tb is None:
        return f"Error occurred with message [{error}]"

    file_name = exc_tb.tb_frame.f_code.co_filename

    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{error}]"
    )


class PipelineException(Exception):
    """
    Base exception class for shared pipeline error handling.
    """

    def __init__(self, error_message: Exception):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)

    def __str__(self) -> str:
        return self.error_message


class ConfigError(PipelineException):
    """Raised when loading runtime configuration fails."""


class SnowflakeConfigError(PipelineException):
    """Raised when creating a Snowflake connection fails."""


class EmbeddingsException(PipelineException):
    """Base exception class for the embeddings package."""


class EmbeddingCreationError(EmbeddingsException):
    """Raised when creating embeddings fails."""


class IngestionException(PipelineException):
    """Base exception class for the ingestion package."""


class SnowflakeConnectionError(IngestionException):
    """Raised when there is an issue connecting to Snowflake or during file transfer."""


class DataLoadError(IngestionException):
    """Raised when a COPY INTO command fails to load data into a table."""


class NodeException(PipelineException):
    """Base exception class for the nodes package."""


class SchemaBuildError(NodeException):
    """Raised when the unified schema build process fails."""


class SQLGenerationError(NodeException):
    """Raised when the SQL generation process fails."""


class SupabaseConnectionError(PipelineException):
    """Raised when there is an issue connecting to Supabase or during file transfer."""


class SupabaseAuthError(PipelineException):
    """Raised when Supabase authentication fails."""


class SupabaseQueryError(PipelineException):
    """Raised when a Supabase database query fails."""


class NetworkRetryExhaustedError(PipelineException):
    """Raised when a Supabase/HTTP client call still fails after transient retries."""


class SnowflakeResultProcessingError(PipelineException):
    """Raised when converting Snowflake cursor rows to dicts fails."""


class PromptSyncError(PipelineException):
    """Raised when the prompt synchronization process fails."""


class LLMException(PipelineException):
    """Base exception class for LLM-related errors."""


class LLMInvocationError(LLMException):
    """Raised when the LLM invocation (chain.invoke) fails."""


class LLMTemplateError(LLMException):
    """Raised when loading or parsing a prompt template fails."""


class YAMLProcessingError(PipelineException):
    """Raised when loading, dumping, or saving YAML files fails."""


class GuardrailException(PipelineException):
    """Base exception class for guardrail-related errors."""


class GuardrailInputError(GuardrailException):
    """Raised when an error occurs while scanning user input."""


class GuardrailOutputError(GuardrailException):
    """Raised when an error occurs while scanning LLM output."""


class MCPException(PipelineException):
    """Base exception class for all MCP server related errors."""


class MCPConfigError(MCPException):
    """Raised when critical MCP credentials or configuration variables are missing."""


class LangGraphExecutionError(MCPException):
    """Raised when communication with the LangGraph run API fails."""


class GraphResponseError(MCPException):
    """Raised when the graph returns an invalid, empty, or unexpected response format."""
