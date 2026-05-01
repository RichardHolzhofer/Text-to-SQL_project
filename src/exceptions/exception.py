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
