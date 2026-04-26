from src.exceptions.exception import PipelineException


class IngestionException(PipelineException):
    """
    Base exception class for the ingestion package.
    All custom pipeline errors inherit from this class.
    """

    def __init__(self, error_message: Exception):
        super().__init__(error_message)


# Specialized exception subclasses for better categorization of failures


class SnowflakeConnectionError(IngestionException):
    """Raised when there is an issue connecting to Snowflake or during file transfer."""

    pass


class ConfigError(IngestionException):
    """Raised when there is an issue with the config.yml file (missing or malformed)."""

    pass


class DataLoadError(IngestionException):
    """Raised when a COPY INTO command fails to load data into a table."""

    pass
