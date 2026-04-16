import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including file name and line number
    from a traceback.

    Args:
        error (Exception): The exception object caught in the 'except' block.
        error_detail (sys): The 'sys' module, which contains the 'exc_info()' method.

    Returns:
        str: A formatted error message string containing file, line, and message.
    """
    # exc_info returns (type, value, traceback). We only need the traceback (exc_tb)
    _, _, exc_tb = error_detail.exc_info()

    # Extract filename from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format the message to clearly show where and why the failure occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class IngestionException(Exception):
    """
    Base exception class for the ingestion package.
    All custom pipeline errors inherit from this class.
    """

    def __init__(self, error_message: str, error_detail: sys):
        # Initialize the base Exception class
        super().__init__(error_message)
        # Store a detailed, formatted error message using our helper function
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        # Return the detailed error message when the exception is printed or logged
        return self.error_message


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
