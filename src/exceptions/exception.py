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
