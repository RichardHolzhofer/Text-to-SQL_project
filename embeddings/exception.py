from src.exceptions.exception import PipelineException


class EmbeddingsException(PipelineException):
    """
    Base exception class for the embeddings package.
    """

    def __init__(self, error_message: Exception):
        super().__init__(error_message)


class EmbeddingCreationError(EmbeddingsException):
    """Raised when creating embeddings fails."""

    pass
