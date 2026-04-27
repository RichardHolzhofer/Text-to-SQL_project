from src.config.config import Config


class EmbeddingsConfig(Config):
    """
    Specialized configuration for the embeddings pipeline.
    Inherits from the centralized Config and defaults to the 'embeddings' logger.
    """

    def __init__(self):
        super().__init__(logger_name="embeddings")


# Single instance to be shared across all embedding modules
config = EmbeddingsConfig()
logger = config.logger
