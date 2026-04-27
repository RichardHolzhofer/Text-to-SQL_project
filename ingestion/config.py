from src.config.config import Config


class IngestionConfig(Config):
    """
    Specialized configuration for the ingestion pipeline.
    Inherits from the centralized Config and defaults to the 'ingestion' logger.
    """

    def __init__(self):
        super().__init__(logger_name="ingestion")


# Single instance to be shared across all ingestion modules
config = IngestionConfig()
logger = config.logger
