import logging
import os
from datetime import datetime

# Define log file name with timestamp or static name
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes to the shared project log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == file_handler.baseFilename
        for handler in logger.handlers
    ):
        logger.addHandler(file_handler)

    return logger


logger = get_logger("main")
