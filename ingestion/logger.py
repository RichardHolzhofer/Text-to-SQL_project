import logging
import os
from datetime import datetime

# Define log file name with timestamp or static name
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Setup named logger to avoid conflicts with other apps
logger = logging.getLogger("ingestion")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
formatter = logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)

# Add only the file handler as per user request
logger.addHandler(file_handler)
