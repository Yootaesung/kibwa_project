import logging
import os
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_level = logging.INFO

# Create a custom logger
logger = logging.getLogger("kibwa_chatbot")
logger.setLevel(log_level)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    LOGS_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Set levels
console_handler.setLevel(log_level)
file_handler.setLevel(log_level)

# Create formatters and add it to handlers
formatter = logging.Formatter(log_format)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
