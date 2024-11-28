import logging
from datetime import datetime
import os

def setup_logging(result_dir: str,
                  name: str) -> logging.Logger:
    # Remove any existing logger with this name
    if name in logging.Logger.manager.loggerDict:
        logging.Logger.manager.loggerDict.pop(name)

    # Create logger with specific name instead of __name__
    logger = logging.getLogger(name)
    # Clear any existing handlers
    logger.handlers = []

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log_file = os.path.join(result_dir, f'{name}.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
