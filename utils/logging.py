"""
Logging setup utilities.
"""

import os
import logging
from datetime import datetime


def init_logging(log_dir="logs"):
    """
    Initialize logging to both console and a timestamped log file.

    Args:
        log_dir: Directory for log files.

    Returns:
        Path to the created log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode="w", encoding="utf-8"),
        ],
    )

    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename
