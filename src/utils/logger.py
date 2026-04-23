"""
Logging configuration for the experiment pipeline.

Sets up a dual-handler logger (console + file) so every pipeline run
is both visible in the terminal and persisted for later inspection.
"""

import logging
import os
from datetime import datetime


def setup_logger(output_dir: str = "results", level: int = logging.INFO) -> logging.Logger:
    """Create and configure the pipeline logger.

    Args:
        output_dir: Directory where log files are saved.
        level: Logging verbosity (default: INFO).

    Returns:
        Configured logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("say_it_differently")
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"experiment_{timestamp}.log")
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
