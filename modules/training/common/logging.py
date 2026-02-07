"""
Logging utilities for training.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Setup logging for training.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_to_file: Whether to log to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger("koe-tts")
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
