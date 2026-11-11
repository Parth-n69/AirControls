"""Logging configuration for AirControl.

Ensures diagnostics are written to disk with rotation, while strictly respecting
user privacy: NO camera frames or biometric images are ever saved, cached, or transmitted.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

LOGGER_NAME = "aircontrol"
LOG_FILENAME = "aircontrol.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3


def setup_logger(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Initialize and configure the root AirControl logger.

    Args:
        log_level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Target file path. Defaults to aircontrol.log in the current directory.
        console_output: Whether to also log to stdout.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if already initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(module)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    target_file = log_file or LOG_FILENAME
    try:
        file_handler = RotatingFileHandler(
            target_file,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:
        print(f"Warning: Failed to setup file logger at {target_file}: {exc}")

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("AirControl logger initialized. Privacy mode active: zero frames logged.")
    return logger


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """Retrieve an AirControl logger instance for a given submodule."""
    base_logger = logging.getLogger(LOGGER_NAME)
    if module_name:
        return base_logger.getChild(module_name)
    return base_logger

