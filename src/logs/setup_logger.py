"""
🪵 Global Logging Configuration Module
--------------------------------------

This module handles the initialization of the Python standard logging system,
establishing a consistent logging environment for the entire application.

🧠 Purpose:
    To define and apply a standardized logging configuration (format, output stream,
    and level) to the root logger. This ensures that all modules emitting logs
    share a unified output style and behavior.

🔧 Core Functionalities:
    • Configure the root logger with a specific severity level (default: INFO)
    • direct log output to the standard output stream (stdout)
    • Apply a structured format including timestamps, log levels, and logger names
    • Implement idempotency to prevent duplicate handlers if called multiple times

🎯 Intended Use:
    • Invoked automatically by `src.logs.logger` upon import
    • Can be called manually at the entry point of the application to override defaults

📁 Dependencies:
    • logging
    • sys

📝 Notes:
    This configuration targets the root logger (`logging.getLogger()`). Consequently,
    all child loggers inherit this configuration unless explicitly overridden.

Author: Andrea Moleri
File Location: src/logs/setup_logger.py
Last Modified: 20/11/2025
"""

import logging
import sys


def setup_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a standardized stream handler and formatter.

    This function establishes the global logging state for the application.
    It is designed to be idempotent: if the root logger already possesses
    configured handlers, the function terminates immediately to prevent
    duplicate logging or configuration conflicts.

    Parameters
    ----------
    level : int, optional
        The threshold level for the logger (e.g., logging.INFO, logging.DEBUG).
        Messages below this priority will be ignored. Defaults to logging.INFO.

    Returns
    -------
    None
    """
    # Retrieve the root logger instance to apply global configuration.
    root = logging.getLogger()

    # Idempotency check: If the root logger already has handlers attached,
    # we assume it has been configured previously. Returning here prevents
    # the accumulation of duplicate handlers (which causes repeated log lines).
    if root.handlers:
        return

    # Set the global logging threshold.
    root.setLevel(level)

    # Define the log message format.
    # Structure: Timestamp | Level (padded to 8 chars) | Logger Name | Message
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Define the timestamp format (YYYY-MM-DD HH:MM:SS).
    date = "%Y-%m-%d %H:%M:%S"

    # Initialize a StreamHandler to direct logs to standard output (console).
    # Using sys.stdout ensures logs are captured by standard pipe mechanisms.
    handler = logging.StreamHandler(sys.stdout)

    # Apply the formatter to the handler to enforce the defined visual structure.
    handler.setFormatter(logging.Formatter(fmt, datefmt=date))

    # Attach the configured handler to the root logger.
    root.addHandler(handler)