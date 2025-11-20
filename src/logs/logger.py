"""
🪵 Logging Accessor Module
-------------------------

This module serves as the primary interface for retrieving logger instances
within the application context, ensuring consistent configuration and access
patterns across the codebase.

🧠 Purpose:
    The module abstracts the instantiation of `logging.Logger` objects, ensuring
    that the underlying logging subsystem is correctly initialized via a side-effect
    import mechanism before any logging operations occur.

🔧 Core Functionalities:
    • Initialize the logging configuration upon module import
    • Provide a factory function to retrieve named logger instances
    • Implement a fallback mechanism to default to the module name if no specific
      name is provided

🎯 Intended Use:
    • To be imported by any module requiring logging capabilities
    • Used to instantiate loggers that adhere to the global configuration defined
      in `setup_logger`

📁 Dependencies:
    • logging
    • src.logs.setup_logger

📝 Notes:
    The initialization call to `setup_logger()` occurs at the module level.
    Therefore, simply importing this module will trigger the configuration logic.
    This design relies on `setup_logger` being idempotent.

Author: Andrea Moleri
File Location: src/logs/logger.py
Last Modified: 20/11/2025
"""

import logging
from .setup_logger import setup_logger

# Execute the logging configuration immediately upon module import.
# This ensures that the logging subsystem is fully initialized before any
# logger is requested or used by consuming modules.
setup_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Retrieve a configured logger instance associated with the specified name.

    This function wraps the standard `logging.getLogger` method, providing a
    convenient default behavior where the logger name defaults to the current
    module's name if no specific identifier is provided.

    Parameters
    ----------
    name : str | None, optional
        The name of the logger to retrieve. This typically corresponds to the
        `__name__` of the calling module to track the source of log messages.
        If `None` or an empty string is provided, the function defaults to using
        `__name__` (the name of this module).

    Returns
    -------
    logging.Logger
        An instance of `logging.Logger` configured according to the global
        logging setup.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started.")
    """
    # Leverage boolean short-circuit logic to select the logger name.
    # If 'name' is falsy (None or empty string), '__name__' is used as the fallback.
    return logging.getLogger(name or __name__)