"""
🪵 Logging Utilities Interface
------------------------------

This module serves as the initialization entry point for the logging package,
defining the public application programming interface (API) for log management.

🧠 Purpose:
    It encapsulates the internal structure of the `logs` package, allowing
    consumer modules to import key functionalities—specifically the logger
    factory—directly from the package namespace without referencing internal
    submodules.

🔧 Core Functionalities:
    • Re-export the `get_logger` factory function for streamlined access
    • explicit declaration of public symbols via `__all__` to control namespace
      pollution

🎯 Intended Use:
    • Standard import path for logging across the application:
      `from src.logs import get_logger`

Author: Andrea Moleri
File Location: src/logs/__init__.py
Last Modified: 20/11/2025
"""

from .logger import get_logger

# explicitly define the public API of the package.
# This restricts the symbols exported when `from src.logs import *` is used,
# ensuring encapsulation of internal implementation details.
__all__ = ["get_logger"]