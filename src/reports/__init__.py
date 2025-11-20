"""
📦 Reports Package Initialization
---------------------------------

This module serves as the initialization layer for the reports package, 
implementing a facade pattern to simplify the import interface for external 
consumers.

🧠 Purpose:
    To flatten the package namespace, allowing developers to import key 
    reporting utilities directly from `src.reports` without referencing 
    internal submodules (e.g., `pdf_report`).

🔧 Core Functionalities:
    • Expose the `generate_pdf_report` function as the primary public API
    • Restrict wildcard imports using `__all__` to maintain namespace cleanliness

🎯 Intended Use:
    • `from src.reports import generate_pdf_report`

Author: Andrea Moleri
File Location: src/reports/__init__.py
Last Modified: 20/11/2025
"""

# Import the primary report generation function to expose it at the package level
from .pdf_report import generate_pdf_report

# Define the public API of this package
# This controls what is imported when using `from src.reports import *`
# and serves as documentation for which symbols are intended for public use.
__all__ = ["generate_pdf_report"]