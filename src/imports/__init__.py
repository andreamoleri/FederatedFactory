"""
💾 Data Component Exports Module
--------------------------------

This module acts as the centralized interface for the data subsystem, aggregating 
and re-exporting essential components related to dataset management and 
data augmentation.

🧠 Purpose:
    To implement a 'Facade' pattern for the package, allowing external consumers 
    to import key classes and functions directly from the package namespace 
    without needing to know the internal file structure (e.g., separating 
    management logic from augmentation logic).

🔧 Core Functionalities:
    • Expose dataset metadata and retrieval factories  
    • Provide noise injection and data augmentation primitives  
    • Export data normalization and transformation utilities  

🎯 Intended Use:
    • Training pipelines requiring streamlined access to data loaders  
    • Preprocessing scripts necessitating consistent transform definitions  
    • Research environments where data abstraction is required  

📁 Dependencies:
    • .data_management  
    • .data_augmentation  

Author: Andrea Moleri
File Location: src/imports/__init__.py
Last Modified: 06/12/2025
"""

# Import data management primitives.
# These components handle the static definition of dataset properties
# and the logic required to instantiate dataset objects from disk or memory.
from .data_management import DATASET_META, get_dataset

# Import data augmentation and transformation strategies.
# These components are responsible for the dynamic modification of data samples,
# including noise injection (Gaussian), pipeline construction, and 
# tensor renormalization.
from .data_augmentation import (
    AddGaussianNoise,
    NoisyCleanDataset,
    build_transform,
    denormalize,
)

# Define the module's public application programming interface (API).
# This list explicitly controls which symbols are exported when a consumer
# executes `from src.imports import *`. It restricts the namespace to only
# the intended architectural components, hiding internal implementation details.
__all__ = [
    "DATASET_META",
    "get_dataset",
    "AddGaussianNoise",
    "NoisyCleanDataset",
    "build_transform",
    "denormalize",
]