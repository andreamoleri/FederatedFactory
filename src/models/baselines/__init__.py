"""
🧩 Federated Learning Baselines Interface
-----------------------------------------

This module serves as the primary entry point for the federated learning 
baselines package. It exposes a collection of standardized algorithms 
and abstract base classes designed to facilitate reproducible research 
and comparative analysis in distributed learning environments.

🧠 Purpose:
    To aggregate and expose core federated learning strategies, allowing 
    researchers to import foundational algorithms from a single, unified 
    namespace without navigating the internal directory structure.

🔧 Core Functionalities:
    • Export the abstract base class for federated strategies
    • Expose implementations of standard algorithms (FedAvg, FedProx)
    • Expose advanced distillation and dynamic regularization methods (FedDF, FedDyn)
    • Expose control variate methods (SCAFFOLD)

🎯 Intended Use:
    • Benchmarking new federated learning algorithms against established standards
    • Educational demonstrations of distributed optimization techniques
    • Modular integration into larger federated simulation frameworks

📁 Dependencies:
    • Internal modules: base, fedavg, fedprox, feddf, feddyn, scaffold

Author: Andrea Moleri
File Location: src/models/baselines/__init__.py
Last Modified: 06/12/2025
"""

# Import the abstract base class which defines the interface for all federated strategies.
# This facilitates polymorphism and enforces a consistent API across different algorithms.
from .base import FederatedBaseline

# Import the Federated Averaging (FedAvg) algorithm, the canonical baseline
# for federated learning (McMahan et al., 2017).
from .fedavg import FedAvgBaseline

# Import FedProx, which introduces a proximal term to the local objective function
# to tackle statistical heterogeneity (Li et al., 2018).
from .fedprox import FedProxBaseline

# Import FedDF (Ensemble Distillation for Robust Model Fusion), which leverages
# knowledge distillation to aggregate models (Lin et al., 2020).
from .feddf import FedDFBaseline

# Import FedDyn (Federated Learning with Dynamic Regularization), which aligns
# global and local objective functions (Acar et al., 2021).
from .feddyn import FedDynBaseline

# Import SCAFFOLD, which uses control variates to reduce client drift
# (Karimireddy et al., 2020).
from .scaffold import ScaffoldBaseline

# Define the public API of this module.
# This list controls the behavior of 'from module import *' and explicitly declares
# which symbols are intended for external use, hiding implementation details.
__all__ = [
    "FederatedBaseline",
    "FedAvgBaseline",
    "FedProxBaseline",
    "FedDFBaseline",
    "FedDynBaseline",
    "ScaffoldBaseline",
]