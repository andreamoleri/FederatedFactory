# src/models/baselines/base.py

"""
🧩 Federated Learning Baseline Interface
----------------------------------------

This module defines the foundational infrastructure for federated learning
baselines. It provides a base class that standardizes the management of
global and local models, along with utility functions for model evaluation
and metric computation.

🧠 Purpose:
    To establish a common interface for various federated optimization
    strategies (e.g., FedAvg, FedProx), ensuring consistent initialization,
    state management, and evaluation protocols across experiments.

🔧 Core Functionalities:
    • centralized model initialization and distribution to clients
    • Standardized evaluation routines for PyTorch models
    • Abstract state management for training history and model weights

🎯 Intended Use:
    • Subclassed by specific algorithms (e.g., `FedAvgBaseline`)
    • Used by orchestration scripts to manage the lifecycle of a federated round

📁 Dependencies:
    • numpy
    • torch
    • models.cnn (SimpleCNN)

📝 Notes:
    This class adheres to a synchronous federated learning paradigm where
    the global model acts as the source of truth for initializing client models.

Author: Andrea Moleri
File Location: src/models/baselines/base.py
Last Modified: 20/11/2025
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.cnn import SimpleCNN

# Configure module-level logger for tracking evaluation events and warnings.
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_single_classifier(
        model: nn.Module, ld, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a single classifier on a given dataset and returns true and
    predicted labels.

    This function performs a forward pass over the data provided by the loader
    without tracking gradients, ensuring memory efficiency during inference.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate.
    ld : Iterable
        The data loader yielding batches of (input, label) pairs.
        Typically a torch.utils.data.DataLoader.
    device : torch.device
        The computational device (CPU or GPU) where the evaluation is performed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - The first array contains the ground truth labels (y_true).
        - The second array contains the predicted class indices (y_pred).
    """
    # Transfer model to the specified computational device and set to evaluation mode.
    # This disables dropout and batch normalization updates.
    model.to(device).eval()

    y_true, y_pred = [], []

    # Context manager disables gradient calculation to reduce memory usage and speed up computations.
    with torch.no_grad():
        for x, y in ld:
            y_true.append(y)
            # Compute predictions: move input to device, perform forward pass,
            # extract the index of the maximum logit (class label), and move result to CPU.
            y_pred.append(model(x.to(device)).argmax(1).cpu())

    # Move the model back to CPU to free up GPU memory for other processes or models.
    model.cpu()

    # Concatenate list of tensors into a single numpy array for metric calculation.
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def ensemble_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the simple classification accuracy between true and predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
        Array of ground truth class labels.
    y_pred : np.ndarray
        Array of predicted class labels.

    Returns
    -------
    float
        The proportion of correct predictions, ranging from 0.0 to 1.0.
    """
    return float((y_true == y_pred).mean())


class FederatedBaseline:
    """
    Minimal base class for federated learning baselines.

    This class encapsulates the shared logic required for federated algorithms,
    such as model initialization and global evaluation. Unlike abstract base
    classes that enforce method implementation via `NotImplementedError`, this
    class allows subclasses (e.g., `FedAvgBaseline`) to implement only the
    specific methods required for their strategy.

    Attributes
    ----------
    args : Any
        Configuration arguments containing hyperparameters and settings.
    num_classes : int
        The number of output classes for the classification task.
    chans : int
        The number of input channels (e.g., 3 for RGB images, 1 for grayscale).
    device : torch.device
        The primary device for tensor computations.
    client_models : Dict[str, nn.Module]
        A registry mapping client identifiers to their respective local models.
    global_model : Optional[nn.Module]
        The central model aggregated from client updates.
    history : Dict[str, Dict]
        A nested dictionary storing training metrics (loss, accuracy) over time.
    """

    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        """
        Initializes the FederatedBaseline instance.

        Parameters
        ----------
        args : Any
            An object or namespace containing runtime configuration parameters.
        num_classes : int
            The dimensionality of the output layer.
        chans : int
            The dimensionality of the input layer (number of channels).
        device : torch.device
            The target device for model operations.
        """
        self.args = args
        self.num_classes = num_classes
        self.chans = chans
        self.device = device

        # Registry for storing individual client models.
        self.client_models: Dict[str, nn.Module] = {}

        # The central model shared among clients; initialized in `initialize_models`.
        self.global_model: Optional[nn.Module] = None

        # Metric tracking container.
        self.history: Dict[str, Dict] = {"train_loss": {}, "val_acc": {}}

    def initialize_models(self, client_names):
        """
        Initializes the global model and instantiates a local model for each client.

        This method ensures synchronization by copying the state dictionary
        of the newly created global model to all client models, effectively
        starting the federation with identical weights (standard FedAvg behavior).

        Parameters
        ----------
        client_names : Iterable[str]
            A list or iterable of unique identifiers for the participating clients.
        """
        # Instantiate (or reset) the global model architecture.
        self.global_model = SimpleCNN(self.chans, self.num_classes)
        global_state = self.global_model.state_dict()

        # Iterate through each client to create their local model instance.
        # Critical: Load the global state dict to ensure identical starting points.
        for client_name in client_names:
            m = SimpleCNN(self.chans, self.num_classes)
            m.load_state_dict(global_state)
            self.client_models[client_name] = m

    def evaluate(self, test_loader) -> float:
        """
        Evaluates the current global model on the provided test dataset.

        Parameters
        ----------
        test_loader : Iterable
            The data loader for the test dataset.

        Returns
        -------
        float
            The classification accuracy of the global model. Returns 0.0 if
            the model is uninitialized or the dataset is empty.
        """
        # Guard clause: Prevent execution if the global model has not been initialized.
        if self.global_model is None:
            logger.warning("[BASELINE] Global model is None, accuracy = 0.0")
            return 0.0

        # Guard clause: Prevent execution on an empty dataset to avoid division by zero errors.
        if len(test_loader.dataset) == 0:
            logger.warning("[BASELINE] Test loader is empty; evaluation cannot proceed.")
            return 0.0

        # Transfer global model to the computation device.
        self.global_model.to(self.device)

        # Perform inference to get ground truth and predictions.
        y_true, y_pred = evaluate_single_classifier(
            self.global_model, test_loader, self.device
        )

        # Compute accuracy metric.
        accuracy = ensemble_accuracy(y_true, y_pred)

        # Return global model to CPU to conserve accelerator memory.
        self.global_model.cpu()

        return accuracy