"""
🤖 SCAFFOLD Baseline Module
---------------------------

This module implements the SCAFFOLD (Stochastic Controlled Averaging for
Federated Learning) algorithm, a method designed to mitigate client drift
in heterogeneous data environments.

🧠 Purpose:
    To improve convergence in Non-IID settings by maintaining and utilizing
    control variates (c_global and c_local). These variates estimate the
    update direction of the global model and correct local gradient updates
    accordingly.

🔧 Core Functionalities:
    • State management for global (server-side) and local (client-side) control variates.
    • Gradient correction during local training: g_corrected = g - c_i + c.
    • Computation of control variate updates based on model drift.
    • Aggregation of both model weights and control variate deltas.

🎯 Intended Use:
    • Federated Learning research scenarios involving high data heterogeneity.
    • Comparative benchmarking against standard FedAvg implementations.

📝 Reference:
    Karimireddy, S. P., et al. (2020). "SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning". International Conference on Machine
    Learning (ICML).

Author: Andrea Moleri
File Location: src/models/scaffold.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import logging
import copy
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

from models.baselines.base import (
    FederatedBaseline,
    evaluate_single_classifier,
    ensemble_accuracy,
)

logger = logging.getLogger(__name__)


class ScaffoldBaseline(FederatedBaseline):
    """
    Implements the SCAFFOLD algorithm for Federated Learning.

    This class extends the FederatedBaseline to include the management of
    control variates, which are used to reduce the variance of local client
    updates caused by data heterogeneity.
    """

    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        """
        Initialize the SCAFFOLD baseline strategy.

        Parameters
        ----------
        args : Any
            Configuration object containing training hyperparameters (e.g.,
            momentum, weight decay, learning rate).
        num_classes : int
            The number of target classes for the classification task.
        chans : int
            The number of input channels for the model.
        device : torch.device
            The computation device (CPU or GPU) where training operations occur.
        """
        super().__init__(args, num_classes, chans, device)
        self.current_round = 0

        # Optimization parameters extracted from configuration
        # SCAFFOLD typically benefits from zero momentum to strictly follow control variates.
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.0))
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 0.0))
        self._max_grad_norm = float(getattr(self.args, "baseline_clip_grad_norm", 0.0))

        # Normalize gradient clipping value; disable if non-positive.
        if self._max_grad_norm <= 0:
            self._max_grad_norm = None

        # --- SCAFFOLD State Initialization ---

        # Global Control Variate (c): Estimates the average gradient direction.
        # Initialized lazily upon the first access since the model structure is not yet known.
        self.global_c: Dict[str, torch.Tensor] = {}

        # Client Control Variates (c_i): Estimates the client-specific gradient direction.
        # Structure: Map[client_name -> Map[param_name -> Tensor]]
        # NOTE: Stored on CPU to minimize GPU memory consumption during the federation lifecycle.
        self.client_controls: Dict[str, Dict[str, torch.Tensor]] = {}

        logger.info(
            "[SCAFFOLD] Initialized | momentum=%.2f (typ. 0) | weight_decay=%.1e",
            self._momentum, self._weight_decay
        )

    def _get_or_init_control(self, client_name: str, model: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieve or initialize the control variates for the global state and the specific client.

        This method ensures that control variates exist and match the shape of the
        provided model parameters. It performs lazy initialization if the variates
        are empty.

        Parameters
        ----------
        client_name : str
            The unique identifier for the client.
        model : nn.Module
            The client's local model instance, used to determine parameter shapes.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            A tuple containing:
            1. The global control variate (c) moved to the active device.
            2. The local client control variate (c_i) moved to the active device.
        """
        # 1. Initialize Global Control Variate if it has not been created yet.
        # We assume zero initialization for the first round.
        if not self.global_c:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.global_c[k] = torch.zeros_like(v, device="cpu")

        # 2. Initialize Client Control Variate if missing for this specific client.
        if client_name not in self.client_controls:
            self.client_controls[client_name] = {}
            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.client_controls[client_name][k] = torch.zeros_like(v, device="cpu")

        # 3. Transfer control variates to the target computation device (e.g., GPU).
        # This is necessary because the canonical storage is on CPU to save memory.
        c_global_dev = {k: v.to(self.device) for k, v in self.global_c.items()}
        c_local_dev = {k: v.to(self.device) for k, v in self.client_controls[client_name].items()}

        return c_global_dev, c_local_dev

    def _effective_lr(self, round_num: int) -> float:
        """
        Calculate the effective learning rate for the current training round.

        Applies an exponential decay factor based on the round number.

        Parameters
        ----------
        round_num : int
            The current federated learning round.

        Returns
        -------
        float
            The calculated learning rate.
        """
        base_lr = float(getattr(self.args, "learning_rate", 0.1))
        round_decay = float(getattr(self.args, "baseline_round_lr_decay", 1.0))

        # Clamp the decay factor to ensure it remains within the valid range [0.0, 1.0].
        round_decay = max(0.0, min(1.0, round_decay))

        return base_lr * (round_decay ** round_num)

    def train_client(
            self,
            client_name: str,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            round_num: int
    ) -> Tuple[Dict[str, Any], int]:
        """
        Perform local training on a specific client using the SCAFFOLD algorithm.

        This method performs standard SGD updates but corrects the gradients using
        the difference between the global and local control variates. It also
        computes the update to the local control variate to be sent back to the server.

        Parameters
        ----------
        client_name : str
            The unique identifier of the client being trained.
        train_loader : DataLoader
            The data loader for the training dataset.
        val_loader : DataLoader
            The data loader for the validation dataset (unused in this specific logic but required by signature).
        round_num : int
            The current federated round index.

        Returns
        -------
        Tuple[Dict[str, Any], int]
            A tuple containing:
            1. A state dictionary containing updated model weights and control variate deltas.
            2. The number of samples in the training dataset.

        Raises
        ------
        RuntimeError
            If the global model has not been initialized prior to training.
        """
        if self.global_model is None:
            raise RuntimeError("[SCAFFOLD] Global model not initialized.")

        model = self.client_models[client_name]

        # Preserve initial weights to calculate model drift later.
        # These must be detached and cloned to avoid reference modification.
        initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.to(self.device)

        # Retrieve the relevant control variates for this client.
        c_global, c_local = self._get_or_init_control(client_name, model)

        # Configure optimizer parameters.
        lr = self._effective_lr(round_num)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Handle edge case: empty training loader.
        if train_loader is None or len(train_loader.dataset) == 0:
            return initial_weights, 0

        num_samples = len(train_loader.dataset)
        epochs = int(getattr(self.args, "baseline_epochs_per_round", 1))

        # --- Local Training Loop ---
        model.train()
        steps_performed = 0

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # --- SCAFFOLD Gradient Correction ---
                # The standard gradient is modified to account for drift.
                # Formula: grad_corrected = grad - c_i + c
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Apply the control variate difference to the gradient.
                        param.grad.data += (c_global[name] - c_local[name])

                # Apply gradient clipping if configured.
                if self._max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm)

                optimizer.step()
                steps_performed += 1

        # --- Update Local Control Variate (c_i) ---
        # We compute the new local control variate based on the drift between
        # the initial global model (x) and the trained local model (y_i).
        # Formula: c_i_new = c_i - c + (1 / (K * eta)) * (x - y_i)

        # Calculate the scaling factor (1 / (K * eta)).
        factor = 1.0 / (steps_performed * lr) if steps_performed > 0 and lr > 0 else 0.0

        # Prepare the state dictionary to return to the server.
        # Note: We must embed the control variate updates (deltas) within this dictionary
        # to transport them via the existing aggregation infrastructure.
        return_state = {k: v.cpu() for k, v in model.state_dict().items()}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # x_param: Initial weights (global model state at round start).
                # y_param: Local weights (after training).
                x_param = initial_weights[name].to(self.device)
                y_param = param.data

                # Compute the new local control variate.
                # c_new = c_local - c_global + factor * (x - y)
                c_new = c_local[name] - c_global[name] + factor * (x_param - y_param)

                # Compute the delta to send to the server: delta_c = c_new - c_local.
                delta_c = c_new - c_local[name]

                # Update the client's persistent local storage for the next round.
                self.client_controls[client_name][name] = c_new.cpu()

                # Pack the delta into the return dictionary with a reserved prefix.
                return_state[f"__ctrl__{name}"] = delta_c.cpu()

        # Offload model to CPU to conserve GPU resources.
        model.cpu()
        return return_state, num_samples

    def aggregate(self, round_num: int, client_updates: Dict[str, Tuple[dict, int]]):
        """
        Aggregate model updates and control variate deltas from participating clients.

        Parameters
        ----------
        round_num : int
            The current federated round number.
        client_updates : Dict[str, Tuple[dict, int]]
            A dictionary mapping client names to tuples of (state_dict, number of samples).
            The state_dict contains both model weights and control variate deltas.
        """
        if not client_updates:
            return

        # 1. Separate Model Weights from Control Variate Updates.
        model_updates = []
        control_updates = []  # List of dictionaries containing control deltas.

        # Total number of clients in the entire federation (N).
        # This is critical for the mathematically correct update of the global control variate.
        total_clients_population = len(self.client_models)

        # Number of clients that participated in this specific round (|S|).
        participating_clients_count = len(client_updates)

        for c_name, (state, n_samples) in client_updates.items():
            # Filter standard model weights.
            weights = {k: v for k, v in state.items() if not k.startswith("__ctrl__")}

            # Filter control deltas (keys prefixed with '__ctrl__').
            ctrls = {k.replace("__ctrl__", ""): v for k, v in state.items() if k.startswith("__ctrl__")}

            # Convert weights to NumPy for Flower aggregation compatibility.
            weights_np = [v.numpy() for v in weights.values()]
            model_updates.append((weights_np, n_samples))

            control_updates.append(ctrls)

        # 2. Aggregate Model Weights using Standard Weighted Averaging (FedAvg).
        if model_updates:
            agg_weights_np = flwr_aggregate(model_updates)

            # Retrieve keys to map aggregated numpy arrays back to the state dict.
            # We use the keys from the first client's update as a reference.
            state_keys = [k for k in client_updates[list(client_updates.keys())[0]][0].keys()
                          if not k.startswith("__ctrl__")]

            new_state = {}
            for k, arr in zip(state_keys, agg_weights_np):
                new_state[k] = torch.from_numpy(arr)

            self.global_model.load_state_dict(new_state)

        # 3. Aggregate Control Variates.
        # Update Rule: c_global_new = c_global + (1/N) * sum_{i in S} (delta_c_i)
        # Note: The scaling factor is 1/N (total population), not 1/|S|.
        # This ensures the global control variate remains an unbiased estimator.

        fraction = 1.0 / total_clients_population

        if control_updates:
            for name in self.global_c:
                delta_sum = torch.zeros_like(self.global_c[name])

                # Sum the deltas from all participating clients.
                for c_deltas in control_updates:
                    if name in c_deltas:
                        delta_sum += c_deltas[name]

                # Update the global control variate.
                self.global_c[name] += delta_sum * fraction

        # 4. Synchronize all clients with the new global model state.
        global_sd = self.global_model.state_dict()
        for c_name in self.client_models:
            self.client_models[c_name].load_state_dict(global_sd)

        logger.info(
            f"[SCAFFOLD] Round {round_num + 1} aggregated. Global Controls updated with scaling factor {fraction:.4f}"
        )

    def evaluate_client(self, client_name: str, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate a specific client's model against a test dataset.

        Parameters
        ----------
        client_name : str
            The identifier of the client to evaluate.
        test_loader : DataLoader
            The dataset loader for evaluation.

        Returns
        -------
        float
            The accuracy of the model on the test set.
        """
        # Standard evaluation procedure.
        model = self.client_models[client_name]

        if len(test_loader.dataset) == 0:
            return 0.0

        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        model.cpu()

        return ensemble_accuracy(y_true, y_pred)

    def get_global_model_accuracy(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate the global model's performance.

        Parameters
        ----------
        test_loader : DataLoader
            The dataset loader for evaluation.

        Returns
        -------
        float
            The accuracy of the global model.
        """
        return self.evaluate(test_loader)