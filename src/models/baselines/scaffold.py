"""
🤖 SCAFFOLD Baseline Module
---------------------------

This module implements the SCAFFOLD algorithm (Stochastic Controlled Averaging
for Federated Learning).

🧠 Purpose:
    To address client drift in heterogeneous data settings by utilizing
    control variates (c_global and c_local) to correct local gradient updates.

🔧 Core Functionalities:
    • Maintain global control variate (c) and per-client control variates (c_i).
    • Correct local gradients: g_corrected = g - c_i + c.
    • Update local controls based on the drift between local and global models.
    • Aggregate model updates and control variate updates.

Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
for Federated Learning" (ICML 2020).
"""

from __future__ import annotations

import logging
import copy
from typing import Dict, Tuple, Optional

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
    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        super().__init__(args, num_classes, chans, device)
        self.current_round = 0

        # Optimization params
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.0))  # SCAFFOLD typically uses 0 momentum
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 0.0))
        self._max_grad_norm = float(getattr(self.args, "baseline_clip_grad_norm", 0.0))
        if self._max_grad_norm <= 0:
            self._max_grad_norm = None

        # --- SCAFFOLD State ---
        # Global Control Variate (c)
        # We initialize this when we first create the global model in initialize_models,
        # but since we don't have the model yet, we defer initialization to the first access.
        self.global_c: Dict[str, torch.Tensor] = {}

        # Client Control Variates (c_i)
        # Mapping: client_name -> {param_name: Tensor}
        # Stored on CPU to save GPU memory.
        self.client_controls: Dict[str, Dict[str, torch.Tensor]] = {}

        logger.info(
            "[SCAFFOLD] Initialized | momentum=%.2f (typ. 0) | weight_decay=%.1e",
            self._momentum, self._weight_decay
        )

    def _get_or_init_control(self, client_name: str, model: nn.Module) -> Tuple[Dict, Dict]:
        """
        Ensure global_c and client_controls[client_name] are initialized
        matching the model shape. Returns copies on the correct device.
        """
        # 1. Initialize Global C if empty
        if not self.global_c:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.global_c[k] = torch.zeros_like(v, device="cpu")

        # 2. Initialize Client C if missing
        if client_name not in self.client_controls:
            self.client_controls[client_name] = {}
            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.client_controls[client_name][k] = torch.zeros_like(v, device="cpu")

        # 3. Return device-bound copies for training
        c_global_dev = {k: v.to(self.device) for k, v in self.global_c.items()}
        c_local_dev = {k: v.to(self.device) for k, v in self.client_controls[client_name].items()}

        return c_global_dev, c_local_dev

    def _effective_lr(self, round_num: int) -> float:
        base_lr = float(getattr(self.args, "learning_rate", 0.1))
        round_decay = float(getattr(self.args, "baseline_round_lr_decay", 1.0))
        # Clamp decay
        round_decay = max(0.0, min(1.0, round_decay))
        return base_lr * (round_decay ** round_num)

    def train_client(
            self,
            client_name: str,
            train_loader,
            val_loader,
            round_num: int
    ) -> Tuple[dict, int]:

        if self.global_model is None:
            raise RuntimeError("[SCAFFOLD] Global model not initialized.")

        model = self.client_models[client_name]
        initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.to(self.device)

        # Get Control Variates
        c_global, c_local = self._get_or_init_control(client_name, model)

        # LR & Optimizer
        lr = self._effective_lr(round_num)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        if train_loader is None or len(train_loader.dataset) == 0:
            return initial_weights, 0

        num_samples = len(train_loader.dataset)
        epochs = int(getattr(self.args, "baseline_epochs_per_round", 1))

        # Training Loop
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
                # grad = grad - c_i + c
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Add control variate difference
                        param.grad.data += (c_global[name] - c_local[name])

                if self._max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm)

                optimizer.step()
                steps_performed += 1

        # --- Update Local Control Variate (c_i) ---
        # Formula: c_i^+ = c_i - c + (1 / K * eta) * (x - y_i)
        # Where K = steps, eta = learning rate
        # But we need to compute delta_c = c_i^+ - c_i to send to server

        # K * eta
        factor = 1.0 / (steps_performed * lr) if steps_performed > 0 and lr > 0 else 0.0

        # We need to compute the update for the control variates
        # And embed it into the state_dict to return it to the aggregator
        # without changing the method signature in baseline_runner.

        return_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Calculate Delta Controls
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Global params (x) are in initial_weights (on CPU)
                # Local params (y_i) are in model (on Device)

                x_param = initial_weights[name].to(self.device)
                y_param = param.data

                # New Local Control: c_new = c_local - c_global + factor * (x - y)
                c_new = c_local[name] - c_global[name] + factor * (x_param - y_param)

                # Delta: dc = c_new - c_local
                delta_c = c_new - c_local[name]

                # Update local storage (for next round)
                self.client_controls[client_name][name] = c_new.cpu()

                # Pack delta into return dict with a special prefix
                return_state[f"__ctrl__{name}"] = delta_c.cpu()

        model.cpu()
        return return_state, num_samples

    def aggregate(self, round_num: int, client_updates: Dict[str, Tuple[dict, int]]):
        if not client_updates:
            return

        # 1. Separate Model Weights from Control Updates
        model_updates = []
        control_updates = []  # List of dicts

        total_clients = len(self.client_models)  # N in paper
        participating_clients = len(client_updates)

        for c_name, (state, n_samples) in client_updates.items():
            # Extract standard weights
            weights = {k: v for k, v in state.items() if not k.startswith("__ctrl__")}

            # Extract control deltas
            ctrls = {k.replace("__ctrl__", ""): v for k, v in state.items() if k.startswith("__ctrl__")}

            # Prepare for Flower Aggregation (NumPy)
            weights_np = [v.numpy() for v in weights.values()]
            model_updates.append((weights_np, n_samples))

            control_updates.append(ctrls)

        # 2. Aggregate Model Weights (FedAvg of weights)
        # Note: SCAFFOLD paper uses simple average for x update if learning rates are tuned
        # but weighted avg is standard for non-IID in frameworks like Flower.
        if model_updates:
            agg_weights_np = flwr_aggregate(model_updates)

            # Load back into global model
            state_keys = [k for k in client_updates[list(client_updates.keys())[0]][0].keys()
                          if not k.startswith("__ctrl__")]

            new_state = {}
            for k, arr in zip(state_keys, agg_weights_np):
                new_state[k] = torch.from_numpy(arr)

            self.global_model.load_state_dict(new_state)

        # 3. Aggregate Control Variates
        # c_global = c_global + (1 / N) * Sum(delta_c_i)
        # Note: In standard SCAFFOLD, we divide by Total Clients (N), not just participating.
        # This assumes non-participating clients have delta_c = 0.

        if control_updates:
            for name in self.global_c:
                delta_sum = torch.zeros_like(self.global_c[name])
                for c_deltas in control_updates:
                    if name in c_deltas:
                        delta_sum += c_deltas[name]

                # Update global control
                # Factor: 1 / |S| is technically aggregation, but SCAFFOLD implies
                # averaging over participating subset effectively estimates the total drift.
                # Standard impl: c <- c + (1/N) * sum(delta)
                # Here we assume partial participation approximation: c <- c + (1/|S|) * sum(delta) * (|S|/N)
                # Simplified: average of deltas.

                avg_delta = delta_sum / participating_clients
                self.global_c[name] += avg_delta

        # 4. Sync Clients
        global_sd = self.global_model.state_dict()
        for c_name in self.client_models:
            self.client_models[c_name].load_state_dict(global_sd)

        logger.info(f"[SCAFFOLD] Round {round_num + 1} aggregated {participating_clients} clients.")

    def evaluate_client(self, client_name, test_loader):
        # Standard evaluation
        model = self.client_models[client_name]
        if len(test_loader.dataset) == 0: return 0.0
        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        model.cpu()
        return ensemble_accuracy(y_true, y_pred)

    def get_global_model_accuracy(self, test_loader):
        return self.evaluate(test_loader)