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

🧪 Implementation Notes:
    • Optimization hyperparameters (momentum, weight decay, gradient clipping)
      now share the SAME defaults as the other baselines (FedAvg/FedProx/FedDyn):
        - baseline_momentum       (default: 0.9)
        - baseline_weight_decay   (default: 1e-4)
        - baseline_clip_grad_norm (default: 5.0)
      For canonical SCAFFOLD behaviour, you may set `baseline_momentum=0.0`
      via the CLI, but the interface is fully aligned for fair comparisons.

    • Numerical stability behaviour matches the other baselines:
        - NaN/Inf checks on loss, gradients, and model parameters.
        - Optional L2 gradient clipping (if baseline_clip_grad_norm > 0).
        - If instability is detected, the client update is discarded by
          restoring initial weights and returning num_samples=0.

Author: Andrea Moleri
File Location: src/models/scaffold.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import numpy as np
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

import math

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

        # Optimization parameters extracted from configuration.
        # Defaults are aligned with other baselines for fairness; users can still
        # set baseline_momentum=0.0 to obtain the more "classical" SCAFFOLD setup.
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.9))
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 1e-4))
        self._max_grad_norm = float(getattr(self.args, "baseline_clip_grad_norm", 5.0))

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
            "[SCAFFOLD] Initialized | momentum=%.3f | weight_decay=%.1e | "
            "clip_grad_norm=%s",
            self._momentum,
            self._weight_decay,
            str(self._max_grad_norm),
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _evaluate_on_loader(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on a specific DataLoader using standard CE loss.

        This uses the same evaluation semantics as other baselines (no
        control variates in the loss), to keep metrics comparable.

        Returns:
            (avg_loss, accuracy), or (NaN, NaN) if loader is empty or unstable.
        """
        if loader is None or len(loader.dataset) == 0:
            return float("nan"), float("nan")

        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = criterion(output, target)

            if not torch.isfinite(loss):
                logger.warning(
                    "[SCAFFOLD] Non-finite validation loss detected (loss=%s). "
                    "Setting val_loss/val_acc to NaN for this batch.",
                    loss.item(),
                )
                return float("nan"), float("nan")

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_examples += batch_size

        if total_examples == 0:
            return float("nan"), float("nan")

        avg_loss = total_loss / total_examples
        acc = total_correct / total_examples
        return float(avg_loss), float(acc)

    import math

    # Inside FedAvgBaseline class
    def _effective_lr(self, round_num: int) -> Tuple[float, float, float]:
        base_lr = float(getattr(self.args, "learning_rate", 0.1))
        max_rounds = int(getattr(self.args, "baseline_max_rounds", 200))

        # Cosine Annealing Logic
        # eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max))
        # Assuming eta_min = 0
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * round_num / max_rounds))

        return float(lr), base_lr, 0.0

    @staticmethod
    def _has_non_finite_params(model: nn.Module) -> bool:
        """
        Check if any parameter in the model contains NaN or Infinite values.
        """
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
        return False

    def _get_or_init_control(
        self,
        client_name: str,
        model: nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieve or initialize the control variates for the global state
        and the specific client.

        Ensures shapes match the model parameters, with lazy initialization.
        """
        # 1. Initialize Global Control Variate if it has not been created yet.
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

        # 3. Transfer control variates to the target computation device.
        c_global_dev = {k: v.to(self.device) for k, v in self.global_c.items()}
        c_local_dev = {
            k: v.to(self.device) for k, v in self.client_controls[client_name].items()
        }

        return c_global_dev, c_local_dev

    def state_dict(self):
        """
        Returns a dictionary containing the full state of the SCAFFOLD algorithm,
        including model weights and both global/local control variates.
        """
        return {
            'global_model': self.global_model.state_dict(),
            'global_c': self.global_c,
            'client_controls': self.client_controls
        }

    def load_state_dict(self, state):
        """
        Restores the SCAFFOLD state from a checkpoint.
        """
        # 1. Load Model Weights
        self.global_model.load_state_dict(state['global_model'])

        # 2. Load Global Control Variates
        self.global_c = state['global_c']

        # 3. Load Client Control Variates
        self.client_controls = state['client_controls']
    # ------------------------------------------------------------------ #
    # Local training (SCAFFOLD)                                          #
    # ------------------------------------------------------------------ #
    def train_client(
        self,
        client_name: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        round_num: int,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Perform local training on a specific client using the SCAFFOLD algorithm.

        The gradient is corrected using the difference between the global and
        local control variates:

            grad_corrected = grad - c_i + c

        Numerical stability behaviour matches the other baselines:
        - Non-finite CE loss / total loss → rollback and discard update.
        - Non-finite gradients → rollback and discard update.
        - Non-finite parameters after optimizer.step → rollback and discard update.
        """
        if self.global_model is None:
            raise RuntimeError("[SCAFFOLD] Global model not initialized.")

        model = self.client_models[client_name]

        # Preserve initial weights to calculate model drift and enable rollback.
        initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.to(self.device)

        # Retrieve control variates for this client.
        c_global, c_local = self._get_or_init_control(client_name, model)

        # Configure optimizer parameters.
        lr = self._effective_lr(round_num)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Handle edge case: empty training loader.
        if train_loader is None or len(train_loader.dataset) == 0:
            logger.warning(
                "[BASELINE/SCAFFOLD] Client %s | round %d: train_loader is empty; "
                "no update sent.",
                client_name,
                round_num + 1,
            )
            model.load_state_dict(initial_weights)
            model.cpu()

            key = f"{client_name}_round{round_num}"
            self.history.setdefault("train_loss", {})[key] = []
            self.history.setdefault("val_loss", {})[key] = []
            self.history.setdefault("val_acc", {})[key] = []

            state_dict_copy = {k: v.clone() for k, v in initial_weights.items()}
            return state_dict_copy, 0

        num_samples = len(train_loader.dataset)
        epochs = int(getattr(self.args, "baseline_epochs_per_round", 1))
        if epochs <= 0:
            logger.warning(
                "[BASELINE/SCAFFOLD] baseline_epochs_per_round=%d invalid; "
                "defaulting to 1 epoch per round.",
                epochs,
            )
            epochs = 1

        logger.info(
            "[BASELINE/SCAFFOLD] Client %s | round %d | epochs_per_round=%d | "
            "lr=%.6f | num_samples=%d | momentum=%.3f | weight_decay=%.1e | "
            "clip_grad_norm=%s",
            client_name,
            round_num + 1,
            epochs,
            lr,
            num_samples,
            self._momentum,
            self._weight_decay,
            str(self._max_grad_norm),
        )

        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_accs = []

        non_finite_detected = False
        steps_performed = 0

        # --- Local Training Loop ---
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                ce_loss = criterion(output, target)

                # Check for numerical instability in CE loss.
                if not torch.isfinite(ce_loss):
                    logger.error(
                        "[BASELINE/SCAFFOLD] Non-finite CE loss detected "
                        "on client %s | round %d | epoch %d: loss=%s. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                        str(ce_loss.item()),
                    )
                    non_finite_detected = True
                    break

                ce_loss.backward()

                # --- SCAFFOLD Gradient Correction ---
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.data += (c_global[name] - c_local[name])

                # Optional gradient clipping (aligned with other baselines).
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self._max_grad_norm
                    )

                # Check for non-finite gradients.
                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    logger.error(
                        "[BASELINE/SCAFFOLD] Non-finite gradients detected "
                        "on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                optimizer.step()

                # Check for non-finite parameters after update.
                if self._has_non_finite_params(model):
                    logger.error(
                        "[BASELINE/SCAFFOLD] Non-finite parameters detected after "
                        "optimizer.step on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                running_loss += float(ce_loss.item())
                batch_count += 1
                steps_performed += 1

            if non_finite_detected:
                break

            epoch_train_loss = (
                running_loss / batch_count if batch_count > 0 else float("nan")
            )
            train_epoch_losses.append(epoch_train_loss)

            # Validation (logging only; does not influence updates).
            val_loss, val_acc = self._evaluate_on_loader(model, val_loader, criterion)
            val_epoch_losses.append(val_loss)
            val_epoch_accs.append(val_acc)

            logger.info(
                "[BASELINE/SCAFFOLD] Client %s | round %d | epoch %d/%d "
                "| train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                client_name,
                round_num + 1,
                epoch + 1,
                epochs,
                epoch_train_loss,
                val_loss,
                val_acc,
            )

        key = f"{client_name}_round{round_num}"

        # Handle numerical failures by rolling back and ignoring the update.
        if non_finite_detected:
            logger.warning(
                "[BASELINE/SCAFFOLD] Client %s | round %d: NaN/Inf detected; "
                "restoring initial weights and skipping contribution "
                "to aggregation for this round.",
                client_name,
                round_num + 1,
            )
            model.load_state_dict(initial_weights)
            model.cpu()

            self.history.setdefault("train_loss", {})[key] = train_epoch_losses
            self.history.setdefault("val_loss", {})[key] = val_epoch_losses
            self.history.setdefault("val_acc", {})[key] = val_epoch_accs

            state_dict_copy = {k: v.clone() for k, v in initial_weights.items()}
            return state_dict_copy, 0  # num_samples=0 → client ignored.

        # --- Update Local Control Variate (c_i) ---
        # c_i_new = c_i - c + (1 / (K * eta)) * (x - y_i)
        factor = (
            1.0 / (steps_performed * lr)
            if steps_performed > 0 and lr > 0.0
            else 0.0
        )

        # Prepare the state dictionary to return to the server.
        return_state = {k: v.cpu() for k, v in model.state_dict().items()}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                x_param = initial_weights[name].to(self.device)
                y_param = param.data

                c_new = c_local[name] - c_global[name] + factor * (x_param - y_param)
                delta_c = c_new - c_local[name]

                # Update persistent local storage and pack delta into state dict.
                self.client_controls[client_name][name] = c_new.cpu()
                return_state[f"__ctrl__{name}"] = delta_c.cpu()

        # Offload model to CPU to conserve GPU resources.
        model.cpu()

        self.history.setdefault("train_loss", {})[key] = train_epoch_losses
        self.history.setdefault("val_loss", {})[key] = val_epoch_losses
        self.history.setdefault("val_acc", {})[key] = val_epoch_accs

        return return_state, num_samples

    # ------------------------------------------------------------------ #
    # Aggregation                                                        #
    # ------------------------------------------------------------------ #
    def aggregate(
        self,
        round_num: int,
        client_updates: Dict[str, Tuple[dict, int]],
    ):
        """
        Aggregate model updates and control variate deltas from participating clients.

        Numerical stability is enforced similarly to other baselines:
        - Clients with non-finite parameters or non-positive num_samples
          are ignored.
        - If the aggregated weights contain NaN/Inf, the global model is
          NOT updated for this round.
        """
        if not client_updates:
            logger.warning("[SCAFFOLD] No client updates to aggregate")
            return

        model_updates = []
        control_updates = []

        total_clients_population = len(self.client_models)

        for c_name, (state, n_samples) in client_updates.items():
            if n_samples is None or n_samples <= 0:
                logger.info(
                    "[SCAFFOLD] Client %s skipped in aggregation (num_samples=%s).",
                    c_name,
                    str(n_samples),
                )
                continue

            # Separate model weights and control deltas.
            weights = {
                k: v for k, v in state.items() if not k.startswith("__ctrl__")
            }
            ctrls = {
                k.replace("__ctrl__", ""): v
                for k, v in state.items()
                if k.startswith("__ctrl__")
            }

            # Sanity check: ensure weights are finite.
            bad_param = False
            for k, tensor in weights.items():
                if not torch.isfinite(tensor).all():
                    logger.warning(
                        "[SCAFFOLD] Client %s has non-finite parameters (%s). "
                        "Ignoring this update in aggregation.",
                        c_name,
                        k,
                    )
                    bad_param = True
                    break
            if bad_param:
                continue

            weights_np = [v.detach().cpu().numpy() for v in weights.values()]
            model_updates.append((weights_np, int(n_samples)))
            control_updates.append(ctrls)

        if not model_updates:
            logger.warning("[SCAFFOLD] No valid client updates after filtering")
            return

        # 2. Aggregate Model Weights using standard FedAvg (weighted by num_samples).
        agg_weights_np = flwr_aggregate(model_updates)

        # Robustness check: aggregated arrays must be finite.
        for i, arr in enumerate(agg_weights_np):
            if not np.all(np.isfinite(arr)):
                logger.error(
                    "[SCAFFOLD] Aggregated parameter index %d is non-finite (NaN/Inf). "
                    "Skipping global model update for this round.",
                    i,
                )
                return

        # Reconstruct state dict for the global model.
        ref_client = next(iter(client_updates.values()))[0]
        state_keys = [k for k in ref_client.keys() if not k.startswith("__ctrl__")]

        new_state = {}
        for k, arr in zip(state_keys, agg_weights_np):
            ref_tensor = ref_client[k]

            # ========================= FIX STARTS HERE =========================
            # Explicitly convert numpy scalars (e.g., float64) to 0-d arrays
            # because torch.from_numpy() does not accept raw scalars.
            if np.isscalar(arr):
                arr = np.array(arr)
            # ===================================================================

            new_state[k] = torch.from_numpy(arr).to(ref_tensor.dtype)

        self.global_model.load_state_dict(new_state)

        # 3. Aggregate Control Variates.
        # Update Rule: c_global_new = c_global + (1/N) * sum_{i in S} (delta_c_i)
        fraction = 1.0 / float(total_clients_population)

        if control_updates:
            for name in self.global_c:
                delta_sum = torch.zeros_like(self.global_c[name])

                for c_deltas in control_updates:
                    if name in c_deltas:
                        delta_sum += c_deltas[name]

                self.global_c[name] += delta_sum * fraction

        # 4. Synchronize all clients with the new global model state.
        global_sd = self.global_model.state_dict()
        for client_name in self.client_models:
            self.client_models[client_name].load_state_dict(global_sd)

        logger.info(
            "[SCAFFOLD] Round %d aggregated. Global controls updated with "
            "scaling factor %.4f",
            round_num + 1,
            fraction,
        )

    # ------------------------------------------------------------------ #
    # Evaluation utilities                                               #
    # ------------------------------------------------------------------ #
    def evaluate_client(
        self,
        client_name: str,
        test_loader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Evaluate a specific client's model against a test dataset.
        """
        model = self.client_models[client_name]

        if len(test_loader.dataset) == 0:
            logger.warning("[SCAFFOLD] Test loader is empty for client %s", client_name)
            return 0.0

        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        model.cpu()

        acc = ensemble_accuracy(y_true, y_pred)
        logger.info(
            "[SCAFFOLD] Client %s evaluation accuracy: %.4f",
            client_name,
            acc,
        )
        return acc

    def get_global_model_accuracy(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Evaluate the global model's performance on a test set.
        """
        return self.evaluate(test_loader)
