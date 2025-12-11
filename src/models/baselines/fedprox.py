"""
🤖 FedProx Baseline Module
--------------------------

This module implements the FedProx algorithm, a federated learning baseline designed
to tackle heterogeneity in federated networks by introducing a proximal term to the
local objective function.

🧠 Purpose:
    To provide a robust implementation of the FedProx algorithm (Li et al., 2020),
    inheriting from the standard Federated Baseline architecture. It addresses
    statistical heterogeneity (non-IID data) and systems heterogeneity (stragglers)
    by constraining local updates to remain close to the global model.

🔧 Core Functionalities:
    • Implements the modified local objective function with an L2 proximal term.
    • Manages local training loops with specific handling for numerical stability (NaN/Inf checks).
    • Utilizes weighted averaging (FedAvg strategy) for global aggregation.
    • Provides detailed logging for training metrics, convergence tracking, and error states.

🎯 Intended Use:
    • Benchmarking new federated learning algorithms against established state-of-the-art methods.
    • Research scenarios involving highly heterogeneous client data distributions.
    • Experiments requiring robust handling of divergent local model updates.

📁 Dependencies:
    • torch
    • numpy
    • flwr (Flower framework)
    • models.baselines.base (FederatedBaseline)

📝 Notes:
    The implementation strictly adheres to the definitions provided in "Federated Optimization
    in Heterogeneous Networks" (MLSys 2020). When the proximal coefficient (mu) is set to 0,
    this implementation mathematically reduces to FedAvg.

Author: Andrea Moleri
File Location: src/models/baselines/fedprox.py
Last Modified: 23/04/2025
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

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

import math

logger = logging.getLogger(__name__)


class FedProxBaseline(FederatedBaseline):
    r"""
    FedProx baseline implementation reusing Flower's FedAvg aggregation logic.

    Reference: Li et al., "Federated Optimization in Heterogeneous Networks"
    (FedProx, MLSys 2020).

    Key distinctions from standard FedAvg:

    1.  **Local Objective Function**:
        For client $k$ at round $t$, the objective is modified to:
        $$ \min_{w} F_k(w) = \mathbb{E}_{(x,y) \sim \mathcal{D}_k} [\ell(w; x,y)] + \frac{\mu}{2} \|w - w^t\|^2 $$
        Where:
        - $\ell$ is the standard loss (CrossEntropy).
        - $\mu$ is the proximal coefficient (`prox_mu`).
        - $w^t$ represents the global weights at the start of round $t$.

    2.  **Aggregation**:
        - Identical to FedAvg: Weighted average of client weights based on the
          number of training examples. Implemented via `flwr_aggregate`.

    Hyperparameters (aligned with FedAvgBaseline):
        - `learning_rate`: Base learning rate.
        - `baseline_round_lr_decay`: Per-round decay factor (clamped to [0, 1]).
        - `baseline_momentum`: SGD momentum (default 0.9).
        - `baseline_weight_decay`: Weight decay (default 1e-4).
        - `baseline_clip_grad_norm`: L2 gradient clipping (default 5.0).
        - `baseline_epochs_per_round`: Local epochs per communication round.

    Specific FedProx Parameter:
        - `prox_mu`: The coefficient $\mu$ for the proximal term.
          If $\mu=0$, the behavior matches FedAvg exactly.
    """

    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        """
        Initialize the FedProx baseline model.

        Args:
            args (Namespace): Parsed command-line arguments containing hyperparameters.
            num_classes (int): Number of target classes for the classification task.
            chans (int): Number of input channels (e.g., 3 for RGB images).
            device (torch.device): The computation device (CPU or GPU).
        """
        super().__init__(args, num_classes, chans, device)
        self.current_round = 0

        # Shared hyperparameters with FedAvgBaseline (maintained for NeurIPS consistency).
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.9))
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 1e-4))
        self._max_grad_norm = float(getattr(self.args, "baseline_clip_grad_norm", 5.0))

        # Disable gradient clipping if the threshold is non-positive.
        if self._max_grad_norm <= 0:
            self._max_grad_norm = None

        # The coefficient mu for the proximal term.
        self.mu = float(getattr(self.args, "fedprox_mu", 0.01))

        logger.info(
            "[FedProx] Initialized with mu=%.4f | momentum=%.3f | weight_decay=%.1e | clip_grad_norm=%s",
            self.mu,
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
            loader,
            criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Evaluate the model on a specific DataLoader.

        Args:
            model (nn.Module): The neural network model to evaluate.
            loader (DataLoader): The torch DataLoader containing validation data.
            criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).

        Returns:
            Tuple[float, float]: A tuple containing (average_loss, accuracy).
            Returns (nan, nan) if the loader is empty or numerical instability is detected.
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

            # Check for numerical stability.
            if not torch.isfinite(loss):
                logger.warning(
                    "[FedProx] Non-finite validation loss detected "
                    "(loss=%s). Setting val_loss/val_acc to NaN for this batch.",
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

        Args:
            model (nn.Module): The model to check.

        Returns:
            bool: True if any non-finite value is found in the parameters; False otherwise.
        """
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
        return False

    # ------------------------------------------------------------------ #
    # Local training (FedProx)                                           #
    # ------------------------------------------------------------------ #
    def train_client(
            self,
            client_name: str,
            train_loader,
            val_loader,
            round_num: int,
    ) -> Tuple[dict, int]:
        r"""
        Train a single client using the FedProx objective.

        Objective for client $k$ at round $t$:
        $$ F_k(w) = \mathbb{E}_{(x,y) \sim \mathcal{D}_k} [\text{CE}(x,y; w)] + \frac{\mu}{2} \|w - w^t\|^2 $$

        Operational Logic:
        - The model initializes with global weights from the current round (synchronized
          previously in `aggregate()`).
        - Performs `epochs_per_round` of SGD on the proximal objective.
        - No local early stopping: Weights are ALWAYS submitted after the FINAL epoch
          (standard FedAvg behavior augmented with the proximal term).
        - Validation is performed strictly for logging purposes.
        - Robustness: If NaN/Inf values appear in loss, gradients, or parameters, the
          update is DISCARDED for this round (returning `num_samples=0`).

        Args:
            client_name (str): Identifier for the client.
            train_loader (DataLoader): DataLoader for local training data.
            val_loader (DataLoader): DataLoader for local validation data.
            round_num (int): Current global round number.

        Returns:
            Tuple[dict, int]: A tuple containing:
                - The updated state dictionary of the model (weights).
                - The number of training samples used (0 if the update is discarded).
        Raises:
            RuntimeError: If `global_model` has not been initialized.
        """
        if self.global_model is None:
            raise RuntimeError(
                "[FedProx] global_model is None. "
                "Ensure initialize_models() is called before training."
            )

        model = self.client_models[client_name]

        # Save initial weights to enable rollback in case of divergence.
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Calculate LR for this round (consistent with FedAvgBaseline policy).
        lr, base_lr, round_decay = self._effective_lr(round_num)

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        model.to(self.device)

        # Handle edge case: empty training loader implies no update.
        if train_loader is None or len(train_loader.dataset) == 0:
            logger.warning(
                "[BASELINE/FedProx] Client %s | round %d: empty train_loader, "
                "no update sent.",
                client_name,
                round_num + 1,
            )
            model.load_state_dict(initial_state)
            model.cpu()
            key = f"{client_name}_round{round_num}"
            self.history.setdefault("train_loss", {})[key] = []
            self.history.setdefault("val_loss", {})[key] = []
            self.history.setdefault("val_acc", {})[key] = []
            state_dict_copy = {k: v.clone() for k, v in initial_state.items()}
            return state_dict_copy, 0

        num_samples = len(train_loader.dataset)
        epochs_per_round = int(getattr(self.args, "baseline_epochs_per_round", 1))

        # Sanity check for epoch count.
        if epochs_per_round <= 0:
            logger.warning(
                "[BASELINE/FedProx] baseline_epochs_per_round=%d invalid; "
                "using 1 epoch per round.",
                epochs_per_round,
            )
            epochs_per_round = 1

        # Create a snapshot of global parameters w_t for the proximal term calculation.
        global_params = [
            p.detach().clone().to(self.device) for p in self.global_model.parameters()
        ]
        mu = float(self.mu)

        logger.info(
            "[BASELINE/FedProx] Client %s | round %d | epochs_per_round=%d | "
            "lr=%.6f (base_lr=%.6f, round_decay=%.4f) | num_samples=%d | "
            "momentum=%.3f | weight_decay=%.1e | mu=%.4f | clip_grad_norm=%s",
            client_name,
            round_num + 1,
            epochs_per_round,
            lr,
            base_lr,
            round_decay,
            num_samples,
            self._momentum,
            self._weight_decay,
            mu,
            str(self._max_grad_norm),
        )

        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_accs = []

        non_finite_detected = False

        for epoch in range(epochs_per_round):
            model.train()
            running_loss = 0.0
            batch_count = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                ce_loss = criterion(output, target)

                if not torch.isfinite(ce_loss):
                    logger.error(
                        "[BASELINE/FedProx] Non-finite CE loss detected "
                        "on client %s | round %d | epoch %d: loss=%s. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                        str(ce_loss.item()),
                    )
                    non_finite_detected = True
                    break

                # Proximal Term Calculation: (mu/2) * Σ ||w_i - w_i^global||^2
                # If mu=0, this term is skipped entirely.
                prox_term = 0.0
                if mu > 0.0:
                    for param, g_param in zip(model.parameters(), global_params):
                        prox_term = prox_term + ((param - g_param) ** 2).sum()

                if mu > 0.0:
                    loss = ce_loss + (mu / 2.0) * prox_term
                else:
                    loss = ce_loss

                if not torch.isfinite(loss):
                    logger.error(
                        "[BASELINE/FedProx] Non-finite total loss detected "
                        "on client %s | round %d | epoch %d: loss=%s. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                        str(loss.item()),
                    )
                    non_finite_detected = True
                    break

                loss.backward()

                # Optional gradient clipping (standard practice in baselines).
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
                        "[BASELINE/FedProx] Non-finite gradients detected "
                        "on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                optimizer.step()

                # Check for non-finite parameters post-update.
                if self._has_non_finite_params(model):
                    logger.error(
                        "[BASELINE/FedProx] Non-finite parameters after optimizer.step "
                        "on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                running_loss += float(loss.item())
                batch_count += 1

            if non_finite_detected:
                break

            epoch_train_loss = (
                running_loss / batch_count if batch_count > 0 else float("nan")
            )
            train_epoch_losses.append(epoch_train_loss)

            # Validation (logging only; does not influence local updates).
            val_loss, val_acc = self._evaluate_on_loader(model, val_loader, criterion)
            val_epoch_losses.append(val_loss)
            val_epoch_accs.append(val_acc)

            logger.info(
                "[BASELINE/FedProx] Client %s | round %d | "
                "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                client_name,
                round_num + 1,
                epoch + 1,
                epochs_per_round,
                epoch_train_loss,
                val_loss,
                val_acc,
            )

        # Handle numerical failures: rollback and discard update.
        if non_finite_detected:
            logger.warning(
                "[BASELINE/FedProx] Client %s | round %d: NaN/Inf detected, "
                "restoring initial weights and skipping contribution "
                "to aggregation for this round.",
                client_name,
                round_num + 1,
            )
            model.load_state_dict(initial_state)
            model.cpu()

            key = f"{client_name}_round{round_num}"
            self.history.setdefault("train_loss", {})[key] = train_epoch_losses
            self.history.setdefault("val_loss", {})[key] = val_epoch_losses
            self.history.setdefault("val_acc", {})[key] = val_epoch_accs

            state_dict_copy = {k: v.clone() for k, v in initial_state.items()}
            return state_dict_copy, 0  # num_samples=0 -> client ignored

        # No best-epoch selection: send weights from the last epoch (canonical FedProx).
        model.cpu()

        key = f"{client_name}_round{round_num}"
        self.history.setdefault("train_loss", {})[key] = train_epoch_losses
        self.history.setdefault("val_loss", {})[key] = val_epoch_losses
        self.history.setdefault("val_acc", {})[key] = val_epoch_accs

        final_state = {k: v.clone() for k, v in model.state_dict().items()}
        return final_state, num_samples

    # ------------------------------------------------------------------ #
    # FedProx aggregation (FedAvg-style)                                 #
    # ------------------------------------------------------------------ #
    def aggregate(
            self,
            round_num: int,
            client_updates: Dict[str, Tuple[dict, int]],
    ) -> None:
        """
        Perform FedProx aggregation using the FedAvg strategy (via Flower).
        """
        if not client_updates:
            logger.warning("[FedProx] No client updates to aggregate")
            return

        # Use the first client's state to define key ordering.
        first_state, _ = next(iter(client_updates.values()))
        param_keys = list(first_state.keys())

        weights_results = []
        total_clients_used = 0
        total_weight = 0

        for client_name, (client_state, num_samples) in client_updates.items():
            if num_samples is None or num_samples <= 0:
                logger.info(
                    "[FedProx] Client %s skipped in aggregation (num_samples=%s).",
                    client_name,
                    str(num_samples),
                )
                continue

            # Sanity check: verify no non-finite parameters exist before numpy conversion.
            bad_param = False
            for k, tensor in client_state.items():
                if not torch.isfinite(tensor).all():
                    logger.warning(
                        "[FedProx] Client %s has non-finite parameters (%s). "
                        "Ignoring this update in aggregation.",
                        client_name,
                        k,
                    )
                    bad_param = True
                    break
            if bad_param:
                continue

            ndarrays = [client_state[k].detach().cpu().numpy() for k in param_keys]
            weights_results.append((ndarrays, int(num_samples)))
            total_clients_used += 1
            total_weight += int(num_samples)

        if not weights_results:
            logger.warning("[FedProx] No valid client updates after filtering")
            return

        logger.info(
            "[FedProx] Round %d: aggregating %d/%d client updates (total weight=%d).",
            round_num + 1,
            total_clients_used,
            len(client_updates),
            total_weight,
        )

        # Flower FedAvg: Standard weighted average.
        aggregated_ndarrays = flwr_aggregate(weights_results)

        # Robustness check: if the aggregate contains NaN/Inf, DO NOT update.
        for i, arr in enumerate(aggregated_ndarrays):
            if not np.all(np.isfinite(arr)):
                logger.error(
                    "[FedProx] Aggregated parameter index %d non-finite (NaN/Inf). "
                    "Skipping global model update for this round.",
                    i,
                )
                return

        # Reconstruct the global model state
        new_global_state = {}
        for key, arr in zip(param_keys, aggregated_ndarrays):
            ref_tensor = first_state[key]

            # --- FIX: Handle numpy scalars (e.g. float64) which torch.from_numpy rejects ---
            if np.isscalar(arr):
                arr = np.array(arr)
            # -------------------------------------------------------------------------------

            new_global_state[key] = torch.from_numpy(arr).to(ref_tensor.dtype)

        # Update global model.
        self.global_model.load_state_dict(new_global_state)

        # Broadcast the global model to all clients.
        global_state = self.global_model.state_dict()
        for client_name in self.client_models:
            self.client_models[client_name].load_state_dict(global_state)

        self.current_round += 1
        logger.info(
            "[FedProx] Aggregation round %d completed via Flower FedAvg "
            "(clients_used=%d).",
            round_num + 1,
            total_clients_used,
        )

    # ------------------------------------------------------------------ #
    # Evaluation Utilities                                               #
    # ------------------------------------------------------------------ #
    def evaluate_client(self, client_name, test_loader):
        """
        Evaluate a single client on a given test_loader.

        Args:
            client_name (str): Name of the client to evaluate.
            test_loader (DataLoader): DataLoader for testing.

        Returns:
            float: Accuracy of the client model on the test set.
        """
        model = self.client_models[client_name]

        if len(test_loader.dataset) == 0:
            logger.warning("[FedProx] Test loader is empty for client %s", client_name)
            return 0.0

        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        accuracy = ensemble_accuracy(y_true, y_pred)
        model.cpu()

        logger.info(
            "[FedProx] Client %s evaluation accuracy: %.4f",
            client_name,
            accuracy,
        )
        return accuracy

    def get_global_model_accuracy(self, test_loader):
        """
        Shortcut method to evaluate the global model.

        Args:
            test_loader (DataLoader): DataLoader for testing.

        Returns:
            float: Accuracy of the global model.
        """
        return self.evaluate(test_loader)