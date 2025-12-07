"""
🤖 Federated Averaging (FedAvg) Baseline Module
-----------------------------------------------

This module implements the canonical Federated Averaging (FedAvg) algorithm, 
serving as a robust and reproducible baseline for Federated Learning experiments. 
It delegates the core aggregation logic to the Flower framework while enforcing 
strict numerical stability checks during local training.

🧠 Purpose:
    To provide a reference implementation of FedAvg that mirrors the "vanilla" 
    design found in standard literature (e.g., NeurIPS papers), augmented with 
    production-grade safeguards against numerical instability (NaN/Inf).

🔧 Core Functionalities:
    • Local Stochastic Gradient Descent (SGD) with configurable momentum and weight decay.
    • Robust NaN/Inf detection for losses, gradients, and model parameters.
    • Dynamic learning rate scheduling based on round decay.
    • Weighted aggregation strategy via `flwr.server.strategy.aggregate`.
    • Graceful failure handling: discards client updates if divergence is detected.

🎯 Intended Use:
    • Academic benchmarking against novel FL algorithms.
    • Stability testing in heterogeneous data environments.
    • Educational analysis of the standard FL lifecycle.

📁 Dependencies:
    • torch (PyTorch)
    • numpy
    • flwr (Flower)
    • models.baselines.base (Internal Base Classes)

📝 Notes:
    This implementation eschews local early stopping to maintain strict adherence 
    to the FedAvg definition: clients always transmit weights after the final 
    local epoch, provided the training remained numerically stable.

Author: Andrea Moleri
File Location: src/models/baselines/fedavg.py
Last Modified: 20/11/2025
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

logger = logging.getLogger(__name__)


class FedAvgBaseline(FederatedBaseline):
    """
    Implements the Federated Averaging (FedAvg) baseline, leveraging Flower for aggregation.

    This class is designed to be a numerically stable and easily reproducible reference point.
    It strictly adheres to the standard FedAvg paradigm while adding engineering safeguards.

    Design Principles (Robust "Vanilla" FedAvg):
    --------------------------------------------
    - **Local Training**:
        - Uses SGD with standard momentum and weight decay.
        - **No local early stopping**: Clients transmit weights strictly after the
          last local epoch (consistent with classical FedAvg).
        - Validation data is utilized exclusively for logging, not for control flow.
        - **Gradient Clipping**: Optional configuration to prevent exploding gradients.
        - **Stability Checks**: Explicitly monitors loss, gradients, and parameters for
          non-finite values (NaN/Inf). If detected, the specific client's update is
          discarded for the current round to protect the global model.

    - **Aggregation**:
        - Performs a weighted average of client parameters based on the number of
          training examples, utilizing `flwr.server.strategy.aggregate.aggregate`.

    - **Learning Rate Schedule**:
        - Uses `args.learning_rate` as the base learning rate ($lr_{base}$).
        - Applies a per-round decay factor `args.baseline_round_lr_decay` (default=1.0):
          $$ lr_t = lr_{base} \times \text{decay}^{t} $$
          where $t$ is the 0-based round number.
        - If decay > 1.0, it is clamped to 1.0 to prevent divergent learning rates.
        - The learning rate remains constant *within* a specific round.
    """

    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        """
        Initialize the FedAvg baseline with model architecture and training hyperparameters.

        Args:
            args: Configuration namespace containing training arguments.
            num_classes (int): The number of target classes for the classification task.
            chans (int): The number of input channels (e.g., 3 for RGB images).
            device (torch.device): The computing device (CPU or GPU) for tensor operations.
        """
        super().__init__(args, num_classes, chans, device)
        self.current_round = 0

        # "Safety-first" hyperparameters for a numerically stable baseline.
        # These can be overridden via CLI arguments but default to sensible values.
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.9))
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 1e-4))

        # Gradient clipping norm: 0 or None implies disabled.
        self._max_grad_norm = float(getattr(self.args, "baseline_clip_grad_norm", 5.0))
        if self._max_grad_norm <= 0:
            self._max_grad_norm = None

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
        Evaluates the model on a specific DataLoader.

        Args:
            model (nn.Module): The neural network model to evaluate.
            loader: The DataLoader containing the dataset.
            criterion (nn.Module): The loss function.

        Returns:
            Tuple[float, float]: A tuple containing (average_loss, accuracy).
            Returns (NaN, NaN) if the loader is empty or instability is detected.
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

            # Stability check for the loss value
            if not torch.isfinite(loss):
                logger.warning(
                    "[FedAvg] Non-finite validation loss detected "
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

    def _effective_lr(self, round_num: int) -> Tuple[float, float, float]:
        """
        Calculates the effective learning rate for the specified round.

        The formula applied is:
        $$ lr_t = lr_{base} \times \gamma^{round\_num} $$

        Args:
            round_num (int): The current federated training round (0-based).

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - The effective learning rate for the round.
                - The base learning rate.
                - The effective decay factor used.
        """
        base_lr = float(getattr(self.args, "learning_rate", 0.1))
        round_decay = float(getattr(self.args, "baseline_round_lr_decay", 1.0))

        # Enforce safety clamps on the decay factor
        if round_decay > 1.0:
            logger.warning(
                "[FedAvg] baseline_round_lr_decay=%.4f > 1.0 detected; "
                "clamping to 1.0 to prevent learning rate growth over time.",
                round_decay,
            )
            round_decay = 1.0
        if round_decay < 0.0:
            logger.warning(
                "[FedAvg] baseline_round_lr_decay=%.4f < 0.0 detected; "
                "clamping to 0.0.",
                round_decay,
            )
            round_decay = 0.0

        # Round indexing is 0-based: round 0 uses base_lr, round 1 uses base_lr * decay, etc.
        lr = base_lr * (round_decay ** round_num)
        return float(lr), base_lr, round_decay

    @staticmethod
    def _has_non_finite_params(model: nn.Module) -> bool:
        """
        Checks if any parameter in the model contains NaN or Inf values.

        Args:
            model (nn.Module): The model to inspect.

        Returns:
            bool: True if non-finite parameters are found, False otherwise.
        """
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
        return False

    # ------------------------------------------------------------------ #
    # Local training                                                     #
    # ------------------------------------------------------------------ #
    def train_client(self, client_name, train_loader, val_loader, round_num):
        """
        Executes local training for a single client using SGD.

        Implements "Vanilla FedAvg" behavior for baseline comparison:
        - The model initializes with global weights from the current round.
        - Performs `epochs_per_round` epochs of SGD on the training loader.
        - Validation is used strictly for logging purposes.
        - The model state returned is ALWAYS taken after the LAST epoch.
        - **Safety Mechanism**: If parameters become non-finite (NaN/Inf) during
          training, the update is discarded to preserve global stability.

        Args:
            client_name (str): Identifier for the client.
            train_loader: DataLoader for the client's training set.
            val_loader: DataLoader for the client's validation set.
            round_num (int): The current global round number.

        Returns:
            Tuple[Dict, int]:
                - A state_dict of the trained model (or initial weights if failed).
                - The number of samples used for training (0 if update is discarded).
        """
        model = self.client_models[client_name]

        # Cache weights at the start of the round to allow rollback if training diverges.
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Calculate LR for this specific round
        lr, base_lr, round_decay = self._effective_lr(round_num)

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        model.to(self.device)

        # Handle edge case: Empty training loader
        if train_loader is None or len(train_loader.dataset) == 0:
            logger.warning(
                "[BASELINE/FedAvg] Client %s | round %d: train_loader is empty; "
                "no update sent.",
                client_name,
                round_num + 1,
            )
            model.load_state_dict(initial_state)
            model.cpu()

            # Log empty history to maintain data consistency
            key = f"{client_name}_round{round_num}"
            self.history.setdefault("train_loss", {})[key] = []
            self.history.setdefault("val_loss", {})[key] = []
            self.history.setdefault("val_acc", {})[key] = []

            state_dict_copy = {k: v.clone() for k, v in initial_state.items()}
            # Returning num_samples=0 ensures the server ignores this client in aggregation.
            return state_dict_copy, 0

        num_samples = len(train_loader.dataset)
        epochs_per_round = int(getattr(self.args, "baseline_epochs_per_round", 1))
        if epochs_per_round <= 0:
            logger.warning(
                "[BASELINE/FedAvg] baseline_epochs_per_round=%d is invalid; "
                "defaulting to 1 epoch per round.",
                epochs_per_round,
            )
            epochs_per_round = 1

        logger.info(
            "[BASELINE/FedAvg] Client %s | round %d | epochs_per_round=%d | "
            "lr=%.6f (base_lr=%.6f, round_decay=%.4f) | num_samples=%d | "
            "momentum=%.3f | weight_decay=%.1e | clip_grad_norm=%s",
            client_name,
            round_num + 1,
            epochs_per_round,
            lr,
            base_lr,
            round_decay,
            num_samples,
            self._momentum,
            self._weight_decay,
            str(self._max_grad_norm),
        )

        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_accs = []

        non_finite_detected = False

        for epoch in range(epochs_per_round):
            # ---------------------- Training Phase ----------------------
            model.train()
            running_loss = 0.0
            batch_count = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)

                # Check for numerical instability in loss calculation
                if not torch.isfinite(loss):
                    logger.error(
                        "[BASELINE/FedAvg] Non-finite loss detected "
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

                # Apply optional gradient clipping (Standard FL Best Practice)
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self._max_grad_norm
                    )

                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError as e:
                        logger.error(f"[CRITICAL] CUDA error detected during backward/clip for {client_name}: {e}")
                        non_finite_detected = True
                        break

                # Check for non-finite gradients (e.g., overflow)
                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    logger.error(
                        "[BASELINE/FedAvg] Non-finite gradients detected "
                        "on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                optimizer.step()

                # Final check: Ensure parameters remain finite after update
                if self._has_non_finite_params(model):
                    logger.error(
                        "[BASELINE/FedAvg] Non-finite parameters detected after optimizer.step "
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

            # ---------------------- Validation Phase (Logging Only) ----
            val_loss, val_acc = self._evaluate_on_loader(model, val_loader, criterion)
            val_epoch_losses.append(val_loss)
            val_epoch_accs.append(val_acc)

            logger.info(
                "[BASELINE/FedAvg] Client %s | round %d | "
                "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                client_name,
                round_num + 1,
                epoch + 1,
                epochs_per_round,
                epoch_train_loss,
                val_loss,
                val_acc,
            )

        # If instability occurred (NaN/Inf), discard the update
        if non_finite_detected:
            logger.warning(
                "[BASELINE/FedAvg] Client %s | round %d: NaN/Inf detected; "
                "restoring initial weights and omitting contribution "
                "to this round's aggregation.",
                client_name,
                round_num + 1,
            )
            model.load_state_dict(initial_state)
            model.cpu()

            key = f"{client_name}_round{round_num}"
            # Save partial history for debugging analysis
            self.history.setdefault("train_loss", {})[key] = train_epoch_losses
            self.history.setdefault("val_loss", {})[key] = val_epoch_losses
            self.history.setdefault("val_acc", {})[key] = val_epoch_accs

            state_dict_copy = {k: v.clone() for k, v in initial_state.items()}
            return state_dict_copy, 0  # num_samples=0 -> client is ignored

        # No local "best-epoch" selection: send weights from the FINAL epoch.
        model.cpu()

        key = f"{client_name}_round{round_num}"
        self.history.setdefault("train_loss", {})[key] = train_epoch_losses
        self.history.setdefault("val_loss", {})[key] = val_epoch_losses
        self.history.setdefault("val_acc", {})[key] = val_epoch_accs

        # Return a deep copy of the state dict to prevent aliasing issues
        final_state = {k: v.clone() for k, v in model.state_dict().items()}
        return final_state, num_samples

    # ------------------------------------------------------------------ #
    # FedAvg aggregation                                                 #
    # ------------------------------------------------------------------ #
    def aggregate(self, round_num: int, client_updates: Dict[str, Tuple[dict, int]]):
        """
        Executes the FedAvg aggregation step using Flower's implementation.

        Workflow:
        1. Converts PyTorch state_dicts from valid clients into lists of NumPy arrays.
        2. Invokes `flwr_aggregate` to compute the weighted average of parameters based
           on the number of local examples.
        3. Converts the aggregated results back into a PyTorch state_dict for the global model.

        Args:
            round_num (int): The current global round number.
            client_updates (Dict[str, Tuple[dict, int]]): A dictionary mapping client names
                to tuples of (model_state_dict, num_samples).
        """
        if not client_updates:
            logger.warning("[FedAvg] No client updates to aggregate")
            return

        # Retrieve the state of the first client to determine parameter key ordering
        first_state, _ = next(iter(client_updates.values()))
        param_keys = list(first_state.keys())

        # Construct the list structure (ndarrays, num_examples) required by Flower
        weights_results = []
        total_clients_used = 0
        total_weight = 0

        for client_name, (client_state, num_samples) in client_updates.items():
            if num_samples is None or num_samples <= 0:
                logger.info(
                    "[FedAvg] Client %s skipped in aggregation (num_samples=%s).",
                    client_name,
                    str(num_samples),
                )
                continue

            # Sanity check: Ensure no non-finite parameters exist before NumPy conversion
            bad_param = False
            for k, tensor in client_state.items():
                if not torch.isfinite(tensor).all():
                    logger.warning(
                        "[FedAvg] Client %s has non-finite parameters (%s). "
                        "Ignoring this update in aggregation.",
                        client_name,
                        k,
                    )
                    bad_param = True
                    break
            if bad_param:
                continue

            # Convert tensors to NumPy arrays (CPU)
            ndarrays = [client_state[k].detach().cpu().numpy() for k in param_keys]

            weights_results.append((ndarrays, int(num_samples)))
            total_clients_used += 1
            total_weight += int(num_samples)

        if not weights_results:
            logger.warning("[FedAvg] No valid client updates after filtering")
            return

        logger.info(
            "[FedAvg] Round %d: aggregating %d/%d client updates (total weight=%d).",
            round_num + 1,
            total_clients_used,
            len(client_updates),
            total_weight,
        )

        # Flower's FedAvg: Performs standard weighted averaging
        aggregated_ndarrays = flwr_aggregate(weights_results)

        # Robustness check: If the aggregate contains NaN/Inf, DO NOT update the global model
        for i, arr in enumerate(aggregated_ndarrays):
            if not np.all(np.isfinite(arr)):
                logger.error(
                    "[FedAvg] Aggregated parameter index %d is non-finite (NaN/Inf). "
                    "Skipping global model update for this round.",
                    i,
                )
                return

        # Reconstruct the PyTorch state_dict from the aggregated arrays
        new_global_state = {}
        for key, arr in zip(param_keys, aggregated_ndarrays):
            ref_tensor = first_state[key]  # Preserve the dtype of the original tensor

            # --- FIX: Handle numpy scalars (e.g. float64) which torch.from_numpy rejects ---
            if np.isscalar(arr):
                arr = np.array(arr)
            # -------------------------------------------------------------------------------

            new_global_state[key] = torch.from_numpy(arr).to(ref_tensor.dtype)

        # Update the global model
        self.global_model.load_state_dict(new_global_state)

        # Propagate the new global state to all client models
        global_state = self.global_model.state_dict()
        for client_name in self.client_models:
            self.client_models[client_name].load_state_dict(global_state)

        self.current_round += 1
        logger.info(
            "[FedAvg] Aggregation round %d completed via Flower FedAvg "
            "(clients_used=%d).",
            round_num + 1,
            total_clients_used,
        )

    # ------------------------------------------------------------------ #
    # Evaluation Utilities                                               #
    # ------------------------------------------------------------------ #
    def evaluate_client(self, client_name, test_loader):
        """
        Evaluates a specific client's model on the provided test loader.

        Args:
            client_name (str): The identifier of the client.
            test_loader: The DataLoader for testing.

        Returns:
            float: The calculated accuracy. Returns 0.0 if the loader is empty.
        """
        model = self.client_models[client_name]

        if len(test_loader.dataset) == 0:
            logger.warning("[FedAvg] Test loader is empty for client %s", client_name)
            return 0.0

        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        accuracy = ensemble_accuracy(y_true, y_pred)
        model.cpu()

        logger.info(
            "[FedAvg] Client %s evaluation accuracy: %.4f",
            client_name,
            accuracy,
        )
        return accuracy

    def get_global_model_accuracy(self, test_loader):
        """
        Evaluates the global model on the provided test loader.

        Args:
            test_loader: The DataLoader for testing.

        Returns:
            float: The accuracy of the global model.
        """
        return self.evaluate(test_loader)