"""
🤖 FedDyn Baseline Module
-------------------------

This module implements the **FedDyn** (Federated Learning based on Dynamic Regularization) 
baseline algorithm. It extends the standard federated learning framework by introducing 
dynamic regularization to handle statistical heterogeneity among clients.

🧠 Purpose:
    To provide a robust implementation of the FedDyn algorithm for comparative 
    analysis in federated learning research. It addresses the objective inconsistency 
    problem in non-IID settings by maintaining dual variables for each client.

🔧 Core Functionalities:
    • Implements the dynamic regularization objective function for local training  
    • Manages server-side dual variables (h_k) for each client  
    • Performs weighted aggregation of client updates adjusted by dynamic terms  
    • Handles gradient clipping, momentum, and learning rate decay  

🎯 Intended Use:
    • Academic benchmarking against other FL algorithms (e.g., FedAvg, FedProx)  
    • Research into convergence properties under non-IID data distributions  
    • Experimental environments requiring stateful client-server interaction  

📁 Dependencies:
    • numpy  
    • torch  
    • models.baselines.base (FederatedBaseline)  

📝 Notes:
    Based on the paper: Acar et al., "Federated Learning based on Dynamic Regularization", 
    NeurIPS 2021. The implementation assumes that client drift is mitigated by 
    aligning local objectives with the global objective via the auxiliary variable h_k.

Author: Andrea Moleri  
File Location: src/models/baselines/feddyn.py  
Last Modified: 06/12/2025
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.baselines.base import (
    FederatedBaseline,
    evaluate_single_classifier,
    ensemble_accuracy,
)

logger = logging.getLogger(__name__)


class FedDynBaseline(FederatedBaseline):
    """
    FedDyn Baseline with Dynamic Regularization.

    This class implements the FedDyn algorithm as described by Acar et al. (NeurIPS 2021).
    It modifies the local objective function and the global aggregation step to ensure
    that local minima converge towards the global stationary point.

    Algorithmic Overview:
    ---------------------
    1. **Local Objective (Client-side):**
       For client k at round t, the loss function is augmented:
       J_k(w) = F_k(w) - <h_k, w> + (alpha / 2) * ||w||^2

       Where:
       - F_k(w) is the standard empirical loss (e.g., CrossEntropy).
       - h_k is a dynamic dual variable vector maintained for client k.
       - alpha is the regularization coefficient.

    2. **Global Aggregation (Server-side):**
       The server updates the global model and the dual variables:
       w^(t+1) = (1 / (alpha * sum(n_k))) * sum(n_k * (alpha * w_k^(t+1) - h_k^(t)))

       h_k^(t+1) = h_k^(t) - alpha * (w^(t+1) - w_k^(t+1))

    Attributes:
        alpha (float): The hyperparameter controlling the strength of dynamic regularization.
        client_h (Dict[str, Dict[str, torch.Tensor]]): Stores the dual variables (h_k) 
                                                       for each client and parameter.
        _momentum (float): Momentum factor for the local optimizer.
        _weight_decay (float): Weight decay (L2 penalty) for the local optimizer.
        _max_grad_norm (float | None): Threshold for gradient clipping.
    """

    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        """
        Initializes the FedDyn baseline model.

        Args:
            args (Namespace): Parsed command-line arguments containing hyperparameters.
            num_classes (int): The number of target classes for classification.
            chans (int): The number of input channels (e.g., 3 for RGB images).
            device (torch.device): The computation device (CPU or GPU).
        """
        super().__init__(args, num_classes, chans, device)
        self.current_round = 0

        # Extract optimization hyperparameters shared with FedAvg/FedProx.
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.9))
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 1e-4))
        self._max_grad_norm = float(
            getattr(self.args, "baseline_clip_grad_norm", 5.0)
        )

        # Disable gradient clipping if the threshold is non-positive.
        if self._max_grad_norm <= 0:
            self._max_grad_norm = None

        # Extract FedDyn-specific alpha coefficient.
        alpha = float(getattr(self.args, "feddyn_alpha", 0.01))

        # Enforce a minimum positive value for alpha to avoid division by zero errors during aggregation.
        if alpha <= 0.0:
            logger.warning(
                "[FedDyn] feddyn_alpha=%.4f <= 0 detected; "
                "clamping to minimum positive value (1e-6).",
                alpha,
            )
            alpha = 1e-6
        self.alpha = alpha

        # Initialize storage for dynamic vectors h_k: dict[client_name][param_name] -> Tensor.
        # These are stored on the CPU to save GPU memory when not in use.
        self.client_h: Dict[str, Dict[str, torch.Tensor]] = {}

        logger.info(
            "[FedDyn] Initialized | alpha=%.4f | momentum=%.3f | "
            "weight_decay=%.1e | clip_grad_norm=%s",
            self.alpha,
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
        Evaluates the model on a specific data loader.

        Computes the average loss and accuracy. Note that this method uses 
        standard CrossEntropyLoss (without FedDyn regularization terms) to ensure 
        validation metrics are comparable across different baselines (e.g., FedAvg).

        Args:
            model (nn.Module): The neural network to evaluate.
            loader (torch.utils.data.DataLoader): The dataset iterator.
            criterion (nn.Module): The loss function (typically CrossEntropyLoss).

        Returns:
            Tuple[float, float]: A tuple containing (average_loss, accuracy). 
                                 Returns (NaN, NaN) if the loader is empty or calculation fails.
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

            # Check for numerical stability issues.
            if not torch.isfinite(loss):
                logger.warning(
                    "[FedDyn] Non-finite validation loss detected "
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
        Calculates the effective learning rate for the current training round.

        Formula: LR_t = base_lr * (round_decay ** round_num)

        Args:
            round_num (int): The current federated learning round (0-based).

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - The calculated effective learning rate.
                - The base learning rate.
                - The effective round decay factor.
        """
        base_lr = float(getattr(self.args, "learning_rate", 0.1))
        round_decay = float(getattr(self.args, "baseline_round_lr_decay", 1.0))

        # Clamp decay factor to standard range [0.0, 1.0].
        if round_decay > 1.0:
            logger.warning(
                "[FedDyn] baseline_round_lr_decay=%.4f > 1.0; "
                "clamping to 1.0 to prevent increasing LR.",
                round_decay,
            )
            round_decay = 1.0
        if round_decay < 0.0:
            logger.warning(
                "[FedDyn] baseline_round_lr_decay=%.4f < 0.0; "
                "clamping to 0.0.",
                round_decay,
            )
            round_decay = 0.0

        lr = base_lr * (round_decay ** round_num)
        return float(lr), base_lr, round_decay

    @staticmethod
    def _has_non_finite_params(model: nn.Module) -> bool:
        """
        Checks if any parameter in the model contains NaN or Inf values.

        Args:
            model (nn.Module): The model to inspect.

        Returns:
            bool: True if non-finite values are found, False otherwise.
        """
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
        return False

    def _ensure_client_h(
            self, client_name: str, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Ensures the dual variable dictionary (h_k) exists for a given client.

        The h_k vectors are maintained on the server (defaulting to CPU storage).
        If initialization is required, zero-tensors matching the model's 
        trainable parameters are created.

        Args:
            client_name (str): The unique identifier for the client.
            model (nn.Module): The client's model (used for shape reference).

        Returns:
            Dict[str, torch.Tensor]: The dictionary containing h_k vectors.
        """
        h_dict = self.client_h.setdefault(client_name, {})
        for name, param in model.named_parameters():
            # Only track trainable parameters; buffers (e.g., BatchNorm) do not require dual variables.
            if not param.requires_grad:
                continue
            if name not in h_dict:
                h_dict[name] = torch.zeros_like(param, device="cpu")
        return h_dict

    # ------------------------------------------------------------------ #
    # Local training (FedDyn)                                           #
    # ------------------------------------------------------------------ #
    def train_client(
            self,
            client_name: str,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            round_num: int,
    ) -> Tuple[dict, int]:
        """
        Executes the local training procedure for a single client using the FedDyn objective.

        The optimizer minimizes the following augmented objective:
            J(w) = Loss(w) - <h, w> + (alpha / 2) * ||w||^2

        Procedure:
            1. Initialize local model with global weights.
            2. Retrieve/Initialize client-specific dual variables (h_k).
            3. Perform SGD epochs using the augmented loss.
            4. Return updated weights to the server.

        Args:
            client_name (str): The ID of the client being trained.
            train_loader (DataLoader): The training data iterator.
            val_loader (DataLoader): The validation data iterator (used for logging only).
            round_num (int): The current global round number.

        Returns:
            Tuple[dict, int]:
                - A state_dict containing the updated model parameters.
                - The number of training samples used (weight for aggregation).

        Raises:
            RuntimeError: If the global model has not been initialized.
        """
        if self.global_model is None:
            raise RuntimeError(
                "[FedDyn] global_model is None. "
                "Ensure initialize_models() is called before training."
            )

        model = self.client_models[client_name]

        # Backup initial state to allow rollback in case of numerical instability.
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Calculate effective learning rate for this round.
        lr, base_lr, round_decay = self._effective_lr(round_num)

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        model.to(self.device)

        # Handle empty training loader edge case.
        if train_loader is None or len(train_loader.dataset) == 0:
            logger.warning(
                "[BASELINE/FedDyn] Client %s | round %d: train_loader empty, "
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

        if epochs_per_round <= 0:
            logger.warning(
                "[BASELINE/FedDyn] baseline_epochs_per_round=%d invalid; "
                "using 1 epoch per round.",
                epochs_per_round,
            )
            epochs_per_round = 1

        # Retrieve h_k vectors (on server CPU) and prepare for device transfer.
        h_dict_cpu = self._ensure_client_h(client_name, model)
        alpha = float(self.alpha)

        logger.info(
            "[BASELINE/FedDyn] Client %s | round %d | epochs_per_round=%d | "
            "lr=%.6f (base_lr=%.6f, round_decay=%.4f) | num_samples=%d | "
            "momentum=%.3f | weight_decay=%.1e | alpha=%.4f | clip_grad_norm=%s",
            client_name,
            round_num + 1,
            epochs_per_round,
            lr,
            base_lr,
            round_decay,
            num_samples,
            self._momentum,
            self._weight_decay,
            alpha,
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
                        "[BASELINE/FedDyn] Non-finite CE loss detected "
                        "on client %s | round %d | epoch %d: loss=%s. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                        str(ce_loss.item()),
                    )
                    non_finite_detected = True
                    break

                # Calculate the Dynamic Regularization term:
                # dyn_reg = sum(<h_k, w>) - (alpha/2) * ||w||^2
                # Note: We subtract this from the loss, effectively minimizing:
                # Loss_total = CE_Loss - dyn_reg
                dyn_reg = 0.0
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    h_param = h_dict_cpu[name].to(self.device)
                    dyn_reg = dyn_reg + torch.sum(h_param * param)
                    if alpha > 0.0:
                        dyn_reg = dyn_reg - (alpha / 2.0) * torch.sum(param ** 2)

                loss = ce_loss - dyn_reg

                if not torch.isfinite(loss):
                    logger.error(
                        "[BASELINE/FedDyn] Non-finite total loss detected "
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

                # Apply optional gradient clipping.
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self._max_grad_norm
                    )

                # Verify finite gradients before optimizer step.
                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    logger.error(
                        "[BASELINE/FedDyn] Non-finite gradients detected "
                        "on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                optimizer.step()

                # Verify finite parameters after update.
                if self._has_non_finite_params(model):
                    logger.error(
                        "[BASELINE/FedDyn] Non-finite parameters detected after optimizer.step "
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

            # Perform validation (purely for logging; does not affect training).
            val_loss, val_acc = self._evaluate_on_loader(model, val_loader, criterion)
            val_epoch_losses.append(val_loss)
            val_epoch_accs.append(val_acc)

            logger.info(
                "[BASELINE/FedDyn] Client %s | round %d | "
                "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                client_name,
                round_num + 1,
                epoch + 1,
                epochs_per_round,
                epoch_train_loss,
                val_loss,
                val_acc,
            )

        # Handle numerical failures by rolling back and ignoring the update.
        if non_finite_detected:
            logger.warning(
                "[BASELINE/FedDyn] Client %s | round %d: NaN/Inf detected, "
                "restoring initial weights and ignoring contribution "
                "for this round.",
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
            return state_dict_copy, 0  # num_samples=0 implies ignored client.

        # Normal completion: Return weights from the last epoch.
        model.cpu()

        key = f"{client_name}_round{round_num}"
        self.history.setdefault("train_loss", {})[key] = train_epoch_losses
        self.history.setdefault("val_loss", {})[key] = val_epoch_losses
        self.history.setdefault("val_acc", {})[key] = val_epoch_accs

        final_state = {k: v.clone() for k, v in model.state_dict().items()}
        return final_state, num_samples

    # ------------------------------------------------------------------ #
    # FedDyn aggregation                                                 #
    # ------------------------------------------------------------------ #
    def aggregate(
            self,
            round_num: int,
            client_updates: Dict[str, Tuple[dict, int]],
    ) -> None:
        """
        Performs FedDyn aggregation and updates global model and dual variables.

        This method implements the server-side logic of FedDyn. It updates the global
        parameters based on the weighted average of client parameters, adjusted by 
        the dual variables (h_k) and regularization coefficient (alpha).

        Mathematical update for trainable parameters:
            w_global = [1 / (alpha * TotalWeight)] * Sum(n_k * (alpha * w_k - h_k))

        Update for dual variables:
            h_k_new = h_k_old - alpha * (w_global - w_k)

        Notes:
            - Buffers (e.g., BatchNorm statistics) are aggregated via standard FedAvg.
            - Updates with non-finite parameters or zero samples are ignored.

        Args:
            round_num (int): The current global round number.
            client_updates (Dict): Dictionary mapping client_name to (state_dict, num_samples).
        """
        if not client_updates:
            logger.warning("[FedDyn] No client updates to aggregate")
            return

        # Use the first client's state to determine the order of parameter keys.
        first_state, _ = next(iter(client_updates.values()))
        param_keys = list(first_state.keys())

        # Identify trainable parameters (these are subject to FedDyn dynamics).
        # Buffers will be handled separately using standard averaging.
        trainable_param_names = {name for name, _ in self.global_model.named_parameters()}

        # Filter out invalid clients (e.g., failed training or numerical instability).
        valid_clients = []
        total_weight = 0

        for client_name, (client_state, num_samples) in client_updates.items():
            if num_samples is None or num_samples <= 0:
                logger.info(
                    "[FedDyn] Client %s skipped in aggregation (num_samples=%s).",
                    client_name,
                    str(num_samples),
                )
                continue

            bad_param = False
            for k, tensor in client_state.items():
                if not torch.isfinite(tensor).all():
                    logger.warning(
                        "[FedDyn] Client %s has non-finite parameters (%s). "
                        "Ignoring this update in aggregation.",
                        client_name,
                        k,
                    )
                    bad_param = True
                    break
            if bad_param:
                continue

            valid_clients.append(client_name)
            total_weight += int(num_samples)

        if not valid_clients or total_weight <= 0:
            logger.warning("[FedDyn] No valid client updates after filtering")
            return

        alpha = float(self.alpha)

        logger.info(
            "[FedDyn] Round %d: aggregating %d/%d client updates (total weight=%d).",
            round_num + 1,
            len(valid_clients),
            len(client_updates),
            total_weight,
        )

        new_global_state: Dict[str, torch.Tensor] = {}

        # Reconstruct the global model state parameter by parameter.
        for key in param_keys:
            numerator = None

            if key in trainable_param_names:
                # Apply FedDyn logic for trainable parameters.
                for client_name in valid_clients:
                    client_state, num_samples = client_updates[client_name]
                    w_k = client_state[key].detach().to(self.device)

                    # Retrieve h_k for this client/parameter (init to zero if missing).
                    h_dict = self.client_h.setdefault(client_name, {})
                    if key not in h_dict:
                        h_dict[key] = torch.zeros_like(w_k, device="cpu")
                    h_k = h_dict[key].to(self.device)

                    # Calculate contribution: n_k * (alpha * w_k - h_k)
                    contrib = int(num_samples) * (alpha * w_k - h_k)

                    if numerator is None:
                        numerator = contrib.clone()
                    else:
                        numerator += contrib

                # Normalize: w_global = numerator / (alpha * total_weight)
                param_global = numerator / (alpha * float(total_weight))
            else:
                # Apply standard FedAvg for non-trainable buffers (e.g., running_mean).
                for client_name in valid_clients:
                    client_state, num_samples = client_updates[client_name]
                    w_k = client_state[key].detach().to(self.device)
                    contrib = int(num_samples) * w_k
                    if numerator is None:
                        numerator = contrib.clone()
                    else:
                        numerator += contrib
                param_global = numerator / float(total_weight)

            # Safety check: do not update if aggregation resulted in NaN/Inf.
            if not torch.isfinite(param_global).all():
                logger.error(
                    "[FedDyn] Aggregated parameter %s is non-finite (NaN/Inf). "
                    "Skipping global model update for this round.",
                    key,
                )
                return

            ref_tensor = first_state[key]
            new_global_state[key] = param_global.to(dtype=ref_tensor.dtype, device="cpu")

        # Apply the new state to the global model.
        self.global_model.load_state_dict(new_global_state)

        # Update the dual variables h_k for all participating clients.
        for client_name in valid_clients:
            client_state, _ = client_updates[client_name]
            h_dict = self.client_h.setdefault(client_name, {})
            for key in param_keys:
                if key not in trainable_param_names:
                    continue

                w_k = client_state[key].detach().to(self.device)
                w_global = new_global_state[key].detach().to(self.device)

                if key not in h_dict:
                    h_dict[key] = torch.zeros_like(w_k, device="cpu")

                h_old = h_dict[key].to(self.device)

                # Update rule: h_new = h_old - alpha * (w_global - w_k)
                h_new = h_old - alpha * (w_global - w_k)
                h_dict[key] = h_new.detach().cpu()

        # Distribute the updated global model to all client instances.
        global_state = self.global_model.state_dict()
        for client_name in self.client_models:
            self.client_models[client_name].load_state_dict(global_state)

        self.current_round += 1
        logger.info(
            "[FedDyn] Aggregation round %d completed (clients_used=%d).",
            round_num + 1,
            len(valid_clients),
        )

    # ------------------------------------------------------------------ #
    # Evaluation Utilities                                               #
    # ------------------------------------------------------------------ #
    def evaluate_client(self, client_name: str, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluates a specific client model on the provided test dataset.

        Args:
            client_name (str): The ID of the client.
            test_loader (DataLoader): The dataset iterator for testing.

        Returns:
            float: The accuracy of the client model [0.0, 1.0]. Returns 0.0 if loader is empty.
        """
        model = self.client_models[client_name]

        if len(test_loader.dataset) == 0:
            logger.warning("[FedDyn] Test loader is empty for client %s", client_name)
            return 0.0

        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        accuracy = ensemble_accuracy(y_true, y_pred)
        model.cpu()

        logger.info(
            "[FedDyn] Client %s evaluation accuracy: %.4f",
            client_name,
            accuracy,
        )
        return accuracy

    def get_global_model_accuracy(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Convenience wrapper to evaluate the global model on a test set.

        Args:
            test_loader (DataLoader): The dataset iterator for testing.

        Returns:
            float: The accuracy of the global model.
        """
        return self.evaluate(test_loader)