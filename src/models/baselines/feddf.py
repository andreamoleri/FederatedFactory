"""
🤖 FedDF Baseline Module
------------------------

This module implements the **FedDF** (Federated Distillation Fusion) algorithm, 
specifically the **FedDF-FA** variant (FedDF + FedAvg), as a robust baseline 
for Federated Learning experiments.

🧠 Purpose:
    To faithfully reproduce the logic of Ensemble Distillation for Robust Model Fusion, 
    combining standard Federated Averaging with server-side knowledge distillation 
    on a shared (proxy) dataset.

🔧 Core Functionalities:
    • Client-side training using SGD with momentum and weight decay
    • Generation of logits on a proxy dataset (validation loader) by clients
    • Server-side aggregation using standard FedAvg
    • Server-side Knowledge Distillation (KD) to refine the global model using 
      client ensembles (Teacher) against the global model (Student)

Start-up Logic:
    1. Clients train locally and compute logits on a proxy dataset.
    2. Server aggregates weights (FedAvg).
    3. Server constructs an ensemble prediction from client logits.
    4. Server refines the global model by minimizing KL Divergence against the ensemble.

🎯 Intended Use:
    • Academic benchmarking of FL algorithms
    • Robustness testing against non-IID data distributions
    • Research into knowledge distillation in distributed settings

📁 Dependencies:
    • numpy
    • torch
    • flwr (Flower)
    • models.baselines.base (FederatedBaseline)

📝 Notes:
    For "faithful" FedDF, all clients must generate logits on the **exact same** proxy dataset (same order, same samples). The code includes logic to detect 
    data alignment and falls back to an approximate method if alignment fails.

Author: Andrea Moleri
File Location: src/models/baselines/feddf.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

from models.baselines.base import (
    FederatedBaseline,
    evaluate_single_classifier,
    ensemble_accuracy,
)

logger = logging.getLogger(__name__)


class FedDFBaseline(FederatedBaseline):
    """
    FedDF baseline implementation (Federated Distillation + FedAvg).

    Reference:
        Lin et al., "Ensemble Distillation for Robust Model Fusion in Federated Learning",
        NeurIPS 2020 (FedDF).

    Implementation Objectives:
    --------------------------
    Faithfully reproduces the FedDF-FA (FedDF + FedAvg) optimization recipe:

    1. **Local Training (Client-side):**
       - Identical to `FedAvgBaseline` in terms of optimization:
         - SGD with momentum and weight decay.
         - Learning Rate Schedule: $LR_t = \text{base\_lr} \times (\text{decay})^{\text{round}}$.
         - Optional gradient clipping.
         - No early stopping: clients always send weights from the last epoch.
       - **FedDF Specifics:**
         - After training, the client computes logits on a **proxy/public dataset** (`val_loader`), which is shared among clients and **not shuffled** (`shuffle=False`).
         - The client sends pairs of $(x, z_k(x))$ (logits) to the server.

    2. **Aggregation (Server-side):**
       - **Step 1:** Classic FedAvg (weighted average via Flower):
         $$w^{(t+1)}_{FA} = \text{aggregate}(\{w_k^{(t+1)}\}, \text{weighted by } n_k)$$
       - **Step 2:** FedDF Distillation:
         - The global model (student) is initialized with $w^{(t+1)}_{FA}$.
         - For every $x$ in the proxy dataset:
           - Calculate client probabilities: $p_k(x) = \text{softmax}(z_k(x) / T)$.
           - Construct ensemble target: $p_{ens}(x) = \frac{1}{K} \sum_k p_k(x)$.
         - The student is trained to minimize:
           $$L = T^2 \cdot \text{KL}( \log \text{softmax}(z_s(x) / T), p_{ens}(x) )$$
           where $z_s(x)$ are the student's logits.

    Attributes:
        temperature (float): Softmax temperature $T$ for distillation.
        distill_epochs (int): Number of epochs for server-side distillation.
        min_clients_for_distill (int): Minimum client updates required to perform distillation.
        distill_batch_size (int): Batch size for the distillation process.
        client_logits (Dict): Buffer to store client inputs and logits for the current round.
    """

    def __init__(self, args, num_classes: int, chans: int, device: torch.device):
        """
        Initialize the FedDF Baseline.

        Args:
            args (Namespace): Configuration object containing hyperparameters.
            num_classes (int): Number of output classes.
            chans (int): Number of input channels.
            device (torch.device): Computational device (CPU/GPU).
        """
        super().__init__(args, num_classes, chans, device)
        self.current_round = 0

        # Hyperparameters aligned with FedAvgBaseline.
        self._momentum = float(getattr(self.args, "baseline_momentum", 0.9))
        self._weight_decay = float(getattr(self.args, "baseline_weight_decay", 1e-4))
        self._max_grad_norm = float(
            getattr(self.args, "baseline_clip_grad_norm", 5.0)
        )
        # Disable clipping if value is non-positive
        if self._max_grad_norm <= 0:
            self._max_grad_norm = None

        # FedDF-specific parameters.
        self.temperature = float(getattr(self.args, "feddf_temperature", 3.0))
        self.distill_epochs = int(getattr(self.args, "feddf_distill_epochs", 3))
        if self.distill_epochs <= 0:
            logger.warning(
                "[FedDF] Invalid feddf_distill_epochs=%d; defaulting to 1 distillation epoch.",
                self.distill_epochs,
            )
            self.distill_epochs = 1

        self.min_clients_for_distill = int(
            getattr(self.args, "feddf_min_clients_for_distill", 2)
        )
        if self.min_clients_for_distill < 1:
            self.min_clients_for_distill = 1

        # Batch size for distillation (used when iterating over tensors in memory).
        self.distill_batch_size = int(
            getattr(
                self.args, "feddf_distill_batch_size", getattr(self.args, "batch_size", 32)
            )
        )
        if self.distill_batch_size <= 0:
            logger.warning(
                "[FedDF] Invalid feddf_distill_batch_size=%d; defaulting to 32.",
                self.distill_batch_size,
            )
            self.distill_batch_size = 32

        # Logits and inputs on validation/proxy for each client:
        # client_logits[client_name] = {"inputs": Tensor[N, ...],
        #                               "logits": Tensor[N, num_classes]}
        #
        # Design Logic: If clients share the same proxy dataset (same order),
        # these tensors will be aligned, allowing the calculation of a true ensemble for each x.
        self.client_logits: Dict[str, Dict[str, torch.Tensor]] = {}

        logger.info(
            "[FedDF] Initialized | momentum=%.3f | weight_decay=%.1e | "
            "clip_grad_norm=%s | T=%.2f | distill_epochs=%d | "
            "min_clients_for_distill=%d | distill_batch_size=%d",
            self._momentum,
            self._weight_decay,
            str(self._max_grad_norm),
            self.temperature,
            self.distill_epochs,
            self.min_clients_for_distill,
            self.distill_batch_size,
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
        Evaluate the model on a given DataLoader.

        Args:
            model (nn.Module): The model to evaluate.
            loader (DataLoader): The data loader containing validation/test data.
            criterion (nn.Module): The loss function.

        Returns:
            Tuple[float, float]: A tuple containing (average_loss, accuracy).
            Returns (nan, nan) if the loader is empty or loss is non-finite.
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
                    "[FedDF] Non-finite validation loss detected "
                    "(loss=%s). Setting val_loss/val_acc to NaN for this loader.",
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
        Calculate the effective learning rate for the current round.

        Formula:
            $LR_t = \text{base\_lr} \times (\text{round\_decay})^{\text{round\_num}}$
            (where round_num is 0-based).

        Args:
            round_num (int): The current training round index.

        Returns:
            Tuple[float, float, float]: (calculated_lr, base_lr, effective_decay).
        """
        base_lr = float(getattr(self.args, "learning_rate", 0.1))
        round_decay = float(getattr(self.args, "baseline_round_lr_decay", 1.0))

        if round_decay > 1.0:
            logger.warning(
                "[FedDF] baseline_round_lr_decay=%.4f > 1.0; "
                "clamping to 1.0 to prevent increasing LR.",
                round_decay,
            )
            round_decay = 1.0
        if round_decay < 0.0:
            logger.warning(
                "[FedDF] baseline_round_lr_decay=%.4f < 0.0; "
                "clamping to 0.0.",
                round_decay,
            )
            round_decay = 0.0

        lr = base_lr * (round_decay ** round_num)
        return float(lr), base_lr, round_decay

    @staticmethod
    def _has_non_finite_params(model: nn.Module) -> bool:
        """
        Check if any parameter in the model contains NaN or Inf values.

        Args:
            model (nn.Module): The model to check.

        Returns:
            bool: True if non-finite parameters exist, False otherwise.
        """
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
        return False

    # ------------------------------------------------------------------ #
    # Local training (FedAvg-like)                                      #
    # ------------------------------------------------------------------ #
    def train_client(
        self,
        client_name: str,
        train_loader,
        val_loader,
        round_num: int,
    ) -> Tuple[dict, int]:
        """
        Train a single client using SGD with momentum, logging, and logit collection.

        Behavior conforms to "Vanilla FedAvg" for the baseline training steps,
        augmented with FedDF logic:

        1. **Initialization**: Model starts with global weights from the current round.
        2. **Training**: Executes `baseline_epochs_per_round` epochs of SGD on `train_loader`.
        3. **Validation**: Used ONLY for logging (Cross-Entropy loss, accuracy).
        4. **FedDF Collection**:
           - After training, if `val_loader` is not empty, the client computes inputs
             and logits $(x, z_k(x))$.
           - These are stored for server-side FedDF distillation.

        Safety Mechanism:
            If NaN/Inf values are detected in loss, gradients, or parameters, the
            update is **discarded** (returns `num_samples=0`) and weights are reset
            to the initial state of the round.

        Args:
            client_name (str): Identifier for the client.
            train_loader (DataLoader): Loader for local training data.
            val_loader (DataLoader): Loader for validation/proxy data.
            round_num (int): Current round number (0-based).

        Returns:
            Tuple[dict, int]: A tuple containing (model_state_dict, number_of_samples).
        """
        model = self.client_models[client_name]

        # Save weights at the start of the round for potential rollback.
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # LR for this round (same policy as FedAvgBaseline).
        lr, base_lr, round_decay = self._effective_lr(round_num)

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        model.to(self.device)

        if train_loader is None or len(train_loader.dataset) == 0:
            logger.warning(
                "[BASELINE/FedDF] Client %s | round %d: train_loader empty, "
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
                "[BASELINE/FedDF] baseline_epochs_per_round=%d invalid; "
                "using 1 epoch per round.",
                epochs_per_round,
            )
            epochs_per_round = 1

        logger.info(
            "[BASELINE/FedDF] Client %s | round %d | epochs_per_round=%d | "
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

        train_epoch_losses: List[float] = []
        val_epoch_losses: List[float] = []
        val_epoch_accs: List[float] = []

        non_finite_detected = False

        for epoch in range(epochs_per_round):
            model.train()
            running_loss = 0.0
            batch_count = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)

                if not torch.isfinite(loss):
                    logger.error(
                        "[BASELINE/FedDF] Non-finite loss detected "
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

                # Optional gradient clipping.
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
                        "[BASELINE/FedDF] Non-finite gradients detected "
                        "on client %s | round %d | epoch %d. "
                        "Discarding client update for this round.",
                        client_name,
                        round_num + 1,
                        epoch + 1,
                    )
                    non_finite_detected = True
                    break

                optimizer.step()

                if self._has_non_finite_params(model):
                    logger.error(
                        "[BASELINE/FedDF] Non-finite parameters detected after optimizer.step "
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

            # Validation (logging only; does not influence local update).
            val_loss, val_acc = self._evaluate_on_loader(model, val_loader, criterion)
            val_epoch_losses.append(val_loss)
            val_epoch_accs.append(val_acc)

            logger.info(
                "[BASELINE/FedDF] Client %s | round %d | "
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
                "[BASELINE/FedDF] Client %s | round %d: NaN/Inf detected, "
                "restoring initial weights and not contributing "
                "to this round's aggregation.",
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
            return state_dict_copy, 0

        # Collect logits on validation/proxy for FedDF distillation.
        # Assumption for faithful FedDF: val_loader is the same proxy dataset
        # for all clients (shuffle=False, same order).
        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            all_inputs = []
            all_logits = []

            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(self.device)
                    logits = model(data)
                    all_inputs.append(data.cpu())
                    all_logits.append(logits.cpu())

            if all_inputs and all_logits:
                inputs_tensor = torch.cat(all_inputs, dim=0)
                logits_tensor = torch.cat(all_logits, dim=0)

                self.client_logits[client_name] = {
                    "inputs": inputs_tensor,
                    "logits": logits_tensor,
                }
                logger.info(
                    "[BASELINE/FedDF] Client %s | round %d: stored %d samples "
                    "for distillation (proxy/validation).",
                    client_name,
                    round_num + 1,
                    inputs_tensor.size(0),
                )

        model.cpu()

        key = f"{client_name}_round{round_num}"
        self.history.setdefault("train_loss", {})[key] = train_epoch_losses
        self.history.setdefault("val_loss", {})[key] = val_epoch_losses
        self.history.setdefault("val_acc", {})[key] = val_epoch_accs

        final_state = {k: v.clone() for k, v in model.state_dict().items()}
        return final_state, num_samples

    # ------------------------------------------------------------------ #
    # FedDF aggregation: FedAvg + distillazione                         #
    # ------------------------------------------------------------------ #
    def aggregate(
        self,
        round_num: int,
        client_updates: Dict[str, Tuple[dict, int]],
    ) -> None:
        """
        Perform FedAvg aggregation followed by server-side FedDF distillation.

        Process:
        1. **Filter**: Remove updates with non-finite parameters or zero samples.
        2. **FedAvg**: Average client weights (weighted by number of samples).
        3. **Distill**: Improve the aggregated model using stored client logits (`_distill_knowledge`).

        Args:
            round_num (int): Current round number.
            client_updates (Dict): Dictionary mapping client names to (state_dict, num_samples).
        """
        if not client_updates:
            logger.warning("[FedDF] No client updates to aggregate")
            return

        # State of the first client to define key order.
        first_state, _ = next(iter(client_updates.values()))
        param_keys = list(first_state.keys())

        # Build list of (ndarrays, num_examples) for Flower.
        weights_results = []
        total_clients_used = 0
        total_weight = 0
        valid_clients: List[str] = []

        for client_name, (client_state, num_samples) in client_updates.items():
            if num_samples is None or num_samples <= 0:
                logger.info(
                    "[FedDF] Client %s skipped in aggregation (num_samples=%s).",
                    client_name,
                    str(num_samples),
                )
                continue

            # Sanity check: Ensure no non-finite parameters before converting to numpy.
            bad_param = False
            for k, tensor in client_state.items():
                if not torch.isfinite(tensor).all():
                    logger.warning(
                        "[FedDF] Client %s has non-finite parameters (%s). "
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
            valid_clients.append(client_name)

        if not weights_results:
            logger.warning("[FedDF] No valid client updates after filtering")
            return

        logger.info(
            "[FedDF] Round %d: FedAvg aggregating %d/%d client updates "
            "(total weight=%d).",
            round_num + 1,
            total_clients_used,
            len(client_updates),
            total_weight,
        )

        # Flower FedAvg: standard weighted average.
        aggregated_ndarrays = flwr_aggregate(weights_results)

        # Robustness check: if the aggregate contains NaN/Inf, DO NOT update.
        for i, arr in enumerate(aggregated_ndarrays):
            if not np.all(np.isfinite(arr)):
                logger.error(
                    "[FedDF] Aggregated parameter index %d is non-finite (NaN/Inf). "
                    "Skipping global model update for this round.",
                    i,
                )
                return

        new_global_state = {}
        for key, arr in zip(param_keys, aggregated_ndarrays):
            ref_tensor = first_state[key]
            new_global_state[key] = torch.from_numpy(arr).to(ref_tensor.dtype)

        # Update global model with FedAvg result.
        self.global_model.load_state_dict(new_global_state)

        # Save a copy of pre-distillation state (for potential rollback).
        pre_distill_state = {
            k: v.detach().clone() for k, v in self.global_model.state_dict().items()
        }

        # Server-side FedDF distillation (if possible).
        distill_success = self._distill_knowledge(
            round_num=round_num,
            participating_clients=valid_clients,
            initial_state=pre_distill_state,
        )

        if not distill_success:
            # In case of failure, ensure the global model is in a valid state (pre-distillation).
            self.global_model.load_state_dict(pre_distill_state)

        # Propagate the global model (post-distillation, if successful) to all clients.
        global_state = self.global_model.state_dict()
        for client_name in self.client_models:
            self.client_models[client_name].load_state_dict(global_state)

        self.current_round += 1
        logger.info(
            "[FedDF] Aggregation + distillation round %d completed "
            "(clients_used=%d, distill_success=%s).",
            round_num + 1,
            len(valid_clients),
            str(distill_success),
        )

    # ------------------------------------------------------------------ #
    # Distillazione server-side (FedDF)                                 #
    # ------------------------------------------------------------------ #
    def _distill_knowledge(
        self,
        round_num: int,
        participating_clients: List[str],
        initial_state: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Perform FedDF distillation from the client "ensemble" to the global model.

        Procedure:
        1. **Selection**: Uses logits stored in `self.client_logits` strictly for clients
           who participated in this round's aggregation.
        2. **Threshold**: If client count < `min_clients_for_distill`, skips distillation.
        3. **Alignment & Ensemble**:
           - **Aligned (Faithful FedDF):** If all clients share the exact same proxy data
             (same tensor shapes and values), calculates a true ensemble:
             $$p_{ens}(x) = \frac{1}{K} \sum_k \text{softmax}(z_k(x)/T)$$
           - **Unaligned (Fallback):** If input data differs (different order or subsets),
             treats every $(x, p_k(x))$ pair as an independent training sample (approximate FedDF).
        4. **Optimization**: Updates global model to minimize KL Divergence against $p_{ens}(x)$.

        Args:
            round_num (int): Current round index.
            participating_clients (List[str]): List of client IDs involved in the aggregation.
            initial_state (Dict): State dict of the model before distillation (for rollback).

        Returns:
            bool: True if distillation was successful, False if skipped or failed.
        """
        if self.global_model is None:
            logger.warning("[FedDF] global_model is None, skip distillation.")
            return False

        # Select only participating clients that have available logits.
        available_clients = [
            c
            for c in participating_clients
            if c in self.client_logits
            and "inputs" in self.client_logits[c]
            and "logits" in self.client_logits[c]
        ]

        if len(available_clients) < self.min_clients_for_distill:
            logger.info(
                "[FedDF] Round %d: only %d clients with available logits "
                "(min required=%d); skipping distillation.",
                round_num + 1,
                len(available_clients),
                self.min_clients_for_distill,
            )
            # Clear old logits to prevent memory leaks.
            for c in available_clients:
                self.client_logits.pop(c, None)
            return False

        # Prepare tensors for each client: inputs_k and p_k(x) = softmax(z_k/T).
        T = float(self.temperature)
        client_inputs_list: List[torch.Tensor] = []
        client_soft_list: List[torch.Tensor] = []

        for client_name in available_clients:
            data = self.client_logits[client_name]
            inputs = data["inputs"]
            logits = data["logits"]

            if inputs.size(0) == 0:
                logger.warning(
                    "[FedDF] Round %d: client %s has 0 proxy samples, ignoring in distillation.",
                    round_num + 1,
                    client_name,
                )
                continue

            soft_probs = F.softmax(logits / T, dim=1)

            client_inputs_list.append(inputs)
            client_soft_list.append(soft_probs)

        if not client_inputs_list:
            logger.warning(
                "[FedDF] Round %d: no valid data for distillation "
                "(all tensors empty).",
                round_num + 1,
            )
            for c in available_clients:
                self.client_logits.pop(c, None)
            return False

        # -------------------------------------------------------------- #
        # Attempt "Faithful" FedDF: Ensemble on shared proxy dataset    #
        # -------------------------------------------------------------- #
        aligned = True
        base_inputs = client_inputs_list[0]
        num_classes = client_soft_list[0].size(1)

        # All clients must have the same shape regarding inputs.
        for idx, inputs in enumerate(client_inputs_list[1:], start=1):
            if inputs.shape != base_inputs.shape:
                aligned = False
                logger.warning(
                    "[FedDF] Round %d: proxy dataset dimensions differ between clients "
                    "(client 0 shape=%s, client %d shape=%s). "
                    "Using fallback 'per-example' (approximate FedDF).",
                    round_num + 1,
                    tuple(base_inputs.shape),
                    idx,
                    tuple(inputs.shape),
                )
                break
            # Stricter check: Input tensors must be identical.
            if not torch.equal(inputs, base_inputs):
                aligned = False
                logger.warning(
                    "[FedDF] Round %d: proxy input tensors are not identical "
                    "between clients (dataset not perfectly shared). "
                    "Using fallback 'per-example' (approximate FedDF).",
                    round_num + 1,
                )
                break

        if aligned:
            # Shared proxy dataset among clients: "True" ensemble.
            # inputs_ens: (N, ...)
            # soft_probs_k: (N, C) for each client
            stacked_probs = torch.stack(client_soft_list, dim=0)  # (K, N, C)
            distill_inputs = base_inputs
            distill_soft_labels = stacked_probs.mean(dim=0)  # (N, C)

            logger.info(
                "[FedDF] Round %d: Faithful FEDDF ensemble distillation "
                "on shared proxy dataset (%d clients, %d samples, %d classes).",
                round_num + 1,
                stacked_probs.size(0),
                distill_inputs.size(0),
                num_classes,
            )
        else:
            # Fallback: each (x, p_k(x)) is an independent example.
            # This implements the previous version logic; less faithful to the
            # paper but mathematically well-defined.
            distill_inputs = torch.cat(client_inputs_list, dim=0)
            distill_soft_labels = torch.cat(client_soft_list, dim=0)

            logger.warning(
                "[FedDF] Round %d: using 'per-example' distillation (fallback), "
                "treating each (x, p_k(x)) pair as independent. "
                "Total samples=%d, classes=%d.",
                round_num + 1,
                distill_inputs.size(0),
                num_classes,
            )

        # Final dimensional consistency check.
        if distill_inputs.size(0) != distill_soft_labels.size(0):
            logger.error(
                "[FedDF] Round %d: mismatch between input count (%d) and "
                "soft-label count (%d) in distillation. Skipping distillation.",
                round_num + 1,
                distill_inputs.size(0),
                distill_soft_labels.size(0),
            )
            for c in available_clients:
                self.client_logits.pop(c, None)
            return False

        num_samples = distill_inputs.size(0)
        logger.info(
            "[FedDF] Round %d: performing distillation with %d effective samples.",
            round_num + 1,
            num_samples,
        )

        # Initialize optimizer for the global model.
        # Uses the same LR as the round (consistent with the optimization recipe).
        lr, base_lr, round_decay = self._effective_lr(round_num)

        self.global_model.to(self.device)
        optimizer = optim.SGD(
            self.global_model.parameters(),
            lr=lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
        criterion_kl = nn.KLDivLoss(reduction="batchmean")

        distill_epoch_losses: List[float] = []
        non_finite_detected = False

        for epoch in range(self.distill_epochs):
            self.global_model.train()
            running_loss = 0.0
            batch_count = 0

            # Implicit shuffle: one could add random permutation here.
            for i in range(0, num_samples, self.distill_batch_size):
                batch_inputs = distill_inputs[i : i + self.distill_batch_size].to(
                    self.device
                )
                batch_soft_labels = distill_soft_labels[
                    i : i + self.distill_batch_size
                ].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                student_logits = self.global_model(batch_inputs)
                student_log_probs = F.log_softmax(student_logits / T, dim=1)

                # Loss = T^2 * KL( log p_s^T(x), p_ens(x) )
                loss = criterion_kl(student_log_probs, batch_soft_labels) * (T * T)

                if not torch.isfinite(loss):
                    logger.error(
                        "[FedDF] Non-finite distillation loss (epoch %d, round %d): %s. "
                        "Restoring global_model to pre-distillation state.",
                        epoch + 1,
                        round_num + 1,
                        str(loss.item()),
                    )
                    non_finite_detected = True
                    break

                loss.backward()

                # Optional gradient clipping.
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.global_model.parameters(), max_norm=self._max_grad_norm
                    )

                # Check for non-finite gradients.
                grads_ok = True
                for p in self.global_model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    logger.error(
                        "[FedDF] Non-finite gradients during distillation "
                        "(epoch %d, round %d). Restoring global_model.",
                        epoch + 1,
                        round_num + 1,
                    )
                    non_finite_detected = True
                    break

                optimizer.step()

                if self._has_non_finite_params(self.global_model):
                    logger.error(
                        "[FedDF] Non-finite parameters after optimizer.step "
                        "in distillation (epoch %d, round %d). Restoring global_model.",
                        epoch + 1,
                        round_num + 1,
                    )
                    non_finite_detected = True
                    break

                running_loss += float(loss.item())
                batch_count += 1

            if non_finite_detected:
                break

            epoch_loss = running_loss / batch_count if batch_count > 0 else float("nan")
            distill_epoch_losses.append(epoch_loss)

            logger.info(
                "[FedDF] Round %d | distill epoch %d/%d | loss=%.4f "
                "(lr=%.6f, base_lr=%.6f, round_decay=%.4f)",
                round_num + 1,
                epoch + 1,
                self.distill_epochs,
                epoch_loss,
                lr,
                base_lr,
                round_decay,
            )

        if non_finite_detected:
            # Restore pre-distillation state.
            self.global_model.load_state_dict(initial_state)
            self.global_model.cpu()
            # Clean up logits used in this round.
            for c in available_clients:
                self.client_logits.pop(c, None)
            return False

        # Distillation successful.
        self.global_model.cpu()
        key = f"round{round_num}"
        self.history.setdefault("distill_loss", {})[key] = distill_epoch_losses

        logger.info(
            "[FedDF] Round %d: distillation completed "
            "(epochs=%d, final_loss=%.4f).",
            round_num + 1,
            self.distill_epochs,
            distill_epoch_losses[-1] if distill_epoch_losses else float("nan"),
        )

        # Clean up logits used in this round to prevent memory leaks.
        for c in available_clients:
            self.client_logits.pop(c, None)

        return True

    # ------------------------------------------------------------------ #
    # Utility per valutazione                                           #
    # ------------------------------------------------------------------ #
    def evaluate_client(self, client_name, test_loader):
        """
        Evaluate a single client model on a provided test loader.

        Args:
            client_name (str): Identifier for the client.
            test_loader (DataLoader): DataLoader containing test data.

        Returns:
            float: The accuracy of the client model.
        """
        model = self.client_models[client_name]

        if len(test_loader.dataset) == 0:
            logger.warning("[FedDF] Test loader is empty for client %s", client_name)
            return 0.0

        model.to(self.device)
        y_true, y_pred = evaluate_single_classifier(model, test_loader, self.device)
        accuracy = ensemble_accuracy(y_true, y_pred)
        model.cpu()

        logger.info(
            "[FedDF] Client %s evaluation accuracy: %.4f",
            client_name,
            accuracy,
        )
        return accuracy

    def get_global_model_accuracy(self, test_loader):
        """
        Shortcut to evaluate the global model.

        Args:
            test_loader (DataLoader): DataLoader containing test data.

        Returns:
            float: Accuracy of the global model.
        """
        return self.evaluate(test_loader)