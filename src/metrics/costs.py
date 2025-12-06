"""
🪵 Experiment Cost and Resource Tracking Module
---------------------------------------------

This module implements a comprehensive system for monitoring and auditing the computational
resources consumed during machine learning experiments. It provides a unified interface
for tracking hardware utilization, energy consumption, and algorithmic complexity.

🧠 Purpose:
    Designed to facilitate reproducible research and operational auditing by capturing
    granular metrics regarding wall-clock time, floating-point operations (FLOPs),
    energy usage (via NVML or CodeCarbon), and I/O throughput.

🔧 Core Functionalities:
    • Real-time energy monitoring via NVIDIA Management Library (NVML) or CodeCarbon
    • Algorithmic complexity estimation (FLOPs) using `fvcore` with custom heuristic extensions
    • Hierarchical phase-based tracking for granular analysis (e.g., distinct training vs. validation phases)
    • Network and disk I/O accounting with configurable tracking modes

🎯 Intended Use:
    • Academic research requiring energy and efficiency baselines
    • CI/CD pipelines for monitoring regression in resource usage
    • Production environments requiring audit logs for model inference costs

📁 Dependencies:
    • torch
    • psutil
    • pynvml (optional, for GPU telemetry)
    • codecarbon (optional, for emission estimation)
    • fvcore (optional, for FLOP counting)

📝 Notes:
    This module employs aggressive fallback mechanisms (swallowing exceptions) to ensure
    that monitoring instrumentation never interrupts the primary execution flow of the experiment.

Author: Andrea Moleri
File Location: src/tracking/cost_tracker.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import atexit
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import subprocess
import torch


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _lazy_import(mod: str, pip_name: Optional[str] = None):
    """
    Dynamically imports a module, attempting to install it via pip if not found.

    This function facilitates the use of optional dependencies without requiring
    pre-installation in the environment.

    Args:
        mod (str): The name of the module to import.
        pip_name (Optional[str]): The package name to install via pip if distinct
            from the module name. Defaults to `mod` if not provided.

    Returns:
        module: The imported module object.

    Raises:
        subprocess.CalledProcessError: If the pip installation fails.
        ImportError: If the module cannot be imported even after installation.
    """
    try:
        return __import__(mod)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name or mod]
        )
        return __import__(mod)


psutil = _lazy_import("psutil")

try:
    codecarbon = __import__("codecarbon")
except ImportError:
    codecarbon = None  # optional (for energy + CO2 estimates)

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None  # optional, we'll gracefully degrade if missing

# Optional NVML (GPU telemetry)
try:
    import pynvml

    _NVML_OK = True
    pynvml.nvmlInit()
except Exception:
    _NVML_OK = False


# ---------------------------------------------------------------------------
# fvcore custom op handles to reduce "Unsupported operator" warnings
# ---------------------------------------------------------------------------

def _first_tensor_numel(obj: Any) -> int:
    """
    Heuristically attempts to retrieve the number of elements from the first
    tensor-like object found within the input.

    Args:
        obj (Any): The object to inspect (Tensor, list, or tuple).

    Returns:
        int: The number of elements (`numel`) if a tensor is found; otherwise 0.
    """
    try:
        if torch.is_tensor(obj):
            return int(obj.numel())
        if isinstance(obj, (list, tuple)) and obj:
            # find first tensor inside
            for el in obj:
                if torch.is_tensor(el):
                    return int(el.numel())
        # fallback: 0 if nothing tensor-like
        return 0
    except Exception:
        return 0


def _handle_unary_elementwise(inputs, outputs):
    # ~1 FLOP per element in output
    return _first_tensor_numel(outputs)


def _handle_binary_elementwise(inputs, outputs):
    # ~1 FLOP per element in output (add/mul: 1 operation per element)
    return _first_tensor_numel(outputs)


def _handle_randn_like(inputs, outputs):
    # Noise generation: we do not count FLOPs for the model logic
    return 0


def _extract_kernel_hw_from_inputs(inputs) -> Optional[Tuple[int, int]]:
    """
    Attempts to extract the kernel dimensions (height, width) from the input
    arguments of a `max_pool2d` operation.

    JIT schemas for pooling operations vary; this function probes common
    argument positions to identify kernel specifications.

    Args:
        inputs (tuple): The input arguments passed to the operator.

    Returns:
        Optional[Tuple[int, int]]: A tuple `(height, width)` if extraction
        is successful, otherwise `None`.
    """
    try:
        # Inputs can be: (x, kernel_size, stride, padding, dilation, ceil_mode)
        # kernel_size can be int or (h,w)
        if len(inputs) >= 2:
            k = inputs[1]
            if isinstance(k, (list, tuple)):
                if len(k) >= 2:
                    return int(k[0]), int(k[1])
                elif len(k) == 1:
                    kh = kw = int(k[0])
                    return kh, kw
            elif isinstance(k, int):
                kh = kw = int(k)
                return kh, kw
    except Exception:
        pass
    return None


def _handle_max_pool2d_like(inputs, outputs):
    """
    Estimates FLOPs for max-pooling operations based on the number of comparisons.

    The cost is approximated as:
    $$ FLOPs \approx N_{output} \times (K_h \times K_w - 1) $$
    where $N_{output}$ is the number of output elements and $K$ represents kernel dimensions.

    Args:
        inputs: Input arguments to the operator.
        outputs: Output tensor(s) from the operator.

    Returns:
        int: Estimated FLOP count.
    """
    numel = _first_tensor_numel(outputs)
    if numel == 0:
        return 0
    khw = _extract_kernel_hw_from_inputs(inputs)
    if khw is not None:
        kh, kw = khw
        per_elem = max(kh * kw - 1, 1)
    else:
        # fallback prudente
        per_elem = 1
    return numel * per_elem


def register_extra_ops(fca):
    """
    Registers custom operator handles on a `FlopCountAnalysis` instance.

    This extends `fvcore`'s capabilities to cover operators that are technically
    simple but officially unsupported, preventing zero-count warnings.

    Args:
        fca (FlopCountAnalysis): The analysis instance to be patched.
    """
    # Some very old versions of fvcore might not have set_op_handle
    set_handle = getattr(fca, "set_op_handle", None)
    if set_handle is None:
        return

    try:
        # Element-wise
        set_handle("aten::exp", _handle_unary_elementwise)
        set_handle("aten::tanh", _handle_unary_elementwise)
        set_handle("aten::mul", _handle_binary_elementwise)
        set_handle("aten::add", _handle_binary_elementwise)

        # Random
        set_handle("aten::randn_like", _handle_randn_like)

        # Pooling
        set_handle("aten::max_pool2d", _handle_max_pool2d_like)
        set_handle("aten::max_pool2d_with_indices", _handle_max_pool2d_like)
    except Exception:
        # Do not interrupt FLOPs analysis if registration fails.
        pass


# ---------------------------------------------------------------------------
# helpers for bytes / energy / NVML
# ---------------------------------------------------------------------------

def _sum_dir_bytes(path: Path) -> int:
    """
    Calculates the total size in bytes of all files within a directory tree.

    Args:
        path (Path): The root directory to scan.

    Returns:
        int: Total size in bytes. Returns 0 if the path does not exist.
    """
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def _nvml_query() -> Dict[str, Any]:
    """
    Retrieves a snapshot of coarse GPU energy statistics via NVML.

    Note:
        `total_energy_mJ_since_boot` represents the cumulative energy consumption
        since the driver was loaded, not merely since the process started.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'available' (bool): Whether NVML query was successful.
            - 'gpus' (List[Dict]): List of GPU details including index, name,
              and total energy in millijoules (if supported).
    """
    if not _NVML_OK:
        return {"available": False}

    info: Dict[str, Any] = {"available": True, "gpus": []}
    try:
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            try:
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
            except Exception:
                name = f"GPU{i}"

            try:
                # mJ since driver load
                energy_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            except Exception:
                energy_mJ = None

            info["gpus"].append(
                {
                    "index": i,
                    "name": name,
                    "total_energy_mJ_since_boot": energy_mJ,
                }
            )
    except Exception:
        info["available"] = False
    return info


def _nvml_delta_kwh(info0: Dict[str, Any], info1: Dict[str, Any]) -> Optional[float]:
    """
    Estimates energy consumption (kWh) between two NVML snapshots.

    Formula:
    $$ E_{kWh} = \frac{\Delta E_{mJ}}{3.6 \times 10^9} $$

    Args:
        info0 (Dict[str, Any]): The initial NVML snapshot.
        info1 (Dict[str, Any]): The final NVML snapshot.

    Returns:
        Optional[float]: The energy delta in kWh, or `None` if data is invalid/missing.
    """
    try:
        if not (info0.get("available") and info1.get("available")):
            return None

        total_mJ = 0.0
        g0_list: List[Dict[str, Any]] = info0.get("gpus", [])
        g1_list: List[Dict[str, Any]] = info1.get("gpus", [])
        for g0, g1 in zip(g0_list, g1_list):
            e0 = g0.get("total_energy_mJ_since_boot")
            e1 = g1.get("total_energy_mJ_since_boot")
            if e0 is None or e1 is None:
                continue
            # Clamp non-negative in case counters reset
            total_mJ += max(0.0, float(e1) - float(e0))

        # If we literally never saw a supported counter, total_mJ will be 0.0.
        # That's a valid "0 kWh" if counters exist. BUT on some H100 setups
        # NVML just returns None for all GPUs, so we never add anything.
        # In that case total_mJ stays 0.0 *because we skipped all GPUs*,
        # which is indistinguishable from "truly zero". We'll handle that
        # later by checking if we had *any* supported counters.
        any_supported = any(
            (g0.get("total_energy_mJ_since_boot") is not None and
             g1.get("total_energy_mJ_since_boot") is not None)
            for g0, g1 in zip(g0_list, g1_list)
        )

        if not any_supported:
            # Means counters weren't available at all; signal "no data"
            return None

        kwh = total_mJ / 3.6e9
        return kwh
    except Exception:
        return None


def _nvml_power_sum_w() -> Optional[float]:
    """
    Calculates the instantaneous total GPU power draw in Watts.

    This sums `nvmlDeviceGetPowerUsage()` across all available devices.
    Used as a fallback when cumulative energy counters are unavailable.

    Returns:
        Optional[float]: Total power in Watts, or `None` if NVML is unavailable.
    """
    if not _NVML_OK:
        return None

    try:
        n = pynvml.nvmlDeviceGetCount()
        total_w = 0.0
        have_any = False
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                # power usage in milliwatts
                mw = pynvml.nvmlDeviceGetPowerUsage(h)
                total_w += mw / 1000.0
                have_any = True
            except Exception:
                # skip GPUs that don't report power
                pass
        if not have_any:
            return None
        return total_w
    except Exception:
        return None


def _pretty_seconds(s: float) -> Dict[str, float]:
    """
    Formats a duration in seconds into a dictionary containing seconds,
    hours, and minutes for human-readable reporting.
    """
    return dict(total_sec=float(s), hours=s / 3600.0, minutes=s / 60.0)


# ---------------------------------------------------------------------------
# FLOPs cache (so we only run fvcore once per model+shape)
# ---------------------------------------------------------------------------

class _FlopsCache:
    """
    Memoization helper to cache forward-pass FLOPs per (model_id, input_shape).

    This prevents the expensive `FlopCountAnalysis` from running repeatedly
    on identical input configurations.
    """

    def __init__(self):
        self.memo: Dict[str, int] = {}

    def key(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> str:
        return f"{id(model)}::{tuple(input_shape)}"

    def get_or_compute(
            self,
            model: torch.nn.Module,
            sample: torch.Tensor,
    ) -> Optional[int]:
        """
        Retrieves cached FLOPs or computes them via `fvcore.nn.FlopCountAnalysis`.

        Args:
            model (torch.nn.Module): The model to analyze.
            sample (torch.Tensor): A sample input tensor to define the shape.

        Returns:
            Optional[int]: The total forward FLOPs, or `None` if analysis fails.

        Note:
            Exceptions are suppressed to ensure FLOP accounting does not terminate training.
        """
        if FlopCountAnalysis is None:
            return None

        k = self.key(model, tuple(sample.shape))
        if k in self.memo:
            return self.memo[k]

        model_device = next(model.parameters()).device
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                try:
                    fca = FlopCountAnalysis(model, sample.to(model_device))
                    # <<< NEW: registers handle for uncovered operators >>>
                    register_extra_ops(fca)
                    f_total = int(fca.total())
                except Exception:
                    return None
            self.memo[k] = f_total
            return f_total
        except Exception:
            return None
        finally:
            if was_training:
                model.train()


_FLOPS_CACHE = _FlopsCache()


# ---------------------------------------------------------------------------
# dataclasses used in the cost report
# ---------------------------------------------------------------------------

@dataclass
class PhaseStats:
    wall_clock_sec: float = 0.0
    flops_total: int = 0
    steps_counted: int = 0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    # new: per-phase energy container (compatible with backfill in main)
    energy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergyStats:
    kwh: float = 0.0
    method: str = "unknown"  # "codecarbon" | "nvml_delta" | "nvml_power_avg" | "unknown"
    emissions_kg_co2e: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BytesStats:
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    artifacts_written_bytes: int = 0
    by_file: Dict[str, int] = field(default_factory=dict)


@dataclass
class FlopsStats:
    total_flops: int = 0  # accumulated over all optimizer steps all models
    per_step_forward_flops: Optional[int] = None
    per_step_backward_flops: Optional[int] = None
    per_step_total_flops: Optional[int] = None  # last registered model
    steps_counted: int = 0
    method: str = (
        "fvcore(FlopCountAnalysis)+heuristic_backward_2x"
    )


@dataclass
class CostReport:
    wall_clock: Dict[str, float] = field(default_factory=dict)
    phases: Dict[str, PhaseStats] = field(default_factory=dict)
    energy: EnergyStats = field(default_factory=EnergyStats)
    bytes: BytesStats = field(default_factory=BytesStats)
    flops: FlopsStats = field(default_factory=FlopsStats)
    environment: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recursive dataclass -> dict helper for JSON serialization
# ---------------------------------------------------------------------------

def _to_dict(obj: Any):
    if hasattr(obj, "__dict__"):
        try:
            return asdict(obj)  # dataclass recursion
        except Exception:
            return dict(obj.__dict__)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# ExperimentCostTracker
# ---------------------------------------------------------------------------

class ExperimentCostTracker:
    """
    A unified context manager for tracking experiment resource utilization.

    Tracks:
      - Wall-clock time
      - Network and disk I/O
      - Artifact file sizes
      - Energy and carbon emissions (via NVML or CodeCarbon)
      - Compute cost (FLOPs)
    """

    def __init__(self, out_dir: Path, args: Optional[Any] = None, network_accounting: str = "psutil"):
        """
        Initializes the tracker, setting up output directories and baseline metrics.

        Args:
            out_dir (Path): Directory where cost reports will be saved.
            args (Optional[Any]): Argument object (e.g., from argparse) to dump into the report.
            network_accounting (str): Strategy for tracking network bytes.
                - 'psutil': Uses system-wide interface counters.
                - 'manual': Relies solely on manual `note_network_bytes` calls.
                - 'mixed': Combines system counters with manual annotations.
        """
        self.out_dir = Path(out_dir)
        self.args = args
        self.rep = CostReport()
        self._phase_nvml0: Dict[str, Any] = {}

        # ── NEW: network accounting mode
        self._network_accounting = str(network_accounting).lower()
        if self._network_accounting not in ("psutil", "manual", "mixed"):
            self._network_accounting = "psutil"

        # runtime bookkeeping
        self._t0: Optional[float] = None
        self._phase_t0: Dict[str, float] = {}

        # FLOPs bookkeeping
        self._cur_per_step_total_flops: int = 0  # last registered model

        # snapshot I/O counters at start
        self._net0 = psutil.net_io_counters() if self._network_accounting in ("psutil", "mixed") else None
        self._disk0 = psutil.disk_io_counters()

        # snapshot size of output dir at start
        self._artifacts_before = _sum_dir_bytes(self.out_dir)

        # file-by-file bytes (for artifacts)
        self._by_file_sizes: Dict[str, int] = {}

        # active phases + snapshot psutil per phase (only if needed)
        self._phase_stack: List[str] = []
        self._phase_net0: Dict[str, Any] = {}
        self._phase_disk0: Dict[str, Any] = {}

        # environment dump
        try:
            gpu_names: list[str] = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        gpu_names.append(torch.cuda.get_device_name(i))
                    except Exception:
                        gpu_names.append(f"cuda:{i}")
            self.rep.environment.update(
                {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "gpus": gpu_names,
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "args": vars(args) if args is not None else None,
                }
            )
        except Exception:
            pass

        # NVML snapshots / power snapshots
        self._nvml0 = _nvml_query()
        self._pwr0_W = _nvml_power_sum_w()

        # optional CodeCarbon tracker
        self._cc_tracker = None
        if codecarbon is not None:
            try:
                # disable CC's own file writes; we serialize manually
                self._cc_tracker = codecarbon.EmissionsTracker(
                    measure_power_secs=1,
                    save_to_file=False,
                    log_level="error",
                )
            except Exception:
                self._cc_tracker = None

        # Safety: ensure finish() still runs at interpreter shutdown
        atexit.register(self._safe_close)

    # ----- context manager hooks -----------------------------------------

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self._cc_tracker is not None:
            try:
                self._cc_tracker.start()
            except Exception:
                # If CodeCarbon start fails, just disable it.
                self._cc_tracker = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()

    def _safe_close(self):
        """
        Called via atexit. Wrap finish() in try/except so interpreter
        shutdown doesn't throw noisy errors.
        """
        try:
            self.finish()
        except Exception:
            pass

    # ----- phase accounting ----------------------------------------------

    def start_phase(self, name: str):
        """
        Marks the beginning of a named execution phase (e.g., "client_003_gen").

        This initializes counters for the phase and pushes it onto the phase stack.
        Specific computation-heavy phases are heuristically identified to exclude
        system-wide network noise.

        Args:
            name (str): The unique identifier for the phase.
        """
        self._phase_t0[name] = time.perf_counter()

        # NEW: create/guarantee the phase container
        ps = self.rep.phases.get(name, PhaseStats())
        self.rep.phases[name] = ps

        # NEW: track active phase (supports nested phases)
        self._phase_stack.append(name)

        # ⚠️ CRITICAL MODIFICATION: DO NOT snapshot psutil for PURE COMPUTATION phases
        # Phases ending with "_gen", "_clf", "_synthesis", "_generation", "_evaluation"
        # are pure computation and should not capture psutil traffic
        is_computation_phase = any(pattern in name for pattern in
                                   ["_gen", "_clf", "_synthesis", "_generation", "_evaluation", "classifier"])

        if self._network_accounting in ("psutil", "mixed") and not is_computation_phase:
            try:
                self._phase_net0[name] = psutil.net_io_counters()
            except Exception:
                self._phase_net0[name] = None
        else:
            self._phase_net0[name] = None

        # For disk, always snapshot (but could be optional)
        try:
            self._phase_disk0[name] = psutil.disk_io_counters()
        except Exception:
            self._phase_disk0[name] = None

        try:
            self._phase_nvml0[name] = _nvml_query()
        except Exception:
            self._phase_nvml0[name] = None

    def end_phase(self, name: str):
        """
        Marks the end of a named phase, calculating and storing the delta
        resources consumed (time, I/O, energy).

        Args:
            name (str): The identifier of the phase to close.
        """
        t0 = self._phase_t0.get(name, None)
        if t0 is None:
            # phase not started: ignore silently
            return

        dt = time.perf_counter() - t0
        ps = self.rep.phases.get(name, PhaseStats())
        ps.wall_clock_sec += dt

        # --- Per-phase I/O Delta
        n0 = self._phase_net0.pop(name, None)
        d0 = self._phase_disk0.pop(name, None)

        # ⚠️ CRITICAL MODIFICATION: for pure computation phases, IGNORE psutil network
        is_computation_phase = any(pattern in name for pattern in
                                   ["_gen", "_clf", "_synthesis", "_generation", "_evaluation", "classifier"])

        # Network: only if psutil/mixed AND not a pure computation phase
        if self._network_accounting in ("psutil", "mixed") and not is_computation_phase:
            try:
                n1 = psutil.net_io_counters()
            except Exception:
                n1 = None
            if n0 is not None and n1 is not None:
                try:
                    ps.network_rx_bytes += int(n1.bytes_recv - n0.bytes_recv)
                    ps.network_tx_bytes += int(n1.bytes_sent - n0.bytes_sent)
                except Exception:
                    pass

        # Disk: always from system counters (optional: you might want to disable it for pure computation)
        try:
            d1 = psutil.disk_io_counters()
        except Exception:
            d1 = None
        if d0 is not None and d1 is not None:
            try:
                ps.disk_read_bytes += int(d1.read_bytes - d0.read_bytes)
                ps.disk_write_bytes += int(d1.write_bytes - d0.write_bytes)
            except Exception:
                pass

        try:
            nv0 = self._phase_nvml0.pop(name, None)
            nv1 = _nvml_query()
            kwh_phase = _nvml_delta_kwh(nv0, nv1) if (nv0 and nv1) else None
            if isinstance(kwh_phase, float) and kwh_phase > 0.0:
                ps.energy.setdefault("method", "nvml_delta")
                ps.energy["kwh"] = float(kwh_phase)
        except Exception:
            pass

        self.rep.phases[name] = ps

        # NEW: close the phase in the stack
        if self._phase_stack and self._phase_stack[-1] == name:
            self._phase_stack.pop()
        else:
            try:
                self._phase_stack.remove(name)
            except ValueError:
                pass

    # ----- FLOPs accounting ----------------------------------------------

    def register_model(
            self,
            model: torch.nn.Module,
            sample_batch: torch.Tensor,
            backward_factor: float = 2.0,
            loss_extra_fwd: float = 0.0,
    ):
        """
        Estimates the per-step FLOPs for a given model.

        Calculates forward FLOPs using `fvcore` on a sample batch and applies heuristic
        multipliers to account for the backward pass and loss computation.

        Args:
            model (torch.nn.Module): The model to analyze.
            sample_batch (torch.Tensor): A sample input batch for shape inference.
            backward_factor (float): Multiplier to estimate backward pass cost based
                on forward cost (default is 2.0).
            loss_extra_fwd (float): Additional multiplier to account for loss computation overhead.
        """
        try:
            fwd = _FLOPS_CACHE.get_or_compute(model, sample_batch)
        except Exception:
            fwd = None

        if fwd is None:
            # Couldn't get FLOPs (unsupported ops, etc.)
            self.rep.flops.method = "heuristic(unknown)"
            self._cur_per_step_total_flops = 0
            return

        per_step_forward = int((1.0 + float(loss_extra_fwd)) * fwd)
        per_step_backward = int(backward_factor * fwd)
        per_step_total = per_step_forward + per_step_backward

        # Report for latest model
        self.rep.flops.per_step_forward_flops = per_step_forward
        self.rep.flops.per_step_backward_flops = per_step_backward
        self.rep.flops.per_step_total_flops = per_step_total

        # Store for accumulation
        self._cur_per_step_total_flops = per_step_total

    def count_train_step(self, steps: int = 1):
        """
        Accumulates FLOP counts based on the registered model's per-step cost.

        Should be called once per optimizer step. Updates both global stats
        and the currently active phase.

        Args:
            steps (int): Number of steps to record (supports gradient accumulation).
        """
        s = int(steps)
        self.rep.flops.steps_counted += s

        per_step = int(self._cur_per_step_total_flops) if self._cur_per_step_total_flops else 0
        if per_step:
            self.rep.flops.total_flops += int(per_step * s)

            # NEW: if an active phase exists, account FLOPs for that phase
            if self._phase_stack:
                active = self._phase_stack[-1]
                ps = self.rep.phases.get(active, PhaseStats())
                ps.steps_counted += s
                ps.flops_total += int(per_step * s)
                self.rep.phases[active] = ps

    # ----- artifacts accounting ------------------------------------------

    def record_artifact(self, path: Path):
        """
        Registers a specific file as a notable artifact, tracking its size individually.

        Args:
            path (Path): Path to the artifact file.
        """
        try:
            p = Path(path)
            if p.exists() and p.is_file():
                self._by_file_sizes[str(p)] = p.stat().st_size
        except Exception:
            pass

    def note_network_bytes(self, tx: int = 0, rx: int = 0, phase: Optional[str] = None):
        """
        Explicitly annotates transmitted/received network bytes.

        Valid across all accounting modes:
          - 'manual': The primary tracking mechanism.
          - 'mixed': Adds to the system-wide psutil delta.
          - 'psutil': Adds to the system-wide delta (useful for rectification).

        Args:
            tx (int): Bytes transmitted.
            rx (int): Bytes received.
            phase (Optional[str]): The phase name to attribute these bytes to.
        """
        tx = int(tx or 0)
        rx = int(rx or 0)

        # global
        self.rep.bytes.network_tx_bytes += tx
        self.rep.bytes.network_rx_bytes += rx

        # per-phase
        if phase:
            ps = self.rep.phases.get(phase, PhaseStats())
            ps.network_tx_bytes += tx
            ps.network_rx_bytes += rx
            self.rep.phases[phase] = ps

    def note_network_file_transfer(self, path: Path, direction: str, phase: Optional[str] = None):
        """
        Helper to count a file's size as network traffic (TX or RX).

        Args:
            path (Path): The file being transferred.
            direction (str): 'tx' for transmission, 'rx' for reception.
            phase (Optional[str]): The phase name to attribute traffic to.
        """
        try:
            sz = int(Path(path).stat().st_size)
        except Exception:
            sz = 0
        if direction.lower() == "tx":
            self.note_network_bytes(tx=sz, rx=0, phase=phase)
        else:
            self.note_network_bytes(tx=0, rx=sz, phase=phase)

    # ----- finalize & persist --------------------------------------------

    def finish(self):
        """
        Finalizes the experiment tracking, aggregates all metrics, and writes the report.

        Prioritizes energy data sources:
        1. CodeCarbon (if successful)
        2. NVML Energy Counters (Delta)
        3. NVML Power Integration (Average Power * Time)

        Generates:
            - `costs.json`: Full detailed report.
            - `costs_summary.csv`: Flattened summary of key metrics.

        This method is idempotent.
        """
        if self._t0 is None:
            # already closed / finalized
            return

        total_time = time.perf_counter() - self._t0
        self._t0 = None  # mark finalized so we won't run twice

        # ---- ENERGY / EMISSIONS -----------------------------------------
        nvml_end = _nvml_query()
        nvml_delta = _nvml_delta_kwh(self._nvml0, nvml_end)

        pwr1_W = _nvml_power_sum_w()

        energy = EnergyStats()
        energy.extra["nvml_start"] = self._nvml0
        energy.extra["nvml_end"] = nvml_end
        energy.extra["nvml_delta_kwh_raw"] = nvml_delta
        energy.extra["nvml_power_start_W"] = self._pwr0_W
        energy.extra["nvml_power_end_W"] = pwr1_W
        energy.extra["runtime_sec"] = total_time

        # attempt CodeCarbon first
        used_codecarbon = False
        if self._cc_tracker is not None:
            try:
                emissions = self._cc_tracker.stop()  # kg CO₂e
                kwh_cc = float("nan")
                try:
                    # CodeCarbon 2.x exposes final_emissions_data dict
                    cc_energy_kwh = getattr(
                        self._cc_tracker, "final_emissions_data", None
                    )
                    if cc_energy_kwh and "energy_consumed" in cc_energy_kwh:
                        # "energy_consumed" is kWh in recent CodeCarbon
                        kwh_cc = float(cc_energy_kwh["energy_consumed"])
                except Exception:
                    pass

                if math.isfinite(kwh_cc):
                    # happy path -> trust CodeCarbon
                    energy.kwh = kwh_cc
                    energy.method = "codecarbon"
                    energy.emissions_kg_co2e = (
                        float(emissions) if emissions is not None else None
                    )
                    used_codecarbon = True
                else:
                    # CodeCarbon gave unusable energy -> fallback
                    self._cc_tracker = None
            except Exception:
                # CodeCarbon choked during .stop()
                self._cc_tracker = None

        if not used_codecarbon:
            # 1. try NVML cumulative energy counter delta, but only if it's >0
            if nvml_delta is not None and nvml_delta > 0.0:
                energy.kwh = float(nvml_delta)
                energy.method = "nvml_delta"
                energy.emissions_kg_co2e = None
            else:
                # 2. fallback: approximate via average GPU power draw
                #    over the run duration: kWh ~= P_avg[W] * time[h] / 1000
                est_kwh = None
                if (
                        self._pwr0_W is not None
                        and pwr1_W is not None
                        and total_time > 0
                ):
                    p_avg_W = 0.5 * (self._pwr0_W + pwr1_W)
                    # hours = total_time / 3600
                    # kWh = (p_avg_W * hours) / 1000
                    est_kwh = p_avg_W * (total_time / 3600.0) / 1000.0
                    energy.extra["nvml_power_est_kwh"] = est_kwh

                if est_kwh is not None and est_kwh > 0.0:
                    energy.kwh = float(est_kwh)
                    energy.method = "nvml_power_avg"
                    energy.emissions_kg_co2e = None
                else:
                    # 3. totally blind fallback
                    energy.kwh = 0.0
                    energy.method = "unknown"
                    energy.emissions_kg_co2e = None

        self.rep.energy = energy

        # ---- network + disk bytes ---------------------------------------
        # Disk: always from psutil (delta run)
        disk1 = psutil.disk_io_counters()
        disk_read_delta = int(disk1.read_bytes - self._disk0.read_bytes)
        disk_write_delta = int(disk1.write_bytes - self._disk0.write_bytes)

        if self._network_accounting in ("psutil", "mixed"):
            # Base: global psutil delta (if we have _net0)
            try:
                net1 = psutil.net_io_counters()
            except Exception:
                net1 = None

            base_rx = int(net1.bytes_recv - self._net0.bytes_recv) if (net1 and self._net0) else 0
            base_tx = int(net1.bytes_sent - self._net0.bytes_sent) if (net1 and self._net0) else 0

            # Add the manually annotated part if 'mixed'
            add_rx = self.rep.bytes.network_rx_bytes if self._network_accounting == "mixed" else 0
            add_tx = self.rep.bytes.network_tx_bytes if self._network_accounting == "mixed" else 0

            by = BytesStats(
                network_rx_bytes=base_rx + add_rx,
                network_tx_bytes=base_tx + add_tx,
                disk_read_bytes=disk_read_delta,
                disk_write_bytes=disk_write_delta,
            )
        else:
            # 'manual': use ONLY what was annotated via note_network_bytes()
            by = BytesStats(
                network_rx_bytes=self.rep.bytes.network_rx_bytes,
                network_tx_bytes=self.rep.bytes.network_tx_bytes,
                disk_read_bytes=disk_read_delta,
                disk_write_bytes=disk_write_delta,
            )

        artifacts_after = _sum_dir_bytes(self.out_dir)
        by.artifacts_written_bytes = max(0, artifacts_after - self._artifacts_before)
        by.by_file = self._by_file_sizes

        self.rep.bytes = by

        # ---- wall clock --------------------------------------------------
        self.rep.wall_clock = _pretty_seconds(total_time)

        # ---- assumptions / docstrings ------------------------------------
        avg_flops_per_step = 0
        if self.rep.flops.steps_counted > 0:
            avg_flops_per_step = int(
                self.rep.flops.total_flops / self.rep.flops.steps_counted
            )

        self.rep.assumptions.update(
            {
                "backward_flops_multiplier": 2.0,
                "flops_tool": self.rep.flops.method,
                "bytes_on_wire_definition": (
                    "network_accounting='psutil' => delta psutil.net_io_counters; "
                    "'manual' => sum of calls tracker.note_network_bytes(); "
                    "'mixed' => sum of both."
                ),
                "network_accounting": self._network_accounting,
                "disk_io_definition": (
                    "psutil.disk_io_counters deltas; artifacts_* = size of files created in output dir"
                ),
                "energy_tool": self.rep.energy.method,
                "avg_flops_per_step": avg_flops_per_step,
            }
        )

        # ---- persist on disk --------------------------------------------
        self.out_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.out_dir / "costs.json"
        try:
            with json_path.open("w") as f:
                json.dump(_to_dict(self.rep), f, indent=2)
        except Exception:
            pass

        csv_path = self.out_dir / "costs_summary.csv"
        try:
            with csv_path.open("w", newline="") as cf:
                w = csv.writer(cf)
                w.writerow(["metric", "value", "unit"])

                w.writerow(
                    [
                        "wall_clock_total",
                        "{:.6f}".format(
                            self.rep.wall_clock.get("total_sec", float("nan"))
                        ),
                        "sec",
                    ]
                )
                w.writerow(["network_rx", by.network_rx_bytes, "bytes"])
                w.writerow(["network_tx", by.network_tx_bytes, "bytes"])
                w.writerow(["disk_read", by.disk_read_bytes, "bytes"])
                w.writerow(["disk_write", by.disk_write_bytes, "bytes"])
                w.writerow(
                    ["artifacts_written", by.artifacts_written_bytes, "bytes"]
                )

                w.writerow(
                    ["flops_total", self.rep.flops.total_flops, "FLOPs"]
                )

                # Report average FLOPs/step over the WHOLE run (all models)
                w.writerow(
                    [
                        "flops_per_step",
                        avg_flops_per_step,
                        "FLOPs/step",
                    ]
                )

                # Energy / Emissions
                # We ALWAYS write a numeric value, even if it's 0.0, so downstream
                # code won't think it's "missing" and print N/A.
                w.writerow(
                    [
                        "energy",
                        "{:.9f}".format(self.rep.energy.kwh),
                        "kWh",
                    ]
                )

                w.writerow(
                    [
                        "emissions",
                        (
                            "{:.9f}".format(self.rep.energy.emissions_kg_co2e)
                            if self.rep.energy.emissions_kg_co2e is not None
                            else ""
                        ),
                        "kgCO2e",
                    ]
                )
        except Exception:
            pass