"""
C:\\Users\\ali_z\\ANU AI\\Akkurat\\cognitive_model_3\\cfc.py
===================================================================

Project Chimera / Akkurat - Production Tensorized CfC Substrate v3.0
--------------------------------------------------------------------

ROLE
----
Continuous-time neural dynamics substrate for cognitive lobes.

This module provides:
- A production TCfC_Policy made from multiple bounded TCfC_Cell units.
- TensorTrain/MPO-backed cell backbones when tn.py is available.
- Graceful dense fallback when configured.
- Deterministic seeded initialization.
- Bounded state updates with dt=0 preserving previous state.
- Health inspection, snapshots, runtime state serialization, and full checkpoints.
- Backward-compatible legacy QE_CfC_Cell and QECfCPolicy classes.

DESIGN BOUNDARIES
-----------------
This file is a neural substrate only. It does not orchestrate tools, plans, the
five-domain cognitive runtime, or digital-twin governance. Those responsibilities
belong to cognitive_lobe_runtime.py and digital_twin_kernel.py.

COMPATIBILITY
-------------
The primary class keeps the original constructor style:

    TCfC_Policy(policy_id, input_size, hidden_size, num_cells, bond_dim=8, ...)

and the original step behavior:

    state = policy.step(x, time_delta=1.0)

New production callers may request diagnostics:

    state, aux = policy.step(x, time_delta=1.0, return_state=True)

Author: Akkurat / Project Chimera
License: permissive
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# TensorTrain import
# =============================================================================

TENSORTRAIN_AVAILABLE = False
TT_IMPORT_ERROR: Optional[Exception] = None

try:
    try:
        from .tn import TensorTrain, TTConfig, get_factors, tensor_train_svd
        TENSORTRAIN_AVAILABLE = True
    except ImportError:
        from tn import TensorTrain, TTConfig, get_factors, tensor_train_svd
        TENSORTRAIN_AVAILABLE = True
except Exception as e:  # pragma: no cover - fallback path
    TT_IMPORT_ERROR = e

    def get_factors(n: int) -> Tuple[int, int]:
        n = int(n)
        if n <= 0:
            return (1, 1)
        a = int(math.isqrt(n))
        while a > 0:
            if n % a == 0:
                return (a, n // a)
            a -= 1
        return (1, n)

    class TensorTrain:  # type: ignore[no-redef]
        pass

    @dataclass(frozen=True)
    class TTConfig:  # type: ignore[no-redef]
        dtype: np.dtype = np.float32
        device: str = "cpu"

    def tensor_train_svd(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        return None


# =============================================================================
# Utilities
# =============================================================================

_EPS = 1e-9
_EMPTY_1D_CACHE: Dict[str, np.ndarray] = {}


def _dtype_key(dtype: np.dtype) -> str:
    return str(np.dtype(dtype))


def _empty_1d(dtype: np.dtype) -> np.ndarray:
    key = _dtype_key(dtype)
    arr = _EMPTY_1D_CACHE.get(key)
    if arr is None:
        arr = np.zeros(0, dtype=np.dtype(dtype))
        _EMPTY_1D_CACHE[key] = arr
    return arr


def _as_dtype(x: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(x, dtype=np.dtype(dtype))


def _coerce_vector(
    x: Any,
    *,
    expected_dim: Optional[int] = None,
    dtype: np.dtype = np.float32,
    name: str = "vector",
    clip: Optional[float] = None,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.dtype(dtype)).reshape(-1).copy()
    if expected_dim is not None and int(arr.size) != int(expected_dim):
        raise ValueError(f"{name} size mismatch: expected {expected_dim}, got {arr.size}")
    if arr.size == 0 and expected_dim not in (None, 0):
        raise ValueError(f"{name} cannot be empty")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.dtype(dtype), copy=False)
    if clip is not None and float(clip) > 0:
        np.clip(arr, -float(clip), float(clip), out=arr)
    return arr


def _sanitize(arr: Any, dtype: np.dtype = np.float32, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.dtype(dtype)).copy()
    mask = ~np.isfinite(out)
    if np.any(mask):
        out[mask] = np.dtype(dtype).type(fill)
    return out.astype(np.dtype(dtype), copy=False)


def _safe_norm(x: Any) -> float:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0
        return float(np.linalg.norm(arr))
    except Exception:
        return 0.0


def _count_nonfinite(x: Any) -> int:
    arr = np.asarray(x)
    return int(arr.size - np.count_nonzero(np.isfinite(arr)))


def _saturation_fraction(x: Any, threshold: float = 0.98) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr) >= float(threshold)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))).astype(np.float32, copy=False)


def _tanh_if_requested(x: np.ndarray, activation: str) -> np.ndarray:
    activation = str(activation).lower().strip()
    if activation == "tanh":
        return np.tanh(x).astype(x.dtype, copy=False)
    if activation == "clip":
        return np.clip(x, -1.0, 1.0).astype(x.dtype, copy=False)
    return x


def _clamp_dt(dt: float, *, lo: float = 0.0, hi: float = 1e3) -> float:
    try:
        v = float(dt)
    except Exception:
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return float(np.clip(v, float(lo), float(hi)))


def _check_factors(size: int, f1: int, f2: int, name: str) -> None:
    if int(f1) * int(f2) != int(size):
        raise ValueError(
            f"[{name}] factorization mismatch: ({f1} x {f2}) != {size}. "
            "Update factorization logic."
        )


def two_factors(n: int) -> Tuple[int, int]:
    """Return a near-square positive factorization of n."""
    n = int(n)
    if n <= 0:
        return (1, 1)
    try:
        a, b = get_factors(n)
        a, b = int(a), int(b)
        if a > 0 and b > 0 and a * b == n:
            return a, b
    except Exception:
        pass
    r = int(math.isqrt(n))
    for f in range(r, 0, -1):
        if n % f == 0:
            return (f, n // f)
    return (1, n)


def _bond_dims_from_spec(spec: Union[int, Sequence[int]], num_cores: int) -> List[int]:
    if num_cores <= 1:
        return []
    needed = num_cores - 1
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        vals = [int(max(1, x)) for x in spec]
        if len(vals) == needed:
            return vals
        if len(vals) == 0:
            return [4] * needed
        return [vals[0]] * needed
    return [int(max(1, int(spec)))] * needed


def _estimate_dense_spectral_norm(mat: np.ndarray, *, iters: int = 16) -> Optional[float]:
    try:
        W = np.asarray(mat, dtype=np.float64)
        if W.ndim != 2 or W.size == 0:
            return 0.0
        rng = np.random.default_rng(12345)
        v = rng.normal(size=(W.shape[1],))
        v /= np.linalg.norm(v) + _EPS
        for _ in range(max(1, int(iters))):
            u = W @ v
            un = np.linalg.norm(u)
            if not np.isfinite(un) or un < _EPS:
                return 0.0
            u /= un
            v = W.T @ u
            vn = np.linalg.norm(v)
            if not np.isfinite(vn) or vn < _EPS:
                return 0.0
            v /= vn
        return float(np.linalg.norm(W @ v))
    except Exception:
        return None


def _stable_dense_init_rect(
    out_dim: int,
    in_dim: int,
    rng: np.random.Generator,
    dtype: np.dtype,
    *,
    base_scale: float = 0.9,
    max_sv: float = 0.95,
) -> np.ndarray:
    """
    Stable rectangular dense initialization with approximate spectral control.
    Shape: (out_dim, in_dim), for y = W @ x.
    """
    out_dim = int(out_dim)
    in_dim = int(in_dim)
    dt = np.dtype(dtype)
    if out_dim <= 0 or in_dim <= 0:
        return np.zeros((max(0, out_dim), max(0, in_dim)), dtype=dt)

    scale = float(base_scale) / math.sqrt(max(1, in_dim))
    W = rng.normal(0.0, scale, size=(out_dim, in_dim)).astype(np.float64)

    # If square and modest, use QR to get a better recurrent/control operator.
    if out_dim == in_dim:
        try:
            A = rng.normal(0.0, 1.0, size=(out_dim, out_dim))
            q, r = np.linalg.qr(A)
            d = np.sign(np.diag(r))
            d[d == 0] = 1.0
            W = q * d * float(base_scale)
        except Exception:
            pass

    smax = _estimate_dense_spectral_norm(W, iters=12)
    if smax is not None and smax > float(max_sv) and smax > 0.0:
        W *= float(max_sv) / float(smax)

    return W.astype(dt, copy=False)


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion for NumPy/CuPy-like arrays."""
    if hasattr(x, "__cuda_array_interface__"):
        try:
            import cupy as cp  # type: ignore
            return cp.asnumpy(x)
        except Exception:
            pass
    return np.asarray(x)


def _tt_param_count(tt_obj: Any) -> Optional[int]:
    for name in ("parameter_count", "num_parameters", "parameters"):
        fn = getattr(tt_obj, name, None)
        try:
            if callable(fn):
                val = int(fn())
                if val >= 0:
                    return val
        except Exception:
            pass
    for attr in ("total_parameters", "n_parameters", "num_params"):
        try:
            val = int(getattr(tt_obj, attr))
            if val >= 0:
                return val
        except Exception:
            pass
    cores = getattr(tt_obj, "cores_data", None)
    if isinstance(cores, (list, tuple)):
        try:
            return int(sum(int(np.asarray(_to_numpy(c)).size) for c in cores))
        except Exception:
            return None
    return None


def _tt_apply(
    tt_obj: Any,
    vec: np.ndarray,
    *,
    out_dim: int,
    dtype: np.dtype,
    strict: bool = True,
) -> np.ndarray:
    """Apply TensorTrain-like object to a flat vector."""
    dt = np.dtype(dtype)
    x = np.asarray(vec, dtype=dt).reshape(-1)
    last_err: Optional[BaseException] = None

    for method in ("apply", "contract_with_vector", "matvec", "__call__"):
        fn = getattr(tt_obj, method, None)
        if fn is None:
            continue
        try:
            y = fn(x)
            y = np.asarray(_to_numpy(y), dtype=dt).reshape(-1)
            if y.size == int(out_dim):
                return _sanitize(y, dt)
            if strict:
                raise ValueError(f"TT output size {y.size} != expected {out_dim}")
            if y.size > int(out_dim):
                return _sanitize(y[: int(out_dim)], dt)
            out = np.zeros((int(out_dim),), dtype=dt)
            out[: y.size] = y
            return _sanitize(out, dt)
        except Exception as e:
            last_err = e

    tm = getattr(tt_obj, "to_matrix", None)
    if callable(tm):
        try:
            try:
                W = np.asarray(tm(force_cpu=True), dtype=dt)
            except TypeError:
                W = np.asarray(tm(), dtype=dt)
            if W.ndim == 2 and W.shape[1] == x.size:
                y = W @ x
                y = np.asarray(y, dtype=dt).reshape(-1)
                if y.size == int(out_dim):
                    return _sanitize(y, dt)
                if strict:
                    raise ValueError(f"TT dense output size {y.size} != expected {out_dim}")
            else:
                raise ValueError(f"to_matrix shape {W.shape} incompatible with input size {x.size}")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"TensorTrain contraction failed; last error: {last_err}")


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class TCfCCellConfig:
    input_size: int
    hidden_size: int
    bond_dim: Union[int, Sequence[int]] = 8
    dtype: str = "float32"
    tt_strict: bool = True
    force_dense: bool = False
    energy_tol: float = 0.999
    input_clip: float = 5.0
    state_clip: float = 1.0
    preactivation_clip: float = 10.0
    candidate_activation: str = "tanh"  # "tanh" | "clip" | "linear"
    backbone_activation: str = "tanh"   # "tanh" | "clip" | "linear"
    time_scale: float = 1.0
    leaky_lambda: float = 1.0
    strict_runtime_checks: bool = True


@dataclass
class TCfCPolicyConfig:
    policy_id: str
    input_size: int
    hidden_size: int
    num_cells: int
    bond_dim: Union[int, Sequence[int]] = 8
    dtype: str = "float32"
    tt_strict: bool = True
    force_dense: bool = False
    energy_tol: float = 0.999
    input_clip: float = 5.0
    state_clip: float = 1.0
    preactivation_clip: float = 10.0
    candidate_activation: str = "tanh"
    backbone_activation: str = "tanh"
    time_scale: float = 1.0
    leaky_lambda: float = 1.0
    strict_runtime_checks: bool = True
    verbose: bool = True
    rng_seed: Optional[int] = None


# =============================================================================
# Legacy classes retained for compatibility
# =============================================================================

class QE_CfC_Cell:
    """
    Legacy Quantum-Enabled CfC cell.

    Preserved for backward compatibility only. New code should use TCfC_Cell.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bond_dim: int = 8,
        dtype: np.dtype = np.float32,
        rng: Optional[np.random.Generator] = None,
        tt_strict: bool = False,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dtype = np.dtype(dtype)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self.tt_strict = bool(tt_strict)

        in_factor1, in_factor2 = two_factors(self.input_size)
        hid_factor1, hid_factor2 = two_factors(self.hidden_size)
        _check_factors(self.input_size, in_factor1, in_factor2, "QE_CfC_Cell.input")
        _check_factors(self.hidden_size, hid_factor1, hid_factor2, "QE_CfC_Cell.hidden")

        self._tt_in_shape = (in_factor1, in_factor2)
        self._tt_out_shape = (hid_factor1, hid_factor2)
        self._tt_out_len = self.hidden_size
        self._use_tt = bool(TENSORTRAIN_AVAILABLE)
        self._tt_init_error: Optional[str] = None

        self.TT_f: Optional[Any] = None
        self.TT_g: Optional[Any] = None
        self.TT_h: Optional[Any] = None
        self.W_f: Optional[np.ndarray] = None
        self.W_g: Optional[np.ndarray] = None
        self.W_h: Optional[np.ndarray] = None

        self._init_legacy_operator("f", bond_dim)
        self._init_legacy_operator("g", bond_dim)
        self._init_legacy_operator("h", bond_dim)

        self.f_bias = np.zeros(self.hidden_size, dtype=self.dtype)
        self.g_bias = np.zeros(self.hidden_size, dtype=self.dtype)
        self.h_bias = np.zeros(self.hidden_size, dtype=self.dtype)

    def _init_legacy_operator(self, which: str, bond_dim: int) -> None:
        dense = _stable_dense_init_rect(self.hidden_size, self.input_size, self.rng, self.dtype)
        if self._use_tt:
            try:
                tt = tensor_train_svd(
                    dense,
                    output_dims=list(self._tt_out_shape),
                    input_dims=list(self._tt_in_shape),
                    max_bond_dim=int(max(1, bond_dim)),
                    dtype=self.dtype,
                    energy_tol=0.999,
                    device="cpu",
                    check_finite=True,
                )
                if tt is None:
                    raise RuntimeError("tensor_train_svd returned None")
                setattr(self, f"TT_{which}", tt)
                setattr(self, f"W_{which}", None)
                return
            except Exception as e:
                self._tt_init_error = str(e)
                if self.tt_strict:
                    raise RuntimeError(f"Legacy TT init failed for {which}: {e}") from e
        setattr(self, f"TT_{which}", None)
        setattr(self, f"W_{which}", dense)
        self._use_tt = False

    def _apply_op(self, which: str, x: np.ndarray) -> np.ndarray:
        tt = getattr(self, f"TT_{which}", None)
        W = getattr(self, f"W_{which}", None)
        if tt is not None:
            return _tt_apply(tt, x, out_dim=self.hidden_size, dtype=self.dtype, strict=True)
        if W is None:
            raise RuntimeError(f"Legacy operator {which} is not initialized")
        return _sanitize(W @ x, self.dtype)

    def forward(self, x_prev: np.ndarray, input_vec: np.ndarray, time_delta: float) -> np.ndarray:
        del x_prev  # kept for legacy signature compatibility
        full_input = _coerce_vector(input_vec, expected_dim=self.input_size, dtype=self.dtype, name="input_vec", clip=5.0)
        dt = _clamp_dt(time_delta)
        if dt <= 0.0:
            return np.zeros((self.hidden_size,), dtype=self.dtype)

        f_out = self._apply_op("f", full_input) + self.f_bias
        g_out = np.tanh(self._apply_op("g", full_input) + self.g_bias).astype(self.dtype, copy=False)
        h_out = np.tanh(self._apply_op("h", full_input) + self.h_bias).astype(self.dtype, copy=False)

        pre = np.clip(-f_out * self.dtype.type(dt), -20.0, 20.0)
        gate = sigmoid(pre)
        x_next = gate * g_out + (self.dtype.type(1.0) - gate) * h_out
        return np.clip(_sanitize(x_next, self.dtype), -1.0, 1.0).astype(self.dtype, copy=False)


class QECfCPolicy:
    """Legacy policy composed of QE_CfC_Cell units."""

    def __init__(
        self,
        policy_id: str,
        input_size: int,
        hidden_size: int,
        num_cells: int,
        bond_dim: int = 8,
        dtype: np.dtype = np.float32,
        verbose: bool = True,
        rng: Optional[np.random.Generator] = None,
        tt_strict: bool = False,
    ):
        self.policy_id = str(policy_id)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_cells = int(num_cells)
        self.dtype = np.dtype(dtype)
        self.verbose = bool(verbose)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

        if self.input_size <= 0 or self.hidden_size <= 0 or self.num_cells <= 0:
            raise ValueError("input_size, hidden_size, and num_cells must be positive")

        cell_input_dim = self.input_size + self.hidden_size * self.num_cells
        self.cells = [
            QE_CfC_Cell(cell_input_dim, self.hidden_size, bond_dim=bond_dim, dtype=self.dtype, rng=self.rng, tt_strict=tt_strict)
            for _ in range(self.num_cells)
        ]
        self.state = np.zeros(self.hidden_size * self.num_cells, dtype=self.dtype)
        if self.verbose:
            self._report_parameters()

    def reset_state(self, value: float = 0.0) -> None:
        self.state.fill(self.dtype.type(value))

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_state(self, new_state: Any) -> None:
        self.state = _coerce_vector(new_state, expected_dim=self.state.size, dtype=self.dtype, name="new_state", clip=1.0)

    def step(self, external_input: Any, time_delta: float = 1.0) -> np.ndarray:
        external_input = _coerce_vector(external_input, expected_dim=self.input_size, dtype=self.dtype, name="external_input", clip=5.0)
        full_input = np.concatenate([external_input, self.state]).astype(self.dtype, copy=False)
        new_states = np.empty((self.num_cells, self.hidden_size), dtype=self.dtype)
        for i, cell in enumerate(self.cells):
            new_states[i] = cell.forward(_empty_1d(self.dtype), full_input, time_delta)
        self.state = new_states.reshape(-1).astype(self.dtype, copy=False)
        return self.state.copy()

    def parameter_count(self) -> int:
        total = 0
        for cell in self.cells:
            for which in ("f", "g", "h"):
                tt = getattr(cell, f"TT_{which}", None)
                W = getattr(cell, f"W_{which}", None)
                if tt is not None:
                    total += int(_tt_param_count(tt) or 0)
                elif W is not None:
                    total += int(W.size)
            total += int(cell.f_bias.size + cell.g_bias.size + cell.h_bias.size)
        return int(total)

    def _report_parameters(self) -> None:
        total = self.parameter_count()
        cell_input_dim = self.input_size + self.hidden_size * self.num_cells
        dense = self.num_cells * 3 * (cell_input_dim * self.hidden_size + self.hidden_size)
        print(f"[Legacy QECfCPolicy:{self.policy_id}] params={total:,} dense_equiv={dense:,}")

    def __repr__(self) -> str:
        return (
            f"QECfCPolicy(id={self.policy_id!r}, input={self.input_size}, "
            f"hidden={self.hidden_size}, cells={self.num_cells}, dtype={self.dtype})"
        )


# =============================================================================
# Production TCfC cell
# =============================================================================

class TCfC_Cell:
    """
    Tensorized Closed-form Continuous-time Cell.

    Runtime input:
        [external_input + this_cell_prev_state]

    Backbone:
        TensorTrain/MPO mapping combined input -> hidden intermediate when available.
        Dense rectangular fallback otherwise.

    Bounded state update:
        dt <= 0 preserves previous state.
        candidate states are bounded by default through tanh.
        update_mix = leaky_lambda * (1 - exp(-dt / time_scale)).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bond_dim: Union[int, Sequence[int]] = 8,
        rng: Optional[np.random.Generator] = None,
        dtype: np.dtype = np.float32,
        *,
        tt_strict: bool = True,
        force_dense: Optional[bool] = None,
        energy_tol: float = 0.999,
        input_clip: float = 5.0,
        state_clip: float = 1.0,
        preactivation_clip: float = 10.0,
        candidate_activation: str = "tanh",
        backbone_activation: str = "tanh",
        time_scale: float = 1.0,
        leaky_lambda: float = 1.0,
        strict_runtime_checks: bool = True,
    ):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bond_dim = bond_dim
        self.dtype = np.dtype(dtype)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self.tt_strict = bool(tt_strict)
        self.energy_tol = float(energy_tol)
        self.input_clip = float(input_clip)
        self.state_clip = float(state_clip)
        self.preactivation_clip = float(preactivation_clip)
        self.candidate_activation = str(candidate_activation).lower().strip()
        self.backbone_activation = str(backbone_activation).lower().strip()
        self.time_scale = float(max(_EPS, time_scale))
        self.leaky_lambda = float(np.clip(leaky_lambda, 0.0, 1.0))
        self.strict_runtime_checks = bool(strict_runtime_checks)

        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")

        env_force_dense = str(os.getenv("CFC_FORCE_DENSE", "")).strip().lower() in ("1", "true", "yes")
        self.force_dense = bool(force_dense) if force_dense is not None else False
        self._force_dense_effective = bool(self.force_dense or env_force_dense)

        self.combined_size = self.input_size + self.hidden_size
        in_f1, in_f2 = two_factors(self.combined_size)
        out_f1, out_f2 = two_factors(self.hidden_size)
        _check_factors(self.combined_size, in_f1, in_f2, "TCfC_Cell.backbone_input")
        _check_factors(self.hidden_size, out_f1, out_f2, "TCfC_Cell.backbone_output")

        self._tt_in_shape = (int(in_f1), int(in_f2))
        self._tt_out_shape = (int(out_f1), int(out_f2))
        self._tt_bond_dims = _bond_dims_from_spec(self.bond_dim, len(self._tt_in_shape))

        self.backbone: Optional[Any] = None
        self.dense_backbone: Optional[np.ndarray] = None
        self._use_tt = bool(TENSORTRAIN_AVAILABLE) and not self._force_dense_effective
        self._tt_init_error: Optional[str] = None
        self._backbone_param_count: int = 0

        self._init_backbone()
        self._init_heads()

        # Runtime caches
        self._last_backbone_out = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_f = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_g = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_h = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_gate = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_candidate = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_update_mix = 0.0
        self._last_dt = 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_backbone(self) -> None:
        dense_init = _stable_dense_init_rect(
            self.hidden_size,
            self.combined_size,
            self.rng,
            self.dtype,
            base_scale=0.9,
            max_sv=0.95,
        )

        if self._force_dense_effective:
            self._use_tt = False
            self.dense_backbone = dense_init
            self.backbone = None
            self._backbone_param_count = int(self.dense_backbone.size)
            return

        if not TENSORTRAIN_AVAILABLE:
            msg = "TensorTrain unavailable"
            if TT_IMPORT_ERROR is not None:
                msg += f": {TT_IMPORT_ERROR}"
            self._tt_init_error = msg
            if self.tt_strict:
                raise RuntimeError(msg)
            self._use_tt = False
            self.dense_backbone = dense_init
            self.backbone = None
            self._backbone_param_count = int(self.dense_backbone.size)
            return

        try:
            max_rank = max(self._tt_bond_dims) if self._tt_bond_dims else int(max(1, self.bond_dim if isinstance(self.bond_dim, int) else 4))
            tt = tensor_train_svd(
                dense_init,
                output_dims=list(self._tt_out_shape),
                input_dims=list(self._tt_in_shape),
                max_bond_dim=int(max_rank),
                dtype=self.dtype,
                energy_tol=self.energy_tol,
                device="cpu",
                check_finite=True,
            )
            if tt is None:
                raise RuntimeError("tensor_train_svd returned None")
            self.backbone = tt
            self.dense_backbone = None
            self._use_tt = True
            self._backbone_param_count = int(_tt_param_count(tt) or 0)
        except Exception as e:
            msg = f"TensorTrain backbone init failed: {e}"
            self._tt_init_error = msg
            if self.tt_strict:
                raise RuntimeError(msg) from e
            self._use_tt = False
            self.backbone = None
            self.dense_backbone = dense_init
            self._backbone_param_count = int(self.dense_backbone.size)

    def _init_heads(self) -> None:
        self.f_head = _stable_dense_init_rect(self.hidden_size, self.hidden_size, self.rng, self.dtype, base_scale=0.7, max_sv=0.9).T
        self.g_head = _stable_dense_init_rect(self.hidden_size, self.hidden_size, self.rng, self.dtype, base_scale=0.8, max_sv=0.9).T
        self.h_head = _stable_dense_init_rect(self.hidden_size, self.hidden_size, self.rng, self.dtype, base_scale=0.8, max_sv=0.9).T
        self.f_bias = np.zeros(self.hidden_size, dtype=self.dtype)
        self.g_bias = np.zeros(self.hidden_size, dtype=self.dtype)
        self.h_bias = np.zeros(self.hidden_size, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def _backbone_apply(self, full_input: np.ndarray) -> np.ndarray:
        x = _coerce_vector(full_input, expected_dim=self.combined_size, dtype=self.dtype, name="full_input", clip=self.input_clip)
        if self._use_tt:
            if self.backbone is None:
                raise RuntimeError("TT backbone is not initialized")
            y = _tt_apply(self.backbone, x, out_dim=self.hidden_size, dtype=self.dtype, strict=True)
        else:
            if self.dense_backbone is None:
                raise RuntimeError("Dense backbone is not initialized")
            y = self.dense_backbone @ x
            y = _sanitize(y, self.dtype)
        return _tanh_if_requested(y.astype(self.dtype, copy=False), self.backbone_activation)

    def forward(
        self,
        input_vec: Any,
        prev_state: Any,
        time_delta: float,
        *,
        return_aux: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        x = _coerce_vector(input_vec, expected_dim=self.input_size, dtype=self.dtype, name="input_vec", clip=self.input_clip)
        prev = _coerce_vector(prev_state, expected_dim=self.hidden_size, dtype=self.dtype, name="prev_state", clip=self.state_clip)
        dt = _clamp_dt(time_delta)
        self._last_dt = float(dt)

        if dt <= 0.0:
            next_state = prev.copy()
            aux = self._aux(next_state=next_state, dt=dt, skipped=True)
            return (next_state, aux) if return_aux else next_state

        full_input = np.concatenate([x, prev]).astype(self.dtype, copy=False)
        z = self._backbone_apply(full_input)

        f_out = z @ self.f_head + self.f_bias
        g_out = z @ self.g_head + self.g_bias
        h_out = z @ self.h_head + self.h_bias

        if self.preactivation_clip > 0:
            f_out = np.clip(f_out, -self.preactivation_clip, self.preactivation_clip).astype(self.dtype, copy=False)
            g_out = np.clip(g_out, -self.preactivation_clip, self.preactivation_clip).astype(self.dtype, copy=False)
            h_out = np.clip(h_out, -self.preactivation_clip, self.preactivation_clip).astype(self.dtype, copy=False)

        g_act = _tanh_if_requested(g_out, self.candidate_activation)
        h_act = _tanh_if_requested(h_out, self.candidate_activation)

        gate = sigmoid(np.clip(-f_out * self.dtype.type(dt), -20.0, 20.0))
        candidate = gate * g_act + (self.dtype.type(1.0) - gate) * h_act
        candidate = _sanitize(candidate, self.dtype)

        temporal_mix = 1.0 - math.exp(-float(dt) / self.time_scale)
        update_mix = float(np.clip(self.leaky_lambda * temporal_mix, 0.0, 1.0))
        next_state = ((1.0 - update_mix) * prev + update_mix * candidate).astype(self.dtype, copy=False)

        if self.state_clip > 0:
            next_state = np.clip(next_state, -self.state_clip, self.state_clip).astype(self.dtype, copy=False)
        next_state = _sanitize(next_state, self.dtype)

        self._last_backbone_out = z.copy()
        self._last_f = _sanitize(f_out, self.dtype)
        self._last_g = _sanitize(g_act, self.dtype)
        self._last_h = _sanitize(h_act, self.dtype)
        self._last_gate = _sanitize(gate, self.dtype)
        self._last_candidate = _sanitize(candidate, self.dtype)
        self._last_update_mix = float(update_mix)

        if self.strict_runtime_checks and _count_nonfinite(next_state) > 0:
            raise FloatingPointError("Non-finite TCfC cell state encountered")

        aux = self._aux(next_state=next_state, dt=dt, skipped=False)
        return (next_state, aux) if return_aux else next_state

    def _aux(self, *, next_state: np.ndarray, dt: float, skipped: bool) -> Dict[str, Any]:
        return {
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "tt_init_error": self._tt_init_error,
            "dt": float(dt),
            "skipped": bool(skipped),
            "update_mix": float(self._last_update_mix),
            "backbone_out": self._last_backbone_out.copy(),
            "f": self._last_f.copy(),
            "g": self._last_g.copy(),
            "h": self._last_h.copy(),
            "gate": self._last_gate.copy(),
            "candidate": self._last_candidate.copy(),
            "next_state_norm": float(np.linalg.norm(next_state)),
        }

    # ------------------------------------------------------------------
    # Diagnostics / maintenance
    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        total = int(self._backbone_param_count)
        total += int(self.f_head.size + self.g_head.size + self.h_head.size)
        total += int(self.f_bias.size + self.g_bias.size + self.h_bias.size)
        return int(total)

    def estimate_backbone_norm(self, *, iters: int = 12) -> Optional[float]:
        if not self._use_tt and self.dense_backbone is not None:
            return _estimate_dense_spectral_norm(self.dense_backbone, iters=iters)
        if self._use_tt and self.backbone is not None:
            try:
                rng = np.random.default_rng(2027)
                v = rng.normal(size=(self.combined_size,)).astype(self.dtype)
                v /= np.linalg.norm(v) + _EPS
                last = 0.0
                # Rectangular operator: estimate ||W|| by applying W and W^T only if adjoint is available.
                # If no adjoint, report average forward gain over random probes.
                gains = []
                for _ in range(max(1, int(iters))):
                    v = rng.normal(size=(self.combined_size,)).astype(self.dtype)
                    v /= np.linalg.norm(v) + _EPS
                    y = self._backbone_apply(v)
                    gains.append(float(np.linalg.norm(y)))
                return float(np.mean(gains)) if gains else last
            except Exception:
                return None
        return None

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        target_norm = float(target_norm)
        if target_norm <= 0:
            raise ValueError("target_norm must be positive")
        if not self._use_tt and self.dense_backbone is not None:
            n = _estimate_dense_spectral_norm(self.dense_backbone)
            if n is not None and n > target_norm and n > 0:
                self.dense_backbone *= self.dtype.type(target_norm / n)
            return
        if self._use_tt and self.backbone is not None:
            n = self.estimate_backbone_norm()
            if n is not None and n > target_norm and n > 0:
                scale = float(target_norm / n)
                if hasattr(self.backbone, "scale"):
                    self.backbone = self.backbone.scale(scale)
                    self._backbone_param_count = int(_tt_param_count(self.backbone) or self._backbone_param_count)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "combined_size": int(self.combined_size),
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "tt_available": bool(TENSORTRAIN_AVAILABLE),
            "tt_in_shape": list(self._tt_in_shape),
            "tt_out_shape": list(self._tt_out_shape),
            "tt_bond_dims": list(self._tt_bond_dims),
            "tt_init_error": self._tt_init_error,
            "parameter_count": int(self.parameter_count()),
            "last_dt": float(self._last_dt),
            "last_update_mix": float(self._last_update_mix),
        }


# =============================================================================
# Production TCfC policy
# =============================================================================

class TCfC_Policy:
    """
    Multi-cell Tensorized Closed-form Continuous-time policy.

    The policy is intentionally substrate-level: it accepts a numeric feature
    vector and returns a bounded recurrent state. Higher-level lobe logic should
    use get_control_vector(...) to read a compact output.
    """

    def __init__(
        self,
        policy_id: str,
        input_size: int,
        hidden_size: int,
        num_cells: int,
        bond_dim: Union[int, Sequence[int]] = 8,
        rng: Optional[np.random.Generator] = None,
        dtype: np.dtype = np.float32,
        verbose: bool = True,
        *,
        tt_strict: bool = True,
        force_dense: Optional[bool] = None,
        energy_tol: float = 0.999,
        input_clip: float = 5.0,
        state_clip: float = 1.0,
        preactivation_clip: float = 10.0,
        candidate_activation: str = "tanh",
        backbone_activation: str = "tanh",
        time_scale: float = 1.0,
        leaky_lambda: float = 1.0,
        strict_runtime_checks: bool = True,
    ):
        self.policy_id = str(policy_id)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_cells = int(num_cells)
        self.bond_dim = bond_dim
        self.dtype = np.dtype(dtype)
        self.verbose = bool(verbose)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
        self.tt_strict = bool(tt_strict)
        self.force_dense = bool(force_dense) if force_dense is not None else False
        self.energy_tol = float(energy_tol)
        self.input_clip = float(input_clip)
        self.state_clip = float(state_clip)
        self.preactivation_clip = float(preactivation_clip)
        self.candidate_activation = str(candidate_activation).lower().strip()
        self.backbone_activation = str(backbone_activation).lower().strip()
        self.time_scale = float(max(_EPS, time_scale))
        self.leaky_lambda = float(np.clip(leaky_lambda, 0.0, 1.0))
        self.strict_runtime_checks = bool(strict_runtime_checks)

        if self.input_size <= 0 or self.hidden_size <= 0 or self.num_cells <= 0:
            raise ValueError("input_size, hidden_size, and num_cells must be positive")

        self.config = TCfCPolicyConfig(
            policy_id=self.policy_id,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_cells=self.num_cells,
            bond_dim=list(bond_dim) if isinstance(bond_dim, Sequence) and not isinstance(bond_dim, (str, bytes)) else int(bond_dim),
            dtype=str(self.dtype),
            tt_strict=self.tt_strict,
            force_dense=self.force_dense,
            energy_tol=self.energy_tol,
            input_clip=self.input_clip,
            state_clip=self.state_clip,
            preactivation_clip=self.preactivation_clip,
            candidate_activation=self.candidate_activation,
            backbone_activation=self.backbone_activation,
            time_scale=self.time_scale,
            leaky_lambda=self.leaky_lambda,
            strict_runtime_checks=self.strict_runtime_checks,
            verbose=self.verbose,
            rng_seed=None,
        )

        self.cells: List[TCfC_Cell] = [
            TCfC_Cell(
                self.input_size,
                self.hidden_size,
                bond_dim=self.bond_dim,
                rng=self.rng,
                dtype=self.dtype,
                tt_strict=self.tt_strict,
                force_dense=self.force_dense,
                energy_tol=self.energy_tol,
                input_clip=self.input_clip,
                state_clip=self.state_clip,
                preactivation_clip=self.preactivation_clip,
                candidate_activation=self.candidate_activation,
                backbone_activation=self.backbone_activation,
                time_scale=self.time_scale,
                leaky_lambda=self.leaky_lambda,
                strict_runtime_checks=self.strict_runtime_checks,
            )
            for _ in range(self.num_cells)
        ]

        self.state = np.zeros((self.hidden_size * self.num_cells,), dtype=self.dtype)
        self._step_count = 0
        self._last_input = np.zeros((self.input_size,), dtype=self.dtype)
        self._last_cell_aux: List[Dict[str, Any]] = []
        self._last_time_delta = 0.0

        if self.verbose:
            mode_counts = self.mode_counts()
            print(
                f"[TCfC_Policy:{self.policy_id}] cells={self.num_cells} hidden={self.hidden_size} "
                f"input={self.input_size} modes={mode_counts} params={self.parameter_count():,}"
            )

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------
    def reset_state(self, value: float = 0.0) -> None:
        v = self.dtype.type(value)
        self.state.fill(v)
        if self.state_clip > 0:
            np.clip(self.state, -self.state_clip, self.state_clip, out=self.state)
        self._step_count = 0
        self._last_input.fill(0)
        self._last_cell_aux = []
        self._last_time_delta = 0.0

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_state(self, new_state: Any) -> None:
        self.state = _coerce_vector(new_state, expected_dim=self.state.size, dtype=self.dtype, name="new_state", clip=self.state_clip)

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "kind": "tcfc_policy_state",
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "num_cells": int(self.num_cells),
            "dtype": str(self.dtype),
            "state": self.state.astype(np.float32).tolist(),
            "step_count": int(self._step_count),
            "last_time_delta": float(self._last_time_delta),
            "mode_counts": self.mode_counts(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        if int(payload.get("hidden_size", self.hidden_size)) != self.hidden_size:
            raise ValueError("hidden_size mismatch in TCfC state payload")
        if int(payload.get("num_cells", self.num_cells)) != self.num_cells:
            raise ValueError("num_cells mismatch in TCfC state payload")
        state = payload.get("state", None)
        if state is None:
            raise ValueError("payload missing 'state'")
        self.set_state(state)
        self._step_count = int(payload.get("step_count", self._step_count))
        self._last_time_delta = float(payload.get("last_time_delta", self._last_time_delta))

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def step(
        self,
        external_input: Any,
        time_delta: float = 1.0,
        *,
        return_state: bool = False,
        return_aux: Optional[bool] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Advance the policy by one bounded continuous-time step.

        Backward-compatible default:
            state = step(x, time_delta)

        Production diagnostic form:
            state, aux = step(x, time_delta, return_state=True)
        """
        wants_aux = bool(return_state if return_aux is None else return_aux)
        x = _coerce_vector(external_input, expected_dim=self.input_size, dtype=self.dtype, name="external_input", clip=self.input_clip)
        dt = _clamp_dt(time_delta)

        self._step_count += 1
        self._last_input = x.copy()
        self._last_time_delta = float(dt)

        state_matrix = self.state.reshape(self.num_cells, self.hidden_size)
        new_states = np.empty_like(state_matrix)
        cell_aux: List[Dict[str, Any]] = []

        for i, cell in enumerate(self.cells):
            if wants_aux:
                ns, aux = cell.forward(x, state_matrix[i], dt, return_aux=True)  # type: ignore[misc]
                cell_aux.append(aux)
            else:
                ns = cell.forward(x, state_matrix[i], dt, return_aux=False)  # type: ignore[assignment]
            ns_arr = np.asarray(ns, dtype=self.dtype).reshape(-1)
            if ns_arr.size != self.hidden_size:
                raise RuntimeError(f"TCfC cell {i} returned size {ns_arr.size}; expected {self.hidden_size}")
            new_states[i, :] = ns_arr

        self.state = new_states.reshape(-1).astype(self.dtype, copy=False)
        self.state = _sanitize(self.state, self.dtype)
        if self.state_clip > 0:
            np.clip(self.state, -self.state_clip, self.state_clip, out=self.state)

        self._last_cell_aux = cell_aux

        if self.strict_runtime_checks and _count_nonfinite(self.state) > 0:
            raise FloatingPointError(f"[{self.policy_id}] non-finite TCfC policy state encountered")

        if wants_aux:
            aux_out = {
                "policy_id": self.policy_id,
                "hidden_state": self.state.copy(),
                "control_vector": self.get_control_vector(out_dim=min(32, self.hidden_size)),
                "cell_means": self.get_cell_means(),
                "cell_norms": self.get_norms(),
                "cell_aux": cell_aux,
                "runtime_flags": {
                    "mode_counts": self.mode_counts(),
                    "strict_runtime_checks": bool(self.strict_runtime_checks),
                    "tt_available": bool(TENSORTRAIN_AVAILABLE),
                },
                "health": self.health_metrics(),
            }
            return self.state.copy(), aux_out
        return self.state.copy()

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------
    def get_control_vector(self, out_dim: int = 32) -> np.ndarray:
        out_dim = int(max(1, out_dim))
        state_mat = self.state.reshape(self.num_cells, self.hidden_size)
        summary = state_mat.mean(axis=0).astype(self.dtype, copy=False)
        if out_dim <= summary.size:
            return summary[:out_dim].astype(self.dtype, copy=False)
        out = np.zeros((out_dim,), dtype=self.dtype)
        out[: summary.size] = summary
        return out

    def get_cell_means(self) -> np.ndarray:
        state_mat = self.state.reshape(self.num_cells, self.hidden_size)
        return state_mat.mean(axis=1).astype(self.dtype, copy=False)

    def get_norms(self) -> np.ndarray:
        state_mat = self.state.reshape(self.num_cells, self.hidden_size)
        return np.linalg.norm(state_mat, axis=1).astype(self.dtype, copy=False)

    # ------------------------------------------------------------------
    # Diagnostics / maintenance
    # ------------------------------------------------------------------
    def mode_counts(self) -> Dict[str, int]:
        tt_count = sum(1 for c in self.cells if c._use_tt)
        return {"tt": int(tt_count), "dense": int(self.num_cells - tt_count)}

    def parameter_count(self) -> int:
        return int(sum(c.parameter_count() for c in self.cells))

    def equivalent_dense_parameter_count(self) -> int:
        combined = self.input_size + self.hidden_size
        per_cell = combined * self.hidden_size + 3 * (self.hidden_size * self.hidden_size + self.hidden_size)
        return int(self.num_cells * per_cell)

    def compression_factor(self) -> Optional[float]:
        params = self.parameter_count()
        dense = self.equivalent_dense_parameter_count()
        if params <= 0:
            return None
        return float(dense / params)

    def health_metrics(self, *, include_expensive: bool = False) -> Dict[str, Any]:
        state = self.state
        norms = self.get_norms()
        metrics: Dict[str, Any] = {
            "policy_id": self.policy_id,
            "step_count": int(self._step_count),
            "mode_counts": self.mode_counts(),
            "state_norm": float(np.linalg.norm(state)),
            "state_mean": float(np.mean(state)) if state.size else 0.0,
            "state_std": float(np.std(state)) if state.size else 0.0,
            "state_min": float(np.min(state)) if state.size else 0.0,
            "state_max": float(np.max(state)) if state.size else 0.0,
            "cell_norm_min": float(np.min(norms)) if norms.size else 0.0,
            "cell_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "cell_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "state_saturation_fraction": _saturation_fraction(state, threshold=max(0.5, 0.98 * self.state_clip)),
            "nonfinite_state_count": _count_nonfinite(state),
            "last_input_norm": float(np.linalg.norm(self._last_input)),
            "last_time_delta": float(self._last_time_delta),
            "parameter_count": int(self.parameter_count()),
            "equivalent_dense_parameter_count": int(self.equivalent_dense_parameter_count()),
            "compression_factor": self.compression_factor(),
        }
        metrics["has_nan"] = bool(metrics["nonfinite_state_count"] > 0)
        metrics["is_saturated"] = bool(metrics["state_saturation_fraction"] >= 0.25)
        metrics["is_stable"] = bool(not metrics["has_nan"] and metrics["state_norm"] < max(1.0, self.state.size * 1.25))

        if include_expensive:
            norms_est: List[Optional[float]] = []
            for c in self.cells:
                norms_est.append(c.estimate_backbone_norm())
            finite = [float(x) for x in norms_est if x is not None and np.isfinite(x)]
            metrics["backbone_norm_mean"] = float(np.mean(finite)) if finite else None
            metrics["backbone_norm_max"] = float(np.max(finite)) if finite else None

        return metrics

    def snapshot(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "kind": "tcfc_policy",
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "num_cells": int(self.num_cells),
            "dtype": str(self.dtype),
            "mode_counts": self.mode_counts(),
            "parameter_count": int(self.parameter_count()),
            "equivalent_dense_parameter_count": int(self.equivalent_dense_parameter_count()),
            "compression_factor": self.compression_factor(),
            "state_norm": float(np.linalg.norm(self.state)),
            "cell_norms": self.get_norms().astype(float).tolist(),
            "step_count": int(self._step_count),
            "config": asdict(self.config),
            "cells": [c.snapshot() for c in self.cells],
        }

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        for c in self.cells:
            c.apply_regularization(target_norm=target_norm)

    def summary(self) -> str:
        cf = self.compression_factor()
        cf_s = "n/a" if cf is None else f"{cf:.2f}x"
        return (
            f"[TCfC Summary]\n"
            f"- id: {self.policy_id}\n"
            f"- input: {self.input_size}\n"
            f"- hidden: {self.hidden_size}\n"
            f"- cells: {self.num_cells}\n"
            f"- dtype: {self.dtype}\n"
            f"- modes: {self.mode_counts()}\n"
            f"- params: {self.parameter_count():,}\n"
            f"- dense-equivalent params: {self.equivalent_dense_parameter_count():,}\n"
            f"- compression: {cf_s}\n"
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Union[str, os.PathLike[str]]) -> None:
        """Save full model weights plus runtime state to a compressed NPZ file."""
        p = Path(path)
        arrays: Dict[str, np.ndarray] = {}
        meta: Dict[str, Any] = {
            "format": "akkurat_tcfc_checkpoint_v1",
            "config": asdict(self.config),
            "policy_id": self.policy_id,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_cells": self.num_cells,
            "dtype": str(self.dtype),
            "step_count": int(self._step_count),
            "last_time_delta": float(self._last_time_delta),
            "cells": [],
        }

        arrays["state"] = self.state.astype(self.dtype, copy=False)

        for i, cell in enumerate(self.cells):
            cmeta = {
                "use_tt": bool(cell._use_tt),
                "tt_in_shape": list(cell._tt_in_shape),
                "tt_out_shape": list(cell._tt_out_shape),
                "tt_bond_dims": list(cell._tt_bond_dims),
                "tt_init_error": cell._tt_init_error,
            }
            arrays[f"cell_{i}_f_head"] = cell.f_head
            arrays[f"cell_{i}_g_head"] = cell.g_head
            arrays[f"cell_{i}_h_head"] = cell.h_head
            arrays[f"cell_{i}_f_bias"] = cell.f_bias
            arrays[f"cell_{i}_g_bias"] = cell.g_bias
            arrays[f"cell_{i}_h_bias"] = cell.h_bias

            if cell._use_tt and cell.backbone is not None:
                cores = getattr(cell.backbone, "cores_data", None)
                if isinstance(cores, (list, tuple)):
                    cmeta["num_tt_cores"] = len(cores)
                    for j, core in enumerate(cores):
                        arrays[f"cell_{i}_tt_core_{j}"] = np.asarray(_to_numpy(core), dtype=self.dtype)
                else:
                    cmeta["num_tt_cores"] = 0
            else:
                cmeta["num_tt_cores"] = 0
                if cell.dense_backbone is not None:
                    arrays[f"cell_{i}_dense_backbone"] = cell.dense_backbone
            meta["cells"].append(cmeta)

        arrays["__meta_json__"] = np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_)
        np.savez_compressed(p, **arrays)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        rng: Optional[np.random.Generator] = None,
        verbose: Optional[bool] = None,
        force_dense: Optional[bool] = None,
        tt_strict: Optional[bool] = None,
    ) -> "TCfC_Policy":
        p = Path(path)
        data = np.load(p, allow_pickle=False)
        meta_raw = str(data["__meta_json__"].item())
        meta = json.loads(meta_raw)
        cfg = meta.get("config", {})

        dtype = np.dtype(meta.get("dtype", cfg.get("dtype", "float32")))
        obj = cls(
            policy_id=str(meta.get("policy_id", cfg.get("policy_id", "loaded_tcfc"))),
            input_size=int(meta.get("input_size", cfg.get("input_size"))),
            hidden_size=int(meta.get("hidden_size", cfg.get("hidden_size"))),
            num_cells=int(meta.get("num_cells", cfg.get("num_cells"))),
            bond_dim=cfg.get("bond_dim", 8),
            rng=rng if isinstance(rng, np.random.Generator) else np.random.default_rng(),
            dtype=dtype,
            verbose=bool(cfg.get("verbose", False)) if verbose is None else bool(verbose),
            tt_strict=bool(cfg.get("tt_strict", False)) if tt_strict is None else bool(tt_strict),
            force_dense=bool(cfg.get("force_dense", False)) if force_dense is None else bool(force_dense),
            energy_tol=float(cfg.get("energy_tol", 0.999)),
            input_clip=float(cfg.get("input_clip", 5.0)),
            state_clip=float(cfg.get("state_clip", 1.0)),
            preactivation_clip=float(cfg.get("preactivation_clip", 10.0)),
            candidate_activation=str(cfg.get("candidate_activation", "tanh")),
            backbone_activation=str(cfg.get("backbone_activation", "tanh")),
            time_scale=float(cfg.get("time_scale", 1.0)),
            leaky_lambda=float(cfg.get("leaky_lambda", 1.0)),
            strict_runtime_checks=bool(cfg.get("strict_runtime_checks", True)),
        )

        obj.state = np.asarray(data["state"], dtype=dtype).reshape(-1)
        obj._step_count = int(meta.get("step_count", 0))
        obj._last_time_delta = float(meta.get("last_time_delta", 0.0))

        for i, cell in enumerate(obj.cells):
            cell.f_head = np.asarray(data[f"cell_{i}_f_head"], dtype=dtype)
            cell.g_head = np.asarray(data[f"cell_{i}_g_head"], dtype=dtype)
            cell.h_head = np.asarray(data[f"cell_{i}_h_head"], dtype=dtype)
            cell.f_bias = np.asarray(data[f"cell_{i}_f_bias"], dtype=dtype)
            cell.g_bias = np.asarray(data[f"cell_{i}_g_bias"], dtype=dtype)
            cell.h_bias = np.asarray(data[f"cell_{i}_h_bias"], dtype=dtype)

            cmeta = meta.get("cells", [{}])[i]
            use_tt = bool(cmeta.get("use_tt", False)) and TENSORTRAIN_AVAILABLE and not bool(force_dense)
            ncores = int(cmeta.get("num_tt_cores", 0))
            if use_tt and ncores > 0:
                cores = [np.asarray(data[f"cell_{i}_tt_core_{j}"], dtype=dtype) for j in range(ncores)]
                try:
                    cell.backbone = TensorTrain(
                        output_dims=list(cmeta.get("tt_out_shape", cell._tt_out_shape)),
                        input_dims=list(cmeta.get("tt_in_shape", cell._tt_in_shape)),
                        cores_data=cores,
                        config=TTConfig(dtype=dtype, device="cpu"),
                    )
                    cell.dense_backbone = None
                    cell._use_tt = True
                    cell._backbone_param_count = int(_tt_param_count(cell.backbone) or 0)
                except Exception:
                    if tt_strict:
                        raise
                    cell._use_tt = False
                    cell.backbone = None
                    key = f"cell_{i}_dense_backbone"
                    if key in data:
                        cell.dense_backbone = np.asarray(data[key], dtype=dtype)
                    else:
                        cell.dense_backbone = _stable_dense_init_rect(cell.hidden_size, cell.combined_size, obj.rng, dtype)
                    cell._backbone_param_count = int(cell.dense_backbone.size)
            else:
                cell._use_tt = False
                cell.backbone = None
                key = f"cell_{i}_dense_backbone"
                if key in data:
                    cell.dense_backbone = np.asarray(data[key], dtype=dtype)
                elif cell.dense_backbone is None:
                    cell.dense_backbone = _stable_dense_init_rect(cell.hidden_size, cell.combined_size, obj.rng, dtype)
                cell._backbone_param_count = int(cell.dense_backbone.size)

        return obj

    def __repr__(self) -> str:
        return (
            f"TCfC_Policy(id={self.policy_id!r}, input={self.input_size}, "
            f"hidden={self.hidden_size}, cells={self.num_cells}, modes={self.mode_counts()}, dtype={self.dtype})"
        )


# Alias with a more standard class-style name for new code.
TensorizedCfCPolicy = TCfC_Policy
TensorizedCfCCell = TCfC_Cell


# =============================================================================
# Demonstration / smoke tests
# =============================================================================

def _self_test() -> None:
    print("=== TCfC production substrate self-test ===")
    rng = np.random.default_rng(42)

    policy = TCfC_Policy(
        policy_id="selftest_tcfc",
        input_size=16,
        hidden_size=32,
        num_cells=4,
        bond_dim=4,
        rng=rng,
        dtype=np.float32,
        verbose=True,
        tt_strict=False,
        force_dense=False,
        state_clip=1.0,
        time_scale=1.0,
        leaky_lambda=1.0,
    )

    x = rng.normal(size=(16,)).astype(np.float32)
    s0 = policy.get_state()
    s_zero = policy.step(x, time_delta=0.0)
    assert np.allclose(s0, s_zero), "dt=0 must preserve state"

    state, aux = policy.step(x, time_delta=0.5, return_state=True)
    assert state.shape == (32 * 4,)
    assert np.isfinite(state).all()
    assert "health" in aux

    payload = policy.serialize_state()
    policy.reset_state()
    policy.load_state(payload)
    assert np.allclose(policy.get_state(), state)

    # Determinism probe
    def build(seed: int) -> TCfC_Policy:
        return TCfC_Policy(
            policy_id=f"det_{seed}",
            input_size=16,
            hidden_size=32,
            num_cells=4,
            bond_dim=4,
            rng=np.random.default_rng(seed),
            dtype=np.float32,
            verbose=False,
            tt_strict=False,
        )

    det_in = np.random.default_rng(99).normal(size=(16,)).astype(np.float32)
    p1 = build(2025)
    p2 = build(2025)
    y1 = p1.step(det_in, time_delta=0.25)
    y2 = p2.step(det_in, time_delta=0.25)
    assert np.allclose(y1, y2, atol=1e-5, rtol=1e-5), "determinism probe failed"

    # Checkpoint probe
    ckpt = Path("tcfc_selftest_checkpoint.npz")
    policy.save_checkpoint(ckpt)
    loaded = TCfC_Policy.load_checkpoint(ckpt, verbose=False, tt_strict=False)
    assert np.allclose(loaded.get_state(), policy.get_state(), atol=1e-6)
    try:
        ckpt.unlink()
    except Exception:
        pass

    print(policy.summary())
    print("Health:", policy.health_metrics())
    print("=== TCfC self-test passed ===")


if __name__ == "__main__":
    _self_test()
