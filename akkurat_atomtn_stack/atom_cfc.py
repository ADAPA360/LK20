#!/usr/bin/env python3
# atom_cfc.py
"""
AtomTN Quantum Closed-form Continuous-time Policy
=================================================

Production-compatible QuantumCfC implementation backed by the shared AtomTN
neuromorphic reservoir runtime.

Architecture
------------
Each QuantumCfC cell contains:

1. A stateful AtomTN quantum reservoir over the vector
       [external_input, previous_cell_state]
2. A holographic readout feature vector measured at the TTN boundary.
3. Three classical CfC heads f, g, h that map quantum features to hidden state.
4. A bounded closed-form continuous-time update with dt=0 preserving state.

The implementation keeps the legacy public names and constructor style:

    QuantumCfC_Policy(policy_id, input_size, hidden_size, num_cells, bond_dim=2)
    state = policy.step(x, time_delta=1.0)

Production callers may request diagnostics:

    state, aux = policy.step(x, time_delta=1.0, return_state=True)

This module deliberately delegates AtomTN mechanics to neuromorphic.py so the
AtomTN runtime family has one maintained reservoir path across neuromorphic.py,
atom_ncp.py, and atom_cfc.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Local import setup
# =============================================================================


def _add_local_paths() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, here.parent.parent):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return here.parent


_MODULE_DIR = _add_local_paths()

_NEUROMORPHIC_OK = False
_NEUROMORPHIC_IMPORT_ERROR: Optional[BaseException] = None

try:
    from neuromorphic import (  # type: ignore
        HolographicReadout,
        HolographicReadoutConfig,
        QuantumReservoir,
        QuantumReservoirConfig,
        ReservoirBuildConfig,
        atomtn_status as _atomtn_status,
        build_quantum_reservoir,
        require_atomtn,
    )

    _NEUROMORPHIC_OK = True
except Exception as exc:  # pragma: no cover - depends on local AtomTN install
    _NEUROMORPHIC_OK = False
    _NEUROMORPHIC_IMPORT_ERROR = exc

    HolographicReadout = Any  # type: ignore
    HolographicReadoutConfig = Any  # type: ignore
    QuantumReservoir = Any  # type: ignore
    QuantumReservoirConfig = Any  # type: ignore
    ReservoirBuildConfig = Any  # type: ignore

    def _atomtn_status() -> Dict[str, Any]:
        return {"available": False, "import_error": repr(_NEUROMORPHIC_IMPORT_ERROR), "module_dir": str(_MODULE_DIR)}

    def require_atomtn() -> None:
        raise RuntimeError(f"neuromorphic.py / AtomTN runtime is not importable: {_NEUROMORPHIC_IMPORT_ERROR!r}")

    def build_quantum_reservoir(cfg: Any) -> Any:
        require_atomtn()


# =============================================================================
# Helpers
# =============================================================================

_EPS = 1e-12


def atom_cfc_status() -> Dict[str, Any]:
    """Return import/runtime status without constructing a policy."""
    status = _atomtn_status()
    return {
        "atom_cfc_available": bool(_NEUROMORPHIC_OK and status.get("available", False)),
        "neuromorphic_import_ok": bool(_NEUROMORPHIC_OK),
        "neuromorphic_import_error": None if _NEUROMORPHIC_IMPORT_ERROR is None else repr(_NEUROMORPHIC_IMPORT_ERROR),
        "atomtn_status": status,
        "module_dir": str(_MODULE_DIR),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {"real": np.real(arr).astype(float).tolist(), "imag": np.imag(arr).astype(float).tolist()}
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "snapshot") and callable(getattr(obj, "snapshot")):
        try:
            return _json_safe(obj.snapshot())
        except Exception:
            pass
    if hasattr(obj, "serialize_state") and callable(getattr(obj, "serialize_state")):
        try:
            return _json_safe(obj.serialize_state())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _coerce_vector(
    x: Any,
    *,
    expected_dim: Optional[int] = None,
    dtype: np.dtype = np.float32,
    name: str = "vector",
    resize: bool = False,
    clip: Optional[float] = None,
) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.dtype(dtype)).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.dtype(dtype))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.dtype(dtype), copy=False)

    if expected_dim is not None and int(arr.size) != int(expected_dim):
        if not resize:
            raise ValueError(f"{name} size mismatch: expected {expected_dim}, got {arr.size}")
        out = np.zeros((int(expected_dim),), dtype=np.dtype(dtype))
        n = min(int(expected_dim), int(arr.size))
        if n > 0:
            out[:n] = arr[:n]
        arr = out

    if clip is not None and float(clip) > 0:
        np.clip(arr, -float(clip), float(clip), out=arr)
    return arr.astype(np.dtype(dtype), copy=False)


def _safe_norm(x: Any) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return 0.0
        v = float(np.linalg.norm(arr.reshape(-1)))
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _count_nonfinite(x: Any) -> int:
    try:
        arr = np.asarray(x)
        return int(arr.size - np.count_nonzero(np.isfinite(arr)))
    except Exception:
        return 0


def _saturation_fraction(x: Any, threshold: float = 0.98) -> float:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(np.mean(np.abs(arr) >= float(threshold)))
    except Exception:
        return 0.0


def _sigmoid(x: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    z = np.asarray(x, dtype=np.dtype(dtype))
    return (1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))).astype(np.dtype(dtype), copy=False)


def _activation(x: np.ndarray, kind: str) -> np.ndarray:
    k = str(kind or "tanh").lower().strip()
    if k == "tanh":
        return np.tanh(x).astype(x.dtype, copy=False)
    if k == "clip":
        return np.clip(x, -1.0, 1.0).astype(x.dtype, copy=False)
    if k == "softsign":
        return (x / (1.0 + np.abs(x))).astype(x.dtype, copy=False)
    return x.astype(x.dtype, copy=False)


def _clamp_dt(dt: Any, *, lo: float = 0.0, hi: float = 1.0e3, default: float = 1.0) -> float:
    v = _finite_float(dt, default)
    return float(np.clip(v, float(lo), float(hi)))


def _stable_head_init(
    out_dim: int,
    in_dim: int,
    rng: np.random.Generator,
    dtype: np.dtype,
    *,
    scale: float = 1.0,
    max_col_norm: float = 2.0,
) -> np.ndarray:
    out_dim = int(out_dim)
    in_dim = int(in_dim)
    dt = np.dtype(dtype)
    if out_dim <= 0 or in_dim <= 0:
        return np.zeros((max(0, out_dim), max(0, in_dim)), dtype=dt)
    W = rng.normal(0.0, float(scale) / math.sqrt(max(1, in_dim)), size=(out_dim, in_dim)).astype(np.float64)
    # Conservative per-row norm cap to keep initial CfC heads bounded.
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    scale_down = np.minimum(1.0, float(max_col_norm) / np.maximum(row_norms, _EPS))
    W *= scale_down
    return W.astype(dt, copy=False)


def _resolve_input(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for key in ("observation", "input", "inputs", "x", "features", "previous_output", "state"):
            if key in payload:
                return payload[key]
        return []
    return payload


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QuantumCfCConfig:
    """Configuration for the AtomTN-backed QuantumCfC policy and cells."""

    # Atom / geometry
    atom_name: str = "H"
    atom_level: int = 1
    noncommutative: bool = False
    fuzzy_l: int = 2
    seed: int = 42
    tree_mode: str = "balanced"
    tree_arity: int = 4

    # Fiber / TTN state
    fiber_d_uniform: int = 2
    fiber_d_min: int = 2
    fiber_d_max: int = 4

    # Vibration bath. Back-compatible naming mirrors neuromorphic.py.
    vibration_kind: str = "linear"
    vibration_strength: float = 0.1
    vibration_temperature: float = 10.0
    vibration_n: int = 16
    vibration_spectral_kind: str = "ohmic"
    vibration_alpha: float = 1.0
    vibration_omega_c: float = 10.0

    # Reservoir / encoder
    reservoir_dt: float = 0.05
    use_time_delta_for_reservoir: bool = False
    memory_damping: float = 0.1
    encoder_scale: float = 1.0
    encoder_sparsity: float = 0.2
    encoder_normalize_input: bool = False
    encoder_output_clip: Optional[float] = 25.0
    encoder_input_clip: Optional[float] = 10.0
    injection_clip: Optional[float] = 25.0

    # Fast motor-control defaults: direct injection + Euler scaffold.
    flow_steps: int = 0
    flow_damping: float = 0.0
    flow_diffusion: float = 0.0
    evolution_method: str = "euler_legacy"
    apply_truncate_rank: int = 4
    post_step_truncate_rank: int = 4
    canonicalize_every: int = 1
    renormalize_every: int = 1
    step_bucket_every: int = 1

    # Hamiltonian scales
    onsite_scale: float = 1.0
    edge_scale: float = 0.5
    vib_scale: float = 0.1
    hop_scale: float = 1.0

    # Holographic readout
    readout_observables: Tuple[str, ...] = ("Z",)
    readout_include_summary_stats: bool = False
    readout_normalize_features: bool = False
    readout_clip_value: Optional[float] = 10.0
    readout_divide_by_norm: bool = False

    # CfC head / state update
    head_init_scale: float = 1.0
    preactivation_clip: float = 10.0
    candidate_activation: str = "tanh"  # "tanh" | "clip" | "softsign" | "linear"
    state_clip: float = 1.0
    input_clip: Optional[float] = 10.0
    time_scale: float = 1.0
    leaky_lambda: float = 1.0
    strict_runtime_checks: bool = True

    # Runtime behavior
    strict: bool = True
    cache_runtime_objects: bool = True
    resize_mismatched_input: bool = False
    reset_reservoir_on_policy_reset: bool = True

    def normalized(self) -> "QuantumCfCConfig":
        dmin = int(max(1, self.fiber_d_min))
        dunif = int(max(dmin, self.fiber_d_uniform))
        dmax = int(max(dunif, self.fiber_d_max))
        return QuantumCfCConfig(
            atom_name=str(self.atom_name),
            atom_level=int(max(1, self.atom_level)),
            noncommutative=bool(self.noncommutative),
            fuzzy_l=int(max(0, self.fuzzy_l)),
            seed=int(self.seed),
            tree_mode=str(self.tree_mode or "balanced"),
            tree_arity=int(max(2, self.tree_arity)),
            fiber_d_uniform=dunif,
            fiber_d_min=dmin,
            fiber_d_max=dmax,
            vibration_kind=str(self.vibration_kind or "linear"),
            vibration_strength=float(max(_EPS, self.vibration_strength)),
            vibration_temperature=float(max(float(self.vibration_strength) + _EPS, self.vibration_temperature)),
            vibration_n=int(max(1, self.vibration_n)),
            vibration_spectral_kind=str(self.vibration_spectral_kind or "ohmic"),
            vibration_alpha=float(self.vibration_alpha),
            vibration_omega_c=float(max(_EPS, self.vibration_omega_c)),
            reservoir_dt=float(max(0.0, self.reservoir_dt)),
            use_time_delta_for_reservoir=bool(self.use_time_delta_for_reservoir),
            memory_damping=float(np.clip(self.memory_damping, 0.0, 1.0)),
            encoder_scale=float(self.encoder_scale),
            encoder_sparsity=float(np.clip(self.encoder_sparsity, 0.0, 1.0)),
            encoder_normalize_input=bool(self.encoder_normalize_input),
            encoder_output_clip=None if self.encoder_output_clip is None else float(max(0.0, self.encoder_output_clip)),
            encoder_input_clip=None if self.encoder_input_clip is None else float(max(0.0, self.encoder_input_clip)),
            injection_clip=None if self.injection_clip is None else float(max(0.0, self.injection_clip)),
            flow_steps=int(max(0, self.flow_steps)),
            flow_damping=float(max(0.0, self.flow_damping)),
            flow_diffusion=float(max(0.0, self.flow_diffusion)),
            evolution_method=str(self.evolution_method or "euler_legacy"),
            apply_truncate_rank=int(max(1, self.apply_truncate_rank)),
            post_step_truncate_rank=int(max(1, self.post_step_truncate_rank)),
            canonicalize_every=int(max(1, self.canonicalize_every)),
            renormalize_every=int(max(1, self.renormalize_every)),
            step_bucket_every=int(max(1, self.step_bucket_every)),
            onsite_scale=float(self.onsite_scale),
            edge_scale=float(self.edge_scale),
            vib_scale=float(self.vib_scale),
            hop_scale=float(self.hop_scale),
            readout_observables=tuple(str(x) for x in (self.readout_observables or ("Z",))),
            readout_include_summary_stats=bool(self.readout_include_summary_stats),
            readout_normalize_features=bool(self.readout_normalize_features),
            readout_clip_value=None if self.readout_clip_value is None else float(max(0.0, self.readout_clip_value)),
            readout_divide_by_norm=bool(self.readout_divide_by_norm),
            head_init_scale=float(max(0.0, self.head_init_scale)),
            preactivation_clip=float(max(0.0, self.preactivation_clip)),
            candidate_activation=str(self.candidate_activation or "tanh"),
            state_clip=float(max(0.0, self.state_clip)),
            input_clip=None if self.input_clip is None else float(max(0.0, self.input_clip)),
            time_scale=float(max(_EPS, self.time_scale)),
            leaky_lambda=float(np.clip(self.leaky_lambda, 0.0, 1.0)),
            strict_runtime_checks=bool(self.strict_runtime_checks),
            strict=bool(self.strict),
            cache_runtime_objects=bool(self.cache_runtime_objects),
            resize_mismatched_input=bool(self.resize_mismatched_input),
            reset_reservoir_on_policy_reset=bool(self.reset_reservoir_on_policy_reset),
        )


# =============================================================================
# Quantum CfC cell
# =============================================================================

class QuantumCfC_Cell:
    """
    A single AtomTN-backed CfC cell.

    The cell owns a quantum reservoir, so its hidden state contains both:
      - explicit classical CfC state `prev_state -> next_state`, and
      - implicit quantum reservoir memory inside AtomTN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bond_dim: int = 2,
        rng: Optional[np.random.Generator] = None,
        dtype: np.dtype = np.float32,
        *,
        dt: float = 0.05,
        damping: float = 0.1,
        noncommutative: bool = False,
        config: Optional[QuantumCfCConfig] = None,
        cell_index: int = 0,
        seed: Optional[int] = None,
    ):
        require_atomtn()

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bond_dim = int(max(1, bond_dim))
        self.dtype = np.dtype(dtype)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(42 if seed is None else int(seed))
        self.cell_index = int(cell_index)

        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")

        base_cfg = config or QuantumCfCConfig()
        base_cfg.reservoir_dt = float(dt if dt is not None else base_cfg.reservoir_dt)
        base_cfg.memory_damping = float(damping if damping is not None else base_cfg.memory_damping)
        base_cfg.noncommutative = bool(noncommutative)
        if seed is not None:
            base_cfg.seed = int(seed)
        self.config = base_cfg.normalized()

        self.backbone_input_dim = self.input_size + self.hidden_size
        self._last_features = np.zeros((0,), dtype=self.dtype)
        self._last_f = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_g = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_h = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_gate = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_candidate = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_state = np.zeros((self.hidden_size,), dtype=self.dtype)
        self._last_update_mix = 0.0
        self._last_dt = 0.0
        self._last_error: Optional[str] = None

        self.reservoir = self._build_reservoir()
        self.atom = self.reservoir.atom
        self.fiber = self.reservoir.fiber

        readout_cfg = self.reservoir.config.readout
        initial_features = HolographicReadout.measure(self.atom.state, readout_cfg)
        if int(initial_features.size) <= 0:
            leaves = list(getattr(getattr(self.atom.state, "tree", None), "leaves", []))
            initial_features = np.zeros((len(leaves) or 64,), dtype=np.float32)
        self.reservoir_size = int(initial_features.size)
        self._last_features = self._features_to_dim(initial_features)

        self.W_f = _stable_head_init(self.hidden_size, self.reservoir_size, self.rng, self.dtype, scale=self.config.head_init_scale)
        self.W_g = _stable_head_init(self.hidden_size, self.reservoir_size, self.rng, self.dtype, scale=self.config.head_init_scale)
        self.W_h = _stable_head_init(self.hidden_size, self.reservoir_size, self.rng, self.dtype, scale=self.config.head_init_scale)
        self.b_f = np.zeros((self.hidden_size,), dtype=self.dtype)
        self.b_g = np.zeros((self.hidden_size,), dtype=self.dtype)
        self.b_h = np.zeros((self.hidden_size,), dtype=self.dtype)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_reservoir(self) -> Any:
        readout_cfg = HolographicReadoutConfig(
            observables=tuple(self.config.readout_observables),
            include_summary_stats=bool(self.config.readout_include_summary_stats),
            normalize_features=bool(self.config.readout_normalize_features),
            clip_value=self.config.readout_clip_value,
            divide_by_norm=bool(self.config.readout_divide_by_norm),
        )
        reservoir_cfg = QuantumReservoirConfig(
            input_dim=int(self.backbone_input_dim),
            encoder_seed=int(self.config.seed),
            encoder_sparsity=float(self.config.encoder_sparsity),
            encoder_scale=float(self.config.encoder_scale),
            encoder_normalize_input=bool(self.config.encoder_normalize_input),
            encoder_output_clip=self.config.encoder_output_clip,
            encoder_input_clip=self.config.encoder_input_clip,
            encoder_resize_mismatched_input=bool(self.config.resize_mismatched_input),
            memory_damping=float(self.config.memory_damping),
            injection_clip=self.config.injection_clip,
            preserve_on_zero_dt=True,
            flow_steps=int(self.config.flow_steps),
            flow_damping=float(self.config.flow_damping),
            flow_diffusion=float(self.config.flow_diffusion),
            onsite_scale=float(self.config.onsite_scale),
            edge_scale=float(self.config.edge_scale),
            vib_scale=float(self.config.vib_scale),
            hop_scale=float(self.config.hop_scale),
            evolution_method=str(self.config.evolution_method),
            apply_truncate_rank=int(max(1, self.config.apply_truncate_rank)),
            canonicalize_every=int(max(1, self.config.canonicalize_every)),
            apply_grouping="lca_routed",
            post_step_truncate_rank=int(max(1, self.config.post_step_truncate_rank)),
            renormalize_every=int(max(1, self.config.renormalize_every)),
            step_bucket_every=int(max(1, self.config.step_bucket_every)),
            readout=readout_cfg,
            cache_runtime_objects=bool(self.config.cache_runtime_objects),
            strict=bool(self.config.strict),
        )
        build_cfg = ReservoirBuildConfig(
            atom_name=str(self.config.atom_name),
            atom_level=int(self.config.atom_level),
            noncommutative=bool(self.config.noncommutative),
            fuzzy_l=int(self.config.fuzzy_l),
            seed=int(self.config.seed),
            tree_mode=str(self.config.tree_mode),
            tree_arity=int(self.config.tree_arity),
            vibration_kind=str(self.config.vibration_kind),
            vibration_strength=float(self.config.vibration_strength),
            vibration_temperature=float(self.config.vibration_temperature),
            vibration_n=int(self.config.vibration_n),
            vibration_spectral_kind=str(self.config.vibration_spectral_kind),
            vibration_alpha=float(self.config.vibration_alpha),
            vibration_omega_c=float(self.config.vibration_omega_c),
            fiber_d_uniform=int(self.config.fiber_d_uniform),
            fiber_d_min=int(self.config.fiber_d_min),
            fiber_d_max=int(self.config.fiber_d_max),
            bond_dim=int(self.bond_dim),
            reservoir=reservoir_cfg,
        )
        return build_quantum_reservoir(build_cfg)

    def _features_to_dim(self, features: Any) -> np.ndarray:
        arr = _coerce_vector(features, dtype=self.dtype, name="reservoir_features")
        if not hasattr(self, "reservoir_size") or int(getattr(self, "reservoir_size", 0)) <= 0:
            return arr.astype(self.dtype, copy=False)
        if arr.size == self.reservoir_size:
            return arr.astype(self.dtype, copy=False)
        out = np.zeros((self.reservoir_size,), dtype=self.dtype)
        n = min(int(arr.size), int(self.reservoir_size))
        if n > 0:
            out[:n] = arr[:n]
        return out

    def _effective_reservoir_dt(self, time_delta: float) -> float:
        if bool(self.config.use_time_delta_for_reservoir):
            return float(max(0.0, time_delta))
        return float(max(0.0, self.config.reservoir_dt))

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def forward(
        self,
        input_vec: Any,
        prev_state: Any,
        time_delta: float,
        *,
        return_aux: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        x = _coerce_vector(
            input_vec,
            expected_dim=self.input_size,
            dtype=self.dtype,
            name="input_vec",
            resize=bool(self.config.resize_mismatched_input),
            clip=self.config.input_clip,
        )
        prev = _coerce_vector(prev_state, expected_dim=self.hidden_size, dtype=self.dtype, name="prev_state", resize=False, clip=self.config.state_clip)
        dt = _clamp_dt(time_delta)
        self._last_dt = float(dt)
        self._last_error = None

        if dt <= 0.0:
            next_state = prev.copy()
            self._last_state = next_state.copy()
            aux = self._aux(next_state=next_state, skipped=True)
            return (next_state, aux) if return_aux else next_state

        backbone_input = np.concatenate([x, prev]).astype(self.dtype, copy=False)
        reservoir_dt = self._effective_reservoir_dt(dt)

        try:
            features = self.reservoir.step(backbone_input, dt=reservoir_dt)
        except Exception as exc:
            self._last_error = f"reservoir_step_failed: {exc!r}"
            if bool(self.config.strict):
                raise
            features = self._last_features.copy()

        z = self._features_to_dim(features)
        self._last_features = z.copy()

        f_out = (self.W_f @ z + self.b_f).astype(self.dtype, copy=False)
        g_out = (self.W_g @ z + self.b_g).astype(self.dtype, copy=False)
        h_out = (self.W_h @ z + self.b_h).astype(self.dtype, copy=False)

        if self.config.preactivation_clip > 0:
            lim = float(self.config.preactivation_clip)
            np.clip(f_out, -lim, lim, out=f_out)
            np.clip(g_out, -lim, lim, out=g_out)
            np.clip(h_out, -lim, lim, out=h_out)

        g_act = _activation(g_out, self.config.candidate_activation)
        h_act = _activation(h_out, self.config.candidate_activation)

        gate = _sigmoid(-f_out * self.dtype.type(dt), dtype=self.dtype)
        candidate = (gate * g_act + (self.dtype.type(1.0) - gate) * h_act).astype(self.dtype, copy=False)
        candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0).astype(self.dtype, copy=False)

        temporal_mix = 1.0 - math.exp(-float(dt) / float(self.config.time_scale))
        update_mix = float(np.clip(float(self.config.leaky_lambda) * temporal_mix, 0.0, 1.0))
        next_state = ((1.0 - update_mix) * prev + update_mix * candidate).astype(self.dtype, copy=False)

        if self.config.state_clip > 0:
            np.clip(next_state, -float(self.config.state_clip), float(self.config.state_clip), out=next_state)
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0).astype(self.dtype, copy=False)

        self._last_f = f_out.copy()
        self._last_g = g_act.copy()
        self._last_h = h_act.copy()
        self._last_gate = gate.copy()
        self._last_candidate = candidate.copy()
        self._last_update_mix = float(update_mix)
        self._last_state = next_state.copy()

        if self.config.strict_runtime_checks and _count_nonfinite(next_state) > 0:
            raise FloatingPointError("Non-finite QuantumCfC cell state encountered")

        aux = self._aux(next_state=next_state, skipped=False)
        return (next_state, aux) if return_aux else next_state

    # ------------------------------------------------------------------
    # Diagnostics / maintenance
    # ------------------------------------------------------------------

    def _aux(self, *, next_state: np.ndarray, skipped: bool) -> Dict[str, Any]:
        health = self.health_metrics()
        return {
            "cell_index": int(self.cell_index),
            "backbone": "atomtn_quantum_reservoir",
            "dt": float(self._last_dt),
            "reservoir_dt": float(self._effective_reservoir_dt(self._last_dt)),
            "skipped": bool(skipped),
            "update_mix": float(self._last_update_mix),
            "features": self._last_features.copy(),
            "f": self._last_f.copy(),
            "g": self._last_g.copy(),
            "h": self._last_h.copy(),
            "gate": self._last_gate.copy(),
            "candidate": self._last_candidate.copy(),
            "next_state_norm": float(np.linalg.norm(next_state)),
            "reservoir_snapshot": self.reservoir.snapshot() if hasattr(self.reservoir, "snapshot") else {},
            "health": health,
            "error": self._last_error,
        }

    def reset_reservoir(self) -> None:
        if hasattr(self.reservoir, "reset_state"):
            self.reservoir.reset_state()
        self._last_features = self._features_to_dim(HolographicReadout.measure(self.atom.state, self.reservoir.config.readout))
        self._last_error = None

    def parameter_count(self) -> int:
        return int(self.W_f.size + self.W_g.size + self.W_h.size + self.b_f.size + self.b_g.size + self.b_h.size)

    def trainable_parameter_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W_f": self.W_f,
            "W_g": self.W_g,
            "W_h": self.W_h,
            "b_f": self.b_f,
            "b_g": self.b_g,
            "b_h": self.b_h,
        }

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        target = float(target_norm)
        if target <= 0:
            raise ValueError("target_norm must be positive")
        for name in ("W_f", "W_g", "W_h"):
            W = getattr(self, name)
            n = float(np.linalg.norm(W))
            if math.isfinite(n) and n > target and n > 0:
                setattr(self, name, (W * self.dtype.type(target / n)).astype(self.dtype, copy=False))

    def health_metrics(self) -> Dict[str, Any]:
        res_h = self.reservoir.health_metrics() if hasattr(self.reservoir, "health_metrics") else {}
        params_finite = all(np.all(np.isfinite(v)) for v in self.trainable_parameter_dict().values())
        feat_finite = bool(self._last_features.size == 0 or np.all(np.isfinite(self._last_features)))
        state_finite = bool(self._last_state.size == 0 or np.all(np.isfinite(self._last_state)))
        stable = bool(params_finite and feat_finite and state_finite and self._last_error is None and bool(res_h.get("is_stable", True)))
        return {
            "kind": "QuantumCfC_Cell",
            "cell_index": int(self.cell_index),
            "is_stable": stable,
            "has_nan": bool((not params_finite) or (not feat_finite) or (not state_finite) or res_h.get("has_nan", False)),
            "feature_norm": _safe_norm(self._last_features),
            "state_norm": _safe_norm(self._last_state),
            "gate_mean": float(np.mean(self._last_gate)) if self._last_gate.size else 0.0,
            "gate_std": float(np.std(self._last_gate)) if self._last_gate.size else 0.0,
            "candidate_norm": _safe_norm(self._last_candidate),
            "head_parameter_count": int(self.parameter_count()),
            "reservoir": res_h,
            "last_error": self._last_error,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "QuantumCfC_Cell",
            "cell_index": int(self.cell_index),
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "backbone_input_dim": int(self.backbone_input_dim),
            "reservoir_size": int(self.reservoir_size),
            "bond_dim": int(self.bond_dim),
            "dtype": str(self.dtype),
            "parameter_count": int(self.parameter_count()),
            "config": _json_safe(self.config),
            "health": self.health_metrics(),
            "reservoir_snapshot": self.reservoir.snapshot() if hasattr(self.reservoir, "snapshot") else {},
        }


# =============================================================================
# Quantum CfC policy
# =============================================================================

class QuantumCfC_Policy:
    """Multi-cell AtomTN-backed CfC policy."""

    def __init__(
        self,
        policy_id: str,
        input_size: int,
        hidden_size: int,
        num_cells: int,
        bond_dim: int = 2,
        rng: Optional[np.random.Generator] = None,
        dtype: np.dtype = np.float32,
        verbose: bool = True,
        *,
        dt: float = 0.05,
        damping: float = 0.1,
        noncommutative: bool = False,
        config: Optional[QuantumCfCConfig] = None,
        **kwargs: Any,
    ):
        require_atomtn()

        self.policy_id = str(policy_id)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_cells = int(num_cells)
        self.bond_dim = int(max(1, bond_dim))
        self.dtype = np.dtype(dtype)
        self.verbose = bool(verbose)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(int(kwargs.get("seed", 42)))

        if self.input_size <= 0 or self.hidden_size <= 0 or self.num_cells <= 0:
            raise ValueError("input_size, hidden_size, and num_cells must be positive")

        base_cfg = config or QuantumCfCConfig()
        base_cfg.reservoir_dt = float(kwargs.get("reservoir_dt", dt if dt is not None else base_cfg.reservoir_dt))
        base_cfg.memory_damping = float(kwargs.get("memory_damping", damping if damping is not None else base_cfg.memory_damping))
        base_cfg.noncommutative = bool(kwargs.get("noncommutative", noncommutative))
        for key, val in kwargs.items():
            if hasattr(base_cfg, key):
                setattr(base_cfg, key, val)
        self.config = base_cfg.normalized()

        self.cells: List[QuantumCfC_Cell] = []
        for i in range(self.num_cells):
            cell_seed = int(self.rng.integers(0, 2**31 - 1))
            cell_cfg = self.config.normalized()
            cell_cfg.seed = int((self.config.seed + 1009 * i + cell_seed) % (2**31 - 1))
            self.cells.append(
                QuantumCfC_Cell(
                    self.input_size,
                    self.hidden_size,
                    bond_dim=self.bond_dim,
                    rng=self.rng,
                    dtype=self.dtype,
                    dt=float(self.config.reservoir_dt),
                    damping=float(self.config.memory_damping),
                    noncommutative=bool(self.config.noncommutative),
                    config=cell_cfg,
                    cell_index=i,
                    seed=cell_cfg.seed,
                )
            )

        self.state = np.zeros((self.hidden_size * self.num_cells,), dtype=self.dtype)
        self._step_count = 0
        self._last_input = np.zeros((self.input_size,), dtype=self.dtype)
        self._last_time_delta = 0.0
        self._last_cell_aux: List[Dict[str, Any]] = []
        self._last_error: Optional[str] = None

        if self.verbose:
            print(
                f"[{self.policy_id}] QuantumCfC ready: input={self.input_size} hidden={self.hidden_size} "
                f"cells={self.num_cells} bond={self.bond_dim} params={self.parameter_count():,}"
            )

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def reset_state(self, value: float = 0.0) -> None:
        v = self.dtype.type(value)
        self.state.fill(v)
        if self.config.state_clip > 0:
            np.clip(self.state, -float(self.config.state_clip), float(self.config.state_clip), out=self.state)
        if bool(self.config.reset_reservoir_on_policy_reset):
            for cell in self.cells:
                cell.reset_reservoir()
        self._step_count = 0
        self._last_input.fill(0)
        self._last_time_delta = 0.0
        self._last_cell_aux = []
        self._last_error = None

    def get_state(self) -> np.ndarray:
        return self.state.astype(self.dtype, copy=True)

    def set_state(self, new_state: Any) -> None:
        self.state = _coerce_vector(new_state, expected_dim=self.state.size, dtype=self.dtype, name="new_state", resize=False, clip=self.config.state_clip)

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "kind": "QuantumCfC_Policy_state",
            "policy_id": self.policy_id,
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "num_cells": int(self.num_cells),
            "dtype": str(self.dtype),
            "state": self.state.astype(np.float32).tolist(),
            "step_count": int(self._step_count),
            "last_time_delta": float(self._last_time_delta),
            "cell_reservoirs": [c.reservoir.serialize_state() if hasattr(c.reservoir, "serialize_state") else {} for c in self.cells],
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if int(payload.get("hidden_size", self.hidden_size)) != self.hidden_size:
            raise ValueError("hidden_size mismatch in QuantumCfC state payload")
        if int(payload.get("num_cells", self.num_cells)) != self.num_cells:
            raise ValueError("num_cells mismatch in QuantumCfC state payload")
        if "state" not in payload:
            raise ValueError("payload missing 'state'")
        self.set_state(payload["state"])
        self._step_count = int(payload.get("step_count", self._step_count))
        self._last_time_delta = float(payload.get("last_time_delta", self._last_time_delta))
        reservoirs = payload.get("cell_reservoirs", [])
        if isinstance(reservoirs, Sequence) and not isinstance(reservoirs, (str, bytes)):
            for cell, rp in zip(self.cells, reservoirs):
                if isinstance(rp, Mapping) and hasattr(cell.reservoir, "load_state"):
                    cell.reservoir.load_state(rp)

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

        Diagnostic form:
            state, aux = step(x, time_delta, return_state=True)
        """
        wants_aux = bool(return_state if return_aux is None else return_aux)
        raw = _resolve_input(external_input)
        x = _coerce_vector(
            raw,
            expected_dim=self.input_size,
            dtype=self.dtype,
            name="external_input",
            resize=bool(self.config.resize_mismatched_input),
            clip=self.config.input_clip,
        )
        dt = _clamp_dt(time_delta)
        self._last_input = x.copy()
        self._last_time_delta = float(dt)
        self._last_error = None
        self._step_count += 1

        state_matrix = self.state.reshape(self.num_cells, self.hidden_size)
        new_states = np.empty_like(state_matrix)
        cell_aux: List[Dict[str, Any]] = []

        for i, cell in enumerate(self.cells):
            try:
                if wants_aux:
                    ns, aux = cell.forward(x, state_matrix[i], dt, return_aux=True)  # type: ignore[misc]
                    cell_aux.append(aux)
                else:
                    ns = cell.forward(x, state_matrix[i], dt, return_aux=False)  # type: ignore[assignment]
            except Exception as exc:
                self._last_error = f"cell_{i}_failed: {exc!r}"
                if bool(self.config.strict):
                    raise
                ns = state_matrix[i].copy()
                if wants_aux:
                    cell_aux.append({"cell_index": i, "error": self._last_error, "skipped": True})

            ns_arr = np.asarray(ns, dtype=self.dtype).reshape(-1)
            if ns_arr.size != self.hidden_size:
                raise RuntimeError(f"QuantumCfC cell {i} returned size {ns_arr.size}; expected {self.hidden_size}")
            new_states[i, :] = ns_arr

        self.state = new_states.reshape(-1).astype(self.dtype, copy=False)
        self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0).astype(self.dtype, copy=False)
        if self.config.state_clip > 0:
            np.clip(self.state, -float(self.config.state_clip), float(self.config.state_clip), out=self.state)

        self._last_cell_aux = cell_aux
        if self.config.strict_runtime_checks and _count_nonfinite(self.state) > 0:
            raise FloatingPointError(f"[{self.policy_id}] non-finite QuantumCfC policy state encountered")

        if wants_aux:
            aux_out = {
                "policy_id": self.policy_id,
                "hidden_state": self.state.copy(),
                "control_vector": self.get_control_vector(out_dim=min(32, self.hidden_size)),
                "cell_means": self.get_cell_means(),
                "cell_norms": self.get_norms(),
                "cell_aux": cell_aux,
                "runtime_flags": {
                    "backbone": "atomtn_quantum_reservoir",
                    "noncommutative": bool(self.config.noncommutative),
                    "strict_runtime_checks": bool(self.config.strict_runtime_checks),
                    "atomtn_status": atom_cfc_status(),
                },
                "health": self.health_metrics(),
                "error": self._last_error,
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
        return self.state.reshape(self.num_cells, self.hidden_size).mean(axis=1).astype(self.dtype, copy=False)

    def get_norms(self) -> np.ndarray:
        return np.linalg.norm(self.state.reshape(self.num_cells, self.hidden_size), axis=1).astype(self.dtype, copy=False)

    # ------------------------------------------------------------------
    # Diagnostics / maintenance
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        return int(sum(c.parameter_count() for c in self.cells))

    def total_parameter_count(self) -> int:
        return self.parameter_count()

    def equivalent_classical_head_parameter_count(self) -> int:
        per_cell = 3 * (self.hidden_size * 64 + self.hidden_size)
        return int(self.num_cells * per_cell)

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        for cell in self.cells:
            cell.apply_regularization(target_norm=target_norm)

    def health_metrics(self) -> Dict[str, Any]:
        state = self.state
        norms = self.get_norms()
        cell_health = [c.health_metrics() for c in self.cells]
        all_cells_stable = all(bool(h.get("is_stable", False)) for h in cell_health)
        metrics: Dict[str, Any] = {
            "kind": "QuantumCfC_Policy",
            "policy_id": self.policy_id,
            "step_count": int(self._step_count),
            "state_norm": float(np.linalg.norm(state)) if state.size else 0.0,
            "state_mean": float(np.mean(state)) if state.size else 0.0,
            "state_std": float(np.std(state)) if state.size else 0.0,
            "state_min": float(np.min(state)) if state.size else 0.0,
            "state_max": float(np.max(state)) if state.size else 0.0,
            "cell_norm_min": float(np.min(norms)) if norms.size else 0.0,
            "cell_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "cell_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "state_saturation_fraction": _saturation_fraction(state, threshold=max(0.5, 0.98 * self.config.state_clip)),
            "nonfinite_state_count": _count_nonfinite(state),
            "last_input_norm": _safe_norm(self._last_input),
            "last_time_delta": float(self._last_time_delta),
            "parameter_count": int(self.parameter_count()),
            "cell_health": cell_health,
            "last_error": self._last_error,
        }
        metrics["has_nan"] = bool(metrics["nonfinite_state_count"] > 0 or any(h.get("has_nan", False) for h in cell_health))
        metrics["is_saturated"] = bool(metrics["state_saturation_fraction"] >= 0.25)
        metrics["is_stable"] = bool((not metrics["has_nan"]) and all_cells_stable and self._last_error is None and metrics["state_norm"] < max(1.0, self.state.size * 1.25))
        return metrics

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "QuantumCfC_Policy",
            "policy_id": self.policy_id,
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "num_cells": int(self.num_cells),
            "bond_dim": int(self.bond_dim),
            "dtype": str(self.dtype),
            "parameter_count": int(self.parameter_count()),
            "state_norm": float(np.linalg.norm(self.state)),
            "cell_norms": self.get_norms().astype(float).tolist(),
            "step_count": int(self._step_count),
            "config": _json_safe(self.config),
            "health": self.health_metrics(),
            "cells": [c.snapshot() for c in self.cells],
        }

    def summary(self) -> str:
        return (
            f"[QuantumCfC Summary]\n"
            f"- id: {self.policy_id}\n"
            f"- input: {self.input_size}\n"
            f"- hidden: {self.hidden_size}\n"
            f"- cells: {self.num_cells}\n"
            f"- dtype: {self.dtype}\n"
            f"- backbone: AtomTN quantum reservoir\n"
            f"- params: {self.parameter_count():,}\n"
            f"- method: {self.config.evolution_method}\n"
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save classical heads and lightweight runtime state to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "format": "akkurat_atomtn_quantum_cfc_checkpoint_v1",
            "ts": _now_iso(),
            "snapshot": self.snapshot(),
            "state": self.serialize_state(),
            "cells": [],
        }
        for c in self.cells:
            payload["cells"].append(
                {
                    "W_f": c.W_f.astype(np.float32).tolist(),
                    "W_g": c.W_g.astype(np.float32).tolist(),
                    "W_h": c.W_h.astype(np.float32).tolist(),
                    "b_f": c.b_f.astype(np.float32).tolist(),
                    "b_g": c.b_g.astype(np.float32).tolist(),
                    "b_h": c.b_h.astype(np.float32).tolist(),
                }
            )
        p.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = payload.get("cells", [])
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise ValueError("checkpoint missing cells list")
        if len(cells) != len(self.cells):
            raise ValueError(f"checkpoint cell count mismatch: expected {len(self.cells)}, got {len(cells)}")
        for c, cp in zip(self.cells, cells):
            if not isinstance(cp, Mapping):
                raise ValueError("invalid cell checkpoint entry")
            for name in ("W_f", "W_g", "W_h", "b_f", "b_g", "b_h"):
                arr = np.asarray(cp[name], dtype=self.dtype)
                target = getattr(c, name)
                if arr.shape != target.shape:
                    raise ValueError(f"{name} shape mismatch: expected {target.shape}, got {arr.shape}")
                setattr(c, name, arr.astype(self.dtype, copy=False))
        state = payload.get("state", None)
        if isinstance(state, Mapping):
            self.load_state(state)

    def __repr__(self) -> str:
        return (
            f"QuantumCfC_Policy(id={self.policy_id!r}, input={self.input_size}, hidden={self.hidden_size}, "
            f"cells={self.num_cells}, backbone='AtomTN', method={self.config.evolution_method!r})"
        )


# Compatibility aliases.
QuantumCfCPolicy = QuantumCfC_Policy
AtomCfCPolicy = QuantumCfC_Policy
AtomCfC_Cell = QuantumCfC_Cell


# =============================================================================
# Demo / CLI
# =============================================================================


def run_quantum_cfc_demo(
    *,
    input_size: int = 3,
    hidden_size: int = 8,
    num_cells: int = 2,
    steps: int = 8,
    seed: int = 42,
    noncommutative: bool = False,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    policy = QuantumCfC_Policy(
        "demo_quantum_cfc",
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        num_cells=int(num_cells),
        bond_dim=2,
        rng=rng,
        verbose=True,
        noncommutative=bool(noncommutative),
        config=QuantumCfCConfig(seed=int(seed), noncommutative=bool(noncommutative), strict=True),
    )

    rows = []
    for t in range(int(steps)):
        x = np.array([math.sin(t / 3.0), math.cos(t / 3.0), 0.1 * math.sin(t)], dtype=np.float32)
        if input_size != 3:
            xx = np.zeros((int(input_size),), dtype=np.float32)
            xx[: min(3, int(input_size))] = x[: min(3, int(input_size))]
            x = xx
        state, aux = policy.step(x, time_delta=0.25, return_state=True)
        rows.append(
            {
                "step": int(t),
                "input": x.astype(float).tolist(),
                "state_norm": _safe_norm(state),
                "control_norm": _safe_norm(aux.get("control_vector", [])),
                "cell_norms": np.asarray(aux.get("cell_norms", []), dtype=float).tolist(),
            }
        )

    return {"ok": bool(policy.health_metrics().get("is_stable", False)), "snapshot": policy.snapshot(), "rows": rows}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AtomTN QuantumCfC policy wrapper")
    p.add_argument("--mode", choices=["status", "demo"], default="demo")
    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--hidden-size", type=int, default=8)
    p.add_argument("--num-cells", type=int, default=2)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noncommutative", action="store_true")
    p.add_argument("--output", type=str, default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.mode == "status":
        status = atom_cfc_status()
        print(json.dumps(_json_safe(status), indent=2, ensure_ascii=False))
        return 0 if status.get("atom_cfc_available", False) else 2

    report = run_quantum_cfc_demo(
        input_size=int(args.input_size),
        hidden_size=int(args.hidden_size),
        num_cells=int(args.num_cells),
        steps=int(args.steps),
        seed=int(args.seed),
        noncommutative=bool(args.noncommutative),
    )
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=False))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
