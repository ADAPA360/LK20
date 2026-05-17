#!/usr/bin/env python3
# atom_ncp.py
"""
AtomTN Quantum Neural Circuit Policy
====================================

Production-compatible QuantumNCP policy backed by the AtomTN neuromorphic
reservoir runtime.

Architecture
------------
1. A deterministic sparse encoder maps policy inputs into AtomTN graph-flow
   excitations.
2. The AtomTN reservoir evolves its TTN quantum state under a flow-induced
   Hamiltonian.
3. Holographic leaf observables are measured into a classical reservoir feature
   vector.
4. A small trainable NumPy readout maps reservoir features to action logits.

Compatibility
-------------
The public class name and constructor style are retained:

    QuantumNCP(policy_id, num_inputs, num_hidden, num_outputs, bond_dim=4, ...)

The step method preserves the legacy return convention:

    logits, aux = policy.step(x)

where aux contains hidden_state and freq_gates-compatible metadata.

This module intentionally delegates reservoir mechanics to neuromorphic.py so
there is one maintained AtomTN runtime path for Akkurat integration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

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


def atom_ncp_status() -> Dict[str, Any]:
    """Return import/runtime status without constructing a policy."""
    status = _atomtn_status()
    return {
        "atom_ncp_available": bool(_NEUROMORPHIC_OK and status.get("available", False)),
        "neuromorphic_import_ok": bool(_NEUROMORPHIC_OK),
        "neuromorphic_import_error": None if _NEUROMORPHIC_IMPORT_ERROR is None else repr(_NEUROMORPHIC_IMPORT_ERROR),
        "atomtn_status": status,
        "module_dir": str(_MODULE_DIR),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


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
        arr = np.asarray(x).reshape(-1)
        if arr.size == 0:
            return 0.0
        v = float(np.linalg.norm(arr))
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _count_nonfinite(x: Any) -> int:
    try:
        arr = np.asarray(x)
        return int(arr.size - np.count_nonzero(np.isfinite(arr)))
    except Exception:
        return 0


def _activation(x: np.ndarray, kind: str) -> np.ndarray:
    k = str(kind or "linear").lower().strip()
    if k == "tanh":
        return np.tanh(x).astype(x.dtype, copy=False)
    if k == "sigmoid":
        z = np.clip(x, -25.0, 25.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(x.dtype, copy=False)
    if k == "softsign":
        return (x / (1.0 + np.abs(x))).astype(x.dtype, copy=False)
    if k == "clip":
        return np.clip(x, -1.0, 1.0).astype(x.dtype, copy=False)
    return x.astype(x.dtype, copy=False)


def _activation_derivative_from_output(y: np.ndarray, kind: str) -> np.ndarray:
    k = str(kind or "linear").lower().strip()
    if k == "tanh":
        return (1.0 - y * y).astype(y.dtype, copy=False)
    if k == "sigmoid":
        return (y * (1.0 - y)).astype(y.dtype, copy=False)
    if k == "softsign":
        # Approximation using output y = x/(1+|x|): derivative = (1-|y|)^2.
        return np.maximum(0.0, 1.0 - np.abs(y)).astype(y.dtype, copy=False) ** 2
    if k == "clip":
        return (np.abs(y) < 1.0).astype(y.dtype, copy=False)
    return np.ones_like(y, dtype=y.dtype)


def _stable_readout_init(
    out_dim: int,
    in_dim: int,
    rng: np.random.Generator,
    dtype: np.dtype,
    *,
    scale: float = 0.1,
) -> np.ndarray:
    out_dim = int(out_dim)
    in_dim = int(in_dim)
    dt = np.dtype(dtype)
    if out_dim <= 0 or in_dim <= 0:
        return np.zeros((max(0, out_dim), max(0, in_dim)), dtype=dt)
    s = float(scale) / math.sqrt(max(1, in_dim))
    W = rng.normal(0.0, s, size=(out_dim, in_dim)).astype(np.float64)
    return W.astype(dt, copy=False)


def _resolve_input(payload: Any) -> Any:
    """Extract an input vector from common runtime payload formats."""
    if isinstance(payload, Mapping):
        for key in (
            "observation",
            "input",
            "inputs",
            "x",
            "features",
            "previous_output",
            "state",
        ):
            if key in payload:
                return payload[key]
        return []
    return payload


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QuantumNCPConfig:
    """Configuration for QuantumNCP beyond the legacy constructor arguments."""

    # Atom / geometry
    atom_name: str = "H"
    atom_level: int = 1
    noncommutative: bool = False
    fuzzy_l: int = 2
    seed: int = 42
    tree_mode: str = "balanced"
    tree_arity: int = 4

    # Fiber / TTN state
    fiber_d_uniform: int = 4
    fiber_d_min: int = 2
    fiber_d_max: int = 8

    # Vibration bath. Back-compatible naming mirrors neuromorphic.py.
    vibration_kind: str = "linear"
    vibration_strength: float = 0.1
    vibration_temperature: float = 10.0
    vibration_n: int = 32
    vibration_spectral_kind: str = "ohmic"
    vibration_alpha: float = 1.0
    vibration_omega_c: float = 10.0

    # Reservoir / encoder
    dt: float = 0.05
    memory_damping: float = 0.1
    encoder_scale: float = 1.0
    encoder_sparsity: float = 0.2
    encoder_normalize_input: bool = False
    encoder_output_clip: Optional[float] = 25.0
    encoder_input_clip: Optional[float] = 10.0
    injection_clip: Optional[float] = 25.0

    # Fast policy defaults: direct injection + Euler scaffold.
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

    # Readout feature extraction
    readout_observables: Tuple[str, ...] = ("Z",)
    readout_include_summary_stats: bool = False
    readout_normalize_features: bool = False
    readout_clip_value: Optional[float] = 10.0
    readout_divide_by_norm: bool = False

    # Decoder/readout learning
    readout_init_scale: float = 0.1
    readout_learning_rate: float = 0.01
    readout_l2: float = 0.0
    gradient_clip: Optional[float] = 10.0
    output_clip: Optional[float] = 50.0
    apply_activation_derivative: bool = False

    # Runtime behavior
    strict: bool = True
    cache_runtime_objects: bool = True
    resize_mismatched_input: bool = True
    return_tuple: bool = True

    def normalized(self) -> "QuantumNCPConfig":
        return QuantumNCPConfig(
            atom_name=str(self.atom_name),
            atom_level=int(max(1, self.atom_level)),
            noncommutative=bool(self.noncommutative),
            fuzzy_l=int(max(0, self.fuzzy_l)),
            seed=int(self.seed),
            tree_mode=str(self.tree_mode or "balanced"),
            tree_arity=int(max(2, self.tree_arity)),
            fiber_d_uniform=int(max(1, self.fiber_d_uniform)),
            fiber_d_min=int(max(1, self.fiber_d_min)),
            fiber_d_max=int(max(1, self.fiber_d_max)),
            vibration_kind=str(self.vibration_kind or "linear"),
            vibration_strength=float(max(_EPS, self.vibration_strength)),
            vibration_temperature=float(max(float(self.vibration_strength) + _EPS, self.vibration_temperature)),
            vibration_n=int(max(1, self.vibration_n)),
            vibration_spectral_kind=str(self.vibration_spectral_kind or "ohmic"),
            vibration_alpha=float(self.vibration_alpha),
            vibration_omega_c=float(max(_EPS, self.vibration_omega_c)),
            dt=float(max(0.0, self.dt)),
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
            readout_init_scale=float(max(0.0, self.readout_init_scale)),
            readout_learning_rate=float(max(0.0, self.readout_learning_rate)),
            readout_l2=float(max(0.0, self.readout_l2)),
            gradient_clip=None if self.gradient_clip is None else float(max(0.0, self.gradient_clip)),
            output_clip=None if self.output_clip is None else float(max(0.0, self.output_clip)),
            apply_activation_derivative=bool(self.apply_activation_derivative),
            strict=bool(self.strict),
            cache_runtime_objects=bool(self.cache_runtime_objects),
            resize_mismatched_input=bool(self.resize_mismatched_input),
            return_tuple=bool(self.return_tuple),
        )


# =============================================================================
# Main policy
# =============================================================================

class QuantumNCP:
    """
    Quantum Neural Circuit Policy backed by an AtomTN quantum reservoir.

    The quantum state is the recurrent hidden state. The trainable parameters are
    intentionally limited to the classical readout layer so this remains a stable
    drop-in reservoir policy for Akkurat control loops.
    """

    def __init__(
        self,
        policy_id: str,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        bond_dim: int = 4,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
        *,
        dt: float = 0.05,
        damping: float = 0.1,
        noncommutative: bool = False,
        readout_learning_rate: float = 0.01,
        output_activation: str = "linear",
        config: Optional[QuantumNCPConfig] = None,
        dtype: np.dtype = np.float32,
        **kwargs: Any,
    ):
        require_atomtn()

        self.policy_id = str(policy_id)
        self.num_inputs = int(num_inputs)
        self.num_hidden = int(num_hidden)
        self.num_outputs = int(num_outputs)
        self.bond_dim = int(max(1, bond_dim))
        self.verbose = bool(verbose)
        self.dtype = np.dtype(dtype)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(int(kwargs.get("seed", 42)))

        if self.num_inputs <= 0 or self.num_outputs <= 0:
            raise ValueError("num_inputs and num_outputs must be positive")

        base_cfg = config or QuantumNCPConfig()
        # Legacy constructor args win unless the caller explicitly supplied a config
        # and also passed overrides through kwargs.
        base_cfg.dt = float(kwargs.get("reservoir_dt", dt if dt is not None else base_cfg.dt))
        base_cfg.memory_damping = float(kwargs.get("memory_damping", damping if damping is not None else base_cfg.memory_damping))
        base_cfg.noncommutative = bool(kwargs.get("noncommutative", noncommutative))
        base_cfg.readout_learning_rate = float(kwargs.get("readout_learning_rate", readout_learning_rate))

        # Optional keyword overrides used by experiments and bridge scripts.
        for key, val in kwargs.items():
            if hasattr(base_cfg, key):
                setattr(base_cfg, key, val)

        self.config = base_cfg.normalized()
        self.dt = float(self.config.dt)
        self.damping = float(self.config.memory_damping)
        self.noncommutative = bool(self.config.noncommutative)
        self.lr = float(self.config.readout_learning_rate)
        self.activation = str(output_activation or kwargs.get("output_activation", "linear")).lower().strip()

        # Build the shared reservoir runtime.
        readout_cfg = HolographicReadoutConfig(
            observables=tuple(self.config.readout_observables),
            include_summary_stats=bool(self.config.readout_include_summary_stats),
            normalize_features=bool(self.config.readout_normalize_features),
            clip_value=self.config.readout_clip_value,
            divide_by_norm=bool(self.config.readout_divide_by_norm),
        )
        reservoir_cfg = QuantumReservoirConfig(
            input_dim=int(self.num_inputs),
            encoder_seed=int(self.config.seed),
            encoder_sparsity=float(self.config.encoder_sparsity),
            encoder_scale=float(self.config.encoder_scale),
            encoder_normalize_input=bool(self.config.encoder_normalize_input),
            encoder_output_clip=self.config.encoder_output_clip,
            encoder_input_clip=self.config.encoder_input_clip,
            encoder_resize_mismatched_input=bool(self.config.resize_mismatched_input),
            memory_damping=float(self.config.memory_damping),
            injection_clip=self.config.injection_clip,
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
        self.reservoir: QuantumReservoir = build_quantum_reservoir(build_cfg)
        self.atom = self.reservoir.atom
        self.fiber = self.reservoir.fiber

        # Determine feature size from the initialized reservoir state.
        initial_features = HolographicReadout.measure(self.atom.state, readout_cfg)
        if int(initial_features.size) <= 0:
            initial_features = np.zeros((len(getattr(self.atom.state.tree, "leaves", [])) or 64,), dtype=np.float32)
        self.reservoir_feature_size = int(initial_features.size)
        self._last_reservoir_state = _coerce_vector(initial_features, dtype=self.dtype, name="initial_features", resize=True)

        self.W_out = _stable_readout_init(
            self.num_outputs,
            self.reservoir_feature_size,
            self.rng,
            self.dtype,
            scale=float(self.config.readout_init_scale),
        )
        self.b_out = np.zeros((self.num_outputs,), dtype=self.dtype)

        self.step_counter = 0
        self.last_input = np.zeros((self.num_inputs,), dtype=self.dtype)
        self.last_features = self._last_reservoir_state.copy()
        self.last_logits = np.zeros((self.num_outputs,), dtype=self.dtype)
        self.last_output = np.zeros((self.num_outputs,), dtype=self.dtype)
        self.last_aux: Dict[str, Any] = {}
        self.last_error: Optional[str] = None

        if self.verbose:
            print(
                f"[{self.policy_id}] QuantumNCP ready: input={self.num_inputs} outputs={self.num_outputs} "
                f"features={self.reservoir_feature_size} method={self.config.evolution_method} "
                f"params={self.parameter_count():,}"
            )

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------

    def reset_state(self, value: float = 0.0) -> None:
        """Reset quantum reservoir memory and cached classical readout state."""
        del value  # retained for API compatibility; quantum reset returns baseline state.
        self.reservoir.reset_state()
        features = HolographicReadout.measure(self.atom.state, self.reservoir.config.readout)
        self.last_features = self._features_to_readout_dim(features)
        self._last_reservoir_state = self.last_features.copy()
        self.last_input.fill(0)
        self.last_logits.fill(0)
        self.last_output.fill(0)
        self.last_aux = {}
        self.last_error = None
        self.step_counter = 0

    def get_state(self) -> np.ndarray:
        """Return the current classical reservoir feature vector."""
        if self.last_features.size:
            return self.last_features.astype(self.dtype, copy=True)
        features = HolographicReadout.measure(self.atom.state, self.reservoir.config.readout)
        return self._features_to_readout_dim(features).astype(self.dtype, copy=True)

    def set_state(self, new_state: Any) -> None:
        """
        Set the policy's cached classical reservoir features.

        This does not overwrite the underlying TTN wavefunction. It is provided
        for compatibility with generic recurrent-policy interfaces that cache and
        restore a flat hidden vector.
        """
        self.last_features = self._features_to_readout_dim(new_state)
        self._last_reservoir_state = self.last_features.copy()

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "kind": "QuantumNCP_state",
            "policy_id": self.policy_id,
            "num_inputs": int(self.num_inputs),
            "num_hidden": int(self.num_hidden),
            "num_outputs": int(self.num_outputs),
            "reservoir_feature_size": int(self.reservoir_feature_size),
            "step_counter": int(self.step_counter),
            "last_input": self.last_input.astype(float).tolist(),
            "last_features": self.last_features.astype(float).tolist(),
            "last_logits": self.last_logits.astype(float).tolist(),
            "last_output": self.last_output.astype(float).tolist(),
            "reservoir": self.reservoir.serialize_state(),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if int(payload.get("num_inputs", self.num_inputs)) != self.num_inputs:
            raise ValueError("num_inputs mismatch in QuantumNCP state payload")
        if int(payload.get("num_outputs", self.num_outputs)) != self.num_outputs:
            raise ValueError("num_outputs mismatch in QuantumNCP state payload")
        if "last_input" in payload:
            self.last_input = _coerce_vector(payload["last_input"], expected_dim=self.num_inputs, dtype=self.dtype, resize=True, name="last_input")
        if "last_features" in payload:
            self.last_features = self._features_to_readout_dim(payload["last_features"])
            self._last_reservoir_state = self.last_features.copy()
        if "last_logits" in payload:
            self.last_logits = _coerce_vector(payload["last_logits"], expected_dim=self.num_outputs, dtype=self.dtype, resize=True, name="last_logits")
        if "last_output" in payload:
            self.last_output = _coerce_vector(payload["last_output"], expected_dim=self.num_outputs, dtype=self.dtype, resize=True, name="last_output")
        self.step_counter = int(payload.get("step_counter", self.step_counter))
        reservoir_payload = payload.get("reservoir", None)
        if isinstance(reservoir_payload, Mapping):
            self.reservoir.load_state(reservoir_payload)

    # ------------------------------------------------------------------
    # Core policy step
    # ------------------------------------------------------------------

    def _features_to_readout_dim(self, features: Any) -> np.ndarray:
        arr = _coerce_vector(features, dtype=self.dtype, name="reservoir_features", resize=False)
        if arr.size == self.reservoir_feature_size:
            return arr.astype(self.dtype, copy=False)
        out = np.zeros((self.reservoir_feature_size,), dtype=self.dtype)
        n = min(int(arr.size), int(self.reservoir_feature_size))
        if n > 0:
            out[:n] = arr[:n]
        return out

    def _effective_dt(self, dt: Optional[float]) -> float:
        # Legacy QuantumNCP callers often pass dt=1.0 as a generic recurrent
        # tick. Preserve old stable behavior by interpreting None/1.0 as the
        # configured reservoir dt.
        if dt is None:
            return float(self.dt)
        d = _finite_float(dt, self.dt)
        if abs(d - 1.0) <= 1e-12:
            return float(self.dt)
        return float(max(0.0, d))

    def decode(self, reservoir_state: Any) -> np.ndarray:
        features = self._features_to_readout_dim(reservoir_state)
        logits = (self.W_out @ features + self.b_out).astype(self.dtype, copy=False)
        if self.config.output_clip is not None and float(self.config.output_clip) > 0:
            np.clip(logits, -float(self.config.output_clip), float(self.config.output_clip), out=logits)
        out = _activation(logits, self.activation).astype(self.dtype, copy=False)
        return out

    def step(
        self,
        inputs: Union[np.ndarray, Mapping[str, Any], Sequence[float]],
        dt: Optional[float] = None,
        return_state: bool = False,
        *,
        return_aux: Optional[bool] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Advance the policy by one quantum-reservoir tick.

        By default this preserves the legacy return convention `(logits, aux)`.
        Set `config.return_tuple=False` or `return_aux=False` for array-only use.
        """
        wants_aux = bool(self.config.return_tuple if return_aux is None else return_aux)
        self.last_error = None

        raw = _resolve_input(inputs)
        x = _coerce_vector(
            raw,
            expected_dim=self.num_inputs,
            dtype=self.dtype,
            name="QuantumNCP input",
            resize=bool(self.config.resize_mismatched_input),
            clip=self.config.encoder_input_clip,
        )
        self.last_input = x.copy()

        dt_eff = self._effective_dt(dt)
        try:
            features = self.reservoir.step(x, dt=dt_eff)
            features = self._features_to_readout_dim(features)
            self.last_features = features.copy()
            self._last_reservoir_state = features.copy()

            logits = (self.W_out @ features + self.b_out).astype(self.dtype, copy=False)
            logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).astype(self.dtype, copy=False)
            if self.config.output_clip is not None and float(self.config.output_clip) > 0:
                np.clip(logits, -float(self.config.output_clip), float(self.config.output_clip), out=logits)
            output = _activation(logits, self.activation).astype(self.dtype, copy=False)

            self.last_logits = logits.copy()
            self.last_output = output.copy()
            self.step_counter += 1

            aux: Dict[str, Any] = {
                "policy_id": self.policy_id,
                "step": int(self.step_counter),
                "dt": float(dt_eff),
                "hidden_state": features.copy() if return_state or wants_aux else None,
                "reservoir_state": features.copy(),
                "quantum_norm_squared": float(self.reservoir.quantum_norm_squared()),
                "flow_energy": float(self.reservoir.flow_energy()),
                "readout_norm": _safe_norm(features),
                "logits": logits.copy(),
                # Legacy compatibility: older control code expected this key.
                "freq_gates": np.full((10,), 0.5, dtype=self.dtype),
                "health": self.health_metrics(),
                "reservoir_snapshot": self.reservoir.snapshot(),
            }
            self.last_aux = aux
            return (output.copy(), aux) if wants_aux else output.copy()

        except Exception as exc:
            self.last_error = repr(exc)
            if self.config.strict:
                raise
            # Non-strict degradation: decode cached state.
            output = self.decode(self.last_features)
            aux = {
                "policy_id": self.policy_id,
                "step": int(self.step_counter),
                "dt": float(dt_eff),
                "hidden_state": self.last_features.copy(),
                "reservoir_state": self.last_features.copy(),
                "freq_gates": np.full((10,), 0.5, dtype=self.dtype),
                "error": self.last_error,
                "health": self.health_metrics(),
            }
            self.last_output = output.copy()
            self.last_aux = aux
            return (output.copy(), aux) if wants_aux else output.copy()

    # ------------------------------------------------------------------
    # Training / maintenance
    # ------------------------------------------------------------------

    def train(self, error_grad: Any, reservoir_state: Optional[Any] = None) -> Dict[str, Any]:
        """
        Apply a supervised update to the classical readout layer.

        error_grad is interpreted as dL/d(output). If config.apply_activation_derivative
        is true, it is converted to an approximate dL/d(logits) using the last output.
        """
        grad = _coerce_vector(error_grad, expected_dim=self.num_outputs, dtype=self.dtype, name="error_grad", resize=True)
        if self.config.apply_activation_derivative:
            grad = (grad * _activation_derivative_from_output(self.last_output.astype(self.dtype), self.activation)).astype(self.dtype)

        if self.config.gradient_clip is not None and float(self.config.gradient_clip) > 0:
            np.clip(grad, -float(self.config.gradient_clip), float(self.config.gradient_clip), out=grad)

        features = self._features_to_readout_dim(self.last_features if reservoir_state is None else reservoir_state)
        dW = np.outer(grad, features).astype(self.dtype, copy=False)
        db = grad.astype(self.dtype, copy=False)

        if self.config.readout_l2 > 0:
            dW = dW + self.dtype.type(self.config.readout_l2) * self.W_out

        lr = self.dtype.type(self.lr)
        self.W_out = (self.W_out - lr * dW).astype(self.dtype, copy=False)
        self.b_out = (self.b_out - lr * db).astype(self.dtype, copy=False)

        return {
            "policy_id": self.policy_id,
            "learning_rate": float(self.lr),
            "grad_norm": _safe_norm(grad),
            "feature_norm": _safe_norm(features),
            "dW_norm": _safe_norm(dW),
            "parameter_count": int(self.parameter_count()),
        }

    def parameter_count(self) -> int:
        return int(self.W_out.size + self.b_out.size)

    def total_parameter_count(self) -> int:
        # Trainable count only. Quantum reservoir has generated physics state,
        # not gradient-trained parameters in this policy wrapper.
        return self.parameter_count()

    def apply_regularization(self, target_norm: float = 1.0) -> None:
        target = float(target_norm)
        if target <= 0:
            raise ValueError("target_norm must be positive")
        n = float(np.linalg.norm(self.W_out))
        if math.isfinite(n) and n > target and n > 0:
            self.W_out *= self.dtype.type(target / n)

    # ------------------------------------------------------------------
    # Diagnostics / persistence
    # ------------------------------------------------------------------

    def health_metrics(self) -> Dict[str, Any]:
        reservoir_health = self.reservoir.health_metrics() if hasattr(self.reservoir, "health_metrics") else {}
        readout_finite = bool(np.all(np.isfinite(self.W_out)) and np.all(np.isfinite(self.b_out)))
        feature_finite = bool(self.last_features.size == 0 or np.all(np.isfinite(self.last_features)))
        output_finite = bool(self.last_output.size == 0 or np.all(np.isfinite(self.last_output)))
        stable = bool(
            readout_finite
            and feature_finite
            and output_finite
            and self.last_error is None
            and bool(reservoir_health.get("is_stable", True))
        )
        return {
            "kind": "QuantumNCP",
            "policy_id": self.policy_id,
            "is_stable": stable,
            "has_nan": bool(not readout_finite or not feature_finite or not output_finite or reservoir_health.get("has_nan", False)),
            "step_counter": int(self.step_counter),
            "readout_weight_norm": _safe_norm(self.W_out),
            "readout_bias_norm": _safe_norm(self.b_out),
            "feature_norm": _safe_norm(self.last_features),
            "output_norm": _safe_norm(self.last_output),
            "nonfinite_parameter_count": int(_count_nonfinite(self.W_out) + _count_nonfinite(self.b_out)),
            "reservoir": reservoir_health,
            "last_error": self.last_error,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "QuantumNCP",
            "policy_id": self.policy_id,
            "num_inputs": int(self.num_inputs),
            "num_hidden": int(self.num_hidden),
            "num_outputs": int(self.num_outputs),
            "reservoir_feature_size": int(self.reservoir_feature_size),
            "bond_dim": int(self.bond_dim),
            "dtype": str(self.dtype),
            "activation": self.activation,
            "trainable_parameter_count": int(self.parameter_count()),
            "step_counter": int(self.step_counter),
            "config": _json_safe(self.config),
            "health": self.health_metrics(),
            "reservoir_snapshot": self.reservoir.snapshot(),
        }

    def save_checkpoint(self, path: Union[str, os.PathLike[str]]) -> None:
        """Save decoder weights plus lightweight reservoir runtime metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "akkurat_atomtn_quantum_ncp_checkpoint_v1",
            "ts": _now_iso(),
            "snapshot": self.snapshot(),
            "state": self.serialize_state(),
            "W_out": self.W_out.astype(np.float32).tolist(),
            "b_out": self.b_out.astype(np.float32).tolist(),
        }
        p.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    def load_checkpoint(self, path: Union[str, os.PathLike[str]]) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        W = payload.get("W_out", None)
        b = payload.get("b_out", None)
        if W is not None:
            arr = np.asarray(W, dtype=self.dtype)
            if arr.shape != self.W_out.shape:
                raise ValueError(f"W_out shape mismatch: expected {self.W_out.shape}, got {arr.shape}")
            self.W_out = arr
        if b is not None:
            arr = np.asarray(b, dtype=self.dtype).reshape(-1)
            if arr.shape != self.b_out.shape:
                raise ValueError(f"b_out shape mismatch: expected {self.b_out.shape}, got {arr.shape}")
            self.b_out = arr
        state = payload.get("state", None)
        if isinstance(state, Mapping):
            self.load_state(state)

    def __repr__(self) -> str:
        return (
            f"QuantumNCP(id={self.policy_id!r}, input={self.num_inputs}, outputs={self.num_outputs}, "
            f"features={self.reservoir_feature_size}, method={self.config.evolution_method!r}, "
            f"readout_params={self.parameter_count()})"
        )


# =============================================================================
# Demo / CLI
# =============================================================================


def run_quantum_ncp_demo(
    *,
    input_size: int = 3,
    output_size: int = 2,
    steps: int = 12,
    seed: int = 42,
    noncommutative: bool = False,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    policy = QuantumNCP(
        "demo_quantum_ncp",
        num_inputs=int(input_size),
        num_hidden=64,
        num_outputs=int(output_size),
        bond_dim=4,
        rng=rng,
        verbose=True,
        noncommutative=bool(noncommutative),
        config=QuantumNCPConfig(seed=int(seed), noncommutative=bool(noncommutative), strict=True),
    )

    rows = []
    for t in range(int(steps)):
        x = np.array([math.sin(t / 3.0), math.cos(t / 3.0), 0.1 * math.sin(t)], dtype=np.float32)
        if input_size != 3:
            xx = np.zeros((int(input_size),), dtype=np.float32)
            xx[: min(3, int(input_size))] = x[: min(3, int(input_size))]
            x = xx
        y, aux = policy.step(x, return_state=True)
        rows.append(
            {
                "step": int(t),
                "input": x.astype(float).tolist(),
                "output": np.asarray(y).astype(float).tolist(),
                "feature_norm": float(aux.get("readout_norm", 0.0)),
                "quantum_norm_squared": float(aux.get("quantum_norm_squared", 0.0)),
                "flow_energy": float(aux.get("flow_energy", 0.0)),
            }
        )

    return {"ok": bool(policy.health_metrics().get("is_stable", False)), "snapshot": policy.snapshot(), "rows": rows}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AtomTN QuantumNCP policy wrapper")
    p.add_argument("--mode", choices=["status", "demo"], default="demo")
    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--output-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noncommutative", action="store_true")
    p.add_argument("--output", type=str, default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.mode == "status":
        print(json.dumps(_json_safe(atom_ncp_status()), indent=2, ensure_ascii=False))
        return 0 if atom_ncp_status().get("atom_ncp_available", False) else 2

    report = run_quantum_ncp_demo(
        input_size=int(args.input_size),
        output_size=int(args.output_size),
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
