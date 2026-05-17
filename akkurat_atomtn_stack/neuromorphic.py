#!/usr/bin/env python3
# neuromorphic.py
"""
AtomTN Neuromorphic Runtime
===========================

Holographic Liquid State Machine (HLSM) powered by AtomTN.

This module turns the AtomTN physics engine into a production-friendly reservoir
computing substrate that can be used standalone or through Akkurat's
atom_adapter_runtime.py bridge.

Production runtime policy
-------------------------
The AtomTN stack now exposes multiple execution profiles.  The default profile is
"fast" because the fully correct zip-up RK4 path is intentionally expensive on
CPU-only NumPy workloads.

Profiles:
  smoke     : initialization/readout/flow injection only; no quantum evolution.
  fast      : Euler legacy evolution using the scaffold apply path; default.
  balanced  : Euler legacy with slightly larger fibers/bonds and one flow relax step.
  accurate  : RK4 + zip-up apply; correctness-oriented, slow, opt-in only.

The expensive path is still available, but it is not used by default.  This
prevents demo/hybrid runs from silently exhausting RAM or appearing to hang.

Public API compatibility
------------------------
Preserves the earlier public names:
  - GeometryEncoder
  - HolographicReadout
  - QuantumReservoir
  - run_neuromorphic_demo()

Expected AtomTN files
---------------------
This file expects these AtomTN modules to be importable from the current
directory or PYTHONPATH:
  atom.py, flow.py, ttn_state.py, evolve.py, fiber.py, vibration.py,
  hamiltonian.py, geometry.py
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Optional AtomTN imports
# =============================================================================

_ATOMTN_OK = False
_ATOMTN_IMPORT_ERROR: Optional[BaseException] = None

try:
    from atom import Atom
    from flow import (
        GraphVectorField,
        GeodesicFlowConfig,
        GeodesicFlowSolver,
        NCGeodesicFlowSolver,
        FlowMonitor,
        FlowDiagnostics,
        GraphCalculus,
    )
    from ttn_state import TTNState
    from evolve import TTNEvolveConfig, TTNTimeEvolver, ApplyConfig
    from fiber import LocalFiberBuilder, LocalFiberConfig, AdinkraConstraint
    from vibration import VibrationModel
    from hamiltonian import HamiltonianBuildConfig, TreeMPOBuilder

    _ATOMTN_OK = True
except Exception as exc:  # pragma: no cover - depends on local AtomTN install
    _ATOMTN_OK = False
    _ATOMTN_IMPORT_ERROR = exc

    # Placeholders keep static tools/import-safe status mode working.
    Atom = Any  # type: ignore
    GraphVectorField = Any  # type: ignore
    GeodesicFlowConfig = Any  # type: ignore
    GeodesicFlowSolver = Any  # type: ignore
    NCGeodesicFlowSolver = Any  # type: ignore
    FlowMonitor = Any  # type: ignore
    FlowDiagnostics = Any  # type: ignore
    GraphCalculus = Any  # type: ignore
    TTNState = Any  # type: ignore
    TTNEvolveConfig = Any  # type: ignore
    TTNTimeEvolver = Any  # type: ignore
    ApplyConfig = Any  # type: ignore
    LocalFiberBuilder = Any  # type: ignore
    LocalFiberConfig = Any  # type: ignore
    AdinkraConstraint = Any  # type: ignore
    VibrationModel = Any  # type: ignore
    HamiltonianBuildConfig = Any  # type: ignore
    TreeMPOBuilder = Any  # type: ignore


# =============================================================================
# Status / requirements
# =============================================================================


def atomtn_status() -> Dict[str, Any]:
    """Return AtomTN import status without throwing."""
    return {
        "available": bool(_ATOMTN_OK),
        "import_error": None if _ATOMTN_IMPORT_ERROR is None else repr(_ATOMTN_IMPORT_ERROR),
        "python": sys.version.split()[0],
        "module": str(Path(__file__).resolve()) if "__file__" in globals() else "neuromorphic.py",
    }


def require_atomtn() -> None:
    """Raise a clear error if AtomTN modules are unavailable."""
    if not _ATOMTN_OK:
        raise RuntimeError(
            "AtomTN modules are not importable. Ensure atom.py, flow.py, "
            "ttn_state.py, evolve.py, fiber.py, vibration.py, and "
            f"hamiltonian.py are on PYTHONPATH. Original error: {_ATOMTN_IMPORT_ERROR!r}"
        )


# =============================================================================
# Generic helpers
# =============================================================================

_EPS = 1e-9


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _as_float32_vector(x: Any, *, expected_dim: Optional[int] = None, name: str = "vector") -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if expected_dim is not None and int(arr.size) != int(expected_dim):
        raise ValueError(f"{name} size mismatch: expected {expected_dim}, got {arr.size}")
    return arr


def _safe_norm(x: Any) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return 0.0
        val = float(np.linalg.norm(arr))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _l2_normalize(x: Any, eps: float = _EPS) -> np.ndarray:
    arr = _as_float32_vector(x)
    n = float(np.linalg.norm(arr)) if arr.size else 0.0
    if not np.isfinite(n) or n <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / n).astype(np.float32, copy=False)


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [{"re": float(np.real(v)), "im": float(np.imag(v))} for v in obj.reshape(-1)]
        return np.asarray(obj, dtype=float).tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.complexfloating):
        z = complex(obj.item())
        return {"re": float(z.real), "im": float(z.imag)}
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _extract_edge_scalar(value: Any) -> float:
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            v = float(value)
        else:
            v = float(np.linalg.norm(np.asarray(value)))
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _get_tree_leaf_count(state: Any) -> int:
    try:
        return int(len(list(getattr(getattr(state, "tree", None), "leaves", []))))
    except Exception:
        return 0


def _state_norm_squared(state: Any) -> float:
    try:
        fn = getattr(state, "amplitude_norm_squared", None)
        if callable(fn):
            val = float(fn())
            return val if np.isfinite(val) else 0.0
    except Exception:
        pass
    return 0.0


def _normalize_state_if_possible(state: Any) -> float:
    try:
        fn = getattr(state, "normalize_in_place", None)
        if callable(fn):
            fn()
    except Exception:
        pass
    return _state_norm_squared(state)


# =============================================================================
# Runtime profiles
# =============================================================================

ProfileName = str


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    evolution_method: str
    description: str
    flow_steps: int
    flow_diffusion: float
    flow_damping: float
    bond_dim: int
    fiber_d_uniform: int
    fiber_d_min: int
    fiber_d_max: int
    vibration_n: int
    apply_truncate_rank: int
    post_step_truncate_rank: int
    canonicalize_every: int
    renormalize_every: int
    strict: bool


RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "smoke": RuntimeProfile(
        name="smoke",
        evolution_method="none",
        description="Import/build/readout smoke mode; no quantum evolution.",
        flow_steps=0,
        flow_diffusion=0.0,
        flow_damping=0.0,
        bond_dim=2,
        fiber_d_uniform=2,
        fiber_d_min=2,
        fiber_d_max=2,
        vibration_n=4,
        apply_truncate_rank=2,
        post_step_truncate_rank=2,
        canonicalize_every=1,
        renormalize_every=1,
        strict=True,
    ),
    "fast": RuntimeProfile(
        name="fast",
        evolution_method="euler_legacy",
        description="Default CPU-safe reservoir mode using legacy scaffold apply.",
        flow_steps=0,
        flow_diffusion=0.0,
        flow_damping=0.0,
        bond_dim=2,
        fiber_d_uniform=2,
        fiber_d_min=2,
        fiber_d_max=2,
        vibration_n=8,
        apply_truncate_rank=2,
        post_step_truncate_rank=2,
        canonicalize_every=1,
        renormalize_every=1,
        strict=True,
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        evolution_method="euler_legacy",
        description="Moderate CPU profile with one scalar flow relax step.",
        flow_steps=1,
        flow_diffusion=0.005,
        flow_damping=0.01,
        bond_dim=3,
        fiber_d_uniform=3,
        fiber_d_min=2,
        fiber_d_max=4,
        vibration_n=16,
        apply_truncate_rank=4,
        post_step_truncate_rank=4,
        canonicalize_every=1,
        renormalize_every=1,
        strict=True,
    ),
    "accurate": RuntimeProfile(
        name="accurate",
        evolution_method="rk4_end_truncate",
        description="Correctness-oriented RK4+zip-up path. Slow; opt-in only.",
        flow_steps=1,
        flow_diffusion=0.01,
        flow_damping=0.02,
        bond_dim=4,
        fiber_d_uniform=4,
        fiber_d_min=2,
        fiber_d_max=6,
        vibration_n=32,
        apply_truncate_rank=6,
        post_step_truncate_rank=6,
        canonicalize_every=1,
        renormalize_every=1,
        strict=True,
    ),
}


def get_runtime_profile(profile: str) -> RuntimeProfile:
    key = str(profile or "fast").lower().strip()
    if key not in RUNTIME_PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choices={sorted(RUNTIME_PROFILES)}")
    return RUNTIME_PROFILES[key]


# =============================================================================
# 1. Encoder: classical vectors -> geometry
# =============================================================================

@dataclass
class GeometryEncoderConfig:
    input_dim: int
    num_edges: int
    seed: int = 0
    sparsity: float = 0.2
    scale: float = 1.0
    normalize_input: bool = False
    output_clip: Optional[float] = 25.0


@dataclass
class GeometryEncoder:
    """
    Maps input vectors to edge excitations on the AtomTN manifold.

    The projection is deterministic and sparse. Inputs are treated as energy /
    curvature injections that perturb graph-flow edge values.
    """

    input_dim: int
    num_edges: int
    seed: int = 0
    sparsity: float = 0.2
    scale: float = 1.0
    normalize_input: bool = False
    output_clip: Optional[float] = 25.0

    def __post_init__(self) -> None:
        self.input_dim = int(self.input_dim)
        self.num_edges = int(self.num_edges)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.num_edges <= 0:
            raise ValueError("num_edges must be positive.")

        self.seed = int(self.seed)
        self.sparsity = float(np.clip(self.sparsity, 0.0, 1.0))
        self.scale = float(self.scale)
        self.rng = np.random.default_rng(self.seed)

        self.W_in = self.rng.normal(0.0, 1.0, size=(self.num_edges, self.input_dim)).astype(np.float32)

        # sparsity means retained fraction; sparsity=0.2 keeps about 20%.
        keep_mask = self.rng.random((self.num_edges, self.input_dim)) <= self.sparsity
        self.W_in[~keep_mask] = 0.0

        denom = max(1.0, math.sqrt(float(self.input_dim)))
        self.W_in *= np.float32(self.scale / denom)

    @classmethod
    def from_config(cls, cfg: GeometryEncoderConfig) -> "GeometryEncoder":
        return cls(**asdict(cfg))

    def encode(self, x: Any) -> np.ndarray:
        x_flat = _as_float32_vector(x, expected_dim=self.input_dim, name="GeometryEncoder input")
        if self.normalize_input:
            x_flat = _l2_normalize(x_flat)

        excitation = (self.W_in @ x_flat).astype(np.float32, copy=False)
        excitation = np.nan_to_num(excitation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        if self.output_clip is not None and self.output_clip > 0:
            np.clip(excitation, -float(self.output_clip), float(self.output_clip), out=excitation)

        return excitation

    def snapshot(self) -> Dict[str, Any]:
        return {
            "input_dim": int(self.input_dim),
            "num_edges": int(self.num_edges),
            "seed": int(self.seed),
            "sparsity": float(self.sparsity),
            "scale": float(self.scale),
            "normalize_input": bool(self.normalize_input),
            "output_clip": None if self.output_clip is None else float(self.output_clip),
            "weight_norm": _safe_norm(self.W_in),
        }


# =============================================================================
# 2. Readout: quantum TTN state -> classical features
# =============================================================================

@dataclass
class HolographicReadoutConfig:
    observables: Tuple[str, ...] = ("Z",)
    include_summary_stats: bool = False
    normalize_features: bool = False
    clip_value: Optional[float] = 10.0
    divide_by_norm: bool = False


class HolographicReadout:
    """
    Extracts classical information from an AtomTN TTN state.

    By default, computes local Pauli-Z-like expectation values at leaves. X/Y/I
    fallbacks are provided when explicit local operators are not supplied.
    """

    @staticmethod
    def _default_local_operator(dim: int, op_key: str) -> np.ndarray:
        d = int(max(1, dim))
        key = str(op_key or "Z").upper()

        if key in {"I", "ID", "IDENTITY"}:
            return np.eye(d, dtype=np.complex128)

        if key == "Z":
            diag = np.ones(d, dtype=np.float64)
            diag[1::2] = -1.0
            return np.diag(diag).astype(np.complex128)

        if key == "X":
            A = np.zeros((d, d), dtype=np.complex128)
            for i in range(d - 1):
                A[i, i + 1] = 1.0
                A[i + 1, i] = 1.0
            if d == 1:
                A[0, 0] = 1.0
            return A

        if key == "Y":
            A = np.zeros((d, d), dtype=np.complex128)
            for i in range(d - 1):
                A[i, i + 1] = -1.0j
                A[i + 1, i] = 1.0j
            return A

        return HolographicReadout._default_local_operator(d, "Z")

    @staticmethod
    def measure_local_observables(
        state: Any,
        op_key: str = "Z",
        local_ops: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        *,
        divide_by_norm: bool = False,
    ) -> np.ndarray:
        """
        Return a vector of real expectation values, one per leaf.

        Uses TTNState.top_down_bond_envs() when available. This matches the
        environment-contraction readout pattern used by AtomTN.
        """
        if state is None:
            return np.zeros((0,), dtype=np.float32)

        try:
            E = state.top_down_bond_envs()
        except Exception:
            E = {}

        leaves = list(getattr(getattr(state, "tree", None), "leaves", []))
        tensors = getattr(state, "tensors", {})
        norm2 = max(_EPS, _state_norm_squared(state)) if divide_by_norm else 1.0

        features: List[float] = []

        for lid in leaves:
            try:
                T_arr = np.asarray(tensors[lid], dtype=np.complex128)
                if T_arr.ndim != 2:
                    features.append(0.0)
                    continue

                Env = E.get(lid, None) if isinstance(E, Mapping) else None
                if Env is None:
                    Env_arr = np.eye(T_arr.shape[1], dtype=np.complex128)
                else:
                    Env_arr = np.asarray(Env, dtype=np.complex128)

                if Env_arr.ndim != 2 or Env_arr.shape != (T_arr.shape[1], T_arr.shape[1]):
                    Env_arr = np.eye(T_arr.shape[1], dtype=np.complex128)

                if local_ops and lid in local_ops and op_key in local_ops[lid]:
                    A = np.asarray(local_ops[lid][op_key], dtype=np.complex128)
                else:
                    A = HolographicReadout._default_local_operator(T_arr.shape[0], op_key)

                if A.shape != (T_arr.shape[0], T_arr.shape[0]):
                    A = HolographicReadout._default_local_operator(T_arr.shape[0], op_key)

                AT = A @ T_arr
                T_dag_AT = T_arr.conj().T @ AT
                val = np.trace(T_dag_AT @ Env_arr)
                real_val = float(np.real(val)) / norm2
                if not np.isfinite(real_val):
                    real_val = 0.0
                features.append(real_val)

            except Exception:
                features.append(0.0)

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def measure(
        state: Any,
        cfg: Optional[HolographicReadoutConfig] = None,
        *,
        local_ops: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    ) -> np.ndarray:
        cfg = cfg or HolographicReadoutConfig()

        chunks: List[np.ndarray] = []
        for op in cfg.observables:
            chunks.append(
                HolographicReadout.measure_local_observables(
                    state,
                    op_key=str(op),
                    local_ops=local_ops,
                    divide_by_norm=bool(cfg.divide_by_norm),
                )
            )

        feat = np.concatenate(chunks, axis=0).astype(np.float32, copy=False) if chunks else np.zeros((0,), dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        if cfg.include_summary_stats:
            stats = np.array(
                [
                    float(np.mean(feat)) if feat.size else 0.0,
                    float(np.std(feat)) if feat.size else 0.0,
                    float(np.min(feat)) if feat.size else 0.0,
                    float(np.max(feat)) if feat.size else 0.0,
                    float(np.linalg.norm(feat)) if feat.size else 0.0,
                ],
                dtype=np.float32,
            )
            feat = np.concatenate([feat, stats], axis=0)

        if cfg.clip_value is not None and cfg.clip_value > 0:
            np.clip(feat, -float(cfg.clip_value), float(cfg.clip_value), out=feat)

        if cfg.normalize_features:
            feat = _l2_normalize(feat)

        return feat.astype(np.float32, copy=False)


# =============================================================================
# 3. Quantum reservoir config and runtime
# =============================================================================

@dataclass
class QuantumReservoirConfig:
    input_dim: int
    profile: str = "fast"

    encoder_seed: int = 0
    encoder_sparsity: float = 0.2
    encoder_scale: float = 5.0
    encoder_normalize_input: bool = False
    encoder_output_clip: Optional[float] = 25.0

    memory_damping: float = 0.1

    flow_dt: Optional[float] = None
    flow_steps: int = 0
    flow_damping: float = 0.0
    flow_diffusion: float = 0.0

    onsite_scale: float = 1.0
    edge_scale: float = 0.5
    vib_scale: float = 0.1

    evolution_method: str = "euler_legacy"  # none | euler_legacy | rk4_legacy | rk2_mid_truncate | rk4_end_truncate | lie_trotter
    apply_truncate_rank: int = 2
    apply_truncate_tol: Optional[float] = None
    canonicalize_every: int = 1
    apply_grouping: str = "lca_routed"
    post_step_truncate_rank: int = 2
    post_step_truncate_tol: Optional[float] = None
    renormalize_every: int = 1

    readout: HolographicReadoutConfig = field(default_factory=HolographicReadoutConfig)

    cache_runtime_objects: bool = True
    strict: bool = True

    @classmethod
    def for_profile(
        cls,
        *,
        input_dim: int,
        profile: str = "fast",
        encoder_seed: int = 0,
        encoder_scale: Optional[float] = None,
        memory_damping: float = 0.1,
        method_override: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> "QuantumReservoirConfig":
        rp = get_runtime_profile(profile)
        return cls(
            input_dim=int(input_dim),
            profile=rp.name,
            encoder_seed=int(encoder_seed),
            encoder_scale=float(encoder_scale if encoder_scale is not None else (5.0 if rp.name != "accurate" else 10.0)),
            memory_damping=float(memory_damping),
            flow_steps=int(rp.flow_steps),
            flow_damping=float(rp.flow_damping),
            flow_diffusion=float(rp.flow_diffusion),
            evolution_method=str(method_override or rp.evolution_method),
            apply_truncate_rank=int(rp.apply_truncate_rank),
            canonicalize_every=int(rp.canonicalize_every),
            post_step_truncate_rank=int(rp.post_step_truncate_rank),
            renormalize_every=int(rp.renormalize_every),
            strict=bool(rp.strict if strict is None else strict),
        )


class QuantumReservoir:
    """
    Stateful reservoir wrapping an AtomTN Atom.

    Maintains graph-flow memory, TTN quantum state memory, deterministic
    input-to-edge encoder, and holographic observable readout.
    """

    def __init__(
        self,
        atom: Any,
        input_dim: int,
        fiber: Any,
        encoder_scale: float = 1.0,
        memory_damping: float = 0.1,
        *,
        config: Optional[QuantumReservoirConfig] = None,
        readout_config: Optional[HolographicReadoutConfig] = None,
    ):
        require_atomtn()

        self.atom = atom
        self.fiber = fiber
        self.config = config or QuantumReservoirConfig(
            input_dim=int(input_dim),
            encoder_scale=float(encoder_scale),
            memory_damping=float(memory_damping),
            readout=readout_config or HolographicReadoutConfig(),
        )

        # Backward-compatible explicit constructor args win.
        self.config.input_dim = int(input_dim)
        self.config.encoder_scale = float(encoder_scale if encoder_scale is not None else self.config.encoder_scale)
        self.config.memory_damping = float(memory_damping if memory_damping is not None else self.config.memory_damping)
        if readout_config is not None:
            self.config.readout = readout_config

        self.input_dim = int(self.config.input_dim)
        self.damping = float(np.clip(self.config.memory_damping, 0.0, 1.0))

        self.edges = list(self.atom.calc.oriented_edges())
        if not self.edges:
            raise RuntimeError("AtomTN atom.calc.oriented_edges() returned no edges.")

        self.encoder = GeometryEncoder(
            input_dim=self.input_dim,
            num_edges=len(self.edges),
            seed=int(self.config.encoder_seed),
            sparsity=float(self.config.encoder_sparsity),
            scale=float(self.config.encoder_scale),
            normalize_input=bool(self.config.encoder_normalize_input),
            output_clip=self.config.encoder_output_clip,
        )

        self.current_flow = self.atom.build_initial_flow(scale=0.0)
        self.step_counter = 0
        self.last_input = np.zeros((self.input_dim,), dtype=np.float32)
        self.last_excitations = np.zeros((len(self.edges),), dtype=np.float32)
        self.last_readout = np.zeros((0,), dtype=np.float32)
        self.last_flow_energy = 0.0
        self.last_diagnostics: Dict[str, Any] = {}
        self.last_error: Optional[str] = None
        self.last_step_seconds = 0.0

        self._flow_solver = None
        self._monitor = None
        self._evolver = None
        self._builder = None

        self._init_runtime_objects()
        self._ensure_state_normalized(reinitialize_on_zero=True)

    # ------------------------------------------------------------------
    # Initialization / caches
    # ------------------------------------------------------------------

    def _init_runtime_objects(self) -> None:
        if not self.config.cache_runtime_objects:
            return

        try:
            if bool(getattr(self.atom, "noncommutative", False)):
                self._monitor = FlowMonitor(backend=self.atom.backend, num_nodes=64)
                self._flow_solver = NCGeodesicFlowSolver(self.atom.backend)
            else:
                self._monitor = FlowMonitor(calc=self.atom.calc, num_nodes=64)
                self._flow_solver = GeodesicFlowSolver(self.atom.calc)
        except Exception:
            self._monitor = None
            self._flow_solver = None

        try:
            self._evolver = TTNTimeEvolver(self.atom.calc)
        except Exception:
            self._evolver = None

        try:
            self._builder = TreeMPOBuilder(
                self.atom.calc,
                self._hamiltonian_config(),
                decomp=getattr(self.atom, "decomp", None),
                projection=getattr(self.atom, "projection", None),
                holonomy=getattr(self.atom, "holonomy", None),
            )
        except Exception:
            self._builder = None

    def _ensure_state_normalized(self, *, reinitialize_on_zero: bool = False) -> float:
        qn = _normalize_state_if_possible(getattr(self.atom, "state", None))
        if qn > _EPS:
            return qn

        if reinitialize_on_zero:
            try:
                # Keep the small bond dimension used by production profiles.
                bond_dim = int(max(1, getattr(self, "_init_bond_dim", self.config.post_step_truncate_rank)))
                self.atom.init_state(fiber=self.fiber, bond_dim=bond_dim)
                qn = _normalize_state_if_possible(getattr(self.atom, "state", None))
            except Exception:
                pass

        if qn <= _EPS and self.config.strict:
            raise RuntimeError("AtomTN state has zero or invalid norm after initialization.")
        return qn

    def _hamiltonian_config(self) -> Any:
        nc = bool(getattr(self.atom, "noncommutative", False))
        return HamiltonianBuildConfig(
            onsite_scale=float(self.config.onsite_scale),
            onsite_mode=("holographic_su2" if nc else "zfield"),
            edge_scale=float(self.config.edge_scale),
            edge_mode=("holonomy_su2" if nc else "zz"),
            vib_scale=float(self.config.vib_scale),
        )

    def _flow_config(self, dt: float) -> Any:
        flow_dt = float(self.config.flow_dt) if self.config.flow_dt is not None else float(dt)
        return GeodesicFlowConfig(
            dt=flow_dt,
            steps=int(max(0, self.config.flow_steps)),
            damping=float(self.config.flow_damping),
            diffusion=float(self.config.flow_diffusion),
        )

    def _apply_config(self) -> Any:
        return ApplyConfig(
            apply_truncate_rank=int(max(1, self.config.apply_truncate_rank)),
            apply_truncate_tol=self.config.apply_truncate_tol,
            canonicalize_every=int(max(1, self.config.canonicalize_every)),
            apply_grouping=str(self.config.apply_grouping),
        )

    def _evolve_config(self, dt: float) -> Any:
        return TTNEvolveConfig(
            dt=float(dt),
            steps=1,
            method=str(self.config.evolution_method),
            renormalize_every=int(max(1, self.config.renormalize_every)),
            apply_config=self._apply_config(),
            post_step_truncate_rank=int(max(1, self.config.post_step_truncate_rank)),
            post_step_truncate_tol=self.config.post_step_truncate_tol,
        )

    # ------------------------------------------------------------------
    # Flow update
    # ------------------------------------------------------------------

    def _current_edge_value(self, edge: Tuple[int, int], *, nc: bool) -> Any:
        try:
            return self.current_flow.edge_values.get(edge, 0.0)
        except Exception:
            if nc:
                k = int(getattr(getattr(self.atom, "backend", None), "k", 1))
                return np.zeros((k, k), dtype=np.complex128)
            return 0.0

    def _inject_geometry(self, excitations: np.ndarray) -> None:
        nc = bool(getattr(self.atom, "noncommutative", False))
        new_vals: Dict[Tuple[int, int], Any] = {}

        if not nc:
            for i, (u, v) in enumerate(self.edges):
                old_val = float(_extract_edge_scalar(self._current_edge_value((u, v), nc=False)))
                val = (1.0 - self.damping) * old_val + float(excitations[i])
                new_vals[(u, v)] = val
                new_vals[(v, u)] = -val

            self.current_flow = GraphVectorField(new_vals, matrix_valued=False)
            return

        k = int(getattr(getattr(self.atom, "backend", None), "k", 1))
        I = np.eye(k, dtype=np.complex128)
        for i, (u, v) in enumerate(self.edges):
            old_M = self._current_edge_value((u, v), nc=True)
            try:
                old_arr = np.asarray(old_M, dtype=np.complex128)
                if old_arr.shape != (k, k):
                    old_arr = np.zeros((k, k), dtype=np.complex128)
            except Exception:
                old_arr = np.zeros((k, k), dtype=np.complex128)

            injection = I * np.complex128(float(excitations[i]))
            new_M = (1.0 - self.damping) * old_arr + injection

            new_vals[(u, v)] = new_M
            new_vals[(v, u)] = -new_M

        try:
            backend = getattr(self.atom, "backend", None)
            if backend is not None and hasattr(backend, "project_twisted_reality_edge"):
                new_vals = backend.project_twisted_reality_edge(new_vals)
        except Exception:
            pass

        self.current_flow = GraphVectorField(new_vals, matrix_valued=True)

    def _relax_flow(self, dt: float) -> None:
        if int(self.config.flow_steps) <= 0:
            return

        try:
            cfg = self._flow_config(dt)
            if self._flow_solver is None:
                if bool(getattr(self.atom, "noncommutative", False)):
                    solver = NCGeodesicFlowSolver(self.atom.backend)
                else:
                    solver = GeodesicFlowSolver(self.atom.calc)
            else:
                solver = self._flow_solver

            for _ in range(int(max(1, self.config.flow_steps))):
                self.current_flow = solver.step(self.current_flow, cfg)
        except Exception as exc:
            self.last_error = f"flow_step_failed: {exc!r}"
            if self.config.strict:
                raise

    # ------------------------------------------------------------------
    # Quantum update
    # ------------------------------------------------------------------

    def _diagnostics(self, dt: float) -> Any:
        try:
            monitor = self._monitor
            if monitor is None:
                if bool(getattr(self.atom, "noncommutative", False)):
                    monitor = FlowMonitor(backend=self.atom.backend, num_nodes=64)
                else:
                    monitor = FlowMonitor(calc=self.atom.calc, num_nodes=64)

            diag = monitor.diagnostics(None, self.current_flow, float(dt))
            self.last_diagnostics = self._diagnostics_to_dict(diag)
            return diag
        except Exception as exc:
            self.last_error = f"diagnostics_failed: {exc!r}"
            self.last_diagnostics = {"error": repr(exc)}
            if self.config.strict:
                raise
            return None

    @staticmethod
    def _diagnostics_to_dict(diag: Any) -> Dict[str, Any]:
        if diag is None:
            return {}
        if isinstance(diag, Mapping):
            return {str(k): _json_safe(v) for k, v in diag.items()}
        out: Dict[str, Any] = {}
        for key in ("energy", "norm", "divergence", "curl", "curvature", "residual", "stable", "alarm_score", "divX_scalar"):
            if hasattr(diag, key):
                try:
                    v = getattr(diag, key)
                    if key == "divX_scalar":
                        arr = np.asarray(v, dtype=float).reshape(-1)
                        out[key] = {
                            "mean": float(np.mean(arr)) if arr.size else 0.0,
                            "max": float(np.max(arr)) if arr.size else 0.0,
                            "norm": float(np.linalg.norm(arr)) if arr.size else 0.0,
                        }
                    else:
                        out[key] = _json_safe(v)
                except Exception:
                    pass
        if not out and hasattr(diag, "__dict__"):
            out = _json_safe(vars(diag))
        return out if isinstance(out, dict) else {}

    def _builder_obj(self) -> Any:
        if self._builder is not None:
            return self._builder

        return TreeMPOBuilder(
            self.atom.calc,
            self._hamiltonian_config(),
            decomp=getattr(self.atom, "decomp", None),
            projection=getattr(self.atom, "projection", None),
            holonomy=getattr(self.atom, "holonomy", None),
        )

    def _evolve_state(self, dt: float) -> None:
        method = str(self.config.evolution_method or "none").lower().strip()
        if method in {"", "none", "off", "no_evolve", "readout_only"}:
            return

        diag = self._diagnostics(dt)
        builder = self._builder_obj()

        def build_op(s: Any, sid: int) -> Any:
            return builder.build(
                s,
                self.fiber,
                self.current_flow,
                getattr(self.atom, "vibration", None),
                diag,
                sid,
            )

        def build_split_ops(s: Any, sid: int) -> List[Any]:
            if hasattr(builder, "build_split_operators"):
                return builder.build_split_operators(
                    state=s,
                    fiber=self.fiber,
                    X=self.current_flow,
                    vib=getattr(self.atom, "vibration", None),
                    diag=diag,
                    step_id=sid,
                    grouping=str(self.config.apply_grouping),
                )
            return [build_op(s, sid)]

        evolver = self._evolver if self._evolver is not None else TTNTimeEvolver(self.atom.calc)
        apply_cfg = self._apply_config()
        sid = int(self.step_counter)

        try:
            if method in {"euler", "euler_legacy"} and hasattr(evolver, "step_euler_legacy"):
                self.atom.state = evolver.step_euler_legacy(self.atom.state, build_op, float(dt), sid, apply_cfg)

            elif method in {"rk4", "rk4_legacy"} and hasattr(evolver, "step_rk4_legacy"):
                self.atom.state = evolver.step_rk4_legacy(self.atom.state, build_op, float(dt), sid, apply_cfg)

            elif method == "rk2_mid_truncate" and hasattr(evolver, "step_rk2_mid_truncate"):
                self.atom.state = evolver.step_rk2_mid_truncate(self.atom.state, build_op, float(dt), sid, apply_cfg)

            elif method == "rk4_end_truncate" and hasattr(evolver, "step_rk4_end_truncate"):
                self.atom.state = evolver.step_rk4_end_truncate(self.atom.state, build_op, float(dt), sid, apply_cfg)

            elif method == "lie_trotter" and hasattr(evolver, "step_lie_trotter"):
                self.atom.state = evolver.step_lie_trotter(self.atom.state, build_split_ops, float(dt), sid, apply_cfg)

            elif hasattr(evolver, "step"):
                evo_cfg = self._evolve_config(dt)
                try:
                    self.atom.state = evolver.step(self.atom.state, build_op, float(dt), sid, evo_cfg)
                except TypeError:
                    self.atom.state = evolver.step(self.atom.state, build_op, float(dt), sid)
            else:
                raise RuntimeError(f"TTNTimeEvolver exposes no compatible method for evolution_method={method!r}.")

            if int(self.config.renormalize_every) > 0 and (self.step_counter + 1) % int(self.config.renormalize_every) == 0:
                _normalize_state_if_possible(self.atom.state)

        except Exception as exc:
            self.last_error = f"quantum_evolve_failed: {exc!r}"
            if self.config.strict:
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, x_in: Any, dt: float = 0.05) -> np.ndarray:
        """Advance the reservoir by one time step and return readout features."""
        t0 = time.perf_counter()
        self.last_error = None

        x_vec = _as_float32_vector(x_in, expected_dim=self.input_dim, name="QuantumReservoir input")
        self.last_input = x_vec.copy()

        dtv = float(dt)
        if not np.isfinite(dtv) or dtv < 0:
            dtv = 0.05

        try:
            excitations = self.encoder.encode(x_vec)
            self.last_excitations = excitations.copy()

            self._inject_geometry(excitations)
            self._relax_flow(dtv)
            self.last_flow_energy = self.flow_energy()

            if dtv > 0:
                self._evolve_state(dtv)

            # Ensure diagnostics exist even when evolution method is none.
            if not self.last_diagnostics:
                try:
                    self._diagnostics(dtv)
                except Exception:
                    pass

            features = HolographicReadout.measure(self.atom.state, self.config.readout)
            self.last_readout = features.astype(np.float32, copy=True)

            self.step_counter += 1
            self.last_step_seconds = float(time.perf_counter() - t0)
            return self.last_readout.copy()

        except Exception:
            self.last_step_seconds = float(time.perf_counter() - t0)
            if self.config.strict:
                raise
            if self.last_readout.size:
                return self.last_readout.copy()
            return np.zeros((_get_tree_leaf_count(getattr(self.atom, "state", None)),), dtype=np.float32)

    def step_sequence(self, X: Any, *, dt: Union[float, Sequence[float]] = 0.05) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"step_sequence expects shape (T,input_dim), got {arr.shape}.")
        if arr.shape[1] != self.input_dim:
            raise ValueError(f"step_sequence input dim mismatch: expected {self.input_dim}, got {arr.shape[1]}.")

        if isinstance(dt, Sequence) and not isinstance(dt, (str, bytes)):
            dts = [float(v) for v in dt]
            if len(dts) != arr.shape[0]:
                raise ValueError("dt sequence length must match number of timesteps.")
        else:
            dts = [float(dt)] * int(arr.shape[0])

        outs: List[np.ndarray] = []
        for t in range(arr.shape[0]):
            outs.append(self.step(arr[t], dt=dts[t]))
        if not outs:
            return np.zeros((0, 0), dtype=np.float32)

        max_dim = max(int(o.size) for o in outs)
        Y = np.zeros((len(outs), max_dim), dtype=np.float32)
        for i, o in enumerate(outs):
            Y[i, : o.size] = o
        return Y

    def reset_state(self) -> None:
        try:
            if hasattr(self.atom, "init_state"):
                self.atom.init_state(fiber=self.fiber, bond_dim=int(max(1, self.config.post_step_truncate_rank)))
                _normalize_state_if_possible(self.atom.state)
        except Exception:
            pass

        try:
            self.current_flow = self.atom.build_initial_flow(scale=0.0)
        except Exception:
            pass

        self.step_counter = 0
        self.last_input = np.zeros((self.input_dim,), dtype=np.float32)
        self.last_excitations = np.zeros((len(self.edges),), dtype=np.float32)
        self.last_readout = np.zeros((0,), dtype=np.float32)
        self.last_flow_energy = 0.0
        self.last_diagnostics = {}
        self.last_error = None
        self.last_step_seconds = 0.0

    def measure(self, op_key: str = "Z") -> np.ndarray:
        return HolographicReadout.measure_local_observables(self.atom.state, op_key=op_key)

    def observables(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for op in self.config.readout.observables:
            out[str(op).upper()] = self.measure(str(op))
        return out

    def flow_energy(self) -> float:
        try:
            vals = []
            for u, v in self.edges:
                vals.append(_extract_edge_scalar(self.current_flow.edge_values.get((u, v), 0.0)))
            if not vals:
                return 0.0
            arr = np.asarray(vals, dtype=np.float64)
            energy = float(np.mean(arr * arr))
            return energy if np.isfinite(energy) else 0.0
        except Exception:
            return 0.0

    def flow_frame(self) -> Dict[str, Any]:
        flow_vectors: List[Dict[str, Any]] = []
        try:
            for i, (u, v) in enumerate(self.edges):
                val = self.current_flow.edge_values.get((u, v), 0.0)
                mag = _extract_edge_scalar(val)
                flow_vectors.append({"edge_idx": int(i), "u": int(u), "v": int(v), "magnitude": float(mag)})
        except Exception:
            pass

        return {
            "type": "atomtn_flow_frame",
            "step": int(self.step_counter),
            "edge_count": int(len(self.edges)),
            "flow_energy": float(self.flow_energy()),
            "flow_vectors": flow_vectors,
            "ts": _now_iso(),
        }

    def get_digital_twin_frame(self) -> Dict[str, Any]:
        activities = HolographicReadout.measure(self.atom.state, self.config.readout)
        flow = self.flow_frame()

        return {
            "type": "neuromorphic_frame",
            "profile": str(self.config.profile),
            "evolution_method": str(self.config.evolution_method),
            "step": int(self.step_counter),
            "node_activities": activities.astype(np.float32).tolist(),
            "flow_vectors": flow.get("flow_vectors", []),
            "metrics": {
                "avg_activity": float(np.mean(activities)) if activities.size else 0.0,
                "complexity": float(np.std(activities)) if activities.size else 0.0,
                "activity_norm": _safe_norm(activities),
                "flow_energy": float(self.flow_energy()),
                "quantum_norm_squared": float(self.quantum_norm_squared()),
                "last_step_seconds": float(self.last_step_seconds),
                "stable": bool(self.health_metrics().get("is_stable", False)),
            },
            "diagnostics": self.last_diagnostics,
            "ts": _now_iso(),
        }

    def quantum_norm_squared(self) -> float:
        return _state_norm_squared(getattr(self.atom, "state", None))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "QuantumReservoir",
            "input_dim": int(self.input_dim),
            "profile": str(self.config.profile),
            "evolution_method": str(self.config.evolution_method),
            "num_edges": int(len(self.edges)),
            "step_counter": int(self.step_counter),
            "quantum_norm_squared": float(self.quantum_norm_squared()),
            "last_readout_dim": int(self.last_readout.size),
            "last_readout_norm": _safe_norm(self.last_readout),
            "last_flow_energy": float(self.last_flow_energy),
            "last_step_seconds": float(self.last_step_seconds),
            "noncommutative": bool(getattr(self.atom, "noncommutative", False)),
            "readout_observables": list(self.config.readout.observables),
            "last_error": self.last_error,
        }

    def health_metrics(self) -> Dict[str, Any]:
        qn = float(self.quantum_norm_squared())
        readout_finite = bool(self.last_readout.size == 0 or np.all(np.isfinite(self.last_readout)))
        flow_energy = float(self.flow_energy())
        stable = bool(
            np.isfinite(qn)
            and qn > 1e-12
            and qn <= 20.0
            and np.isfinite(flow_energy)
            and flow_energy < 1e8
            and readout_finite
            and self.last_error is None
        )
        return {
            "kind": "QuantumReservoir",
            "is_stable": stable,
            "has_nan": bool(not readout_finite or not np.isfinite(qn) or not np.isfinite(flow_energy)),
            "quantum_norm_squared": qn,
            "readout_norm": _safe_norm(self.last_readout),
            "readout_dim": int(self.last_readout.size),
            "flow_energy": flow_energy,
            "last_step_seconds": float(self.last_step_seconds),
            "step_counter": int(self.step_counter),
            "last_error": self.last_error,
        }

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "adapter_kind": "quantum_reservoir",
            "step_counter": int(self.step_counter),
            "last_input": self.last_input.astype(float).tolist(),
            "last_excitations": self.last_excitations.astype(float).tolist(),
            "last_readout": self.last_readout.astype(float).tolist(),
            "last_flow_energy": float(self.last_flow_energy),
            "last_step_seconds": float(self.last_step_seconds),
            "last_diagnostics": _json_safe(self.last_diagnostics),
            "config": _json_safe(self.config),
            "encoder": self.encoder.snapshot(),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        self.step_counter = int(payload.get("step_counter", self.step_counter))
        if "last_input" in payload:
            self.last_input = _as_float32_vector(payload["last_input"])
        if "last_excitations" in payload:
            self.last_excitations = _as_float32_vector(payload["last_excitations"])
        if "last_readout" in payload:
            self.last_readout = _as_float32_vector(payload["last_readout"])
        self.last_flow_energy = float(payload.get("last_flow_energy", self.last_flow_energy))
        self.last_step_seconds = float(payload.get("last_step_seconds", self.last_step_seconds))
        diag = payload.get("last_diagnostics", {})
        self.last_diagnostics = dict(diag) if isinstance(diag, Mapping) else {}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(self.serialize_state()), indent=2, ensure_ascii=False), encoding="utf-8")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_state(payload)


# =============================================================================
# Convenience builders
# =============================================================================

@dataclass
class ReservoirBuildConfig:
    atom_name: str = "H"
    atom_level: int = 1
    noncommutative: bool = False
    seed: int = 42
    tree_mode: str = "balanced"
    arity: int = 4

    vibration_kind: str = "linear"
    vibration_w_min: float = 0.1
    vibration_w_max: float = 10.0
    vibration_n: int = 8

    fiber_d_uniform: int = 2
    fiber_d_min: int = 2
    fiber_d_max: int = 2
    bond_dim: int = 2

    reservoir: QuantumReservoirConfig = field(default_factory=lambda: QuantumReservoirConfig(input_dim=3))

    @classmethod
    def for_profile(
        cls,
        *,
        input_dim: int,
        profile: str = "fast",
        seed: int = 42,
        encoder_scale: Optional[float] = None,
        memory_damping: float = 0.1,
        noncommutative: bool = False,
        method_override: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> "ReservoirBuildConfig":
        rp = get_runtime_profile(profile)
        reservoir_cfg = QuantumReservoirConfig.for_profile(
            input_dim=int(input_dim),
            profile=rp.name,
            encoder_seed=int(seed),
            encoder_scale=encoder_scale,
            memory_damping=float(memory_damping),
            method_override=method_override,
            strict=strict,
        )
        return cls(
            noncommutative=bool(noncommutative),
            seed=int(seed),
            vibration_n=int(rp.vibration_n),
            fiber_d_uniform=int(rp.fiber_d_uniform),
            fiber_d_min=int(rp.fiber_d_min),
            fiber_d_max=int(rp.fiber_d_max),
            bond_dim=int(rp.bond_dim),
            reservoir=reservoir_cfg,
        )


def build_quantum_reservoir(cfg: ReservoirBuildConfig) -> QuantumReservoir:
    require_atomtn()

    atom = Atom(str(cfg.atom_name), int(cfg.atom_level), noncommutative=bool(cfg.noncommutative), seed=int(cfg.seed))
    # Some atom.py variants accept arity; older ones may not.
    try:
        atom.setup(tree_mode=str(cfg.tree_mode), arity=int(cfg.arity))
    except TypeError:
        atom.setup(tree_mode=str(cfg.tree_mode))

    vib = VibrationModel.build(
        str(cfg.vibration_kind),
        float(cfg.vibration_w_min),
        float(cfg.vibration_w_max),
        n=int(cfg.vibration_n),
        spectral_kind="ohmic",
        alpha=1.0,
        omega_c=max(float(cfg.vibration_w_max), 1e-6),
        seed=int(cfg.seed),
    )
    atom.attach_vibration(vib)

    d_uniform = int(cfg.fiber_d_uniform)
    d_min = int(cfg.fiber_d_min)
    d_max = int(cfg.fiber_d_max)
    if bool(cfg.noncommutative) and getattr(atom, "backend", None) is not None:
        k = int(getattr(atom.backend, "k", d_uniform))
        d_uniform = min(d_uniform, k)
        d_min = min(d_min, d_uniform)
        d_max = min(d_max, k)

    fiber_cfg = LocalFiberConfig(
        d_uniform=d_uniform,
        d_min=d_min,
        d_max=d_max,
        adaptive_strength=0.0,
        vib_influence=0.0,
        seed=int(cfg.seed),
    )
    fiber = LocalFiberBuilder(
        fiber_cfg,
        adinkra=AdinkraConstraint(seed=int(cfg.seed)),
        projection=getattr(atom, "projection", None),
        include_pauli_fallback=True,
    )
    atom.init_state(fiber=fiber, bond_dim=int(cfg.bond_dim))
    _normalize_state_if_possible(atom.state)

    reservoir = QuantumReservoir(
        atom,
        input_dim=int(cfg.reservoir.input_dim),
        fiber=fiber,
        encoder_scale=float(cfg.reservoir.encoder_scale),
        memory_damping=float(cfg.reservoir.memory_damping),
        config=cfg.reservoir,
    )
    reservoir._init_bond_dim = int(cfg.bond_dim)  # type: ignore[attr-defined]
    return reservoir


# =============================================================================
# Demo / benchmark
# =============================================================================


def run_neuromorphic_demo(
    *,
    input_size: int = 3,
    time_steps: int = 25,
    dt: float = 0.05,
    seed: int = 42,
    encoder_scale: Optional[float] = None,
    memory_damping: float = 0.1,
    noncommutative: bool = False,
    profile: str = "fast",
    method: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    max_step_seconds_warn: float = 10.0,
) -> Dict[str, Any]:
    require_atomtn()

    rp = get_runtime_profile(profile)
    print("\n" + "=" * 60)
    print("ATOM-TN: HOLOGRAPHIC LIQUID STATE MACHINE (HLSM) DEMO")
    print("Turning a Physics Simulation into a Computational Brain")
    print("=" * 60 + "\n")

    print("[1] Initializing AtomTN Kernel...")
    print(f"    Profile: {rp.name} | method={method or rp.evolution_method}")
    print(f"    Profile note: {rp.description}")

    if rp.name == "accurate":
        print("    WARNING: accurate profile uses RK4 + zip-up apply and may be slow on CPU.")

    print("[1.1] Attaching Phonon Bath...")
    print("[1.2] Building Hilbert Space (Fiber)...")

    build_cfg = ReservoirBuildConfig.for_profile(
        input_dim=int(input_size),
        profile=rp.name,
        seed=int(seed),
        encoder_scale=encoder_scale,
        memory_damping=float(memory_damping),
        noncommutative=bool(noncommutative),
        method_override=method,
    )
    reservoir = build_quantum_reservoir(build_cfg)
    print(f"    State Initialized: {reservoir.quantum_norm_squared():.6f} norm")

    print(f"\n[2] Constructing Quantum Reservoir (Input Dim: {input_size})...")
    print(f"    Edges: {len(reservoir.edges)} | Readout leaves: {_get_tree_leaf_count(reservoir.atom.state)}")

    print("\n[3] Running Time Series Processing...")
    print("    Task: Process a Sine Wave and generate feature embeddings.")

    rng = np.random.default_rng(seed)
    t_space = np.linspace(0.0, 4.0 * np.pi, int(time_steps), dtype=np.float32)

    data_sin = np.sin(t_space).astype(np.float32)
    data_cos = np.cos(t_space).astype(np.float32)

    if input_size <= 1:
        inputs = data_sin.reshape(-1, 1)
    else:
        cols = [data_sin, data_cos]
        while len(cols) < input_size:
            cols.append(rng.normal(0.0, 0.1, size=int(time_steps)).astype(np.float32))
        inputs = np.stack(cols[:input_size], axis=1).astype(np.float32)

    activity_log: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    step_seconds: List[float] = []

    print(f"\n{'Step':<5} | {'Input (Sin)':<12} | {'Reservoir Mean <Z>':<20} | {'Complexity (Std <Z>)':<20} | {'Q Norm Sq.':<10} | {'sec':<8}")
    print("-" * 101)

    for t in range(int(time_steps)):
        x_t = inputs[t]
        features = reservoir.step(x_t, dt=float(dt))
        activity_log.append(features)
        step_seconds.append(float(reservoir.last_step_seconds))

        mean_activity = float(np.mean(features)) if features.size else 0.0
        complexity = float(np.std(features)) if features.size else 0.0
        qn = reservoir.quantum_norm_squared()

        rows.append(
            {
                "step": int(t),
                "input": x_t.astype(float).tolist(),
                "mean_activity": mean_activity,
                "complexity": complexity,
                "quantum_norm_squared": qn,
                "feature_norm": _safe_norm(features),
                "step_seconds": float(reservoir.last_step_seconds),
                "last_error": reservoir.last_error,
            }
        )

        print(
            f"{t:02d}    | {float(x_t[0]):<12.4f} | {mean_activity:<20.4f} | "
            f"{complexity:<20.4f} | {qn:<10.4f} | {reservoir.last_step_seconds:<8.3f}"
        )

        if float(max_step_seconds_warn) > 0 and reservoir.last_step_seconds > float(max_step_seconds_warn):
            print(
                f"    WARNING: step {t} took {reservoir.last_step_seconds:.2f}s. "
                "Use --profile fast or --profile smoke for CPU-safe testing."
            )

    print("\n[4] Analysis")
    activity_matrix = np.stack(activity_log, axis=0) if activity_log else np.zeros((0, 0), dtype=np.float32)

    mean_state_traj = np.mean(activity_matrix, axis=1) if activity_matrix.size else np.zeros((0,), dtype=np.float32)
    corr = 0.0
    if mean_state_traj.size == data_sin.size and mean_state_traj.size > 1:
        try:
            c = np.corrcoef(mean_state_traj, data_sin)[0, 1]
            corr = float(c) if np.isfinite(c) else 0.0
        except Exception:
            corr = 0.0

    print(f"    Reservoir State Shape: {activity_matrix.shape}")
    print(f"    Correlation with Input Signal: {corr:.4f}")
    print(f"    Mean step seconds: {float(np.mean(step_seconds)) if step_seconds else 0.0:.4f}")
    print(f"    Max step seconds: {float(np.max(step_seconds)) if step_seconds else 0.0:.4f}")

    if abs(corr) > 0.1:
        print("    >> SUCCESS: The Quantum State is tracking the Input Signal.")
    else:
        print("    >> NOTE: Low correlation. This can be normal in smoke/fast mode; increase encoder scale or use balanced profile.")

    report = {
        "ok": bool(reservoir.health_metrics().get("is_stable", False)),
        "profile": rp.name,
        "evolution_method": str(reservoir.config.evolution_method),
        "correlation_with_input": corr,
        "state_shape": list(activity_matrix.shape),
        "mean_step_seconds": float(np.mean(step_seconds)) if step_seconds else 0.0,
        "max_step_seconds": float(np.max(step_seconds)) if step_seconds else 0.0,
        "health": reservoir.health_metrics(),
        "snapshot": reservoir.snapshot(),
        "rows": rows,
        "frame": reservoir.get_digital_twin_frame(),
        "ts": _now_iso(),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[5] Report saved: {path}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    return report


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AtomTN Holographic Liquid State Machine runtime.")
    p.add_argument("--mode", choices=["demo", "status", "benchmark", "export-frame"], default="demo")
    p.add_argument("--profile", choices=sorted(RUNTIME_PROFILES.keys()), default="fast")
    p.add_argument(
        "--method",
        choices=["none", "euler_legacy", "rk4_legacy", "rk2_mid_truncate", "rk4_end_truncate", "lie_trotter"],
        default="",
        help="Override profile evolution method. Use rk4_end_truncate only for small accurate tests.",
    )
    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder-scale", type=float, default=None)
    p.add_argument("--memory-damping", type=float, default=0.1)
    p.add_argument("--noncommutative", action="store_true")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--max-step-seconds-warn", type=float, default=10.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "status":
        status = atomtn_status()
        status["profiles"] = {k: _json_safe(v) for k, v in RUNTIME_PROFILES.items()}
        print(json.dumps(_json_safe(status), indent=2, ensure_ascii=False))
        return 0 if _ATOMTN_OK else 2

    output = args.output or ("neuromorphic_report.json" if args.mode in {"benchmark", "export-frame"} else "")

    try:
        report = run_neuromorphic_demo(
            input_size=int(args.input_size),
            time_steps=int(args.steps),
            dt=float(args.dt),
            seed=int(args.seed),
            encoder_scale=args.encoder_scale,
            memory_damping=float(args.memory_damping),
            noncommutative=bool(args.noncommutative),
            profile=str(args.profile),
            method=(str(args.method) if args.method else None),
            output_path=(output if output else None),
            max_step_seconds_warn=float(args.max_step_seconds_warn),
        )

        if args.mode == "export-frame":
            frame_path = Path(output or "neuromorphic_report.json").with_name("neuromorphic_frame.json")
            frame_path.write_text(json.dumps(_json_safe(report.get("frame", {})), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[EXPORT] Frame saved: {frame_path}")

        return 0 if report.get("ok", False) else 2

    except Exception as exc:
        print(f"[RESULT] FAIL: neuromorphic runtime crashed: {exc!r}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
