#!/usr/bin/env python3
# atom_adapter_runtime.py
"""
Akkurat / AtomTN adapter runtime
================================

Production-oriented bridge between the AtomTN neuromorphic reservoir family and
Akkurat's governed digital twin kernel.

Design goals
------------
- Import-safe: importing this module must not build AtomTN objects or run heavy
  tensor contractions.
- Default-fast: when an AtomTN reservoir is requested, use CPU-safe defaults
  compatible with the current fast neuromorphic.py profile.
- Explicit degradation: if AtomTN or the digital twin cannot attach, preserve the
  error in attachment_errors instead of failing silently.
- Stable public API:
    - AtomAdapterConfig
    - AtomAdapterStepResult
    - AtomTNAdapter
    - AtomAdapterRuntime
    - build_adapter(...)
    - _build_atomtn_reservoir(...)
- Digital twin compatible: exports AtomTN quantum/flow/observable frames to the
  Akkurat DigitalTwinsBuilder taxonomy when enabled.

Typical use
-----------
    from atom_adapter_runtime import AtomAdapterConfig, build_adapter

    rt = build_adapter(AtomAdapterConfig(input_dim=3, enable_atomtn=True))
    result = rt.step([0.0, 1.0, 0.1])
    print(result.ok, result.features.shape, rt.summary())

The module intentionally avoids any PyTorch/autograd dependency.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Path bootstrap
# =============================================================================


def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _add_path(p: Union[str, os.PathLike, None]) -> Optional[Path]:
    if p is None:
        return None
    try:
        path = Path(p).expanduser().resolve()
    except Exception:
        return None

    if path.exists():
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
        return path

    return None


def configure_paths(
    *,
    atomtn_root: Optional[Union[str, os.PathLike]] = None,
    akkurat_root: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Optional[str]]:
    """
    Add likely AtomTN/Akkurat roots to sys.path without importing heavy modules.
    """
    here = _module_dir()

    candidates_atom = [
        atomtn_root,
        os.environ.get("ATOMTN_ROOT"),
        os.environ.get("AKKURAT_ATOMTN_ROOT"),
        here,
        here.parent / "AtomTN",
        here.parent.parent / "AtomTN",
        Path.cwd(),
    ]
    candidates_akk = [
        akkurat_root,
        os.environ.get("AKKURAT_ROOT"),
        os.environ.get("AKKURAT_COGNITIVE_ROOT"),
        here,
        Path.cwd(),
    ]

    found_atom = None
    for c in candidates_atom:
        p = _add_path(c)
        if p is not None and (p / "neuromorphic.py").exists():
            found_atom = p
            break

    found_akk = None
    for c in candidates_akk:
        p = _add_path(c)
        if p is not None and (p / "digital_twin_kernel.py").exists():
            found_akk = p
            break

    _add_path(here)

    return {
        "atomtn_root": None if found_atom is None else str(found_atom),
        "akkurat_root": None if found_akk is None else str(found_akk),
    }


_PATHS = configure_paths()


# =============================================================================
# Optional digital twin imports, exposed for compatibility
# =============================================================================

_DIGITAL_TWIN_IMPORT_ERROR: Optional[BaseException] = None

DigitalTwinsBuilder = None
TreeTensorNetwork = None
AkkuratInterface = None
Action = None

try:
    _dtk = importlib.import_module("digital_twin_kernel")
    DigitalTwinsBuilder = getattr(_dtk, "DigitalTwinsBuilder", None)
    TreeTensorNetwork = getattr(_dtk, "TreeTensorNetwork", None)
    AkkuratInterface = getattr(_dtk, "AkkuratInterface", None)
    Action = getattr(_dtk, "Action", None)
except BaseException as exc:  # pragma: no cover - depends on caller paths
    _DIGITAL_TWIN_IMPORT_ERROR = exc


# =============================================================================
# Generic helpers
# =============================================================================

_EPS = 1e-9


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_norm(x: Any) -> float:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return 0.0
        v = float(np.linalg.norm(a.reshape(-1)))
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _as_float32_vector(
    x: Any,
    *,
    expected_dim: Optional[int] = None,
    name: str = "vector",
    resize: bool = True,
) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.zeros((0,), dtype=np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if expected_dim is not None:
        d = int(expected_dim)
        if arr.size != d:
            if not resize:
                raise ValueError(f"{name} size mismatch: expected {d}, got {arr.size}")
            out = np.zeros((d,), dtype=np.float32)
            if arr.size:
                n = min(d, int(arr.size))
                out[:n] = arr[:n]
            arr = out

    return arr


def _l2_normalize(x: Any, eps: float = _EPS) -> np.ndarray:
    arr = _as_float32_vector(x)
    if arr.size == 0:
        return arr

    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n <= eps:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr / n).astype(np.float32, copy=False)


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)

    if isinstance(obj, np.generic):
        return _json_safe(obj.item())

    if isinstance(obj, np.ndarray):
        arr = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(float).tolist()

    if is_dataclass(obj):
        return _json_safe(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))

    return str(obj)


def _dataclass_kwargs(cls: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return only kwargs accepted by a dataclass constructor.
    """
    try:
        names = {f.name for f in dataclasses.fields(cls)}
    except Exception:
        return dict(values)

    return {k: v for k, v in values.items() if k in names}


def _stable_seed(seed: int, salt: str) -> int:
    h = 2166136261 ^ int(seed)
    for b in salt.encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


# =============================================================================
# Status helpers
# =============================================================================


def atomtn_status() -> Dict[str, Any]:
    """
    Return AtomTN availability without constructing a reservoir.
    """
    configure_paths()

    try:
        mod = importlib.import_module("neuromorphic")

        if hasattr(mod, "atomtn_status"):
            base = dict(mod.atomtn_status())  # type: ignore[attr-defined]
        else:
            base = {"available": True, "import_error": None}

        base.setdefault("module", getattr(mod, "__file__", ""))
        base.setdefault("python", platform.python_version())
        return base

    except BaseException as exc:
        return {
            "available": False,
            "import_error": repr(exc),
            "module": None,
            "python": platform.python_version(),
        }


def digital_twin_status() -> Dict[str, Any]:
    """
    Return digital_twin_kernel availability without constructing a twin.
    """
    global DigitalTwinsBuilder, TreeTensorNetwork, AkkuratInterface, Action, _DIGITAL_TWIN_IMPORT_ERROR

    if DigitalTwinsBuilder is not None:
        return {
            "available": True,
            "import_error": None,
            "module": getattr(sys.modules.get("digital_twin_kernel"), "__file__", ""),
        }

    try:
        mod = importlib.import_module("digital_twin_kernel")
        DigitalTwinsBuilder = getattr(mod, "DigitalTwinsBuilder", None)
        TreeTensorNetwork = getattr(mod, "TreeTensorNetwork", None)
        AkkuratInterface = getattr(mod, "AkkuratInterface", None)
        Action = getattr(mod, "Action", None)
        _DIGITAL_TWIN_IMPORT_ERROR = None

        return {
            "available": DigitalTwinsBuilder is not None,
            "import_error": None,
            "module": getattr(mod, "__file__", ""),
        }

    except BaseException as exc:
        _DIGITAL_TWIN_IMPORT_ERROR = exc
        return {
            "available": False,
            "import_error": repr(exc),
            "module": None,
        }


# =============================================================================
# Config and result containers
# =============================================================================


@dataclass
class AtomAdapterConfig:
    """
    Configuration for the AtomTN -> Akkurat adapter runtime.
    """

    adapter_id: str = "atomtn_adapter"

    # I/O sizes
    input_dim: int = 3
    feature_dim: int = 64
    output_dim: int = 64

    # Attachment flags
    enable_atomtn: bool = True
    enable_digital_twin: bool = False
    enable_governance_updates: bool = True

    # AtomTN reservoir profile
    profile: str = "fast"  # smoke | fast | balanced | accurate
    method: str = "euler_legacy"  # none | euler_legacy | rk4_legacy | rk2_mid_truncate | rk4_end_truncate | lie_trotter
    dt: float = 0.05
    seed: int = 2027

    encoder_scale: float = 20.0
    encoder_sparsity: float = 0.2
    encoder_normalize_input: bool = False
    encoder_output_clip: Optional[float] = 25.0
    memory_damping: float = 0.1
    noncommutative: bool = False

    bond_dim: int = 2
    fiber_d_uniform: int = 2
    fiber_d_min: int = 2
    fiber_d_max: int = 4

    vibration_kind: str = "linear"
    vibration_strength: float = 0.1
    vibration_temperature: float = 10.0
    vibration_n: int = 16

    # Evolution / apply safety
    flow_steps: int = 1
    flow_damping: float = 0.0
    flow_diffusion: float = 0.01

    onsite_scale: float = 1.0
    edge_scale: float = 0.5
    vib_scale: float = 0.1

    apply_truncate_rank: int = 4
    post_step_truncate_rank: int = 4
    canonicalize_every: int = 1
    renormalize_every: int = 1
    cache_runtime_objects: bool = True

    # Readout
    readout_observables: Tuple[str, ...] = ("Z",)
    readout_include_summary_stats: bool = False
    readout_normalize_features: bool = False
    readout_clip_value: Optional[float] = 10.0
    readout_divide_by_norm: bool = False

    # Digital twin
    digital_twin_vector_dim: int = 128
    digital_twin_sketch_dim: Optional[int] = 64
    digital_twin_history_capacity: int = 64
    digital_twin_use_tn_projection: bool = False
    digital_twin_latent_geometry: str = "euclidean"
    digital_twin_peer_id: str = "atomtn_adapter"

    # Fallback reservoir, used if AtomTN is unavailable or disabled
    fallback_leak: float = 0.15
    fallback_noise: float = 0.0
    fallback_normalize: bool = False

    # Runtime behavior
    strict: bool = False
    fail_fast: bool = False
    max_step_seconds_warn: float = 1.0

    atomtn_root: str = ""
    akkurat_root: str = ""

    def normalized(self) -> "AtomAdapterConfig":
        cfg = copy.deepcopy(self)

        cfg.input_dim = int(max(1, cfg.input_dim))
        cfg.feature_dim = int(max(1, cfg.feature_dim))
        cfg.output_dim = int(max(1, cfg.output_dim))

        cfg.seed = int(cfg.seed)
        cfg.profile = str(cfg.profile or "fast").lower().strip()
        cfg.method = str(cfg.method or "euler_legacy").lower().strip()

        cfg.dt = max(0.0, _safe_float(cfg.dt, 0.05))
        cfg.memory_damping = float(np.clip(_safe_float(cfg.memory_damping, 0.1), 0.0, 1.0))
        cfg.encoder_sparsity = float(np.clip(_safe_float(cfg.encoder_sparsity, 0.2), 0.0, 1.0))

        cfg.bond_dim = int(max(1, cfg.bond_dim))
        cfg.fiber_d_uniform = int(max(1, cfg.fiber_d_uniform))
        cfg.fiber_d_min = int(max(1, cfg.fiber_d_min))
        cfg.fiber_d_max = int(max(cfg.fiber_d_min, cfg.fiber_d_max))

        cfg.apply_truncate_rank = int(max(1, cfg.apply_truncate_rank))
        cfg.post_step_truncate_rank = int(max(1, cfg.post_step_truncate_rank))
        cfg.canonicalize_every = int(max(1, cfg.canonicalize_every))
        cfg.renormalize_every = int(max(1, cfg.renormalize_every))

        cfg.digital_twin_vector_dim = int(max(8, cfg.digital_twin_vector_dim))
        if cfg.digital_twin_sketch_dim is not None:
            cfg.digital_twin_sketch_dim = int(
                max(8, min(int(cfg.digital_twin_sketch_dim), cfg.digital_twin_vector_dim))
            )
        cfg.digital_twin_history_capacity = int(max(1, cfg.digital_twin_history_capacity))

        return cfg


@dataclass
class AtomAdapterStepResult:
    ok: bool
    step: int
    ts: str
    features: np.ndarray

    frame: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=dict)
    digital_twin: Dict[str, Any] = field(default_factory=dict)

    elapsed_s: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(self)


# =============================================================================
# Fallback deterministic reservoir
# =============================================================================


class _FallbackReservoir:
    """
    Small deterministic NumPy reservoir used when AtomTN is disabled/unavailable.
    """

    def __init__(self, cfg: AtomAdapterConfig):
        self.cfg = cfg.normalized()

        rng = np.random.default_rng(_stable_seed(self.cfg.seed, "fallback_reservoir"))
        self.W_in = rng.normal(
            0.0,
            1.0 / math.sqrt(self.cfg.input_dim),
            size=(self.cfg.feature_dim, self.cfg.input_dim),
        ).astype(np.float32)

        self.W_rec = rng.normal(
            0.0,
            1.0 / math.sqrt(self.cfg.feature_dim),
            size=(self.cfg.feature_dim, self.cfg.feature_dim),
        ).astype(np.float32)

        # Spectral-radius safety.
        try:
            s = np.linalg.svd(self.W_rec.astype(np.float64), compute_uv=False)[0]
            if s > 0:
                self.W_rec *= np.float32(0.75 / s)
        except Exception:
            pass

        self.state = np.zeros((self.cfg.feature_dim,), dtype=np.float32)
        self.step_counter = 0
        self.last_error: Optional[str] = None

    def step(self, x: Any, dt: float = 0.05) -> np.ndarray:
        xv = _as_float32_vector(x, expected_dim=self.cfg.input_dim, name="fallback input")

        leak = float(
            np.clip(
                self.cfg.fallback_leak * max(0.1, dt / max(self.cfg.dt, 1e-8)),
                0.0,
                1.0,
            )
        )

        drive = self.W_in @ xv + self.W_rec @ self.state
        z = np.tanh(drive).astype(np.float32)

        self.state = ((1.0 - leak) * self.state + leak * z).astype(np.float32)

        if self.cfg.fallback_noise > 0:
            rng = np.random.default_rng(_stable_seed(self.cfg.seed, f"fallback_noise::{self.step_counter}"))
            self.state += rng.normal(
                0.0,
                float(self.cfg.fallback_noise),
                size=self.state.shape,
            ).astype(np.float32)

        if self.cfg.fallback_normalize:
            self.state = _l2_normalize(self.state)

        self.step_counter += 1
        return self.state.copy()

    def get_digital_twin_frame(self) -> Dict[str, Any]:
        st = self.state.copy()

        return {
            "type": "fallback_reservoir_frame",
            "step": int(self.step_counter),
            "node_activities": st.astype(float).tolist(),
            "flow_vectors": [],
            "metrics": {
                "avg_activity": float(np.mean(st)) if st.size else 0.0,
                "complexity": float(np.std(st)) if st.size else 0.0,
                "activity_norm": _safe_norm(st),
                "flow_energy": 0.0,
                "quantum_norm_squared": 1.0,
                "stable": True,
            },
            "diagnostics": {"fallback": True},
            "ts": _now_iso(),
        }

    def health_metrics(self) -> Dict[str, Any]:
        return {
            "kind": "FallbackReservoir",
            "is_stable": bool(np.all(np.isfinite(self.state))),
            "has_nan": bool(not np.all(np.isfinite(self.state))),
            "quantum_norm_squared": 1.0,
            "readout_norm": _safe_norm(self.state),
            "readout_dim": int(self.state.size),
            "flow_energy": 0.0,
            "step_counter": int(self.step_counter),
            "last_error": self.last_error,
        }

    def reset_state(self) -> None:
        self.state.fill(0.0)
        self.step_counter = 0
        self.last_error = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "FallbackReservoir",
            "input_dim": int(self.cfg.input_dim),
            "feature_dim": int(self.cfg.feature_dim),
            "step_counter": int(self.step_counter),
            "last_readout_norm": _safe_norm(self.state),
        }


# =============================================================================
# AtomTN reservoir builder
# =============================================================================


def _profile_defaults(profile: str) -> Dict[str, Any]:
    p = str(profile or "fast").lower().strip()

    if p == "smoke":
        return {
            "method": "none",
            "bond_dim": 1,
            "fiber_d_uniform": 2,
            "fiber_d_min": 2,
            "fiber_d_max": 2,
            "vibration_n": 4,
            "apply_truncate_rank": 2,
            "post_step_truncate_rank": 2,
            "flow_steps": 0,
        }

    if p == "balanced":
        return {
            "method": "rk2_mid_truncate",
            "bond_dim": 3,
            "fiber_d_uniform": 3,
            "fiber_d_min": 2,
            "fiber_d_max": 4,
            "vibration_n": 16,
            "apply_truncate_rank": 6,
            "post_step_truncate_rank": 6,
            "flow_steps": 1,
        }

    if p == "accurate":
        return {
            "method": "rk4_end_truncate",
            "bond_dim": 4,
            "fiber_d_uniform": 4,
            "fiber_d_min": 2,
            "fiber_d_max": 8,
            "vibration_n": 32,
            "apply_truncate_rank": 8,
            "post_step_truncate_rank": 8,
            "flow_steps": 1,
        }

    # fast
    return {
        "method": "euler_legacy",
        "bond_dim": 2,
        "fiber_d_uniform": 2,
        "fiber_d_min": 2,
        "fiber_d_max": 4,
        "vibration_n": 16,
        "apply_truncate_rank": 4,
        "post_step_truncate_rank": 4,
        "flow_steps": 0,
    }


def _effective_config(cfg: AtomAdapterConfig) -> AtomAdapterConfig:
    out = cfg.normalized()
    defaults = _profile_defaults(out.profile)

    if out.method in ("", "auto", "profile") or (out.profile == "smoke" and out.method == "euler_legacy"):
        out.method = str(defaults["method"])

    # Keep explicit user values, but profile supplies conservative defaults.
    if cfg.bond_dim == AtomAdapterConfig.bond_dim:
        out.bond_dim = int(defaults["bond_dim"])
    if cfg.fiber_d_uniform == AtomAdapterConfig.fiber_d_uniform:
        out.fiber_d_uniform = int(defaults["fiber_d_uniform"])
    if cfg.fiber_d_min == AtomAdapterConfig.fiber_d_min:
        out.fiber_d_min = int(defaults["fiber_d_min"])
    if cfg.fiber_d_max == AtomAdapterConfig.fiber_d_max:
        out.fiber_d_max = int(defaults["fiber_d_max"])
    if cfg.vibration_n == AtomAdapterConfig.vibration_n:
        out.vibration_n = int(defaults["vibration_n"])
    if cfg.apply_truncate_rank == AtomAdapterConfig.apply_truncate_rank:
        out.apply_truncate_rank = int(defaults["apply_truncate_rank"])
    if cfg.post_step_truncate_rank == AtomAdapterConfig.post_step_truncate_rank:
        out.post_step_truncate_rank = int(defaults["post_step_truncate_rank"])
    if cfg.flow_steps == AtomAdapterConfig.flow_steps:
        out.flow_steps = int(defaults["flow_steps"])

    return out.normalized()


def _build_atomtn_reservoir(cfg: AtomAdapterConfig) -> Any:
    """
    Build a QuantumReservoir from neuromorphic.py using best-effort compatible kwargs.

    This function is intentionally lazy. It imports neuromorphic only when called.
    """
    cfg = _effective_config(cfg)

    if cfg.atomtn_root:
        configure_paths(atomtn_root=cfg.atomtn_root, akkurat_root=cfg.akkurat_root or None)
    else:
        configure_paths(akkurat_root=cfg.akkurat_root or None)

    neu = importlib.import_module("neuromorphic")

    if hasattr(neu, "require_atomtn"):
        neu.require_atomtn()  # type: ignore[attr-defined]

    HolographicReadoutConfig = getattr(neu, "HolographicReadoutConfig")
    QuantumReservoirConfig = getattr(neu, "QuantumReservoirConfig")
    ReservoirBuildConfig = getattr(neu, "ReservoirBuildConfig")
    build_quantum_reservoir = getattr(neu, "build_quantum_reservoir")

    readout_kwargs = _dataclass_kwargs(
        HolographicReadoutConfig,
        {
            "observables": tuple(cfg.readout_observables),
            "include_summary_stats": bool(cfg.readout_include_summary_stats),
            "normalize_features": bool(cfg.readout_normalize_features),
            "clip_value": cfg.readout_clip_value,
            "divide_by_norm": bool(cfg.readout_divide_by_norm),
        },
    )
    readout_cfg = HolographicReadoutConfig(**readout_kwargs)

    reservoir_kwargs = _dataclass_kwargs(
        QuantumReservoirConfig,
        {
            "input_dim": int(cfg.input_dim),
            "encoder_seed": int(cfg.seed),
            "encoder_sparsity": float(cfg.encoder_sparsity),
            "encoder_scale": float(cfg.encoder_scale),
            "encoder_normalize_input": bool(cfg.encoder_normalize_input),
            "encoder_output_clip": cfg.encoder_output_clip,
            "memory_damping": float(cfg.memory_damping),
            "flow_steps": int(cfg.flow_steps),
            "flow_damping": float(cfg.flow_damping),
            "flow_diffusion": float(cfg.flow_diffusion),
            "onsite_scale": float(cfg.onsite_scale),
            "edge_scale": float(cfg.edge_scale),
            "vib_scale": float(cfg.vib_scale),
            "evolution_method": str(cfg.method),
            "apply_truncate_rank": int(cfg.apply_truncate_rank),
            "canonicalize_every": int(cfg.canonicalize_every),
            "post_step_truncate_rank": int(cfg.post_step_truncate_rank),
            "renormalize_every": int(cfg.renormalize_every),
            "readout": readout_cfg,
            "cache_runtime_objects": bool(cfg.cache_runtime_objects),
            "strict": bool(cfg.strict),
        },
    )
    reservoir_cfg = QuantumReservoirConfig(**reservoir_kwargs)

    build_kwargs = _dataclass_kwargs(
        ReservoirBuildConfig,
        {
            "atom_name": "H",
            "atom_level": 1,
            "noncommutative": bool(cfg.noncommutative),
            "seed": int(cfg.seed),
            "tree_mode": "balanced",
            "vibration_kind": str(cfg.vibration_kind),
            "vibration_strength": float(cfg.vibration_strength),
            "vibration_temperature": float(cfg.vibration_temperature),
            "vibration_n": int(cfg.vibration_n),
            "fiber_d_uniform": int(cfg.fiber_d_uniform),
            "fiber_d_min": int(cfg.fiber_d_min),
            "fiber_d_max": int(cfg.fiber_d_max),
            "bond_dim": int(cfg.bond_dim),
            "reservoir": reservoir_cfg,
        },
    )
    build_cfg = ReservoirBuildConfig(**build_kwargs)

    return build_quantum_reservoir(build_cfg)


# =============================================================================
# AtomTN adapter wrapper
# =============================================================================


class AtomTNAdapter:
    """
    Thin resilience wrapper around neuromorphic.QuantumReservoir.
    """

    def __init__(self, cfg: AtomAdapterConfig, *, reservoir: Any = None):
        self.cfg = _effective_config(cfg)
        self.reservoir = reservoir
        self.fallback = _FallbackReservoir(self.cfg)

        self.attachment_error: Optional[str] = None
        self.attached = False
        self.step_counter = 0

        if self.cfg.enable_atomtn:
            if self.reservoir is None:
                try:
                    self.reservoir = _build_atomtn_reservoir(self.cfg)
                except BaseException as exc:
                    self.attachment_error = repr(exc)
                    if self.cfg.fail_fast or self.cfg.strict:
                        raise
                    self.reservoir = None

            self.attached = self.reservoir is not None
        else:
            self.attachment_error = "AtomTN disabled by config."
            self.attached = False

    def step(self, x: Any, *, dt: Optional[float] = None) -> np.ndarray:
        dtv = self.cfg.dt if dt is None else _safe_float(dt, self.cfg.dt)
        x_vec = _as_float32_vector(x, expected_dim=self.cfg.input_dim, name="AtomTNAdapter input")

        if self.attached and self.reservoir is not None:
            try:
                # Smoke/no-evolve mode avoids Hamiltonian evolution.
                step_dt = 0.0 if self.cfg.method == "none" else dtv
                out = self.reservoir.step(x_vec, dt=float(step_dt))
                y = _as_float32_vector(out)
                self.step_counter += 1
                return self._coerce_feature_dim(y)
            except BaseException as exc:
                self.attachment_error = f"reservoir_step_failed: {exc!r}"
                if self.cfg.fail_fast or self.cfg.strict:
                    raise

        y = self.fallback.step(x_vec, dt=dtv)
        self.step_counter += 1
        return self._coerce_feature_dim(y)

    def _coerce_feature_dim(self, y: Any) -> np.ndarray:
        arr = _as_float32_vector(y)
        if arr.size == self.cfg.output_dim:
            return arr

        out = np.zeros((self.cfg.output_dim,), dtype=np.float32)
        if arr.size:
            n = min(out.size, arr.size)
            out[:n] = arr[:n]
        return out

    def get_digital_twin_frame(self) -> Dict[str, Any]:
        if self.attached and self.reservoir is not None:
            fn = getattr(self.reservoir, "get_digital_twin_frame", None)
            if callable(fn):
                try:
                    frame = fn()
                    if isinstance(frame, Mapping):
                        return dict(frame)
                except Exception as exc:
                    return {
                        "type": "atomtn_frame_error",
                        "error": repr(exc),
                        "ts": _now_iso(),
                    }

        return self.fallback.get_digital_twin_frame()

    def health_metrics(self) -> Dict[str, Any]:
        if self.attached and self.reservoir is not None:
            fn = getattr(self.reservoir, "health_metrics", None)
            if callable(fn):
                try:
                    h = fn()
                    if isinstance(h, Mapping):
                        out = dict(h)
                        out.setdefault("atom_attached", True)
                        out.setdefault("attachment_error", self.attachment_error)
                        return _json_safe(out)
                except Exception as exc:
                    return {
                        "kind": "AtomTNAdapter",
                        "is_stable": False,
                        "last_error": repr(exc),
                        "atom_attached": True,
                    }

        h = self.fallback.health_metrics()
        h.update(
            {
                "atom_attached": False,
                "attachment_error": self.attachment_error,
            }
        )
        return h

    def reset_state(self) -> None:
        self.step_counter = 0

        if self.attached and self.reservoir is not None:
            fn = getattr(self.reservoir, "reset_state", None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception as exc:
                    self.attachment_error = f"reset_failed: {exc!r}"
                    if self.cfg.fail_fast or self.cfg.strict:
                        raise

        self.fallback.reset_state()

    def snapshot(self) -> Dict[str, Any]:
        if self.attached and self.reservoir is not None:
            fn = getattr(self.reservoir, "snapshot", None)
            if callable(fn):
                try:
                    snap = fn()
                    if isinstance(snap, Mapping):
                        out = dict(snap)
                        out["atom_attached"] = True
                        out["attachment_error"] = self.attachment_error
                        return _json_safe(out)
                except Exception as exc:
                    return {
                        "kind": "AtomTNAdapter",
                        "atom_attached": True,
                        "snapshot_error": repr(exc),
                    }

        out = self.fallback.snapshot()
        out.update(
            {
                "atom_attached": False,
                "attachment_error": self.attachment_error,
            }
        )
        return out


# =============================================================================
# Runtime orchestration
# =============================================================================


class AtomAdapterRuntime:
    """
    Stateful adapter that optionally updates an Akkurat digital twin.
    """

    def __init__(
        self,
        cfg: Optional[AtomAdapterConfig] = None,
        *,
        atom_adapter: Optional[AtomTNAdapter] = None,
        digital_twin: Any = None,
        control_plane: Any = None,
    ):
        self.cfg = _effective_config(cfg or AtomAdapterConfig())

        configure_paths(
            atomtn_root=(self.cfg.atomtn_root or None),
            akkurat_root=(self.cfg.akkurat_root or None),
        )

        self.attachment_errors: Dict[str, Optional[str]] = {
            "atom": None,
            "digital_twin": None,
        }
        self.audit_log: List[Dict[str, Any]] = []

        self.step_count = 0
        self.last_result: Optional[AtomAdapterStepResult] = None

        self.atom_adapter = atom_adapter if atom_adapter is not None else AtomTNAdapter(self.cfg)

        if not self.atom_adapter.attached:
            self.attachment_errors["atom"] = self.atom_adapter.attachment_error

        self.digital_twin = digital_twin
        self.control_plane = control_plane

        if self.cfg.enable_digital_twin and self.digital_twin is None:
            self._attach_digital_twin()
        elif self.digital_twin is not None and self.control_plane is None:
            self._attach_control_plane_for_existing_twin()

    @property
    def atom_attached(self) -> bool:
        return bool(self.atom_adapter is not None and self.atom_adapter.attached)

    @property
    def digital_twin_attached(self) -> bool:
        return bool(self.digital_twin is not None)

    def _attach_digital_twin(self) -> None:
        global DigitalTwinsBuilder, AkkuratInterface

        try:
            status = digital_twin_status()
            if not status.get("available", False) or DigitalTwinsBuilder is None:
                raise RuntimeError(f"digital_twin_kernel unavailable: {status.get('import_error')}")

            self.digital_twin = DigitalTwinsBuilder.build_platform(
                vector_dim=int(self.cfg.digital_twin_vector_dim),
                seed=int(self.cfg.seed),
                sketch_dim=self.cfg.digital_twin_sketch_dim,
                history_capacity=int(self.cfg.digital_twin_history_capacity),
                use_tn_projection=bool(self.cfg.digital_twin_use_tn_projection),
                latent_geometry=str(self.cfg.digital_twin_latent_geometry),
            )

            if AkkuratInterface is not None:
                self.control_plane = AkkuratInterface(self.digital_twin, strict=False)

            self.attachment_errors["digital_twin"] = None

        except BaseException as exc:
            self.digital_twin = None
            self.control_plane = None
            self.attachment_errors["digital_twin"] = repr(exc)

            if self.cfg.fail_fast or self.cfg.strict:
                raise

    def _attach_control_plane_for_existing_twin(self) -> None:
        global AkkuratInterface

        try:
            if AkkuratInterface is None:
                digital_twin_status()

            if AkkuratInterface is not None:
                self.control_plane = AkkuratInterface(self.digital_twin, strict=False)

        except BaseException as exc:
            self.attachment_errors["digital_twin"] = f"control_plane_attach_failed: {exc!r}"

            if self.cfg.fail_fast or self.cfg.strict:
                raise

    def step(
        self,
        x: Any,
        *,
        dt: Optional[float] = None,
        note: str = "atom_adapter_step",
    ) -> AtomAdapterStepResult:
        t0 = time.perf_counter()

        warnings: List[str] = []
        err: Optional[str] = None
        ok = True

        try:
            x_vec = _as_float32_vector(
                x,
                expected_dim=self.cfg.input_dim,
                name="AtomAdapterRuntime input",
            )

            features = self.atom_adapter.step(x_vec, dt=dt)
            frame = self.atom_adapter.get_digital_twin_frame()
            health = self.atom_adapter.health_metrics()

            metrics = self._metrics_from_features(features, frame=frame, health=health)
            digital = self._update_digital_twin(
                features,
                frame=frame,
                health=health,
                note=note,
            )

            if not bool(health.get("is_stable", True)):
                ok = False
                warnings.append("health_unstable")

        except BaseException as exc:
            if self.cfg.fail_fast or self.cfg.strict:
                raise

            ok = False
            err = repr(exc)
            features = np.zeros((self.cfg.output_dim,), dtype=np.float32)
            frame = {
                "type": "adapter_step_error",
                "error": err,
                "ts": _now_iso(),
            }
            health = {
                "kind": "AtomAdapterRuntime",
                "is_stable": False,
                "last_error": err,
            }
            metrics = self._metrics_from_features(features, frame=frame, health=health)
            digital = {}

        elapsed = float(time.perf_counter() - t0)

        if elapsed > float(self.cfg.max_step_seconds_warn):
            warnings.append(f"slow_step({elapsed:.3f}s)")

        result = AtomAdapterStepResult(
            ok=bool(ok),
            step=int(self.step_count),
            ts=_now_iso(),
            features=features.astype(np.float32, copy=False),
            frame=_json_safe(frame),
            metrics=_json_safe(metrics),
            health=_json_safe(health),
            digital_twin=_json_safe(digital),
            elapsed_s=elapsed,
            error=err,
            warnings=warnings,
        )

        self.last_result = result
        self.step_count += 1

        self.audit_log.append(
            {
                "ts": result.ts,
                "event": "step",
                "ok": bool(result.ok),
                "elapsed_s": elapsed,
                "warnings": list(warnings),
            }
        )

        return result

    def step_sequence(
        self,
        X: Any,
        *,
        dt: Union[float, Sequence[float], None] = None,
    ) -> List[AtomAdapterStepResult]:
        arr = np.asarray(X, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError(f"step_sequence expects 1D or 2D input, got shape {arr.shape}")

        outs: List[AtomAdapterStepResult] = []

        if isinstance(dt, Sequence) and not isinstance(dt, (str, bytes)):
            dts = list(dt)
            if len(dts) != arr.shape[0]:
                raise ValueError("dt sequence length must match number of rows")
        else:
            dts = [dt] * int(arr.shape[0])

        for i in range(arr.shape[0]):
            outs.append(self.step(arr[i], dt=dts[i]))

        return outs

    @staticmethod
    def _metrics_from_features(
        features: np.ndarray,
        *,
        frame: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        y = _as_float32_vector(features)
        fmetrics = frame.get("metrics", {}) if isinstance(frame, Mapping) else {}

        return {
            "feature_dim": int(y.size),
            "feature_mean": float(np.mean(y)) if y.size else 0.0,
            "feature_std": float(np.std(y)) if y.size else 0.0,
            "feature_norm": _safe_norm(y),
            "activity_norm": (
                _safe_float(fmetrics.get("activity_norm", _safe_norm(y)))
                if isinstance(fmetrics, Mapping)
                else _safe_norm(y)
            ),
            "flow_energy": (
                _safe_float(fmetrics.get("flow_energy", 0.0))
                if isinstance(fmetrics, Mapping)
                else 0.0
            ),
            "quantum_norm_squared": (
                _safe_float(
                    health.get(
                        "quantum_norm_squared",
                        fmetrics.get("quantum_norm_squared", 0.0)
                        if isinstance(fmetrics, Mapping)
                        else 0.0,
                    )
                )
                if isinstance(health, Mapping)
                else 0.0
            ),
            "stable": bool(health.get("is_stable", True)) if isinstance(health, Mapping) else True,
        }

    def _update_digital_twin(
        self,
        features: np.ndarray,
        *,
        frame: Mapping[str, Any],
        health: Mapping[str, Any],
        note: str,
    ) -> Dict[str, Any]:
        if self.digital_twin is None or not self.cfg.enable_governance_updates:
            return {
                "attached": bool(self.digital_twin is not None),
                "updated": False,
                "reason": "disabled_or_unattached",
            }

        updates: List[Dict[str, Any]] = []

        try:
            # Primary AtomTN branch IDs from digital_twin_kernel.DigitalTwinsBuilder.
            if hasattr(self.digital_twin, "update_node_data"):
                self.digital_twin.update_node_data(
                    "3.2.3.5.1",
                    {
                        "quantum_norm_squared": (
                            _safe_float(health.get("quantum_norm_squared", 0.0))
                            if isinstance(health, Mapping)
                            else 0.0
                        ),
                        "stable": bool(health.get("is_stable", True))
                        if isinstance(health, Mapping)
                        else True,
                        "step": int(self.step_count),
                    },
                    note=f"{note}:quantum_state",
                )
                updates.append({"node_id": "3.2.3.5.1", "kind": "quantum_state"})

                self.digital_twin.update_node_data(
                    "3.2.3.5.2",
                    {
                        "flow_vectors": list(frame.get("flow_vectors", []))
                        if isinstance(frame, Mapping)
                        else [],
                        "flow_energy": (
                            _safe_float((frame.get("metrics", {}) or {}).get("flow_energy", 0.0))
                            if isinstance(frame, Mapping)
                            else 0.0
                        ),
                        "step": int(self.step_count),
                    },
                    note=f"{note}:flow_field",
                )
                updates.append({"node_id": "3.2.3.5.2", "kind": "flow_field"})

                self.digital_twin.update_node_data(
                    "3.2.3.5.3",
                    {
                        "node_activities": _as_float32_vector(
                            frame.get("node_activities", features)
                            if isinstance(frame, Mapping)
                            else features
                        ).astype(float).tolist(),
                        "feature_mean": float(np.mean(features)) if features.size else 0.0,
                        "feature_std": float(np.std(features)) if features.size else 0.0,
                        "feature_norm": _safe_norm(features),
                        "step": int(self.step_count),
                    },
                    note=f"{note}:observables",
                )
                updates.append({"node_id": "3.2.3.5.3", "kind": "observables"})

                self.digital_twin.update_node_data(
                    "3.2.3.5.5",
                    {
                        "adapter_health": _json_safe(health),
                        "attachment_errors": _json_safe(self.attachment_errors),
                        "step": int(self.step_count),
                    },
                    note=f"{note}:governance",
                )
                updates.append({"node_id": "3.2.3.5.5", "kind": "governance"})

            root_hash = None
            if hasattr(self.digital_twin, "merkle_root") and hasattr(self.digital_twin, "merkle"):
                try:
                    root_hash = self.digital_twin.merkle.hex64(self.digital_twin.merkle_root())
                except Exception:
                    root_hash = None

            return {
                "attached": True,
                "updated": True,
                "updates": updates,
                "root_hash": root_hash,
            }

        except BaseException as exc:
            self.attachment_errors["digital_twin"] = f"update_failed: {exc!r}"

            if self.cfg.fail_fast or self.cfg.strict:
                raise

            return {
                "attached": True,
                "updated": False,
                "error": repr(exc),
                "updates": updates,
            }

    def reset_state(self) -> None:
        self.step_count = 0
        self.last_result = None

        if self.atom_adapter is not None:
            self.atom_adapter.reset_state()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "AtomAdapterRuntime",
            "adapter_id": self.cfg.adapter_id,
            "step_count": int(self.step_count),
            "atom_attached": self.atom_attached,
            "digital_twin_attached": self.digital_twin_attached,
            "attachment_errors": dict(self.attachment_errors),
            "atom": self.atom_adapter.snapshot() if self.atom_adapter is not None else None,
            "last_result": None if self.last_result is None else self.last_result.to_dict(),
        }

    def summary(self) -> str:
        health = self.atom_adapter.health_metrics() if self.atom_adapter is not None else {}
        stable = bool(health.get("is_stable", False)) if isinstance(health, Mapping) else False

        lines = [
            "[AtomAdapterRuntime]",
            f"- id: {self.cfg.adapter_id}",
            f"- input_dim: {self.cfg.input_dim}",
            f"- output_dim: {self.cfg.output_dim}",
            f"- profile: {self.cfg.profile}",
            f"- method: {self.cfg.method}",
            f"- step_count: {self.step_count}",
            f"- stable: {stable}",
            f"- atom_attached: {self.atom_attached}",
            f"- digital_twin_attached: {self.digital_twin_attached}",
        ]

        if any(v for v in self.attachment_errors.values()):
            lines.append(
                f"- attachment_errors: {json.dumps(_json_safe(self.attachment_errors), ensure_ascii=False)}"
            )

        return "\n".join(lines) + "\n"

    def save_snapshot(self, path: Union[str, os.PathLike]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(_json_safe(self.snapshot()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# =============================================================================
# Public builder
# =============================================================================


def build_adapter(config: Optional[AtomAdapterConfig] = None, **overrides: Any) -> AtomAdapterRuntime:
    """
    Build an AtomAdapterRuntime with optional dataclass-field overrides.
    """
    cfg = copy.deepcopy(config or AtomAdapterConfig())

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise TypeError(f"Unknown AtomAdapterConfig field: {k}")

    return AtomAdapterRuntime(cfg)


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akkurat AtomTN adapter runtime.")

    p.add_argument("--mode", choices=["status", "demo", "smoke", "export-snapshot"], default="demo")
    p.add_argument("--input-dim", type=int, default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2027)

    p.add_argument("--profile", choices=["smoke", "fast", "balanced", "accurate"], default="fast")
    p.add_argument("--method", default="euler_legacy")

    p.add_argument("--disable-atomtn", action="store_true")
    p.add_argument("--enable-digital-twin", action="store_true")

    p.add_argument("--atomtn-root", default="")
    p.add_argument("--akkurat-root", default="")
    p.add_argument("--output", default="")

    return p


def _demo_inputs(input_dim: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    steps = int(max(1, steps))

    t = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32)

    cols = [np.sin(t), np.cos(t)]
    while len(cols) < int(input_dim):
        cols.append(rng.normal(0.0, 0.05, size=steps).astype(np.float32))

    return np.stack(cols[: int(input_dim)], axis=1).astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    configure_paths(
        atomtn_root=args.atomtn_root or None,
        akkurat_root=args.akkurat_root or None,
    )

    if args.mode == "status":
        print(
            json.dumps(
                {
                    "atomtn": atomtn_status(),
                    "digital_twin": digital_twin_status(),
                    "paths": _PATHS,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    cfg = AtomAdapterConfig(
        adapter_id="atom_adapter_cli",
        input_dim=int(args.input_dim),
        output_dim=64,
        feature_dim=64,
        enable_atomtn=not bool(args.disable_atomtn),
        enable_digital_twin=bool(args.enable_digital_twin),
        profile=str(args.profile),
        method=str(args.method),
        dt=float(args.dt),
        seed=int(args.seed),
        atomtn_root=str(args.atomtn_root or ""),
        akkurat_root=str(args.akkurat_root or ""),
    )

    if args.mode == "smoke":
        cfg.profile = "smoke"
        cfg.method = "none"
        cfg.enable_digital_twin = bool(args.enable_digital_twin)

    rt = build_adapter(cfg)

    X = _demo_inputs(cfg.input_dim, int(args.steps), cfg.seed)

    rows = []
    for i, x in enumerate(X):
        res = rt.step(x, dt=cfg.dt)
        rows.append(
            {
                "step": i,
                "ok": res.ok,
                "feature_norm": res.metrics.get("feature_norm"),
                "elapsed_s": res.elapsed_s,
                "warnings": res.warnings,
                "error": res.error,
            }
        )

    report = {
        "ok": all(bool(r["ok"]) for r in rows),
        "summary": rt.summary(),
        "rows": rows,
        "snapshot": rt.snapshot(),
    }

    if args.output or args.mode == "export-snapshot":
        out = Path(args.output or "atom_adapter_snapshot.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_json_safe(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report["output"] = str(out)

    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())