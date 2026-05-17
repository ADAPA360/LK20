#!/usr/bin/env python3
# cognitive_lobe_runtime.py
r"""
Project Chimera / Akkurat - Cognitive Lobe Runtime
==================================================

Production composition layer for the Akkurat cognitive architecture.

This module connects:
  - tn.py                         tensor-network backend
  - ncp.py                        TensorizedNCP recurrent neural substrate
  - cfc.py                        TCfC_Policy continuous-time substrate
  - digital_twin_kernel.py         optional governed digital-twin state layer
  - atom_adapter_runtime.py        optional AtomTN external physics/reservoir modules

Core runtime
------------
Akkurat ships with five paired cognitive domains.  Each domain owns one NCP and
one CfC module, giving ten native neural lobes:

    sensory_ncp      + sensory_cfc
    memory_ncp       + memory_cfc
    semantic_ncp     + semantic_cfc
    planning_ncp     + planning_cfc
    regulation_ncp   + regulation_cfc

The runtime also supports:
  - standalone NCP/CfC creation,
  - external adapters such as AtomTN reservoirs,
  - task-specific neural assemblies,
  - phase-synchronous signal routing,
  - regulation-driven metacognitive routing,
  - assembly feedback and consolidation,
  - governed digital-twin writes,
  - JSON state persistence.

Design constraints
------------------
- CPU-first / NumPy-first.
- Fixed-width communication bus.
- No hard dependency on AtomTN.
- Existing substrate scripts are treated as high-level APIs.
- External modules are registered through a formal adapter contract instead of
  direct dictionary mutation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Constants and utilities
# =============================================================================

_EPS = 1e-9


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_hash_u32(s: str) -> int:
    h = 2166136261
    for b in (s or "").encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _rng_from_key(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng((_stable_hash_u32(key) ^ (int(seed) & 0xFFFFFFFF)) & 0xFFFFFFFF)


def _as_vector(x: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=dtype)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)


def _l2_normalize(x: Any, eps: float = _EPS, *, dtype: np.dtype = np.float32) -> np.ndarray:
    v = _as_vector(x, dtype=dtype)
    if v.size == 0:
        return v
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return np.zeros_like(v, dtype=dtype)
    return (v / n).astype(dtype, copy=False)


def _clip_norm(x: Any, max_norm: float = 1.0, eps: float = _EPS, *, dtype: np.dtype = np.float32) -> np.ndarray:
    v = _as_vector(x, dtype=dtype)
    if v.size == 0 or max_norm <= 0:
        return v
    n = float(np.linalg.norm(v))
    if np.isfinite(n) and n > max_norm and n > eps:
        v = v * np.asarray(float(max_norm) / n, dtype=dtype)
    return v.astype(dtype, copy=False)


def _sigmoid(x: Any) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def _coerce_kind(value: Any) -> "ModuleKind":
    if isinstance(value, ModuleKind):
        return value
    s = str(value).strip().lower()
    for k in ModuleKind:
        if s == k.value or s == k.name.lower():
            return k
    if s in {"atom", "atomtn", "quantum", "reservoir"}:
        return ModuleKind.RESERVOIR
    return ModuleKind.EXTERNAL


def _add_project_paths() -> None:
    """Best-effort local project path setup."""
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent]
    if here.name.lower() == "cognitive_model_3":
        candidates.append(here.parent.parent / "AtomTN")
    for c in candidates:
        try:
            s = str(c.resolve())
        except Exception:
            s = str(c)
        if s and s not in sys.path:
            sys.path.insert(0, s)


_add_project_paths()


# =============================================================================
# Optional substrate imports
# =============================================================================

_NCP_OK = False
_CFC_OK = False
_NCP_ERR = ""
_CFC_ERR = ""

try:
    try:
        from .ncp import TensorizedNCP  # type: ignore
    except Exception:
        from ncp import TensorizedNCP  # type: ignore
    _NCP_OK = True
except Exception as e:
    TensorizedNCP = None  # type: ignore
    _NCP_ERR = repr(e)

try:
    try:
        from .cfc import TCfC_Policy, TensorizedCfCPolicy  # type: ignore
    except Exception:
        from cfc import TCfC_Policy, TensorizedCfCPolicy  # type: ignore
    _CFC_OK = True
except Exception as e:
    TCfC_Policy = None  # type: ignore
    TensorizedCfCPolicy = None  # type: ignore
    _CFC_ERR = repr(e)


# =============================================================================
# Deterministic signed projection
# =============================================================================

class SignedProjector:
    """
    Deterministic zero-parameter projection.

    Maps arbitrary vector lengths to a target dimension using signed feature
    hashing.  This keeps all module interfaces fixed-width without trainable
    adapter matrices.
    """

    def __init__(self, *, seed: int = 2027):
        self.seed = int(seed)

    def project(self, x: Any, *, target_dim: int, key: str, normalize: bool = True) -> np.ndarray:
        D = int(max(0, target_dim))
        out = np.zeros(D, dtype=np.float32)
        if D <= 0:
            return out

        arr = _as_vector(x, dtype=np.float32)
        if arr.size == 0:
            return out

        base = _stable_hash_u32(f"signed_project::{self.seed}::{key}::{arr.size}->{D}")
        for i, val in enumerate(arr):
            if not np.isfinite(val):
                continue
            h = _stable_hash_u32(f"{base}:{i}")
            j = int(h % D)
            sign = -1.0 if (h & 1) else 1.0
            out[j] += np.float32(sign) * np.float32(val)

        return _l2_normalize(out) if normalize else out.astype(np.float32, copy=False)

    def fuse(self, vectors: Sequence[Any], *, target_dim: int, key: str, weights: Optional[Sequence[float]] = None) -> np.ndarray:
        if not vectors:
            return np.zeros(int(target_dim), dtype=np.float32)
        acc = np.zeros(int(target_dim), dtype=np.float32)
        if weights is None:
            weights = [1.0] * len(vectors)
        total = 0.0
        for i, (v, w) in enumerate(zip(vectors, weights)):
            ww = float(w)
            if not np.isfinite(ww) or ww == 0.0:
                continue
            acc += np.float32(ww) * self.project(v, target_dim=target_dim, key=f"{key}:{i}", normalize=True)
            total += abs(ww)
        if total <= 0:
            return np.zeros(int(target_dim), dtype=np.float32)
        return _l2_normalize(acc)


# =============================================================================
# Protocols and configuration
# =============================================================================

class ModuleKind(str, Enum):
    NCP = "ncp"
    CFC = "cfc"
    EXTERNAL = "external"
    RESERVOIR = "reservoir"
    PHYSICS = "physics"


class NeuralAdapter(Protocol):
    module_id: str
    kind: ModuleKind
    input_dim: int
    output_dim: int

    def step(self, x: Any, *, dt: float = 1.0, return_state: bool = False) -> Any: ...
    def reset_state(self, value: float = 0.0) -> None: ...
    def get_state(self) -> np.ndarray: ...
    def set_state(self, state: Any) -> None: ...
    def snapshot(self) -> Dict[str, Any]: ...
    def health_metrics(self) -> Dict[str, Any]: ...
    def serialize_state(self) -> Dict[str, Any]: ...
    def load_state(self, payload: Mapping[str, Any]) -> None: ...


@dataclass
class NCPModuleSpec:
    module_id: str
    process_role: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    bond_dim: Union[int, Sequence[int]] = 4
    tau_min: float = 1.0
    tau_max: float = 30.0
    tt_modes: Optional[Sequence[int]] = None
    output_activation: str = "tanh"
    output_scale: float = 1.0
    tau_jitter_frac: float = 0.0
    safety_clamp: Optional[float] = 5.0
    tt_strict: bool = False
    leaky_lambda: float = 0.25
    num_freq_gates: int = 0
    num_mvm_outputs: int = 0
    integration_method: str = "euler"
    learn_dt_gate: bool = False
    force_dense: bool = False
    verbose: bool = False


@dataclass
class CFCModuleSpec:
    module_id: str
    process_role: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_cells: int = 3
    bond_dim: int = 6
    tt_strict: bool = False
    force_dense: bool = False
    learnable_tau: bool = True
    fused_heads: bool = True
    time_scale: float = 1.0
    output_activation: str = "tanh"
    verbose: bool = False


@dataclass
class LobePairSpec:
    domain_id: str
    process_role: str
    bus_dim: int
    ncp: NCPModuleSpec
    cfc: CFCModuleSpec
    twin_node_id: Optional[str] = None
    emit_to_twin: bool = False
    phase_frequency: float = 0.01
    fusion_gate_bias: float = 0.0


@dataclass
class ExternalModuleSpec:
    module_id: str
    kind: ModuleKind = ModuleKind.EXTERNAL
    process_role: str = "external module"
    input_dim: int = 64
    output_dim: int = 64
    capabilities: Dict[str, Any] = field(default_factory=dict)
    twin_node_id: Optional[str] = None
    emit_to_twin: bool = False


@dataclass
class LobeSignal:
    source_id: str
    target_id: str
    vector: np.ndarray
    strength: float = 1.0
    phase: float = 0.0
    ttl: int = 3
    decay: float = 0.85
    created_tick: int = 0
    signal_type: str = "latent"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_strength(self, tick: int) -> float:
        age = max(0, int(tick) - int(self.created_tick))
        if age >= int(self.ttl):
            return 0.0
        return float(self.strength) * (float(self.decay) ** age)


@dataclass
class LobeStepResult:
    domain_id: str
    ncp_output: np.ndarray
    cfc_output: np.ndarray
    fused_output: np.ndarray
    outgoing_signals: List[LobeSignal]
    health: Dict[str, Any]
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblySpec:
    assembly_id: str
    module_ids: List[str]
    output_dim: int
    strategy: str = "sequential"
    created_at_tick: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyRecord:
    spec: AssemblySpec
    usage_count: int = 0
    feedback_ema: float = 0.0
    last_output_norm: float = 0.0
    consolidated_module_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)


@dataclass
class CognitiveRuntimeConfig:
    bus_dim: int = 64
    seed: int = 2027
    strict_imports: bool = True
    signal_ttl: int = 4
    signal_decay: float = 0.85
    signal_strength: float = 1.0
    route_all_to_all: bool = True
    phase_sync_enabled: bool = True
    phase_coupling: float = 0.35
    dynamic_routing_enabled: bool = True
    routing_threshold: float = 0.48
    routing_top_k_per_source: int = 2
    routing_blend: float = 0.65
    twin_context_mix: float = 0.25
    incoming_signal_mix: float = 0.35
    governed_write_interval: int = 1
    assembly_feedback_beta: float = 0.35
    assembly_min_usage_for_consolidation: int = 16
    assembly_min_score_for_consolidation: float = 0.85
    consolidated_hidden_dim: int = 96
    consolidated_num_cells: int = 3
    consolidated_bond_dim: int = 4
    max_global_norm: float = 1.0


# =============================================================================
# Adapter validation and capability discovery
# =============================================================================

def validate_neural_adapter(adapter: Any) -> Dict[str, Any]:
    required_attrs = ["module_id", "kind", "input_dim", "output_dim"]
    required_methods = [
        "step",
        "reset_state",
        "get_state",
        "set_state",
        "snapshot",
        "health_metrics",
        "serialize_state",
        "load_state",
    ]

    missing: List[str] = []
    for attr in required_attrs:
        if not hasattr(adapter, attr):
            missing.append(attr)
    for meth in required_methods:
        if not callable(getattr(adapter, meth, None)):
            missing.append(meth)

    ok = not missing
    return {
        "ok": bool(ok),
        "missing": missing,
        "module_id": str(getattr(adapter, "module_id", "")),
        "kind": str(getattr(getattr(adapter, "kind", ""), "value", getattr(adapter, "kind", ""))),
        "input_dim": int(getattr(adapter, "input_dim", -1)) if hasattr(adapter, "input_dim") else -1,
        "output_dim": int(getattr(adapter, "output_dim", -1)) if hasattr(adapter, "output_dim") else -1,
    }


def discover_adapter_capabilities(adapter: Any) -> Dict[str, Any]:
    return {
        "observable_provider": callable(getattr(adapter, "measure_observables", None)),
        "flow_provider": callable(getattr(adapter, "get_flow_frame", None)),
        "twin_frame_provider": callable(getattr(adapter, "get_digital_twin_frame", None)),
        "physics_simulator": callable(getattr(adapter, "simulate_action", None)),
        "checkpointable": callable(getattr(adapter, "save_checkpoint", None)) and callable(getattr(adapter, "load_checkpoint", None)),
        "state_serializable": callable(getattr(adapter, "serialize_state", None)) and callable(getattr(adapter, "load_state", None)),
        "snapshot_provider": callable(getattr(adapter, "snapshot", None)),
        "health_provider": callable(getattr(adapter, "health_metrics", None)),
    }


# =============================================================================
# Native neural adapters
# =============================================================================

class NCPAdapter:
    def __init__(self, spec: NCPModuleSpec, *, rng: np.random.Generator, projector: SignedProjector):
        if not _NCP_OK or TensorizedNCP is None:
            raise ImportError(f"TensorizedNCP unavailable: {_NCP_ERR}")

        self.spec = spec
        self.module_id = spec.module_id
        self.kind = ModuleKind.NCP
        self.input_dim = int(spec.input_dim)
        self.output_dim = int(spec.output_dim)
        self.projector = projector
        self.rng = rng

        kwargs = dict(
            policy_id=spec.module_id,
            num_inputs=spec.input_dim,
            num_hidden=spec.hidden_dim,
            num_outputs=spec.output_dim,
            bond_dim=spec.bond_dim,
            tau_min=spec.tau_min,
            tau_max=spec.tau_max,
            tt_modes=spec.tt_modes,
            rng=rng,
            verbose=spec.verbose,
            output_activation=spec.output_activation,
            output_scale=spec.output_scale,
            tau_jitter_frac=spec.tau_jitter_frac,
            safety_clamp=spec.safety_clamp,
            tt_strict=spec.tt_strict,
            leaky_lambda=spec.leaky_lambda,
            num_freq_gates=spec.num_freq_gates,
            num_mvm_outputs=spec.num_mvm_outputs,
            force_dense=spec.force_dense,
        )
        # Newer ncp.py supports integration/dt options.  Fallback gracefully.
        for extra in ("integration_method", "learn_dt_gate"):
            kwargs[extra] = getattr(spec, extra)

        try:
            self.impl = TensorizedNCP(**kwargs)
        except TypeError:
            kwargs.pop("integration_method", None)
            kwargs.pop("learn_dt_gate", None)
            try:
                self.impl = TensorizedNCP(**kwargs)
            except TypeError:
                kwargs.pop("force_dense", None)
                self.impl = TensorizedNCP(**kwargs)

    def step(self, x: Any, *, dt: float = 1.0, return_state: bool = False) -> Any:
        inp = self.projector.project(x, target_dim=self.input_dim, key=f"{self.module_id}:in")
        try:
            result = self.impl.step(inp, dt=dt, return_state=return_state)
        except TypeError:
            result = self.impl.step(inp, dt=dt)
            if return_state:
                result = (result, {"health": self.health_metrics()})

        if return_state:
            y, aux = result
            out = self.projector.project(y, target_dim=self.output_dim, key=f"{self.module_id}:out")
            return out, aux
        return self.projector.project(result, target_dim=self.output_dim, key=f"{self.module_id}:out")

    def reset_state(self, value: float = 0.0) -> None:
        self.impl.reset_state(value)

    def get_state(self) -> np.ndarray:
        try:
            return _as_vector(self.impl.get_state())
        except Exception:
            return np.zeros(self.output_dim, dtype=np.float32)

    def set_state(self, state: Any) -> None:
        self.impl.set_state(_as_vector(state, dtype=np.float64))

    def snapshot(self) -> Dict[str, Any]:
        try:
            snap = dict(self.impl.snapshot())
        except Exception:
            snap = {}
        snap.update({"module_id": self.module_id, "kind": self.kind.value, "process_role": self.spec.process_role})
        return snap

    def health_metrics(self) -> Dict[str, Any]:
        try:
            h = dict(self.impl.health_metrics())
        except Exception as e:
            h = {"has_nan": True, "is_stable": False, "error": repr(e)}
        h.update({"module_id": self.module_id, "kind": self.kind.value, "is_stable": bool(h.get("is_stable", not h.get("has_nan", False)))})
        return h

    def serialize_state(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"adapter_kind": self.kind.value, "spec": _json_safe(self.spec)}
        try:
            payload["impl_state"] = self.impl.serialize_state()
        except Exception:
            payload["impl_state"] = {"hidden_state": self.get_state().tolist()}
        return payload

    def load_state(self, payload: Mapping[str, Any]) -> None:
        impl_state = payload.get("impl_state", payload)
        try:
            self.impl.load_state(dict(impl_state))
        except Exception:
            if isinstance(impl_state, Mapping) and "hidden_state" in impl_state:
                self.set_state(impl_state["hidden_state"])


class CFCAdapter:
    def __init__(self, spec: CFCModuleSpec, *, rng: np.random.Generator, projector: SignedProjector):
        if not _CFC_OK or TCfC_Policy is None:
            raise ImportError(f"TCfC_Policy unavailable: {_CFC_ERR}")

        self.spec = spec
        self.module_id = spec.module_id
        self.kind = ModuleKind.CFC
        self.input_dim = int(spec.input_dim)
        self.output_dim = int(spec.output_dim)
        self.projector = projector
        self.rng = rng

        kwargs = dict(
            policy_id=spec.module_id,
            input_size=spec.input_dim,
            hidden_size=spec.hidden_dim,
            num_cells=spec.num_cells,
            bond_dim=spec.bond_dim,
            rng=rng,
            dtype=np.float32,
            verbose=spec.verbose,
        )

        # Newer cfc.py accepts these.  Older versions do not.
        optional = {
            "output_size": spec.output_dim,
            "tt_strict": spec.tt_strict,
            "force_dense": spec.force_dense,
            "learnable_tau": spec.learnable_tau,
            "fused_heads": spec.fused_heads,
            "time_scale": spec.time_scale,
            "output_activation": spec.output_activation,
        }
        self.impl = None
        try:
            self.impl = TCfC_Policy(**kwargs, **optional)
        except TypeError:
            try:
                reduced = {k: optional[k] for k in ("output_size", "tt_strict", "force_dense") if k in optional}
                self.impl = TCfC_Policy(**kwargs, **reduced)
            except TypeError:
                self.impl = TCfC_Policy(**kwargs)

    def step(self, x: Any, *, dt: float = 1.0, return_state: bool = False) -> Any:
        inp = self.projector.project(x, target_dim=self.input_dim, key=f"{self.module_id}:in")
        aux: Dict[str, Any] = {}
        out_raw: Any

        try:
            result = self.impl.step(inp, time_delta=dt, return_state=return_state)
        except TypeError:
            try:
                result = self.impl.step(inp, dt=dt, return_state=return_state)
            except TypeError:
                result = self.impl.step(inp, time_delta=dt)

        if return_state and isinstance(result, tuple) and len(result) == 2:
            out_raw, aux = result
        else:
            out_raw = result
            if return_state:
                aux = {"health": self.health_metrics()}

        # Prefer dedicated control/readout if policy returns full internal state.
        if hasattr(self.impl, "get_control_vector"):
            try:
                out_raw = self.impl.get_control_vector(out_dim=self.output_dim)
            except TypeError:
                try:
                    out_raw = self.impl.get_control_vector(self.output_dim)
                except Exception:
                    pass
            except Exception:
                pass

        out = self.projector.project(out_raw, target_dim=self.output_dim, key=f"{self.module_id}:out")
        if return_state:
            return out, aux
        return out

    def reset_state(self, value: float = 0.0) -> None:
        self.impl.reset_state(value)

    def get_state(self) -> np.ndarray:
        try:
            return _as_vector(self.impl.get_state())
        except Exception:
            return np.zeros(self.output_dim, dtype=np.float32)

    def set_state(self, state: Any) -> None:
        self.impl.set_state(_as_vector(state))

    def snapshot(self) -> Dict[str, Any]:
        try:
            snap = dict(self.impl.snapshot())
        except Exception:
            snap = {
                "state_norm": float(np.linalg.norm(self.get_state())),
                "state_size": int(self.get_state().size),
            }
        snap.update({"module_id": self.module_id, "kind": self.kind.value, "process_role": self.spec.process_role})
        return snap

    def health_metrics(self) -> Dict[str, Any]:
        try:
            h = dict(self.impl.health_metrics())
        except Exception as e:
            st = self.get_state()
            h = {
                "is_stable": bool(np.all(np.isfinite(st)) and np.linalg.norm(st) < max(10.0, st.size * 2.0)),
                "has_nan": bool(not np.all(np.isfinite(st))),
                "state_norm": float(np.linalg.norm(st)),
                "error": repr(e),
            }
        h.update({"module_id": self.module_id, "kind": self.kind.value, "is_stable": bool(h.get("is_stable", not h.get("has_nan", False)))})
        return h

    def serialize_state(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"adapter_kind": self.kind.value, "spec": _json_safe(self.spec)}
        try:
            payload["impl_state"] = self.impl.serialize_state()
        except Exception:
            payload["impl_state"] = {"state": self.get_state().tolist()}
        return payload

    def load_state(self, payload: Mapping[str, Any]) -> None:
        impl_state = payload.get("impl_state", payload)
        try:
            self.impl.load_state(dict(impl_state))
        except Exception:
            if isinstance(impl_state, Mapping) and "state" in impl_state:
                self.set_state(impl_state["state"])


# =============================================================================
# Signal bus
# =============================================================================

class InterLobeSignalBus:
    def __init__(
        self,
        *,
        bus_dim: int,
        seed: int,
        ttl: int = 4,
        decay: float = 0.85,
        phase_sync_enabled: bool = True,
        phase_coupling: float = 0.35,
    ):
        self.bus_dim = int(bus_dim)
        self.seed = int(seed)
        self.ttl = int(ttl)
        self.decay = float(decay)
        self.phase_sync_enabled = bool(phase_sync_enabled)
        self.phase_coupling = float(phase_coupling)
        self.pending: Dict[str, List[LobeSignal]] = {}
        self.published_count = 0
        self.collected_count = 0
        self.last_phase_metrics: Dict[str, Any] = {"mean_alignment": 0.0, "target_metrics": {}}
        self.created_at = _now_iso()

    @staticmethod
    def _phase_alignment(a: float, b: float) -> float:
        # Cosine phase alignment in [0, 1].
        return float(0.5 + 0.5 * math.cos(2.0 * math.pi * (float(a) - float(b))))

    def publish(self, signal: LobeSignal) -> None:
        sig = LobeSignal(
            source_id=str(signal.source_id),
            target_id=str(signal.target_id),
            vector=_l2_normalize(signal.vector),
            strength=float(signal.strength),
            phase=float(signal.phase) % 1.0,
            ttl=int(signal.ttl if signal.ttl is not None else self.ttl),
            decay=float(signal.decay if signal.decay is not None else self.decay),
            created_tick=int(signal.created_tick),
            signal_type=str(signal.signal_type),
            metadata=dict(signal.metadata),
        )
        self.pending.setdefault(sig.target_id, []).append(sig)
        self.published_count += 1

    def publish_many(self, signals: Iterable[LobeSignal]) -> None:
        for sig in signals:
            self.publish(sig)

    def collect(self, target_id: str, *, tick: int, target_phase: float = 0.0) -> np.ndarray:
        signals = self.pending.get(str(target_id), [])
        if not signals:
            self.last_phase_metrics.setdefault("target_metrics", {})[str(target_id)] = {"count": 0, "mean_alignment": 0.0}
            return np.zeros(self.bus_dim, dtype=np.float32)

        kept: List[LobeSignal] = []
        acc = np.zeros(self.bus_dim, dtype=np.float32)
        weight_sum = 0.0
        alignments: List[float] = []

        for sig in signals:
            eff = sig.effective_strength(tick)
            if eff <= 0:
                continue
            kept.append(sig)
            v = _l2_normalize(sig.vector)
            if v.size != self.bus_dim:
                # This should be rare because publisher projects into bus_dim.
                proj = SignedProjector(seed=self.seed)
                v = proj.project(v, target_dim=self.bus_dim, key=f"bus_collect:{sig.source_id}->{target_id}")
            alignment = self._phase_alignment(sig.phase, target_phase)
            alignments.append(alignment)
            if self.phase_sync_enabled:
                eff *= (1.0 + self.phase_coupling * (2.0 * alignment - 1.0))
            acc += np.float32(max(0.0, eff)) * v.astype(np.float32, copy=False)
            weight_sum += abs(float(eff))

        self.pending[str(target_id)] = kept
        self.collected_count += len(kept)

        mean_align = float(np.mean(alignments)) if alignments else 0.0
        self.last_phase_metrics.setdefault("target_metrics", {})[str(target_id)] = {
            "count": len(kept),
            "mean_alignment": mean_align,
            "target_phase": float(target_phase),
        }
        all_metrics = self.last_phase_metrics.get("target_metrics", {})
        vals = [float(m.get("mean_alignment", 0.0)) for m in all_metrics.values() if isinstance(m, Mapping) and int(m.get("count", 0)) > 0]
        self.last_phase_metrics["mean_alignment"] = float(np.mean(vals)) if vals else 0.0

        if weight_sum <= 0.0:
            return np.zeros(self.bus_dim, dtype=np.float32)
        return _l2_normalize(acc)

    def prune(self, *, tick: int) -> None:
        for tgt in list(self.pending.keys()):
            self.pending[tgt] = [s for s in self.pending[tgt] if s.effective_strength(tick) > 0.0]
            if not self.pending[tgt]:
                self.pending.pop(tgt, None)

    def snapshot(self) -> Dict[str, Any]:
        pending_targets = {k: len(v) for k, v in self.pending.items()}
        active_signals = int(sum(pending_targets.values()))
        return {
            "bus_dim": self.bus_dim,
            "published_count": int(self.published_count),
            "collected_count": int(self.collected_count),
            "pending_targets": pending_targets,
            "active_signals": active_signals,
            "phase_sync_enabled": bool(self.phase_sync_enabled),
            "phase_coherence": dict(self.last_phase_metrics),
            "mean_phase_coherence": float(self.last_phase_metrics.get("mean_alignment", 0.0)),
        }

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "published_count": int(self.published_count),
            "collected_count": int(self.collected_count),
            "pending": {
                tgt: [
                    {
                        "source_id": s.source_id,
                        "target_id": s.target_id,
                        "vector": _as_vector(s.vector).tolist(),
                        "strength": s.strength,
                        "phase": s.phase,
                        "ttl": s.ttl,
                        "decay": s.decay,
                        "created_tick": s.created_tick,
                        "signal_type": s.signal_type,
                        "metadata": _json_safe(s.metadata),
                    }
                    for s in sigs
                ]
                for tgt, sigs in self.pending.items()
            },
            "last_phase_metrics": _json_safe(self.last_phase_metrics),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        self.published_count = int(payload.get("published_count", self.published_count))
        self.collected_count = int(payload.get("collected_count", self.collected_count))
        self.pending = {}
        for tgt, sigs in dict(payload.get("pending", {})).items():
            self.pending[str(tgt)] = []
            for s in sigs or []:
                if not isinstance(s, Mapping):
                    continue
                self.pending[str(tgt)].append(
                    LobeSignal(
                        source_id=str(s.get("source_id", "")),
                        target_id=str(s.get("target_id", tgt)),
                        vector=np.asarray(s.get("vector", []), dtype=np.float32),
                        strength=float(s.get("strength", 1.0)),
                        phase=float(s.get("phase", 0.0)),
                        ttl=int(s.get("ttl", self.ttl)),
                        decay=float(s.get("decay", self.decay)),
                        created_tick=int(s.get("created_tick", 0)),
                        signal_type=str(s.get("signal_type", "latent")),
                        metadata=dict(s.get("metadata", {})) if isinstance(s.get("metadata", {}), Mapping) else {},
                    )
                )
        if isinstance(payload.get("last_phase_metrics"), Mapping):
            self.last_phase_metrics = dict(payload["last_phase_metrics"])


# =============================================================================
# Cognitive lobe pair
# =============================================================================

class CognitiveLobePair:
    def __init__(
        self,
        spec: LobePairSpec,
        *,
        ncp: NCPAdapter,
        cfc: CFCAdapter,
        projector: SignedProjector,
        rng: np.random.Generator,
    ):
        self.spec = spec
        self.domain_id = spec.domain_id
        self.ncp = ncp
        self.cfc = cfc
        self.projector = projector
        self.rng = rng
        self.bus_dim = int(spec.bus_dim)
        self.last_fused = np.zeros(self.bus_dim, dtype=np.float32)
        self.last_health: Dict[str, Any] = {}
        self.step_count = 0

    def phase_at(self, tick: int) -> float:
        return float((float(tick) * float(self.spec.phase_frequency)) % 1.0)

    def _fuse(self, ncp_out: np.ndarray, cfc_out: np.ndarray, incoming: np.ndarray, twin_context: np.ndarray) -> np.ndarray:
        n = self.projector.project(ncp_out, target_dim=self.bus_dim, key=f"{self.domain_id}:fuse:ncp")
        c = self.projector.project(cfc_out, target_dim=self.bus_dim, key=f"{self.domain_id}:fuse:cfc")
        inc = self.projector.project(incoming, target_dim=self.bus_dim, key=f"{self.domain_id}:fuse:incoming")
        twin = self.projector.project(twin_context, target_dim=self.bus_dim, key=f"{self.domain_id}:fuse:twin")

        # Gate is stable and deterministic from the CfC/NCP difference.
        diff = float(np.dot(n, c))
        gate = float(_sigmoid(np.asarray([diff + self.spec.fusion_gate_bias], dtype=np.float32))[0])
        fused = gate * n + (1.0 - gate) * c + 0.25 * inc + 0.15 * twin
        return _l2_normalize(fused)

    def step(
        self,
        *,
        external_input: Any,
        incoming_signal: Any,
        twin_context: Any,
        dt: float,
        tick: int,
    ) -> LobeStepResult:
        self.step_count += 1

        x = self.projector.fuse(
            [external_input, incoming_signal, twin_context, self.last_fused],
            target_dim=self.bus_dim,
            key=f"{self.domain_id}:pair_input:{tick}",
            weights=[1.0, 0.35, 0.25, 0.20],
        )

        ncp_out, ncp_aux = self.ncp.step(x, dt=dt, return_state=True)
        cfc_in = self.projector.fuse([x, ncp_out], target_dim=self.bus_dim, key=f"{self.domain_id}:cfc_input:{tick}", weights=[1.0, 0.5])
        cfc_out, cfc_aux = self.cfc.step(cfc_in, dt=dt, return_state=True)

        fused = self._fuse(ncp_out, cfc_out, _as_vector(incoming_signal), _as_vector(twin_context))
        self.last_fused = fused

        health = {
            "domain_id": self.domain_id,
            "stable": bool(
                self.ncp.health_metrics().get("is_stable", True)
                and self.cfc.health_metrics().get("is_stable", True)
                and np.all(np.isfinite(fused))
            ),
            "ncp": self.ncp.health_metrics(),
            "cfc": self.cfc.health_metrics(),
            "fused_norm": float(np.linalg.norm(fused)),
            "step_count": int(self.step_count),
        }
        self.last_health = health

        return LobeStepResult(
            domain_id=self.domain_id,
            ncp_output=_as_vector(ncp_out),
            cfc_output=_as_vector(cfc_out),
            fused_output=fused,
            outgoing_signals=[],
            health=health,
            aux={"ncp": ncp_aux, "cfc": cfc_aux},
        )

    def reset_state(self) -> None:
        self.ncp.reset_state()
        self.cfc.reset_state()
        self.last_fused.fill(0.0)
        self.step_count = 0


# =============================================================================
# Registry
# =============================================================================

class NeuralModuleRegistry:
    def __init__(self):
        self.modules: Dict[str, NeuralAdapter] = {}
        self.pairs: Dict[str, CognitiveLobePair] = {}
        self.assemblies: Dict[str, AssemblyRecord] = {}
        self.specs: Dict[str, Any] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self.external_specs: Dict[str, ExternalModuleSpec] = {}

    def add_module(self, adapter: NeuralAdapter, spec: Optional[Any] = None, capabilities: Optional[Dict[str, Any]] = None) -> None:
        module_id = str(getattr(adapter, "module_id"))
        self.modules[module_id] = adapter
        if spec is not None:
            self.specs[module_id] = spec
        self.capabilities[module_id] = dict(capabilities or discover_adapter_capabilities(adapter))

    def add_pair(self, pair: CognitiveLobePair) -> None:
        self.pairs[pair.domain_id] = pair
        self.add_module(pair.ncp, pair.ncp.spec, discover_adapter_capabilities(pair.ncp))
        self.add_module(pair.cfc, pair.cfc.spec, discover_adapter_capabilities(pair.cfc))

    def create_assembly(self, spec: AssemblySpec) -> AssemblyRecord:
        rec = AssemblyRecord(spec=spec)
        self.assemblies[spec.assembly_id] = rec
        return rec


# =============================================================================
# Runtime
# =============================================================================

class CognitiveLobeSystem:
    def __init__(
        self,
        *,
        config: Optional[CognitiveRuntimeConfig] = None,
        twin_network: Optional[Any] = None,
        control_plane: Optional[Any] = None,
    ):
        self.config = config or CognitiveRuntimeConfig()
        self.bus_dim = int(self.config.bus_dim)
        self.seed = int(self.config.seed)
        self.rng = np.random.default_rng(self.seed)
        self.projector = SignedProjector(seed=self.seed)
        self.registry = NeuralModuleRegistry()
        self.signal_bus = InterLobeSignalBus(
            bus_dim=self.bus_dim,
            seed=self.seed,
            ttl=self.config.signal_ttl,
            decay=self.config.signal_decay,
            phase_sync_enabled=self.config.phase_sync_enabled,
            phase_coupling=self.config.phase_coupling,
        )
        self.twin_network = twin_network
        self.control_plane = control_plane
        self.tick = 0
        self.global_state = np.zeros(self.bus_dim, dtype=np.float32)
        self.tick_history: List[Dict[str, Any]] = []
        self.consolidation_log: List[Dict[str, Any]] = []
        self.routing_domains: List[str] = []
        self.routing_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.created_at = _now_iso()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_default(
        cls,
        *,
        bus_dim: int = 64,
        seed: int = 2027,
        strict_imports: bool = True,
        twin_network: Optional[Any] = None,
        control_plane: Optional[Any] = None,
    ) -> "CognitiveLobeSystem":
        cfg = CognitiveRuntimeConfig(bus_dim=int(bus_dim), seed=int(seed), strict_imports=bool(strict_imports))
        system = cls(config=cfg, twin_network=twin_network, control_plane=control_plane)
        for spec in default_lobe_pair_specs(bus_dim=bus_dim):
            system.add_lobe_pair(spec)
        system._ensure_routing_matrix()
        return system

    def add_lobe_pair(self, spec: LobePairSpec) -> CognitiveLobePair:
        ncp_rng = _rng_from_key(self.seed, f"ncp::{spec.ncp.module_id}")
        cfc_rng = _rng_from_key(self.seed, f"cfc::{spec.cfc.module_id}")
        ncp_adapter = NCPAdapter(spec.ncp, rng=ncp_rng, projector=self.projector)
        cfc_adapter = CFCAdapter(spec.cfc, rng=cfc_rng, projector=self.projector)
        pair = CognitiveLobePair(spec, ncp=ncp_adapter, cfc=cfc_adapter, projector=self.projector, rng=_rng_from_key(self.seed, f"pair::{spec.domain_id}"))
        self.registry.add_pair(pair)
        self._ensure_routing_matrix()
        return pair

    def add_standalone_ncp(
        self,
        *,
        module_id: str,
        process_role: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bond_dim: Union[int, Sequence[int]] = 4,
        **kwargs: Any,
    ) -> NCPAdapter:
        spec = NCPModuleSpec(
            module_id=module_id,
            process_role=process_role,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bond_dim=bond_dim,
            **kwargs,
        )
        adapter = NCPAdapter(spec, rng=_rng_from_key(self.seed, f"standalone_ncp::{module_id}"), projector=self.projector)
        self.registry.add_module(adapter, spec)
        return adapter

    def add_standalone_cfc(
        self,
        *,
        module_id: str,
        process_role: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_cells: int = 3,
        bond_dim: int = 6,
        **kwargs: Any,
    ) -> CFCAdapter:
        spec = CFCModuleSpec(
            module_id=module_id,
            process_role=process_role,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_cells=num_cells,
            bond_dim=bond_dim,
            **kwargs,
        )
        adapter = CFCAdapter(spec, rng=_rng_from_key(self.seed, f"standalone_cfc::{module_id}"), projector=self.projector)
        self.registry.add_module(adapter, spec)
        return adapter

    def register_external_module(
        self,
        adapter: NeuralAdapter,
        *,
        process_role: str = "external module",
        capabilities: Optional[Dict[str, Any]] = None,
        twin_node_id: Optional[str] = None,
        emit_to_twin: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Register an external adapter such as AtomTN as a first-class module.

        The adapter must satisfy the NeuralAdapter-shaped protocol.  Optional
        capability methods are discovered and stored in registry.capabilities.
        """
        report = validate_neural_adapter(adapter)
        if validate and not report["ok"]:
            raise TypeError(f"External adapter does not satisfy NeuralAdapter protocol: {report}")

        # Normalize attributes where possible.
        if not hasattr(adapter, "kind"):
            try:
                setattr(adapter, "kind", ModuleKind.EXTERNAL)
            except Exception:
                pass
        else:
            try:
                setattr(adapter, "kind", _coerce_kind(getattr(adapter, "kind")))
            except Exception:
                pass

        module_id = str(getattr(adapter, "module_id"))
        caps = discover_adapter_capabilities(adapter)
        if capabilities:
            caps.update(capabilities)

        spec = ExternalModuleSpec(
            module_id=module_id,
            kind=_coerce_kind(getattr(adapter, "kind", ModuleKind.EXTERNAL)),
            process_role=str(process_role),
            input_dim=int(getattr(adapter, "input_dim", self.bus_dim)),
            output_dim=int(getattr(adapter, "output_dim", self.bus_dim)),
            capabilities=caps,
            twin_node_id=twin_node_id,
            emit_to_twin=bool(emit_to_twin),
        )
        self.registry.add_module(adapter, spec, caps)
        self.registry.external_specs[module_id] = spec
        return {"registered": True, "module_id": module_id, "validation": report, "capabilities": caps}

    # Backward-compatible aliases.
    add_external_adapter = register_external_module
    register_external_adapter = register_external_module

    # ------------------------------------------------------------------
    # Assemblies
    # ------------------------------------------------------------------

    def create_assembly(
        self,
        *,
        assembly_id: str,
        module_ids: Sequence[str],
        output_dim: Optional[int] = None,
        strategy: str = "sequential",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssemblyRecord:
        missing = [m for m in module_ids if m not in self.registry.modules]
        if missing:
            raise KeyError(f"Unknown modules in assembly {assembly_id}: {missing}")
        spec = AssemblySpec(
            assembly_id=str(assembly_id),
            module_ids=[str(m) for m in module_ids],
            output_dim=int(output_dim or self.bus_dim),
            strategy=str(strategy),
            created_at_tick=int(self.tick),
            metadata=metadata or {},
        )
        return self.registry.create_assembly(spec)

    def run_assembly(self, assembly_id: str, x: Any, *, dt: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        if assembly_id not in self.registry.assemblies:
            raise KeyError(f"Unknown assembly_id: {assembly_id}")
        rec = self.registry.assemblies[assembly_id]
        spec = rec.spec
        strategy = spec.strategy.lower().strip()
        trace: Dict[str, Any] = {"assembly_id": assembly_id, "strategy": strategy, "modules": []}

        if strategy == "parallel":
            outs = []
            for module_id in spec.module_ids:
                adapter = self.registry.modules[module_id]
                inp = self.projector.project(x, target_dim=int(adapter.input_dim), key=f"assembly:{assembly_id}:{module_id}:parallel_in")
                out, aux = self._step_module(adapter, inp, dt=dt)
                outs.append(self.projector.project(out, target_dim=spec.output_dim, key=f"assembly:{assembly_id}:{module_id}:parallel_out"))
                trace["modules"].append({"module_id": module_id, "output_norm": float(np.linalg.norm(outs[-1])), "aux": _json_safe(aux)})
            y = self.projector.fuse(outs, target_dim=spec.output_dim, key=f"assembly:{assembly_id}:parallel_fuse")
        else:
            cur = self.projector.project(x, target_dim=self.bus_dim, key=f"assembly:{assembly_id}:start")
            for module_id in spec.module_ids:
                adapter = self.registry.modules[module_id]
                inp = self.projector.project(cur, target_dim=int(adapter.input_dim), key=f"assembly:{assembly_id}:{module_id}:in")
                out, aux = self._step_module(adapter, inp, dt=dt)
                cur = self.projector.project(out, target_dim=spec.output_dim, key=f"assembly:{assembly_id}:{module_id}:out")
                trace["modules"].append({"module_id": module_id, "output_norm": float(np.linalg.norm(cur)), "aux": _json_safe(aux)})
            y = cur

        y = self.projector.project(y, target_dim=spec.output_dim, key=f"assembly:{assembly_id}:final")
        rec.usage_count += 1
        rec.last_output_norm = float(np.linalg.norm(y))
        trace["output_norm"] = rec.last_output_norm
        trace["usage_count"] = rec.usage_count
        return y, trace

    def record_assembly_feedback(self, assembly_id: str, score: float) -> Dict[str, Any]:
        if assembly_id not in self.registry.assemblies:
            raise KeyError(f"Unknown assembly_id: {assembly_id}")
        rec = self.registry.assemblies[assembly_id]
        s = float(score)
        if not np.isfinite(s):
            s = 0.0
        beta = float(self.config.assembly_feedback_beta)
        if rec.usage_count <= 1 and rec.feedback_ema == 0.0:
            rec.feedback_ema = s
        else:
            rec.feedback_ema = (1.0 - beta) * rec.feedback_ema + beta * s

        consolidated = None
        if (
            rec.consolidated_module_id is None
            and rec.usage_count >= int(self.config.assembly_min_usage_for_consolidation)
            and rec.feedback_ema >= float(self.config.assembly_min_score_for_consolidation)
        ):
            consolidated = self._consolidate_assembly(rec)

        return {
            "assembly_id": assembly_id,
            "score": s,
            "feedback_ema": float(rec.feedback_ema),
            "usage_count": int(rec.usage_count),
            "consolidated_module_id": consolidated,
        }

    def _consolidate_assembly(self, rec: AssemblyRecord) -> str:
        # If the assembly contains an external/physics module, keep it intact and
        # create a helper module beside it.  We do not pretend to compress a
        # physics simulator into an NCP/CfC without explicit distillation data.
        contains_external = any(_coerce_kind(getattr(self.registry.modules[m], "kind", ModuleKind.EXTERNAL)) not in {ModuleKind.NCP, ModuleKind.CFC} for m in rec.spec.module_ids)

        module_id = f"consolidated_{rec.spec.assembly_id}_{self.tick}"
        if contains_external:
            helper = self.add_standalone_cfc(
                module_id=module_id,
                process_role=f"consolidated helper for hybrid assembly {rec.spec.assembly_id}",
                input_dim=self.bus_dim,
                hidden_dim=int(self.config.consolidated_hidden_dim),
                output_dim=int(rec.spec.output_dim),
                num_cells=int(self.config.consolidated_num_cells),
                bond_dim=int(self.config.consolidated_bond_dim),
                tt_strict=False,
            )
            mode = "hybrid_helper_cfc"
        else:
            helper = self.add_standalone_ncp(
                module_id=module_id,
                process_role=f"consolidated approximation of assembly {rec.spec.assembly_id}",
                input_dim=self.bus_dim,
                hidden_dim=int(self.config.consolidated_hidden_dim),
                output_dim=int(rec.spec.output_dim),
                bond_dim=int(self.config.consolidated_bond_dim),
                tt_strict=False,
            )
            mode = "native_approximation_ncp"

        rec.consolidated_module_id = helper.module_id
        event = {
            "ts": _now_iso(),
            "assembly_id": rec.spec.assembly_id,
            "consolidated_module_id": helper.module_id,
            "mode": mode,
            "usage_count": rec.usage_count,
            "feedback_ema": rec.feedback_ema,
            "contains_external": bool(contains_external),
            "module_ids": list(rec.spec.module_ids),
        }
        self.consolidation_log.append(event)
        return helper.module_id

    # ------------------------------------------------------------------
    # Runtime stepping
    # ------------------------------------------------------------------

    def _step_module(self, adapter: NeuralAdapter, x: Any, *, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            result = adapter.step(x, dt=dt, return_state=True)
        except TypeError:
            result = adapter.step(x, dt=dt)

        if isinstance(result, tuple) and len(result) == 2:
            y, aux = result
        else:
            y, aux = result, {"health": adapter.health_metrics() if callable(getattr(adapter, "health_metrics", None)) else {}}

        out = self.projector.project(y, target_dim=int(getattr(adapter, "output_dim", self.bus_dim)), key=f"module_step:{getattr(adapter, 'module_id', 'unknown')}")
        return out, dict(aux) if isinstance(aux, Mapping) else {"aux": _json_safe(aux)}

    def _twin_context(self) -> np.ndarray:
        if self.twin_network is None:
            return np.zeros(self.bus_dim, dtype=np.float32)
        try:
            gs = self.twin_network.global_state()
            return self.projector.project(gs, target_dim=self.bus_dim, key=f"twin_context:{self.tick}")
        except Exception:
            return np.zeros(self.bus_dim, dtype=np.float32)

    def _ensure_routing_matrix(self) -> None:
        domains = sorted(self.registry.pairs.keys())
        n = len(domains)
        if domains == self.routing_domains and self.routing_matrix.shape == (n, n):
            return
        self.routing_domains = domains
        mat = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(mat, 0.0)
        self.routing_matrix = mat

    def _update_routing_from_regulation(self, regulation_output: Optional[np.ndarray]) -> None:
        self._ensure_routing_matrix()
        if not self.config.dynamic_routing_enabled or self.routing_matrix.size == 0:
            return

        n = len(self.routing_domains)
        if regulation_output is None or _as_vector(regulation_output).size == 0:
            return
        raw = self.projector.project(regulation_output, target_dim=n * n, key=f"dynamic_routing:{self.tick}", normalize=False)
        new = _sigmoid(raw.reshape(n, n)).astype(np.float32)
        np.fill_diagonal(new, 0.0)

        # Guarantee top-k outgoing routes per source so the system never isolates
        # a domain purely due to initial random projections.
        k = max(0, min(n - 1, int(self.config.routing_top_k_per_source)))
        if k > 0:
            for i in range(n):
                row = new[i].copy()
                row[i] = -1.0
                top = np.argsort(row)[-k:]
                for j in top:
                    if i != j:
                        new[i, j] = max(new[i, j], np.float32(self.config.routing_threshold + 0.02))

        blend = float(np.clip(self.config.routing_blend, 0.0, 1.0))
        self.routing_matrix = (1.0 - blend) * self.routing_matrix + blend * new
        np.fill_diagonal(self.routing_matrix, 0.0)

    def _targets_for_source(self, source_domain: str) -> List[str]:
        domains = self.routing_domains
        if source_domain not in domains:
            return [d for d in domains if d != source_domain]
        i = domains.index(source_domain)

        if self.config.route_all_to_all and not self.config.dynamic_routing_enabled:
            return [d for d in domains if d != source_domain]

        if self.routing_matrix.shape != (len(domains), len(domains)):
            return [d for d in domains if d != source_domain]

        row = self.routing_matrix[i]
        targets = [domains[j] for j, val in enumerate(row) if j != i and float(val) >= float(self.config.routing_threshold)]
        k = max(0, min(len(domains) - 1, int(self.config.routing_top_k_per_source)))
        if len(targets) < k:
            order = np.argsort(row)[::-1]
            for j in order:
                if j == i:
                    continue
                d = domains[j]
                if d not in targets:
                    targets.append(d)
                if len(targets) >= k:
                    break

        if self.config.route_all_to_all and not targets:
            targets = [d for d in domains if d != source_domain]
        return targets

    def _submit_governed_actions(self, results: Mapping[str, LobeStepResult]) -> Dict[str, Any]:
        if self.twin_network is None:
            return {"submitted": 0, "approved": 0, "executed": {"applied": 0, "skipped": 0, "errors": 0}, "mode": "no_twin"}

        actions = []
        try:
            from digital_twin_kernel import Action  # type: ignore
        except Exception:
            Action = None  # type: ignore

        for domain_id, result in results.items():
            pair = self.registry.pairs.get(domain_id)
            if pair is None or not pair.spec.emit_to_twin or not pair.spec.twin_node_id:
                continue
            payload = result.fused_output.astype(np.float32).tolist()
            if Action is not None:
                actions.append(Action(kind="update_node_latent", node_id=pair.spec.twin_node_id, payload=payload, note=f"cognitive_lobe:{domain_id}:tick:{self.tick}"))
            else:
                actions.append({"kind": "update_node_latent", "node_id": pair.spec.twin_node_id, "payload": payload, "note": f"cognitive_lobe:{domain_id}:tick:{self.tick}"})

        # External modules may also opt into twin emission.
        for module_id, spec in self.registry.external_specs.items():
            if not spec.emit_to_twin or not spec.twin_node_id:
                continue
            adapter = self.registry.modules.get(module_id)
            if adapter is None:
                continue
            try:
                payload = self.projector.project(adapter.get_state(), target_dim=getattr(self.twin_network, "vector_dim", self.bus_dim), key=f"external_twin:{module_id}:{self.tick}").tolist()
            except Exception:
                continue
            if Action is not None:
                actions.append(Action(kind="update_node_latent", node_id=spec.twin_node_id, payload=payload, note=f"external_module:{module_id}:tick:{self.tick}"))
            else:
                actions.append({"kind": "update_node_latent", "node_id": spec.twin_node_id, "payload": payload, "note": f"external_module:{module_id}:tick:{self.tick}"})

        if not actions:
            return {"submitted": 0, "approved": 0, "executed": {"applied": 0, "skipped": 0, "errors": 0}, "mode": "no_actions"}

        if self.control_plane is not None and callable(getattr(self.control_plane, "governed_actions", None)):
            try:
                res = self.control_plane.governed_actions(actions, intent=f"cognitive_runtime_tick_{self.tick}")
                executed = res.get("execute", res.get("executed", {"applied": 0, "skipped": 0, "errors": 0})) if isinstance(res, Mapping) else {}
                return {
                    "submitted": len(actions),
                    "approved": int(bool(res.get("approved", False))) if isinstance(res, Mapping) else 0,
                    "executed": executed,
                    "mode": "governed_actions",
                    "raw": _json_safe(res),
                }
            except Exception as e:
                return {"submitted": len(actions), "approved": 0, "executed": {"applied": 0, "skipped": 0, "errors": 1}, "mode": "governed_error", "error": repr(e)}

        # Fallback direct writes if no control plane is available.
        applied = 0
        errors = 0
        for act in actions:
            try:
                node_id = getattr(act, "node_id", None) if not isinstance(act, Mapping) else act.get("node_id")
                payload = getattr(act, "payload", None) if not isinstance(act, Mapping) else act.get("payload")
                self.twin_network.update_node_latent(str(node_id), np.asarray(payload, dtype=np.float32), note="cognitive_runtime_direct_fallback")
                applied += 1
            except Exception:
                errors += 1
        return {"submitted": len(actions), "approved": 1 if errors == 0 else 0, "executed": {"applied": applied, "skipped": 0, "errors": errors}, "mode": "direct_fallback"}

    def step_once(self, external_input: Any, *, dt: float = 1.0) -> Dict[str, LobeStepResult]:
        self.tick += 1
        self._ensure_routing_matrix()
        self.signal_bus.prune(tick=self.tick)

        twin_context = self._twin_context()
        results: Dict[str, LobeStepResult] = {}

        # Step paired domains in deterministic order.
        for domain_id in self.routing_domains:
            pair = self.registry.pairs[domain_id]
            incoming = self.signal_bus.collect(domain_id, tick=self.tick, target_phase=pair.phase_at(self.tick))
            result = pair.step(
                external_input=external_input,
                incoming_signal=incoming,
                twin_context=twin_context,
                dt=dt,
                tick=self.tick,
            )
            results[domain_id] = result

        # Update global state before routing.
        if results:
            self.global_state = self.projector.fuse([r.fused_output for r in results.values()], target_dim=self.bus_dim, key=f"global_state:{self.tick}")
        else:
            self.global_state = np.zeros(self.bus_dim, dtype=np.float32)

        # Regulation lobe controls the next routing matrix.
        reg = results.get("regulation")
        self._update_routing_from_regulation(reg.fused_output if reg else None)

        # Publish inter-lobe signals.
        all_signals: List[LobeSignal] = []
        for domain_id, result in results.items():
            pair = self.registry.pairs[domain_id]
            source_phase = pair.phase_at(self.tick)
            for target in self._targets_for_source(domain_id):
                sig = LobeSignal(
                    source_id=domain_id,
                    target_id=target,
                    vector=result.fused_output.copy(),
                    strength=float(self.config.signal_strength),
                    phase=source_phase,
                    ttl=int(self.config.signal_ttl),
                    decay=float(self.config.signal_decay),
                    created_tick=int(self.tick),
                    signal_type="domain_fused",
                    metadata={"tick": self.tick},
                )
                all_signals.append(sig)
            result.outgoing_signals = [s for s in all_signals if s.source_id == domain_id]
        self.signal_bus.publish_many(all_signals)

        governance = {"submitted": 0, "approved": 0, "executed": {"applied": 0, "skipped": 0, "errors": 0}, "mode": "skipped_interval"}
        if self.config.governed_write_interval > 0 and self.tick % int(self.config.governed_write_interval) == 0:
            governance = self._submit_governed_actions(results)

        record = {
            "tick": self.tick,
            "ts": _now_iso(),
            "stable": all(bool(r.health.get("stable", True)) for r in results.values()),
            "global_state_norm": float(np.linalg.norm(self.global_state)),
            "domains": list(results.keys()),
            "bus": self.signal_bus.snapshot(),
            "routing": self.routing_snapshot(),
            "governance": governance,
        }
        self.tick_history.append(record)
        return results

    def run_ticks(self, inputs: Sequence[Any], *, dt: float = 1.0) -> List[Dict[str, LobeStepResult]]:
        return [self.step_once(x, dt=dt) for x in inputs]

    # ------------------------------------------------------------------
    # External capability helpers
    # ------------------------------------------------------------------

    def external_capabilities(self) -> Dict[str, Dict[str, Any]]:
        return {
            mid: caps
            for mid, caps in self.registry.capabilities.items()
            if _coerce_kind(getattr(self.registry.modules.get(mid), "kind", ModuleKind.EXTERNAL)) not in {ModuleKind.NCP, ModuleKind.CFC}
        }

    def module_capabilities(self, module_id: str) -> Dict[str, Any]:
        return dict(self.registry.capabilities.get(module_id, {}))

    def collect_external_twin_frames(self) -> Dict[str, Any]:
        frames: Dict[str, Any] = {}
        for module_id, adapter in self.registry.modules.items():
            if callable(getattr(adapter, "get_digital_twin_frame", None)):
                try:
                    frames[module_id] = adapter.get_digital_twin_frame()
                except Exception as e:
                    frames[module_id] = {"error": repr(e)}
        return frames

    def simulate_external_action(self, module_id: str, action: Any, *, horizon: int = 3, dt: float = 0.1) -> Dict[str, Any]:
        adapter = self.registry.modules.get(module_id)
        if adapter is None:
            raise KeyError(f"Unknown module_id: {module_id}")
        fn = getattr(adapter, "simulate_action", None)
        if not callable(fn):
            return {"ok": False, "module_id": module_id, "reason": "adapter_has_no_simulate_action"}
        return dict(fn(action, horizon=horizon, dt=dt))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def routing_snapshot(self) -> Dict[str, Any]:
        return {
            "domains": list(self.routing_domains),
            "matrix": self.routing_matrix.astype(float).tolist(),
            "dynamic_routing_enabled": bool(self.config.dynamic_routing_enabled),
            "last_update_tick": int(self.tick),
        }

    def health_metrics(self) -> Dict[str, Any]:
        module_health: Dict[str, Any] = {}
        stable = True
        for module_id, adapter in self.registry.modules.items():
            try:
                h = adapter.health_metrics()
            except Exception as e:
                h = {"is_stable": False, "has_nan": True, "error": repr(e)}
            module_health[module_id] = h
            stable = stable and bool(h.get("is_stable", not h.get("has_nan", False)))

        pair_health = {domain: pair.last_health for domain, pair in self.registry.pairs.items()}
        return {
            "stable": bool(stable and np.all(np.isfinite(self.global_state))),
            "tick": int(self.tick),
            "bus_dim": int(self.bus_dim),
            "global_state_norm": float(np.linalg.norm(self.global_state)),
            "module_count": len(self.registry.modules),
            "pair_count": len(self.registry.pairs),
            "assembly_count": len(self.registry.assemblies),
            "consolidation_count": len(self.consolidation_log),
            "modules": module_health,
            "pairs": pair_health,
            "bus": self.signal_bus.snapshot(),
            "routing": self.routing_snapshot(),
            "external_capabilities": self.external_capabilities(),
        }

    def global_cognitive_state(self) -> np.ndarray:
        return self.global_state.copy()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "tick": int(self.tick),
            "config": _json_safe(self.config),
            "global_state": self.global_state.astype(float).tolist(),
            "health": self.health_metrics(),
            "assemblies": {k: _json_safe(v) for k, v in self.registry.assemblies.items()},
            "capabilities": _json_safe(self.registry.capabilities),
            "consolidation_log": _json_safe(self.consolidation_log),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def serialize_state(self) -> Dict[str, Any]:
        return {
            "version": "cognitive_lobe_runtime_v2_external",
            "tick": int(self.tick),
            "config": _json_safe(self.config),
            "global_state": self.global_state.astype(float).tolist(),
            "signal_bus": self.signal_bus.serialize_state(),
            "routing_domains": list(self.routing_domains),
            "routing_matrix": self.routing_matrix.astype(float).tolist(),
            "modules": {
                module_id: {
                    "kind": str(getattr(getattr(adapter, "kind", ""), "value", getattr(adapter, "kind", ""))),
                    "state": _json_safe(adapter.serialize_state() if callable(getattr(adapter, "serialize_state", None)) else {}),
                    "capabilities": _json_safe(self.registry.capabilities.get(module_id, {})),
                    "spec": _json_safe(self.registry.specs.get(module_id, self.registry.external_specs.get(module_id))),
                }
                for module_id, adapter in self.registry.modules.items()
            },
            "pairs": {domain: {"spec": _json_safe(pair.spec), "last_fused": pair.last_fused.astype(float).tolist(), "step_count": pair.step_count} for domain, pair in self.registry.pairs.items()},
            "assemblies": {aid: _json_safe(rec) for aid, rec in self.registry.assemblies.items()},
            "consolidation_log": _json_safe(self.consolidation_log),
            "tick_history_tail": _json_safe(self.tick_history[-20:]),
        }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        self.tick = int(payload.get("tick", self.tick))
        if "global_state" in payload:
            self.global_state = self.projector.project(payload["global_state"], target_dim=self.bus_dim, key="load_global_state")
        if isinstance(payload.get("signal_bus"), Mapping):
            self.signal_bus.load_state(payload["signal_bus"])
        if "routing_domains" in payload:
            self.routing_domains = [str(x) for x in payload.get("routing_domains", [])]
        if "routing_matrix" in payload:
            mat = np.asarray(payload.get("routing_matrix", []), dtype=np.float32)
            if mat.ndim == 2:
                self.routing_matrix = mat
        if isinstance(payload.get("modules"), Mapping):
            for module_id, rec in payload["modules"].items():
                if module_id not in self.registry.modules:
                    continue
                try:
                    self.registry.modules[module_id].load_state(rec.get("state", {}))
                except Exception:
                    pass
                if isinstance(rec.get("capabilities"), Mapping):
                    self.registry.capabilities[module_id] = dict(rec["capabilities"])

        # Restore assembly usage/feedback for existing assemblies; create records
        # when specs are serializable enough.
        if isinstance(payload.get("assemblies"), Mapping):
            for aid, rec_payload in payload["assemblies"].items():
                if not isinstance(rec_payload, Mapping):
                    continue
                spec_payload = rec_payload.get("spec", {})
                if aid not in self.registry.assemblies and isinstance(spec_payload, Mapping):
                    mids = [m for m in spec_payload.get("module_ids", []) if m in self.registry.modules]
                    if mids:
                        self.create_assembly(
                            assembly_id=str(aid),
                            module_ids=mids,
                            output_dim=int(spec_payload.get("output_dim", self.bus_dim)),
                            strategy=str(spec_payload.get("strategy", "sequential")),
                            metadata=dict(spec_payload.get("metadata", {})) if isinstance(spec_payload.get("metadata", {}), Mapping) else {},
                        )
                if aid in self.registry.assemblies:
                    rec = self.registry.assemblies[aid]
                    rec.usage_count = int(rec_payload.get("usage_count", rec.usage_count))
                    rec.feedback_ema = float(rec_payload.get("feedback_ema", rec.feedback_ema))
                    rec.last_output_norm = float(rec_payload.get("last_output_norm", rec.last_output_norm))
                    rec.consolidated_module_id = rec_payload.get("consolidated_module_id", rec.consolidated_module_id)

        if isinstance(payload.get("consolidation_log"), list):
            self.consolidation_log = list(payload["consolidation_log"])
        self._ensure_routing_matrix()

    def save_state_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(self.serialize_state()), indent=2, ensure_ascii=False), encoding="utf-8")

    def load_state_json(self, path: Union[str, Path]) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_state(payload)


# =============================================================================
# Default cognitive map
# =============================================================================

def default_lobe_pair_specs(*, bus_dim: int = 64) -> List[LobePairSpec]:
    b = int(bus_dim)
    return [
        LobePairSpec(
            domain_id="sensory",
            process_role="sensory integration, salience filtering, input stabilization",
            bus_dim=b,
            ncp=NCPModuleSpec("sensory_ncp", "bounded sensory recurrent integration", b, max(192, b * 3), b, bond_dim=4, tau_min=1.0, tau_max=18.0, leaky_lambda=0.30, num_freq_gates=2, tt_strict=False, integration_method="heun"),
            cfc=CFCModuleSpec("sensory_cfc", "continuous sensory smoothing and rhythm tracking", b, max(96, b), b, num_cells=4, bond_dim=6, tt_strict=False, learnable_tau=True, fused_heads=True),
            twin_node_id="3.2.3.4.1",
            emit_to_twin=True,
            phase_frequency=0.010,
        ),
        LobePairSpec(
            domain_id="memory",
            process_role="working memory, temporal association, recurrent context",
            bus_dim=b,
            ncp=NCPModuleSpec("memory_ncp", "bounded recurrent working memory", b, max(320, b * 5), b, bond_dim=5, tau_min=3.0, tau_max=48.0, leaky_lambda=0.18, num_mvm_outputs=4, tt_strict=False, integration_method="heun"),
            cfc=CFCModuleSpec("memory_cfc", "continuous temporal persistence and decay", b, max(160, b * 2), b, num_cells=6, bond_dim=6, tt_strict=False, learnable_tau=True, fused_heads=True),
            twin_node_id="3.2.3.4.2",
            emit_to_twin=True,
            phase_frequency=0.007,
        ),
        LobePairSpec(
            domain_id="semantic",
            process_role="semantic memory, conceptual association, language-ready meaning state",
            bus_dim=b,
            ncp=NCPModuleSpec("semantic_ncp", "bounded semantic recurrent reasoning", b, max(384, b * 6), b, bond_dim=5, tau_min=2.0, tau_max=40.0, leaky_lambda=0.22, num_mvm_outputs=4, tt_strict=False, integration_method="heun"),
            cfc=CFCModuleSpec("semantic_cfc", "semantic activation timing and concept persistence", b, max(128, b * 2), b, num_cells=4, bond_dim=6, tt_strict=False, learnable_tau=True, fused_heads=True),
            twin_node_id="3.2.3.4.3",
            emit_to_twin=True,
            phase_frequency=0.009,
        ),
        LobePairSpec(
            domain_id="planning",
            process_role="executive planning, action policy, goal-directed control",
            bus_dim=b,
            ncp=NCPModuleSpec("planning_ncp", "bounded policy generation and action selection", b, max(320, b * 5), b, bond_dim=5, tau_min=1.5, tau_max=32.0, leaky_lambda=0.25, num_freq_gates=2, num_mvm_outputs=4, tt_strict=False, integration_method="heun"),
            cfc=CFCModuleSpec("planning_cfc", "action timing and sequencing", b, max(160, b * 2), b, num_cells=5, bond_dim=6, tt_strict=False, learnable_tau=True, fused_heads=True),
            twin_node_id="3.2.3.4.4",
            emit_to_twin=True,
            phase_frequency=0.012,
        ),
        LobePairSpec(
            domain_id="regulation",
            process_role="homeostasis, coherence monitoring, safety gating, damping",
            bus_dim=b,
            ncp=NCPModuleSpec("regulation_ncp", "bounded anomaly interpretation and safety gating", b, max(160, b * 3), b, bond_dim=4, tau_min=1.0, tau_max=22.0, leaky_lambda=0.35, num_freq_gates=2, tt_strict=False, integration_method="heun"),
            cfc=CFCModuleSpec("regulation_cfc", "continuous health rhythm and damping", b, max(96, b), b, num_cells=3, bond_dim=5, tt_strict=False, learnable_tau=True, fused_heads=True),
            twin_node_id="3.2.3.4.5",
            emit_to_twin=True,
            phase_frequency=0.015,
        ),
    ]


# =============================================================================
# Smoke test
# =============================================================================

def _smoke_test() -> None:
    print("=== Cognitive Lobe Runtime smoke test ===")
    system = CognitiveLobeSystem.build_default(bus_dim=64, seed=2027, strict_imports=False)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.normal(size=64).astype(np.float32)
        res = system.step_once(x, dt=0.1)
        print(f"tick={system.tick} stable={system.health_metrics()['stable']} domains={sorted(res.keys())}")

    system.create_assembly(
        assembly_id="smoke_assembly",
        module_ids=["sensory_ncp", "memory_ncp", "planning_cfc"],
        output_dim=64,
        strategy="sequential",
    )
    y, trace = system.run_assembly("smoke_assembly", rng.normal(size=64).astype(np.float32), dt=0.1)
    print("assembly output:", y.shape, "norm=", round(float(np.linalg.norm(y)), 6))
    print("health stable:", system.health_metrics()["stable"])
    print("=== Done ===")


if __name__ == "__main__":
    _smoke_test()
