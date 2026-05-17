#!/usr/bin/env python3
# akkurat_atom_hybrid.py
"""
Akkurat / AtomTN Hybrid Runtime
===============================

Production-ready orchestrator that fuses:

1. AtomTN neuromorphic reservoir features via atom_adapter_runtime.py
2. A lightweight tensorized/Numpy CfC-style controller
3. Optional Akkurat digital twin governance updates

The runtime is deliberately import-safe:
- importing this file does not construct AtomTN objects
- heavy AtomTN objects are built only when AkkuratAtomHybridRuntime is created
- every optional attachment records explicit status/error metadata

Primary public API
------------------
- AkkuratAtomHybridConfig
- AkkuratAtomHybridStepResult
- AkkuratAtomHybridRuntime
- build_hybrid_runtime(...)

CLI examples
------------
From:
    C:\\Users\\ali_z\\ANU AI\\Akkurat\\cognitive_model_3

Run fast demo:
    python akkurat_atom_hybrid.py

Check attachment status:
    python akkurat_atom_hybrid.py --mode status

Run with AtomTN smoke profile:
    python akkurat_atom_hybrid.py --profile smoke --steps 3

Run without AtomTN:
    python akkurat_atom_hybrid.py --disable-atom

Run with explicit AtomTN root:
    python akkurat_atom_hybrid.py --atomtn-root "C:\\Users\\ali_z\\ANU AI\\AtomTN"
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
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

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
    if not path.exists():
        return None

    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
    return path


def configure_paths(
    *,
    atomtn_root: Optional[Union[str, os.PathLike]] = None,
    akkurat_root: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Optional[str]]:
    """
    Add likely AtomTN and Akkurat paths to sys.path.

    Handles the local layout:
        C:\\Users\\ali_z\\ANU AI\\AtomTN
        C:\\Users\\ali_z\\ANU AI\\Akkurat\\cognitive_model_3
    """
    here = _module_dir()
    anu_root = here.parent.parent if here.name == "cognitive_model_3" else here.parent

    atom_candidates = [
        atomtn_root,
        os.environ.get("ATOMTN_ROOT"),
        os.environ.get("AKKURAT_ATOMTN_ROOT"),
        anu_root / "AtomTN",
        here.parent / "AtomTN",
        here.parent.parent / "AtomTN",
        Path.cwd(),
    ]

    akkurat_candidates = [
        akkurat_root,
        os.environ.get("AKKURAT_ROOT"),
        os.environ.get("AKKURAT_COGNITIVE_ROOT"),
        here,
        Path.cwd(),
    ]

    found_atom = None
    for c in atom_candidates:
        p = _add_path(c)
        if p is not None and (p / "neuromorphic.py").exists():
            found_atom = p
            break

    found_akkurat = None
    for c in akkurat_candidates:
        p = _add_path(c)
        if p is not None and (p / "atom_adapter_runtime.py").exists():
            found_akkurat = p
            break

    _add_path(here)

    return {
        "atomtn_root": None if found_atom is None else str(found_atom),
        "akkurat_root": None if found_akkurat is None else str(found_akkurat),
    }


_PATHS = configure_paths()


# =============================================================================
# Generic helpers
# =============================================================================

_EPS = 1e-9


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_seed(seed: int, salt: str) -> int:
    h = 2166136261 ^ int(seed)
    for b in salt.encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_norm(x: Any) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return 0.0
        val = float(np.linalg.norm(arr.reshape(-1)))
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _sigmoid_stable(x: Any) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32)
    z = np.clip(z, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _as_float32_vector(
    x: Any,
    *,
    expected_dim: Optional[int] = None,
    resize: bool = True,
    name: str = "vector",
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
                n = min(int(arr.size), d)
                out[:n] = arr[:n]
            arr = out

    return arr


def _l2_normalize(x: Any, eps: float = _EPS) -> np.ndarray:
    arr = _as_float32_vector(x)
    n = _safe_norm(arr)
    if n <= eps:
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
        return np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()

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
    try:
        names = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in values.items() if k in names}
    except Exception:
        return dict(values)


def _resize_feature(y: Any, dim: int) -> np.ndarray:
    arr = _as_float32_vector(y)
    d = int(max(1, dim))
    if arr.size == d:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((d,), dtype=np.float32)
    if arr.size:
        n = min(d, int(arr.size))
        out[:n] = arr[:n]
    return out


# =============================================================================
# Optional module status
# =============================================================================


def atom_adapter_status() -> Dict[str, Any]:
    configure_paths()
    try:
        mod = importlib.import_module("atom_adapter_runtime")
        status_fn = getattr(mod, "atomtn_status", None)
        atom_status = status_fn() if callable(status_fn) else {"available": True}
        return {
            "available": True,
            "module": getattr(mod, "__file__", ""),
            "atomtn": _json_safe(atom_status),
        }
    except BaseException as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
            "atomtn": {"available": False, "import_error": repr(exc)},
        }


def tensor_network_status() -> Dict[str, Any]:
    configure_paths()
    try:
        mod = importlib.import_module("tn")
        return {
            "available": True,
            "module": getattr(mod, "__file__", ""),
            "has_tensor_train": hasattr(mod, "TensorTrain"),
            "backend_status": (
                mod.optional_backend_status() if hasattr(mod, "optional_backend_status") else {}
            ),
        }
    except BaseException as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
        }


def digital_twin_status() -> Dict[str, Any]:
    configure_paths()
    try:
        mod = importlib.import_module("digital_twin_kernel")
        return {
            "available": hasattr(mod, "DigitalTwinsBuilder"),
            "module": getattr(mod, "__file__", ""),
        }
    except BaseException as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
        }


# =============================================================================
# Config and result containers
# =============================================================================


@dataclass
class AkkuratAtomHybridConfig:
    runtime_id: str = "akkurat_atom_hybrid_demo"

    mode: str = "hybrid"        # hybrid | atom_only | tcfc_only | passthrough
    fusion: str = "gated"       # gated | additive | average | concat | atom_priority | tcfc_priority

    input_dim: int = 3
    control_dim: int = 16
    output_dim: int = 16

    enable_atom: bool = True
    enable_tcfc: bool = True
    enable_digital_twin: bool = False

    atomtn_root: str = ""
    akkurat_root: str = ""

    atom_profile: str = "fast"
    atom_method: str = "euler_legacy"
    atom_output_dim: int = 64
    atom_dt: float = 0.05
    atom_encoder_scale: float = 20.0
    atom_memory_damping: float = 0.1
    atom_noncommutative: bool = False
    atom_fail_fast: bool = False
    atom_strict: bool = False

    seed: int = 2027
    tcfc_backend: str = "auto"       # auto | tensor_train | dense
    tcfc_hidden_dim: int = 16
    tcfc_max_bond_dim: int = 4
    tcfc_energy_tol: Optional[float] = 0.999
    tcfc_leak_floor: float = 0.03
    tcfc_leak_scale: float = 1.0
    tcfc_input_clip: float = 25.0
    tcfc_state_clip: float = 25.0
    tcfc_regularize_norm: float = 10.0

    gate_bias: float = 0.0
    fusion_normalize_inputs: bool = False
    output_activation: str = "tanh"
    output_clip: Optional[float] = 10.0

    digital_twin_vector_dim: int = 128
    digital_twin_sketch_dim: Optional[int] = 64
    digital_twin_history_capacity: int = 64
    digital_twin_use_tn_projection: bool = False
    governance_updates: bool = True

    strict: bool = False
    fail_fast: bool = False
    max_step_seconds_warn: float = 1.0

    def normalized(self) -> "AkkuratAtomHybridConfig":
        cfg = copy.deepcopy(self)

        cfg.mode = str(cfg.mode or "hybrid").lower().strip()
        if cfg.mode not in {"hybrid", "atom_only", "tcfc_only", "passthrough"}:
            cfg.mode = "hybrid"

        cfg.fusion = str(cfg.fusion or "gated").lower().strip()
        if cfg.fusion not in {"gated", "additive", "average", "concat", "atom_priority", "tcfc_priority"}:
            cfg.fusion = "gated"

        cfg.input_dim = int(max(1, cfg.input_dim))
        cfg.control_dim = int(max(1, cfg.control_dim))
        cfg.output_dim = int(max(1, cfg.output_dim))
        cfg.atom_output_dim = int(max(1, cfg.atom_output_dim))
        cfg.tcfc_hidden_dim = int(max(1, cfg.tcfc_hidden_dim))

        cfg.seed = int(cfg.seed)
        cfg.atom_dt = max(0.0, _safe_float(cfg.atom_dt, 0.05))
        cfg.atom_profile = str(cfg.atom_profile or "fast").lower().strip()
        cfg.atom_method = str(cfg.atom_method or "euler_legacy").lower().strip()

        cfg.tcfc_max_bond_dim = int(max(1, cfg.tcfc_max_bond_dim))
        cfg.tcfc_leak_floor = float(np.clip(_safe_float(cfg.tcfc_leak_floor, 0.03), 0.0, 1.0))
        cfg.tcfc_leak_scale = max(0.0, _safe_float(cfg.tcfc_leak_scale, 1.0))
        cfg.tcfc_input_clip = max(0.0, _safe_float(cfg.tcfc_input_clip, 25.0))
        cfg.tcfc_state_clip = max(0.0, _safe_float(cfg.tcfc_state_clip, 25.0))
        cfg.tcfc_regularize_norm = max(0.0, _safe_float(cfg.tcfc_regularize_norm, 10.0))

        cfg.digital_twin_vector_dim = int(max(8, cfg.digital_twin_vector_dim))
        if cfg.digital_twin_sketch_dim is not None:
            cfg.digital_twin_sketch_dim = int(
                max(8, min(int(cfg.digital_twin_sketch_dim), cfg.digital_twin_vector_dim))
            )
        cfg.digital_twin_history_capacity = int(max(1, cfg.digital_twin_history_capacity))

        return cfg


@dataclass
class AkkuratAtomHybridStepResult:
    ok: bool
    step: int
    ts: str

    output: np.ndarray
    atom_features: np.ndarray
    tcfc_state: np.ndarray
    fused_state: np.ndarray

    metrics: Dict[str, Any]
    health: Dict[str, Any]
    atom: Dict[str, Any]
    digital_twin: Dict[str, Any]

    elapsed_s: float = 0.0
    error: Optional[str] = None
    warnings: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(self)


# =============================================================================
# Lightweight tensorized / dense CfC controller
# =============================================================================


class _HybridCfCController:
    """
    Small deterministic continuous-time controller.

    It prefers tn.TensorTrain for the main projections when available, but always
    falls back to dense NumPy matrices.
    """

    def __init__(self, cfg: AkkuratAtomHybridConfig):
        self.cfg = cfg.normalized()
        self.input_dim = int(self.cfg.input_dim)
        self.hidden_dim = int(self.cfg.tcfc_hidden_dim)
        self.ctrl_in_dim = self.input_dim + self.hidden_dim

        self.rng = np.random.default_rng(_stable_seed(self.cfg.seed, "hybrid_tcfc"))
        self.state = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.step_counter = 0

        self.backend = "dense"
        self.attachment_error: Optional[str] = None
        self.tt_f = None
        self.tt_g = None
        self.tt_h = None

        scale = 1.0 / math.sqrt(max(1, self.ctrl_in_dim))
        self.W_f = self.rng.normal(0.0, scale, size=(self.hidden_dim, self.ctrl_in_dim)).astype(np.float32)
        self.W_g = self.rng.normal(0.0, scale, size=(self.hidden_dim, self.ctrl_in_dim)).astype(np.float32)
        self.W_h = self.rng.normal(0.0, scale, size=(self.hidden_dim, self.ctrl_in_dim)).astype(np.float32)

        self.b_f = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.b_g = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.b_h = np.zeros((self.hidden_dim,), dtype=np.float32)

        self._try_build_tensor_train_backend()

    @property
    def attached(self) -> bool:
        return self.backend in {"tensor_train", "dense"}

    def _factor_modes(self, n: int, tn_mod: Any) -> List[int]:
        if hasattr(tn_mod, "factorize_into_modes"):
            try:
                modes = list(tn_mod.factorize_into_modes(int(n), 2))
                if int(np.prod(modes)) == int(n):
                    return [int(x) for x in modes]
            except Exception:
                pass

        a = int(math.isqrt(int(n)))
        while a > 1:
            if int(n) % a == 0:
                return [a, int(n) // a]
            a -= 1
        return [1, int(n)]

    def _try_build_tensor_train_backend(self) -> None:
        want = str(self.cfg.tcfc_backend or "auto").lower().strip()
        if want == "dense":
            self.backend = "dense"
            return

        try:
            tn_mod = importlib.import_module("tn")
            TensorTrain = getattr(tn_mod, "TensorTrain", None)
            if TensorTrain is None:
                raise RuntimeError("tn.TensorTrain is unavailable.")

            out_dims = self._factor_modes(self.hidden_dim, tn_mod)
            in_dims = self._factor_modes(self.ctrl_in_dim, tn_mod)

            def build_tt(W: np.ndarray) -> Any:
                if hasattr(TensorTrain, "from_dense"):
                    return TensorTrain.from_dense(
                        W.astype(np.float64),
                        output_dims=out_dims,
                        input_dims=in_dims,
                        max_bond_dim=int(self.cfg.tcfc_max_bond_dim),
                        dtype=np.float32,
                        energy_tol=self.cfg.tcfc_energy_tol,
                        device="cpu",
                    )
                raise RuntimeError("TensorTrain.from_dense is unavailable.")

            self.tt_f = build_tt(self.W_f)
            self.tt_g = build_tt(self.W_g)
            self.tt_h = build_tt(self.W_h)
            self.backend = "tensor_train"
            self.attachment_error = None

        except BaseException as exc:
            self.attachment_error = repr(exc)
            if want == "tensor_train" and (self.cfg.fail_fast or self.cfg.strict):
                raise
            self.backend = "dense"
            self.tt_f = self.tt_g = self.tt_h = None

    def _linear(self, W: np.ndarray, b: np.ndarray, tt: Any, x: np.ndarray) -> np.ndarray:
        if self.backend == "tensor_train" and tt is not None:
            try:
                y = tt.apply(x.astype(np.float32))
                return (_as_float32_vector(y, expected_dim=self.hidden_dim) + b).astype(np.float32)
            except Exception as exc:
                self.attachment_error = f"tt_apply_failed: {exc!r}"
                self.backend = "dense"

        return (W @ x + b).astype(np.float32)

    def step(self, external_input: Any, *, dt: float = 0.05) -> np.ndarray:
        x = _as_float32_vector(external_input, expected_dim=self.input_dim, name="TCfC input")
        if self.cfg.tcfc_input_clip > 0:
            np.clip(x, -self.cfg.tcfc_input_clip, self.cfg.tcfc_input_clip, out=x)

        z_in = np.concatenate([x, self.state], axis=0).astype(np.float32)

        f = self._linear(self.W_f, self.b_f, self.tt_f, z_in)
        g = self._linear(self.W_g, self.b_g, self.tt_g, z_in)
        h = self._linear(self.W_h, self.b_h, self.tt_h, z_in)

        g = np.tanh(g).astype(np.float32)
        h = np.tanh(h).astype(np.float32)

        dtv = max(0.0, _safe_float(dt, 0.05))
        leak_arg = -f * np.float32(max(self.cfg.tcfc_leak_floor, dtv * self.cfg.tcfc_leak_scale))
        leak_arg = np.clip(leak_arg, -30.0, 30.0)
        gate = _sigmoid_stable(leak_arg)

        self.state = (gate * g + (1.0 - gate) * h).astype(np.float32)

        if self.cfg.tcfc_state_clip > 0:
            np.clip(self.state, -self.cfg.tcfc_state_clip, self.cfg.tcfc_state_clip, out=self.state)

        if self.cfg.tcfc_regularize_norm > 0:
            n = _safe_norm(self.state)
            if n > self.cfg.tcfc_regularize_norm:
                self.state *= np.float32(self.cfg.tcfc_regularize_norm / max(n, _EPS))

        self.step_counter += 1
        return self.state.copy()

    def reset_state(self, value: float = 0.0) -> None:
        self.state.fill(np.float32(value))
        self.step_counter = 0

    def health_metrics(self) -> Dict[str, Any]:
        finite = bool(np.all(np.isfinite(self.state)))
        return {
            "kind": "HybridCfCController",
            "is_stable": finite,
            "has_nan": not finite,
            "backend": self.backend,
            "hidden_dim": int(self.hidden_dim),
            "state_norm": _safe_norm(self.state),
            "step_counter": int(self.step_counter),
            "attachment_error": self.attachment_error,
        }

    def parameter_count(self) -> int:
        return int(self.W_f.size + self.W_g.size + self.W_h.size + self.b_f.size + self.b_g.size + self.b_h.size)


# =============================================================================
# Hybrid runtime
# =============================================================================


class AkkuratAtomHybridRuntime:
    """
    Hybrid controller that fuses AtomTN reservoir features with a CfC controller.
    """

    def __init__(
        self,
        cfg: Optional[AkkuratAtomHybridConfig] = None,
        *,
        atom_runtime: Any = None,
        tcfc_controller: Optional[_HybridCfCController] = None,
    ):
        self.cfg = (cfg or AkkuratAtomHybridConfig()).normalized()

        configure_paths(
            atomtn_root=(self.cfg.atomtn_root or None),
            akkurat_root=(self.cfg.akkurat_root or None),
        )

        self.step_count = 0
        self.last_result: Optional[AkkuratAtomHybridStepResult] = None

        self.attachment_errors: Dict[str, Optional[str]] = {
            "atom": None,
            "tcfc": None,
            "digital_twin": None,
        }

        self.atom_runtime = atom_runtime
        self.tcfc = tcfc_controller

        self._init_fusion_weights()

        if self.cfg.enable_atom and self.atom_runtime is None:
            self._attach_atom_runtime()

        if self.cfg.enable_tcfc and self.tcfc is None:
            self._attach_tcfc_controller()

    @property
    def atom_attached(self) -> bool:
        if self.atom_runtime is None:
            return False
        try:
            return bool(getattr(self.atom_runtime, "atom_attached", True))
        except Exception:
            return True

    @property
    def tcfc_attached(self) -> bool:
        return bool(self.tcfc is not None and self.tcfc.attached)

    @property
    def digital_twin_attached(self) -> bool:
        if self.atom_runtime is not None:
            try:
                return bool(getattr(self.atom_runtime, "digital_twin_attached", False))
            except Exception:
                return False
        return False

    def _attach_atom_runtime(self) -> None:
        try:
            mod = importlib.import_module("atom_adapter_runtime")
            AtomAdapterConfig = getattr(mod, "AtomAdapterConfig")
            build_adapter = getattr(mod, "build_adapter")

            adapter_kwargs = _dataclass_kwargs(
                AtomAdapterConfig,
                {
                    "adapter_id": f"{self.cfg.runtime_id}_atom",
                    "input_dim": int(self.cfg.input_dim),
                    "feature_dim": int(self.cfg.atom_output_dim),
                    "output_dim": int(self.cfg.atom_output_dim),
                    "enable_atomtn": bool(self.cfg.enable_atom),
                    "enable_digital_twin": bool(self.cfg.enable_digital_twin),
                    "profile": str(self.cfg.atom_profile),
                    "method": str(self.cfg.atom_method),
                    "dt": float(self.cfg.atom_dt),
                    "seed": int(self.cfg.seed),
                    "encoder_scale": float(self.cfg.atom_encoder_scale),
                    "memory_damping": float(self.cfg.atom_memory_damping),
                    "noncommutative": bool(self.cfg.atom_noncommutative),
                    "digital_twin_vector_dim": int(self.cfg.digital_twin_vector_dim),
                    "digital_twin_sketch_dim": self.cfg.digital_twin_sketch_dim,
                    "digital_twin_history_capacity": int(self.cfg.digital_twin_history_capacity),
                    "digital_twin_use_tn_projection": bool(self.cfg.digital_twin_use_tn_projection),
                    "enable_governance_updates": bool(self.cfg.governance_updates),
                    "strict": bool(self.cfg.atom_strict),
                    "fail_fast": bool(self.cfg.atom_fail_fast or self.cfg.fail_fast),
                    "atomtn_root": str(self.cfg.atomtn_root or ""),
                    "akkurat_root": str(self.cfg.akkurat_root or ""),
                },
            )

            adapter_cfg = AtomAdapterConfig(**adapter_kwargs)
            self.atom_runtime = build_adapter(adapter_cfg)
            self.attachment_errors["atom"] = None

            if getattr(self.atom_runtime, "attachment_errors", None):
                err = self.atom_runtime.attachment_errors.get("atom")
                if err:
                    self.attachment_errors["atom"] = err

        except BaseException as exc:
            self.atom_runtime = None
            self.attachment_errors["atom"] = repr(exc)
            if self.cfg.fail_fast or self.cfg.strict:
                raise

    def _attach_tcfc_controller(self) -> None:
        try:
            self.tcfc = _HybridCfCController(self.cfg)
            self.attachment_errors["tcfc"] = self.tcfc.attachment_error
        except BaseException as exc:
            self.tcfc = None
            self.attachment_errors["tcfc"] = repr(exc)
            if self.cfg.fail_fast or self.cfg.strict:
                raise

    def _init_fusion_weights(self) -> None:
        rng = np.random.default_rng(_stable_seed(self.cfg.seed, "hybrid_fusion"))

        atom_dim = int(self.cfg.atom_output_dim)
        ctrl = int(self.cfg.control_dim)
        hidden = int(self.cfg.tcfc_hidden_dim)
        inp = int(self.cfg.input_dim)

        self.atom_proj = rng.normal(0.0, 1.0 / math.sqrt(max(1, atom_dim)), size=(ctrl, atom_dim)).astype(np.float32)
        self.tcfc_proj = rng.normal(0.0, 1.0 / math.sqrt(max(1, hidden)), size=(ctrl, hidden)).astype(np.float32)
        self.concat_proj = rng.normal(0.0, 1.0 / math.sqrt(max(1, 2 * ctrl)), size=(ctrl, 2 * ctrl)).astype(np.float32)

        gate_in = 2 * ctrl + inp
        self.gate_W = rng.normal(0.0, 1.0 / math.sqrt(max(1, gate_in)), size=(ctrl, gate_in)).astype(np.float32)
        self.gate_b = np.full((ctrl,), float(self.cfg.gate_bias), dtype=np.float32)

        self.output_W = rng.normal(0.0, 1.0 / math.sqrt(max(1, ctrl)), size=(self.cfg.output_dim, ctrl)).astype(np.float32)
        self.output_b = np.zeros((self.cfg.output_dim,), dtype=np.float32)

    def _project_atom(self, atom_features: Any) -> np.ndarray:
        a = _resize_feature(atom_features, self.cfg.atom_output_dim)
        if self.cfg.fusion_normalize_inputs:
            a = _l2_normalize(a)
        return np.tanh(self.atom_proj @ a).astype(np.float32)

    def _project_tcfc(self, tcfc_state: Any) -> np.ndarray:
        h = _resize_feature(tcfc_state, self.cfg.tcfc_hidden_dim)
        if self.cfg.fusion_normalize_inputs:
            h = _l2_normalize(h)
        return np.tanh(self.tcfc_proj @ h).astype(np.float32)

    def _fuse(self, x: np.ndarray, atom_features: np.ndarray, tcfc_state: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.mode)
        fusion = str(self.cfg.fusion)

        atom_ctrl = self._project_atom(atom_features)
        tcfc_ctrl = self._project_tcfc(tcfc_state)

        if mode == "atom_only":
            return atom_ctrl
        if mode == "tcfc_only":
            return tcfc_ctrl
        if mode == "passthrough":
            passthrough = np.zeros((self.cfg.control_dim,), dtype=np.float32)
            n = min(self.cfg.control_dim, x.size)
            passthrough[:n] = x[:n]
            return passthrough

        if fusion == "atom_priority":
            return atom_ctrl if self.atom_attached else tcfc_ctrl

        if fusion == "tcfc_priority":
            return tcfc_ctrl if self.tcfc_attached else atom_ctrl

        if fusion == "additive":
            return np.tanh(atom_ctrl + tcfc_ctrl).astype(np.float32)

        if fusion == "average":
            return (0.5 * (atom_ctrl + tcfc_ctrl)).astype(np.float32)

        if fusion == "concat":
            z = np.concatenate([atom_ctrl, tcfc_ctrl], axis=0).astype(np.float32)
            return np.tanh(self.concat_proj @ z).astype(np.float32)

        gate_in = np.concatenate([atom_ctrl, tcfc_ctrl, x], axis=0).astype(np.float32)
        gate = _sigmoid_stable(self.gate_W @ gate_in + self.gate_b)
        return (gate * atom_ctrl + (1.0 - gate) * tcfc_ctrl).astype(np.float32)

    def _decode_output(self, fused: np.ndarray) -> np.ndarray:
        y = (self.output_W @ fused + self.output_b).astype(np.float32)

        act = str(self.cfg.output_activation or "tanh").lower().strip()
        if act == "tanh":
            y = np.tanh(y).astype(np.float32)
        elif act == "sigmoid":
            y = _sigmoid_stable(y)
        elif act != "linear":
            y = np.tanh(y).astype(np.float32)

        if self.cfg.output_clip is not None and self.cfg.output_clip > 0:
            np.clip(y, -float(self.cfg.output_clip), float(self.cfg.output_clip), out=y)

        return y.astype(np.float32, copy=False)

    def step(self, external_input: Any, *, dt: Optional[float] = None) -> AkkuratAtomHybridStepResult:
        t0 = time.perf_counter()
        err = None
        ok = True
        warnings: List[str] = []

        x = _as_float32_vector(
            external_input,
            expected_dim=self.cfg.input_dim,
            name="AkkuratAtomHybridRuntime input",
        )
        dtv = self.cfg.atom_dt if dt is None else _safe_float(dt, self.cfg.atom_dt)

        atom_features = np.zeros((self.cfg.atom_output_dim,), dtype=np.float32)
        tcfc_state = np.zeros((self.cfg.tcfc_hidden_dim,), dtype=np.float32)
        atom_payload: Dict[str, Any] = {}

        try:
            if self.cfg.enable_atom and self.atom_runtime is not None:
                atom_res = self.atom_runtime.step(x, dt=dtv)
                atom_features = _resize_feature(getattr(atom_res, "features", atom_res), self.cfg.atom_output_dim)
                atom_payload = atom_res.to_dict() if hasattr(atom_res, "to_dict") else _json_safe(atom_res)
            elif self.cfg.mode in {"hybrid", "atom_only"}:
                warnings.append("atom_unattached")
        except BaseException as exc:
            self.attachment_errors["atom"] = f"step_failed: {exc!r}"
            warnings.append("atom_step_failed")
            if self.cfg.fail_fast or self.cfg.strict:
                raise

        try:
            if self.cfg.enable_tcfc and self.tcfc is not None:
                tcfc_state = self.tcfc.step(x, dt=dtv)
            elif self.cfg.mode in {"hybrid", "tcfc_only"}:
                warnings.append("tcfc_unattached")
        except BaseException as exc:
            self.attachment_errors["tcfc"] = f"step_failed: {exc!r}"
            warnings.append("tcfc_step_failed")
            if self.cfg.fail_fast or self.cfg.strict:
                raise

        try:
            fused = self._fuse(x, atom_features, tcfc_state)
            output = self._decode_output(fused)

            health = self.health_metrics()
            metrics = self._metrics(output, atom_features, tcfc_state, fused)

            if not bool(health.get("stable", False)):
                ok = False
                warnings.append("hybrid_unstable")

            digital = self._digital_twin_metadata(atom_payload)

        except BaseException as exc:
            if self.cfg.fail_fast or self.cfg.strict:
                raise
            err = repr(exc)
            ok = False
            output = np.zeros((self.cfg.output_dim,), dtype=np.float32)
            fused = np.zeros((self.cfg.control_dim,), dtype=np.float32)
            health = {"stable": False, "last_error": err}
            metrics = {}
            digital = {}

        elapsed = float(time.perf_counter() - t0)
        if elapsed > float(self.cfg.max_step_seconds_warn):
            warnings.append(f"slow_step({elapsed:.3f}s)")

        result = AkkuratAtomHybridStepResult(
            ok=bool(ok),
            step=int(self.step_count),
            ts=_now_iso(),
            output=output.astype(np.float32, copy=False),
            atom_features=atom_features.astype(np.float32, copy=False),
            tcfc_state=tcfc_state.astype(np.float32, copy=False),
            fused_state=fused.astype(np.float32, copy=False),
            metrics=_json_safe(metrics),
            health=_json_safe(health),
            atom=_json_safe(atom_payload),
            digital_twin=_json_safe(digital),
            elapsed_s=elapsed,
            error=err,
            warnings=warnings,
        )

        self.last_result = result
        self.step_count += 1
        return result

    def step_sequence(
        self,
        X: Any,
        *,
        dt: Union[float, Sequence[float], None] = None,
    ) -> List[AkkuratAtomHybridStepResult]:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"step_sequence expects 1D/2D input, got shape {arr.shape}")

        if isinstance(dt, Sequence) and not isinstance(dt, (str, bytes)):
            dts = list(dt)
            if len(dts) != arr.shape[0]:
                raise ValueError("dt sequence length must match input rows")
        else:
            dts = [dt] * int(arr.shape[0])

        return [self.step(arr[i], dt=dts[i]) for i in range(arr.shape[0])]

    def _metrics(
        self,
        output: np.ndarray,
        atom_features: np.ndarray,
        tcfc_state: np.ndarray,
        fused: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "output_dim": int(output.size),
            "output_norm": _safe_norm(output),
            "output_mean": float(np.mean(output)) if output.size else 0.0,
            "output_std": float(np.std(output)) if output.size else 0.0,
            "atom_feature_dim": int(atom_features.size),
            "atom_feature_norm": _safe_norm(atom_features),
            "tcfc_state_dim": int(tcfc_state.size),
            "tcfc_state_norm": _safe_norm(tcfc_state),
            "fused_dim": int(fused.size),
            "fused_norm": _safe_norm(fused),
        }

    def _digital_twin_metadata(self, atom_payload: Any) -> Dict[str, Any]:
        if self.atom_runtime is None:
            return {"attached": False, "updated": False}

        try:
            if isinstance(atom_payload, Mapping):
                digital = atom_payload.get("digital_twin", {})
                if isinstance(digital, Mapping):
                    return dict(digital)
        except Exception:
            pass

        return {"attached": self.digital_twin_attached, "updated": False}

    def health_metrics(self) -> Dict[str, Any]:
        atom_health = {}
        tcfc_health = {}

        if self.atom_runtime is not None:
            try:
                snap = self.atom_runtime.snapshot() if hasattr(self.atom_runtime, "snapshot") else {}
                atom_health = _json_safe(snap)
            except Exception as exc:
                atom_health = {"error": repr(exc)}

        if self.tcfc is not None:
            tcfc_health = self.tcfc.health_metrics()

        finite_last = True
        if self.last_result is not None:
            finite_last = bool(
                np.all(np.isfinite(self.last_result.output))
                and np.all(np.isfinite(self.last_result.fused_state))
            )

        stable = bool(
            finite_last
            and (not self.cfg.enable_tcfc or self.tcfc_attached)
            and (self.cfg.mode != "atom_only" or self.atom_attached)
        )

        return {
            "kind": "AkkuratAtomHybridRuntime",
            "stable": stable,
            "atom_attached": self.atom_attached,
            "tcfc_attached": self.tcfc_attached,
            "digital_twin_attached": self.digital_twin_attached,
            "step_count": int(self.step_count),
            "attachment_errors": dict(self.attachment_errors),
            "atom": atom_health,
            "tcfc": tcfc_health,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "AkkuratAtomHybridRuntime",
            "config": _json_safe(self.cfg),
            "step_count": int(self.step_count),
            "health": self.health_metrics(),
            "last_result": None if self.last_result is None else self.last_result.to_dict(),
        }

    def summary(self) -> str:
        h = self.health_metrics()
        lines = [
            "[AkkuratAtomHybridRuntime]",
            f"- id: {self.cfg.runtime_id}",
            f"- mode: {self.cfg.mode}",
            f"- fusion: {self.cfg.fusion}",
            f"- input_dim: {self.cfg.input_dim}",
            f"- control_dim: {self.cfg.control_dim}",
            f"- output_dim: {self.cfg.output_dim}",
            f"- step_count: {self.step_count}",
            f"- stable: {bool(h.get('stable', False))}",
            f"- atom_attached: {self.atom_attached}",
            f"- tcfc_attached: {self.tcfc_attached}",
            f"- digital_twin_attached: {self.digital_twin_attached}",
        ]

        errs = {k: v for k, v in self.attachment_errors.items() if v}
        if errs:
            lines.append(f"- attachment_errors: {json.dumps(_json_safe(errs), ensure_ascii=False)}")

        if self.tcfc is not None:
            lines.append(f"- tcfc_backend: {self.tcfc.backend}")

        return "\n".join(lines) + "\n"

    def reset_state(self, value: float = 0.0) -> None:
        self.step_count = 0
        self.last_result = None

        if self.atom_runtime is not None and hasattr(self.atom_runtime, "reset_state"):
            try:
                self.atom_runtime.reset_state()
            except Exception as exc:
                self.attachment_errors["atom"] = f"reset_failed: {exc!r}"
                if self.cfg.fail_fast or self.cfg.strict:
                    raise

        if self.tcfc is not None:
            self.tcfc.reset_state(value=value)

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


def build_hybrid_runtime(
    config: Optional[AkkuratAtomHybridConfig] = None,
    **overrides: Any,
) -> AkkuratAtomHybridRuntime:
    cfg = copy.deepcopy(config or AkkuratAtomHybridConfig())

    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown AkkuratAtomHybridConfig field: {k}")
        setattr(cfg, k, v)

    return AkkuratAtomHybridRuntime(cfg)


# Back-compat alias.
build_runtime = build_hybrid_runtime


# =============================================================================
# CLI
# =============================================================================


def _demo_inputs(input_dim: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    steps = int(max(1, steps))

    t = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32)
    cols = [np.sin(t), np.cos(t)]

    while len(cols) < int(input_dim):
        cols.append(rng.normal(0.0, 0.05, size=steps).astype(np.float32))

    return np.stack(cols[: int(input_dim)], axis=1).astype(np.float32)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akkurat / AtomTN hybrid runtime.")

    p.add_argument("--mode", choices=["status", "demo", "smoke", "export-snapshot"], default="demo")
    p.add_argument("--runtime-mode", choices=["hybrid", "atom_only", "tcfc_only", "passthrough"], default="hybrid")
    p.add_argument("--fusion", choices=["gated", "additive", "average", "concat", "atom_priority", "tcfc_priority"], default="gated")

    p.add_argument("--input-dim", type=int, default=3)
    p.add_argument("--control-dim", type=int, default=16)
    p.add_argument("--output-dim", type=int, default=16)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2027)

    p.add_argument("--profile", choices=["smoke", "fast", "balanced", "accurate"], default="fast")
    p.add_argument("--method", default="euler_legacy")

    p.add_argument("--disable-atom", action="store_true")
    p.add_argument("--disable-tcfc", action="store_true")
    p.add_argument("--enable-digital-twin", action="store_true")

    p.add_argument("--tcfc-backend", choices=["auto", "tensor_train", "dense"], default="auto")

    p.add_argument("--atomtn-root", default="")
    p.add_argument("--akkurat-root", default="")
    p.add_argument("--output", default="")

    return p


def _status_report() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "paths": configure_paths(),
        "atom_adapter": atom_adapter_status(),
        "tensor_network": tensor_network_status(),
        "digital_twin": digital_twin_status(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    configure_paths(
        atomtn_root=(args.atomtn_root or None),
        akkurat_root=(args.akkurat_root or None),
    )

    if args.mode == "status":
        print(json.dumps(_json_safe(_status_report()), indent=2, ensure_ascii=False))
        return 0

    cfg = AkkuratAtomHybridConfig(
        runtime_id="akkurat_atom_hybrid_demo",
        mode=str(args.runtime_mode),
        fusion=str(args.fusion),
        input_dim=int(args.input_dim),
        control_dim=int(args.control_dim),
        output_dim=int(args.output_dim),
        enable_atom=not bool(args.disable_atom),
        enable_tcfc=not bool(args.disable_tcfc),
        enable_digital_twin=bool(args.enable_digital_twin),
        atom_profile=str(args.profile),
        atom_method=str(args.method),
        atom_dt=float(args.dt),
        seed=int(args.seed),
        tcfc_backend=str(args.tcfc_backend),
        atomtn_root=str(args.atomtn_root or ""),
        akkurat_root=str(args.akkurat_root or ""),
    )

    if args.mode == "smoke":
        cfg.atom_profile = "smoke"
        cfg.atom_method = "none"

    rt = build_hybrid_runtime(cfg)

    X = _demo_inputs(cfg.input_dim, int(args.steps), cfg.seed)
    results = rt.step_sequence(X, dt=cfg.atom_dt)

    report = {
        "ok": all(bool(r.ok) for r in results),
        "summary": rt.summary(),
        "rows": [
            {
                "step": int(r.step),
                "ok": bool(r.ok),
                "output_norm": r.metrics.get("output_norm"),
                "atom_feature_norm": r.metrics.get("atom_feature_norm"),
                "tcfc_state_norm": r.metrics.get("tcfc_state_norm"),
                "elapsed_s": r.elapsed_s,
                "warnings": r.warnings,
                "error": r.error,
            }
            for r in results
        ],
        "snapshot": rt.snapshot(),
    }

    if args.output or args.mode == "export-snapshot":
        out = Path(args.output or "akkurat_atom_hybrid_snapshot.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_json_safe(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report["output"] = str(out)

    print(json.dumps(_json_safe({"ok": report["ok"], "summary": report["summary"]}), indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())