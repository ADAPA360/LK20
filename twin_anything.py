#!/usr/bin/env python3
# twin_anything.py
r"""
Project Chimera / Akkurat - Twin Anything Factory
=================================================

Production factory for creating governed digital twins for arbitrary objects,
systems, workflows, software processes, plans, environments, machines,
organizations, curriculum artefacts, or abstract operations.

This module is a topology/materialization adapter above:

    digital_twin_kernel.py

and is tensor-network aware through the proprietary local library:

    tn.py

It does not reimplement either library.

Strategic role
--------------
This module builds a universal "thing/process" branch inside a governed
TreeTensorNetwork and annotates the resulting twin with tensor-network
capabilities detected from tn.py.

It can be consumed by:

    lk20_kernel.py
    akkurat_agentic_bridge.py
    success_estimation.py
    kognitive_coder.py
    plan_executioner.py
    akkurat_atom_hybrid.py

Dependency policy
-----------------
Required:
    digital_twin_kernel.py

Optional but recommended:
    tn.py

The digital_twin_kernel already uses tn.py internally for TensorTrain projection
and Tucker-like fusion when available. This factory therefore keeps tn.py as a
detected capability and configuration source, but does not duplicate tensor
operations.

Universal branch
----------------
    5.0 Thing Twin
    ├── 5.1  Identity / Purpose
    ├── 5.2  Environment / Context
    ├── 5.3  Components / Structure
    ├── 5.4  Observables / Telemetry
    ├── 5.5  Process / Behavior Model
    ├── 5.6  Resources
    ├── 5.7  Constraints / Governance
    ├── 5.8  Success Criteria
    ├── 5.9  Risks / Failure Modes
    ├── 5.10 Actions / Actuators
    ├── 5.11 Memory / Logs / Snapshots
    └── 5.12 Tensor-Network Runtime Profile

Public API
----------
- TwinAnythingConfig
- TwinInput
- TwinBuildResult
- TwinAnythingFactory
- build_anything_twin(...)
- create_twin(...)
- status_report(...)

CLI examples
------------
Status:

    python twin_anything.py --mode status

Create a generic twin:

    python twin_anything.py --mode create ^
      --thing-name "Data Center Migration Plan" ^
      --thing-type plan ^
      --description "A staged migration with rollback."

Create and save:

    python twin_anything.py --mode create ^
      --thing-name "CNC Mill 01" ^
      --thing-type industrial_machine ^
      --output cnc_mill_twin.json

Load input JSON:

    python twin_anything.py --mode create ^
      --input-json twin_input.json ^
      --output twin_snapshot.json

Create a tensor-network capability report:

    python twin_anything.py --mode tn-status
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import mimetypes
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Version
# =============================================================================

TWIN_ANYTHING_VERSION = "twin-anything-v3.1-tn-aware-production"


# =============================================================================
# Path bootstrap
# =============================================================================

def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _add_path(path: Union[str, os.PathLike, None], *, prepend: bool = True) -> Optional[Path]:
    if path is None:
        return None

    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        return None

    if not p.exists():
        return None

    s = str(p)
    if s not in sys.path:
        if prepend:
            sys.path.insert(0, s)
        else:
            sys.path.append(s)

    return p


def configure_paths(
    *,
    project_root: Optional[Union[str, os.PathLike]] = None,
    akkurat_root: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Optional[str]]:
    """
    Locate project-local modules.

    Expected LK20 project folder:
        C:\\Users\\ali_z\\ANU AI\\LK20

    This function is deliberately permissive because this module may be run from
    a CLI, imported from another runtime, or called from notebooks.
    """
    here = _module_dir()

    candidates = [
        project_root,
        akkurat_root,
        os.environ.get("LK20_ROOT"),
        os.environ.get("AKKURAT_ROOT"),
        os.environ.get("AKKURAT_COGNITIVE_ROOT"),
        here,
        here.parent,
        here.parent / "app",
        here.parent / "Akkurat",
        here.parent / "Akkurat" / "qwen3.5_4b",
        Path.cwd(),
        Path("/mnt/data"),
    ]

    found_dtk = None
    found_tn = None
    found_self = None

    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue

        if found_dtk is None and (p / "digital_twin_kernel.py").exists():
            found_dtk = p

        if found_tn is None and (p / "tn.py").exists():
            found_tn = p

        if found_self is None and (p / "twin_anything.py").exists():
            found_self = p

    _add_path(here, prepend=True)

    return {
        "module_dir": str(here),
        "project_root": None if project_root is None else str(Path(project_root).expanduser()),
        "digital_twin_kernel_root": None if found_dtk is None else str(found_dtk),
        "tn_root": None if found_tn is None else str(found_tn),
        "twin_anything_root": None if found_self is None else str(found_self),
    }


_PATHS = configure_paths()


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_hash_u32(s: str) -> int:
    h = 2166136261
    for b in (s or "").encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    b = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode(
        "utf-8",
        errors="ignore",
    )
    return hashlib.sha256(b).hexdigest()


def _slug(value: Any, default: str = "thing", max_len: int = 96) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = default
    if s[0].isdigit():
        s = f"{default}_{s}"
    return s[: int(max_len)]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_mapping(x: Any) -> Dict[str, Any]:
    if isinstance(x, Mapping):
        return dict(x)
    return {}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, set):
        return list(x)
    return [x]


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
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "describe") and callable(obj.describe):
        try:
            return _json_safe(obj.describe())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return str(obj)


def _atomic_write_text(path: Union[str, os.PathLike], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, p)


def _write_json(path: Union[str, os.PathLike], payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
    )


def _read_json(path: Union[str, os.PathLike], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(default)

    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(default)


def _sha256_file(path: Union[str, os.PathLike]) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None

    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def _dataclass_kwargs(cls: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        names = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in dict(values).items() if k in names}
    except Exception:
        return dict(values)


def _guess_mime(filename: str) -> str:
    if not filename:
        return "application/octet-stream"
    guess, _ = mimetypes.guess_type(filename)
    return guess or "application/octet-stream"


# =============================================================================
# Dependency status
# =============================================================================

def digital_twin_status() -> Dict[str, Any]:
    configure_paths()

    try:
        dtk = importlib.import_module("digital_twin_kernel")
        return {
            "available": True,
            "module": getattr(dtk, "__file__", ""),
            "attrs": {
                "DigitalTwinsBuilder": hasattr(dtk, "DigitalTwinsBuilder"),
                "TreeTensorNetwork": hasattr(dtk, "TreeTensorNetwork"),
                "AkkuratInterface": hasattr(dtk, "AkkuratInterface"),
                "NodeKind": hasattr(dtk, "NodeKind"),
                "FusionConfig": hasattr(dtk, "FusionConfig"),
                "FusionMode": hasattr(dtk, "FusionMode"),
                "LatentGeometry": hasattr(dtk, "LatentGeometry"),
                "MerkleHasher": hasattr(dtk, "MerkleHasher"),
                "Action": hasattr(dtk, "Action"),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
            "attrs": {},
        }


def tn_status() -> Dict[str, Any]:
    configure_paths()

    try:
        tn = importlib.import_module("tn")

        attrs = {
            "TensorTrain": hasattr(tn, "TensorTrain"),
            "TTConfig": hasattr(tn, "TTConfig"),
            "TuckerTensor": hasattr(tn, "TuckerTensor"),
            "TuckerFusionLayer": hasattr(tn, "TuckerFusionLayer"),
            "TreeTensorFusion": hasattr(tn, "TreeTensorFusion"),
            "DendriticProcessor": hasattr(tn, "DendriticProcessor"),
            "tensor_train_svd": hasattr(tn, "tensor_train_svd"),
            "find_optimal_permutations": hasattr(tn, "find_optimal_permutations"),
            "get_factors": hasattr(tn, "get_factors"),
            "factorize_into_modes": hasattr(tn, "factorize_into_modes"),
        }

        smoke: Dict[str, Any] = {"ok": False}
        try:
            if attrs["TensorTrain"] and attrs["TTConfig"]:
                cfg = tn.TTConfig(dtype=np.float32, device="cpu")
                tt = tn.TensorTrain(
                    output_dims=[2, 2],
                    input_dims=[2, 2],
                    bond_dims=[2],
                    config=cfg,
                    rng=np.random.default_rng(0),
                    init_scale=1e-3,
                    metadata={"source": "twin_anything_status_smoke"},
                )
                x = np.ones((4,), dtype=np.float32)
                y = tt.apply(x)
                smoke = {
                    "ok": True,
                    "operator": _json_safe(tt.describe() if hasattr(tt, "describe") else {}),
                    "output_shape": list(np.asarray(y).shape),
                    "output_norm": float(np.linalg.norm(np.asarray(y))),
                }
        except Exception as exc:
            smoke = {"ok": False, "error": repr(exc)}

        return {
            "available": True,
            "module": getattr(tn, "__file__", ""),
            "attrs": attrs,
            "smoke": smoke,
        }

    except Exception as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
            "attrs": {},
            "smoke": {"ok": False},
        }


def status_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "ts": _now_iso(),
        "version": TWIN_ANYTHING_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "paths": configure_paths(),
        "digital_twin_kernel": digital_twin_status(),
        "tn": tn_status(),
    }


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class TensorNetworkRuntimeConfig:
    """
    Declarative tensor-network profile for created twins.

    The actual low-level tensor operations remain owned by tn.py and
    digital_twin_kernel.py. This config tells the factory how to annotate the
    twin and how aggressively to request tensorized projection/fusion.
    """

    enable_tn_profile: bool = True
    prefer_tn_projection: bool = True
    prefer_tucker_fusion: bool = True
    tt_rank: int = 8
    tucker_rank: int = 16
    tt_num_modes: int = 2
    dtype: str = "float32"
    device: str = "cpu"
    check_finite: bool = True
    save_operator_profiles: bool = False

    def normalized(self) -> "TensorNetworkRuntimeConfig":
        cfg = copy.deepcopy(self)

        cfg.enable_tn_profile = bool(cfg.enable_tn_profile)
        cfg.prefer_tn_projection = bool(cfg.prefer_tn_projection)
        cfg.prefer_tucker_fusion = bool(cfg.prefer_tucker_fusion)
        cfg.tt_rank = int(max(1, cfg.tt_rank))
        cfg.tucker_rank = int(max(1, cfg.tucker_rank))
        cfg.tt_num_modes = int(max(1, cfg.tt_num_modes))

        cfg.dtype = str(cfg.dtype or "float32").lower().strip()
        if cfg.dtype not in {"float16", "float32", "float64"}:
            cfg.dtype = "float32"

        cfg.device = str(cfg.device or "cpu").lower().strip()
        if cfg.device not in {"cpu", "gpu"}:
            cfg.device = "cpu"

        cfg.check_finite = bool(cfg.check_finite)
        cfg.save_operator_profiles = bool(cfg.save_operator_profiles)

        return cfg


@dataclass
class TwinAnythingConfig:
    vector_dim: int = 256
    sketch_dim: int = 96
    history_capacity: int = 64
    seed: int = 2027
    use_tn_projection: bool = True
    latent_geometry: str = "euclidean"

    install_platform_taxonomy: bool = True
    install_success_defaults: bool = True
    install_tensor_network_profile: bool = True

    thing_root_id: str = "5.0"
    project_root: str = ""

    tn_runtime: TensorNetworkRuntimeConfig = field(default_factory=TensorNetworkRuntimeConfig)

    def normalized(self) -> "TwinAnythingConfig":
        cfg = copy.deepcopy(self)

        cfg.vector_dim = int(max(8, cfg.vector_dim))
        cfg.sketch_dim = int(max(8, min(int(cfg.sketch_dim), cfg.vector_dim)))
        cfg.history_capacity = int(max(1, cfg.history_capacity))
        cfg.seed = int(cfg.seed)

        cfg.latent_geometry = str(cfg.latent_geometry or "euclidean").lower().strip()
        if cfg.latent_geometry not in {"euclidean", "hyperbolic", "poincare", "poincaré"}:
            cfg.latent_geometry = "euclidean"

        cfg.use_tn_projection = bool(cfg.use_tn_projection)
        cfg.install_platform_taxonomy = bool(cfg.install_platform_taxonomy)
        cfg.install_success_defaults = bool(cfg.install_success_defaults)
        cfg.install_tensor_network_profile = bool(cfg.install_tensor_network_profile)

        cfg.thing_root_id = str(cfg.thing_root_id or "5.0")
        cfg.project_root = str(cfg.project_root or "")
        cfg.tn_runtime = cfg.tn_runtime.normalized()

        return cfg


@dataclass
class TwinInput:
    thing_name: str
    thing_type: str = "generic"
    description: str = ""

    observables: Dict[str, Any] = field(default_factory=dict)
    process_steps: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    components: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    risks: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)

    source_files: List[Dict[str, Any]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TwinInput":
        p = dict(payload or {})

        return cls(
            thing_name=str(p.get("thing_name", p.get("name", "Unnamed Thing"))),
            thing_type=str(p.get("thing_type", p.get("type", "generic"))),
            description=str(p.get("description", "")),
            observables=_as_mapping(p.get("observables", {})),
            process_steps=[str(x) for x in _as_list(p.get("process_steps", []))],
            resources=_as_mapping(p.get("resources", {})),
            constraints=[str(x) for x in _as_list(p.get("constraints", []))],
            components=_as_mapping(p.get("components", {})),
            environment=_as_mapping(p.get("environment", {})),
            success_criteria=_as_mapping(p.get("success_criteria", {})),
            risks=_as_list(p.get("risks", [])),
            actions=_as_list(p.get("actions", [])),
            source_files=[_as_mapping(x) for x in _as_list(p.get("source_files", []))],
            links=[str(x) for x in _as_list(p.get("links", []))],
            tags=[str(x) for x in _as_list(p.get("tags", []))],
            metadata=_as_mapping(p.get("metadata", {})),
        )

    def normalized(self) -> "TwinInput":
        t = copy.deepcopy(self)
        t.thing_name = str(t.thing_name or "Unnamed Thing")
        t.thing_type = _slug(t.thing_type or "generic", default="generic")
        t.description = str(t.description or "")
        t.observables = _as_mapping(t.observables)
        t.process_steps = [str(x) for x in _as_list(t.process_steps)]
        t.resources = _as_mapping(t.resources)
        t.constraints = [str(x) for x in _as_list(t.constraints)]
        t.components = _as_mapping(t.components)
        t.environment = _as_mapping(t.environment)
        t.success_criteria = _as_mapping(t.success_criteria)
        t.risks = _as_list(t.risks)
        t.actions = _as_list(t.actions)
        t.source_files = [_as_mapping(x) for x in _as_list(t.source_files)]
        t.links = [str(x) for x in _as_list(t.links)]
        t.tags = [str(x) for x in _as_list(t.tags)]
        t.metadata = _as_mapping(t.metadata)
        return t


@dataclass
class SourceFileManifest:
    file_path: str = ""
    filename: str = ""
    mime_type: str = ""
    sha256: str = ""
    role: str = "source"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceFileManifest":
        p = dict(payload or {})
        m = cls(
            file_path=str(p.get("file_path", p.get("path", ""))),
            filename=str(p.get("filename", "")),
            mime_type=str(p.get("mime_type", "")),
            sha256=str(p.get("sha256", p.get("file_sha256", ""))),
            role=str(p.get("role", "source")),
            metadata=_as_mapping(p.get("metadata", {})),
        )
        return m.normalized()

    def normalized(self) -> "SourceFileManifest":
        m = copy.deepcopy(self)

        if m.file_path and not m.filename:
            m.filename = Path(m.file_path).name

        if m.file_path and not m.sha256:
            m.sha256 = _sha256_file(m.file_path) or ""

        if not m.mime_type:
            m.mime_type = _guess_mime(m.filename)

        m.role = _slug(m.role or "source", default="source")

        return m


@dataclass
class TwinBuildResult:
    ok: bool
    network: Any
    control_plane: Any
    metadata: Dict[str, Any]
    node_map: Dict[str, str]
    snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "metadata": _json_safe(self.metadata),
            "node_map": _json_safe(self.node_map),
            "snapshot": _json_safe(self.snapshot),
        }


# =============================================================================
# Tensor-network capability adapter
# =============================================================================

class TensorNetworkProfileBuilder:
    """
    Lightweight introspection and profile builder for proprietary tn.py.

    This class does not reimplement TensorTrain, TT-SVD, or Tucker fusion. It
    builds metadata profiles and small optional smoke-test summaries that can be
    attached to a digital twin node.
    """

    def __init__(self, cfg: TensorNetworkRuntimeConfig, *, seed: int = 2027, vector_dim: int = 256):
        self.cfg = cfg.normalized()
        self.seed = int(seed)
        self.vector_dim = int(max(8, vector_dim))

    def available(self) -> bool:
        return bool(tn_status().get("available", False))

    def build_profile(self) -> Dict[str, Any]:
        status = tn_status()
        profile = {
            "enabled": bool(self.cfg.enable_tn_profile),
            "available": bool(status.get("available", False)),
            "status": status,
            "runtime_config": _json_safe(self.cfg),
            "recommended_usage": {
                "projection": "digital_twin_kernel.ProjectionEngine uses tn.TensorTrain when available.",
                "fusion": "digital_twin_kernel can use tn.TuckerFusionLayer for bounded pairwise fusion.",
                "storage": "TensorTrain.save_npz/load_npz available when tn.py v3.1 is installed.",
            },
            "operator_profile": {},
        }

        if not bool(self.cfg.enable_tn_profile) or not bool(status.get("available", False)):
            return profile

        try:
            tn = importlib.import_module("tn")
            if hasattr(tn, "get_factors"):
                profile["operator_profile"]["vector_dim_factors"] = list(
                    tn.get_factors(int(self.vector_dim))
                )

            if hasattr(tn, "factorize_into_modes"):
                profile["operator_profile"]["vector_dim_modes"] = list(
                    tn.factorize_into_modes(
                        int(self.vector_dim),
                        int(self.cfg.tt_num_modes),
                    )
                )

            if hasattr(tn, "TTConfig"):
                dt = getattr(np, self.cfg.dtype, np.float32)
                tt_cfg = tn.TTConfig(
                    dtype=dt,
                    device="cpu",
                    check_finite=bool(self.cfg.check_finite),
                )
                profile["operator_profile"]["tt_config"] = _json_safe(tt_cfg)

            if hasattr(tn, "TensorTrain") and hasattr(tn, "TTConfig"):
                dim_a, dim_b = 2, 2
                rng = np.random.default_rng(self.seed)
                dt = getattr(np, self.cfg.dtype, np.float32)
                tt_cfg = tn.TTConfig(dtype=dt, device="cpu", check_finite=bool(self.cfg.check_finite))
                tt = tn.TensorTrain(
                    output_dims=[dim_a, dim_b],
                    input_dims=[dim_a, dim_b],
                    bond_dims=[min(2, int(self.cfg.tt_rank))],
                    config=tt_cfg,
                    rng=rng,
                    init_scale=1e-3,
                    metadata={"source": "TwinAnythingFactory.tensor_network_profile"},
                )
                x = np.ones((dim_a * dim_b,), dtype=dt)
                y = tt.apply(x)
                profile["operator_profile"]["tt_smoke"] = {
                    "ok": True,
                    "describe": _json_safe(tt.describe() if hasattr(tt, "describe") else {}),
                    "input_norm": float(np.linalg.norm(x)),
                    "output_norm": float(np.linalg.norm(np.asarray(y))),
                }

            if hasattr(tn, "TuckerFusionLayer"):
                rng = np.random.default_rng(self.seed + 1)
                layer = tn.TuckerFusionLayer(
                    input_dims=[4, 4],
                    output_rank=min(4, int(self.cfg.tucker_rank)),
                    rng=rng,
                    dtype=np.float32,
                )
                fused = layer.contract(
                    [
                        np.ones((4,), dtype=np.float32),
                        np.ones((4,), dtype=np.float32),
                    ]
                )
                profile["operator_profile"]["tucker_smoke"] = {
                    "ok": True,
                    "output_shape": list(np.asarray(fused).shape),
                    "output_norm": float(np.linalg.norm(np.asarray(fused))),
                    "parameter_count": int(getattr(layer, "parameter_count", 0)),
                }

        except Exception as exc:
            profile["operator_profile"]["error"] = repr(exc)
            profile["operator_profile"]["traceback"] = traceback.format_exc()

        return profile


# =============================================================================
# Factory
# =============================================================================

class TwinAnythingFactory:
    """
    Creates a governed digital twin branch for any object/process while preserving
    the standard Akkurat platform taxonomy and attaching tensor-network runtime
    capabilities where available.
    """

    VERSION = TWIN_ANYTHING_VERSION

    UNIVERSAL_NODE_NAMES: Tuple[Tuple[str, str, str], ...] = (
        ("5.1", "Identity / Purpose", "data"),
        ("5.2", "Environment / Context", "physical"),
        ("5.3", "Components / Structure", "entity"),
        ("5.4", "Observables / Telemetry", "observable"),
        ("5.5", "Process / Behavior Model", "model"),
        ("5.6", "Resources", "resource"),
        ("5.7", "Constraints / Governance", "governance"),
        ("5.8", "Success Criteria", "observable"),
        ("5.9", "Risks / Failure Modes", "governance"),
        ("5.10", "Actions / Actuators", "actuator"),
        ("5.11", "Memory / Logs / Snapshots", "storage"),
        ("5.12", "Tensor-Network Runtime Profile", "model"),
    )

    def __init__(
        self,
        *,
        vector_dim: int = 256,
        sketch_dim: int = 96,
        history_capacity: int = 64,
        seed: int = 2027,
        use_tn_projection: bool = True,
        latent_geometry: str = "euclidean",
        install_platform_taxonomy: bool = True,
        install_success_defaults: bool = True,
        install_tensor_network_profile: bool = True,
        project_root: Optional[Union[str, os.PathLike]] = None,
        config: Optional[TwinAnythingConfig] = None,
    ):
        self.cfg = (
            config
            or TwinAnythingConfig(
                vector_dim=vector_dim,
                sketch_dim=sketch_dim,
                history_capacity=history_capacity,
                seed=seed,
                use_tn_projection=use_tn_projection,
                latent_geometry=latent_geometry,
                install_platform_taxonomy=install_platform_taxonomy,
                install_success_defaults=install_success_defaults,
                install_tensor_network_profile=install_tensor_network_profile,
                project_root="" if project_root is None else str(project_root),
            )
        ).normalized()

        configure_paths(project_root=self.cfg.project_root or None)

    # -------------------------------------------------------------------------
    # Main creation API
    # -------------------------------------------------------------------------

    def create_twin(
        self,
        *,
        thing_name: str,
        thing_type: str = "generic",
        description: str = "",
        observables: Optional[Mapping[str, Any]] = None,
        process_steps: Optional[Sequence[str]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        constraints: Optional[Sequence[str]] = None,
        components: Optional[Mapping[str, Any]] = None,
        environment: Optional[Mapping[str, Any]] = None,
        success_criteria: Optional[Mapping[str, Any]] = None,
        risks: Optional[Sequence[Any]] = None,
        actions: Optional[Sequence[Any]] = None,
        source_files: Optional[Sequence[Mapping[str, Any]]] = None,
        links: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        twin_input = TwinInput(
            thing_name=str(thing_name),
            thing_type=str(thing_type or "generic"),
            description=str(description or ""),
            observables=dict(observables or {}),
            process_steps=[str(x) for x in list(process_steps or [])],
            resources=dict(resources or {}),
            constraints=[str(x) for x in list(constraints or [])],
            components=dict(components or {}),
            environment=dict(environment or {}),
            success_criteria=dict(success_criteria or {}),
            risks=list(risks or []),
            actions=list(actions or []),
            source_files=[dict(x) for x in list(source_files or [])],
            links=[str(x) for x in list(links or [])],
            tags=[str(x) for x in list(tags or [])],
            metadata=dict(metadata or {}),
        ).normalized()

        result = self.create_twin_from_input(twin_input)

        return {
            "network": result.network,
            "control_plane": result.control_plane,
            "metadata": result.metadata,
            "node_map": result.node_map,
            "snapshot": result.snapshot,
        }

    def create_twin_from_input(self, twin_input: TwinInput) -> TwinBuildResult:
        dtk = importlib.import_module("digital_twin_kernel")
        twin_input = twin_input.normalized()

        ttn = self._build_base_network(dtk)
        cp = dtk.AkkuratInterface(ttn, strict=False)

        node_map = self._install_universal_branch(dtk, ttn, twin_input)
        self._seed_universal_branch(ttn, twin_input, node_map)
        self._install_type_specific_nodes(dtk, ttn, twin_input, node_map)
        self._install_source_file_nodes(dtk, ttn, twin_input, node_map)
        self._install_success_defaults_if_available(ttn, twin_input, node_map)
        self._install_tensor_network_profile_if_available(ttn, twin_input, node_map)
        self._finalize_histories(ttn)

        snapshot = self._snapshot(dtk, ttn)

        metadata = {
            "factory": "TwinAnythingFactory",
            "factory_version": self.VERSION,
            "thing_name": twin_input.thing_name,
            "thing_type": twin_input.thing_type,
            "thing_slug": _slug(twin_input.thing_name),
            "root_node": self.cfg.thing_root_id,
            "node_count": len(getattr(ttn, "nodes", {})),
            "created_at": _now_iso(),
            "digital_twin_kernel": getattr(dtk, "__file__", ""),
            "tn": tn_status(),
            "config": _json_safe(self.cfg),
            "input_hash": _payload_hash(asdict(twin_input)),
        }

        return TwinBuildResult(
            ok=True,
            network=ttn,
            control_plane=cp,
            metadata=metadata,
            node_map=node_map,
            snapshot=snapshot,
        )

    # -------------------------------------------------------------------------
    # Base network / branch installation
    # -------------------------------------------------------------------------

    def _build_base_network(self, dtk: Any) -> Any:
        geometry = (
            dtk.LatentGeometry.HYPERBOLIC.value
            if self.cfg.latent_geometry in {"hyperbolic", "poincare", "poincaré"}
            else dtk.LatentGeometry.EUCLIDEAN.value
        )

        use_tn_projection = bool(
            self.cfg.use_tn_projection
            and self.cfg.tn_runtime.prefer_tn_projection
            and tn_status().get("available", False)
        )

        if bool(self.cfg.install_platform_taxonomy):
            kwargs = {
                "vector_dim": int(self.cfg.vector_dim),
                "seed": int(self.cfg.seed),
                "sketch_dim": int(self.cfg.sketch_dim),
                "history_capacity": int(self.cfg.history_capacity),
                "use_tn_projection": use_tn_projection,
                "latent_geometry": geometry,
            }

            try:
                return dtk.DigitalTwinsBuilder.build_platform(**kwargs)
            except TypeError:
                kwargs.pop("use_tn_projection", None)
                kwargs.pop("latent_geometry", None)
                return dtk.DigitalTwinsBuilder.build_platform(**kwargs)

        return dtk.TreeTensorNetwork(
            vector_dim=int(self.cfg.vector_dim),
            seed=int(self.cfg.seed),
            root_name="Digital Twin",
            sketch_dim=int(self.cfg.sketch_dim),
            history_capacity=int(self.cfg.history_capacity),
            use_tn_projection=use_tn_projection,
            latent_geometry=geometry,
        )

    def _node_kind(self, dtk: Any, kind_name: str) -> Any:
        name = str(kind_name or "generic").lower().strip()

        try:
            mapping = {
                "root": dtk.NodeKind.ROOT,
                "physical": dtk.NodeKind.PHYSICAL,
                "virtual": dtk.NodeKind.VIRTUAL,
                "data": dtk.NodeKind.DATA,
                "entity": dtk.NodeKind.ENTITY,
                "resource": dtk.NodeKind.RESOURCE,
                "sensor": dtk.NodeKind.SENSOR,
                "actuator": dtk.NodeKind.ACTUATOR,
                "property": dtk.NodeKind.PROPERTY,
                "model": dtk.NodeKind.MODEL,
                "pipeline": dtk.NodeKind.PIPELINE,
                "storage": dtk.NodeKind.STORAGE,
                "visual": dtk.NodeKind.VISUAL,
                "cognitive": dtk.NodeKind.COGNITIVE,
                "atomtn": dtk.NodeKind.ATOMTN,
                "quantum": dtk.NodeKind.QUANTUM,
                "flow": dtk.NodeKind.FLOW,
                "governance": dtk.NodeKind.GOVERNANCE,
                "observable": dtk.NodeKind.OBSERVABLE,
                "generic": dtk.NodeKind.GENERIC,
            }
            return mapping.get(name, dtk.NodeKind.GENERIC)
        except Exception:
            return dtk.NodeKind.GENERIC

    def _add_node(
        self,
        dtk: Any,
        ttn: Any,
        node_id: str,
        name: str,
        *,
        parent_id: str,
        kind: str = "generic",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if node_id in getattr(ttn, "nodes", {}):
            try:
                ttn.nodes[node_id].metadata.update(dict(metadata or {}))
                ttn.bump_tick()
            except Exception:
                pass
            return

        ttn.add_node(
            node_id,
            name,
            parent_id=parent_id,
            kind=self._node_kind(dtk, kind),
            metadata=dict(metadata or {}),
        )

    def _install_universal_branch(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> Dict[str, str]:
        root = str(self.cfg.thing_root_id)
        node_map = {"root": root}

        if root not in ttn.nodes:
            ttn.add_node(
                root,
                f"Thing Twin: {twin_input.thing_name}",
                parent_id="0",
                kind=self._node_kind(dtk, "generic"),
                metadata={
                    "thing_name": twin_input.thing_name,
                    "thing_type": twin_input.thing_type,
                    "description": twin_input.description,
                    "metadata": _json_safe(twin_input.metadata),
                    "tags": list(twin_input.tags),
                    "links": list(twin_input.links),
                    "installed_by": self.VERSION,
                    "created_at": _now_iso(),
                },
            )

        for node_id, name, kind_name in self.UNIVERSAL_NODE_NAMES:
            if node_id not in ttn.nodes:
                ttn.add_node(
                    node_id,
                    name,
                    parent_id=root,
                    kind=self._node_kind(dtk, kind_name),
                    metadata={
                        "universal_branch": True,
                        "installed_by": self.VERSION,
                    },
                )

            key = name.lower().split("/")[0].strip().replace(" ", "_").replace("-", "_")
            node_map[key] = node_id

        node_map.update(
            {
                "identity": "5.1",
                "purpose": "5.1",
                "environment": "5.2",
                "context": "5.2",
                "components": "5.3",
                "structure": "5.3",
                "observables": "5.4",
                "telemetry": "5.4",
                "process": "5.5",
                "behavior": "5.5",
                "resources": "5.6",
                "constraints": "5.7",
                "governance": "5.7",
                "success": "5.8",
                "success_criteria": "5.8",
                "risks": "5.9",
                "failure_modes": "5.9",
                "actions": "5.10",
                "actuators": "5.10",
                "memory": "5.11",
                "logs": "5.11",
                "snapshots": "5.11",
                "tn_profile": "5.12",
                "tensor_network_profile": "5.12",
            }
        )

        self._install_universal_fusion(dtk, ttn)

        return node_map

    def _install_universal_fusion(self, dtk: Any, ttn: Any) -> None:
        fusion_specs = {
            "5.0": dict(attention_beta=2.6, residual_mix=0.16),
            "5.5": dict(attention_beta=2.4, residual_mix=0.20),
            "5.7": dict(attention_beta=2.8, residual_mix=0.14),
            "5.8": dict(attention_beta=2.6, residual_mix=0.16),
            "5.9": dict(attention_beta=2.7, residual_mix=0.14),
            "5.12": dict(attention_beta=2.9, residual_mix=0.12),
        }

        for node_id, spec in fusion_specs.items():
            if node_id not in getattr(ttn, "nodes", {}):
                continue

            try:
                ttn.set_fusion(
                    node_id,
                    dtk.FusionConfig(
                        mode=dtk.FusionMode.ATTENTION,
                        attention_beta=float(spec["attention_beta"]),
                        residual_mix=float(spec["residual_mix"]),
                        auto_tune_beta=True,
                    ),
                )
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------

    def _seed_universal_branch(self, ttn: Any, twin_input: TwinInput, node_map: Mapping[str, str]) -> None:
        self._safe_update(
            ttn,
            node_map["identity"],
            {
                "thing_name": twin_input.thing_name,
                "thing_type": twin_input.thing_type,
                "description": twin_input.description,
                "metadata": twin_input.metadata,
                "tags": twin_input.tags,
                "links": twin_input.links,
                "slug": _slug(twin_input.thing_name),
                "input_hash": _payload_hash(asdict(twin_input)),
            },
            note="thing_identity",
        )

        self._safe_update(
            ttn,
            node_map["environment"],
            twin_input.environment or self._default_environment(twin_input),
            note="thing_environment",
        )

        self._safe_update(
            ttn,
            node_map["components"],
            twin_input.components or self._infer_components(twin_input),
            note="thing_components",
        )

        self._safe_update(
            ttn,
            node_map["observables"],
            twin_input.observables or self._default_observables(twin_input),
            note="thing_observables",
        )

        self._safe_update(
            ttn,
            node_map["process"],
            {
                "process_steps": twin_input.process_steps or self._default_process_steps(twin_input),
                "process_type": twin_input.thing_type,
                "state_model": "universal_governed_process_branch",
            },
            note="thing_process_model",
        )

        self._safe_update(
            ttn,
            node_map["resources"],
            twin_input.resources or self._default_resources(twin_input),
            note="thing_resources",
        )

        self._safe_update(
            ttn,
            node_map["constraints"],
            {
                "constraints": twin_input.constraints or self._default_constraints(twin_input),
                "governance_policy": "Actions require governed simulation and success-estimation approval.",
                "audit_required": True,
                "tensor_network_policy": {
                    "low_level_ops_owned_by": "tn.py",
                    "world_model_owned_by": "digital_twin_kernel.py",
                    "branch_materialization_owned_by": "twin_anything.py",
                },
            },
            note="thing_constraints",
        )

        self._safe_update(
            ttn,
            node_map["success"],
            twin_input.success_criteria or self._default_success_criteria(twin_input),
            note="thing_success_criteria",
        )

        self._safe_update(
            ttn,
            node_map["risks"],
            {
                "risks": twin_input.risks or self._default_risks(twin_input),
                "risk_policy": "High-risk candidates require explicit mitigation and rollback planning.",
            },
            note="thing_risk_model",
        )

        self._safe_update(
            ttn,
            node_map["actions"],
            {
                "actions": twin_input.actions or self._default_actions(twin_input),
                "action_policy": "Propose -> simulate -> estimate success -> approve -> execute.",
            },
            note="thing_action_model",
        )

        self._safe_update(
            ttn,
            node_map["memory"],
            {
                "created_by": "TwinAnythingFactory",
                "factory_version": self.VERSION,
                "created_for": twin_input.thing_name,
                "created_at": _now_iso(),
                "event_log": [],
                "snapshots": [],
                "source_files": _json_safe(twin_input.source_files),
            },
            note="thing_memory_initialization",
        )

    def _safe_update(self, ttn: Any, node_id: str, payload: Any, *, note: str) -> None:
        try:
            ttn.update_node_data(node_id, payload, note=note)
        except TypeError:
            ttn.update_node_data(node_id, payload)
        except Exception:
            raise

    # -------------------------------------------------------------------------
    # Specialization
    # -------------------------------------------------------------------------

    def _install_type_specific_nodes(
        self,
        dtk: Any,
        ttn: Any,
        twin_input: TwinInput,
        node_map: Mapping[str, str],
    ) -> None:
        thing_type = str(twin_input.thing_type or "generic").lower().strip()

        if thing_type in {"plan", "project_plan", "migration_plan"} or "plan" in thing_type:
            self._install_plan_nodes(dtk, ttn, twin_input)

        elif thing_type in {
            "software_process",
            "pipeline",
            "ai_runtime",
            "hybrid_ai_runtime",
            "digital_twin_runtime",
        } or any(x in thing_type for x in ["software", "pipeline", "runtime", "ai"]):
            self._install_software_process_nodes(dtk, ttn, twin_input)

        elif thing_type in {"machine", "industrial_machine", "robot", "device", "equipment"} or any(
            x in thing_type for x in ["machine", "robot", "device", "equipment", "cnc"]
        ):
            self._install_machine_nodes(dtk, ttn, twin_input)

        elif thing_type in {"person", "operator", "team", "organization", "school", "municipality"} or any(
            x in thing_type for x in ["person", "team", "org", "operator", "school", "kommune", "county"]
        ):
            self._install_human_org_nodes(dtk, ttn, twin_input)

        elif thing_type in {
            "curriculum",
            "curriculum_plan",
            "lk20",
            "education_programme",
            "school_curriculum",
        } or any(x in thing_type for x in ["curriculum", "lk20", "udir", "grep", "education"]):
            self._install_curriculum_nodes(dtk, ttn, twin_input)

        elif thing_type in {
            "document",
            "knowledge_base",
            "corpus",
            "policy_document",
        } or any(x in thing_type for x in ["document", "corpus", "knowledge", "policy"]):
            self._install_document_nodes(dtk, ttn, twin_input)

        else:
            self._install_generic_nodes(dtk, ttn, twin_input)

    def _install_plan_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.5.1",
                "Plan Phases",
                "model",
                {"phases": twin_input.process_steps or self._default_process_steps(twin_input)},
            ),
            (
                "5.5.2",
                "Dependencies / Critical Path",
                "model",
                {"dependencies": twin_input.metadata.get("dependencies", [])},
            ),
            (
                "5.5.3",
                "Rollback / Recovery Path",
                "governance",
                {
                    "rollback_required": True,
                    "rollback_plan": twin_input.metadata.get("rollback_plan", ""),
                },
            ),
            (
                "5.8.1",
                "Plan Acceptance Criteria",
                "observable",
                twin_input.success_criteria or self._default_success_criteria(twin_input),
            ),
            (
                "5.9.1",
                "Plan Failure Modes",
                "governance",
                {"risks": twin_input.risks or self._default_risks(twin_input)},
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.5")

    def _install_software_process_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.5.1",
                "Execution Pipeline",
                "pipeline",
                {"steps": twin_input.process_steps or self._default_process_steps(twin_input)},
            ),
            (
                "5.5.2",
                "Runtime State",
                "model",
                {
                    "runtime": twin_input.metadata.get("runtime", "python"),
                    "language": twin_input.metadata.get("language", "python"),
                },
            ),
            (
                "5.5.3",
                "Validation / Test Harness",
                "governance",
                {
                    "tests_required": True,
                    "validation": twin_input.metadata.get("validation", {}),
                },
            ),
            (
                "5.4.1",
                "Software Telemetry",
                "observable",
                twin_input.observables or self._default_observables(twin_input),
            ),
            (
                "5.10.1",
                "Automation Hooks",
                "actuator",
                {"actions": twin_input.actions or self._default_actions(twin_input)},
            ),
            (
                "5.12.1",
                "Tensorized Runtime Hooks",
                "model",
                {
                    "tn_projection": bool(self.cfg.use_tn_projection),
                    "tn_status": tn_status(),
                },
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.5")

    def _install_machine_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.3.1",
                "Machine Components",
                "entity",
                twin_input.components or self._infer_components(twin_input),
            ),
            (
                "5.4.1",
                "Sensor Telemetry",
                "sensor",
                twin_input.observables or self._default_observables(twin_input),
            ),
            (
                "5.10.1",
                "Controllers / Actuators",
                "actuator",
                {"actions": twin_input.actions or self._default_actions(twin_input)},
            ),
            (
                "5.9.1",
                "Fault Modes",
                "governance",
                {"risks": twin_input.risks or self._default_risks(twin_input)},
            ),
            (
                "5.6.1",
                "Maintenance Resources",
                "resource",
                twin_input.resources or self._default_resources(twin_input),
            ),
            (
                "5.5.1",
                "State Estimation Model",
                "model",
                {
                    "model_type": "machine_state_estimation",
                    "requires_telemetry": True,
                    "uses_baselines": True,
                },
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.3")

    def _install_human_org_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.3.1",
                "Roles / Stakeholders",
                "entity",
                twin_input.components or self._infer_components(twin_input),
            ),
            (
                "5.6.1",
                "Capabilities / Resources",
                "resource",
                twin_input.resources or self._default_resources(twin_input),
            ),
            (
                "5.7.1",
                "Norms / Governance",
                "governance",
                {"constraints": twin_input.constraints or self._default_constraints(twin_input)},
            ),
            (
                "5.8.1",
                "Outcome Criteria",
                "observable",
                twin_input.success_criteria or self._default_success_criteria(twin_input),
            ),
            (
                "5.9.1",
                "Organizational Risks",
                "governance",
                {"risks": twin_input.risks or self._default_risks(twin_input)},
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.3")

    def _install_curriculum_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.3.1",
                "Curriculum Structure",
                "entity",
                twin_input.components
                or {
                    "layers": [
                        "framework",
                        "subject",
                        "grade_or_stage",
                        "competence_aim",
                        "assessment",
                        "local_overlay",
                    ]
                },
            ),
            (
                "5.4.1",
                "Curriculum Observables",
                "observable",
                twin_input.observables
                or {
                    "required": [
                        "coverage",
                        "alignment",
                        "gaps",
                        "version_drift",
                        "assessment_binding",
                    ]
                },
            ),
            (
                "5.5.1",
                "Curriculum Binding Process",
                "pipeline",
                {
                    "steps": twin_input.process_steps
                    or [
                        "ingest canonical curriculum",
                        "ingest local artefact",
                        "extract structure",
                        "bind to competence aims",
                        "score coverage",
                        "approve or quarantine",
                    ]
                },
            ),
            (
                "5.7.1",
                "Curriculum Governance Policy",
                "governance",
                {
                    "canonical_source": twin_input.metadata.get("canonical_source", "Udir/Grep"),
                    "local_overlay_policy": "Local artefacts do not mutate canonical curriculum nodes.",
                    "constraints": twin_input.constraints or self._default_constraints(twin_input),
                },
            ),
            (
                "5.8.1",
                "Curriculum Success Criteria",
                "observable",
                twin_input.success_criteria
                or {
                    "minimum_coverage_ratio": 0.90,
                    "requires_subject_binding": True,
                    "requires_grade_binding": True,
                    "requires_version_match": True,
                },
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.5")

    def _install_document_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.3.1",
                "Document Structure",
                "entity",
                twin_input.components
                or {
                    "sections": [],
                    "source_file_count": len(twin_input.source_files),
                },
            ),
            (
                "5.4.1",
                "Document Observables",
                "observable",
                twin_input.observables
                or {
                    "required": [
                        "source_hash",
                        "version",
                        "coverage",
                        "semantic_alignment",
                        "extraction_confidence",
                    ]
                },
            ),
            (
                "5.5.1",
                "Document Processing Pipeline",
                "pipeline",
                {
                    "steps": twin_input.process_steps
                    or [
                        "register source",
                        "extract text",
                        "parse structure",
                        "embed content",
                        "bind to target taxonomy",
                        "persist snapshot",
                    ]
                },
            ),
            (
                "5.7.1",
                "Document Governance",
                "governance",
                {
                    "constraints": twin_input.constraints or self._default_constraints(twin_input),
                    "requires_source_hash": True,
                },
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.5")

    def _install_generic_nodes(self, dtk: Any, ttn: Any, twin_input: TwinInput) -> None:
        additions = [
            (
                "5.5.1",
                "Generic State Model",
                "model",
                {"description": twin_input.description},
            ),
            (
                "5.4.1",
                "Generic Measurements",
                "observable",
                twin_input.observables or self._default_observables(twin_input),
            ),
        ]
        self._add_seeded_nodes(dtk, ttn, additions, parent_fallback="5.5")

    def _add_seeded_nodes(
        self,
        dtk: Any,
        ttn: Any,
        additions: Sequence[Tuple[str, str, str, Any]],
        *,
        parent_fallback: str,
    ) -> None:
        for node_id, name, kind_name, payload in additions:
            if node_id not in ttn.nodes:
                parent = ".".join(node_id.split(".")[:-1])
                if parent not in ttn.nodes:
                    parent = parent_fallback

                ttn.add_node(
                    node_id,
                    name,
                    parent_id=parent,
                    kind=self._node_kind(dtk, kind_name),
                )

            self._safe_update(
                ttn,
                node_id,
                payload,
                note=f"type_specific_seed:{_slug(name)}",
            )

    # -------------------------------------------------------------------------
    # Source files
    # -------------------------------------------------------------------------

    def _install_source_file_nodes(
        self,
        dtk: Any,
        ttn: Any,
        twin_input: TwinInput,
        node_map: Mapping[str, str],
    ) -> None:
        if not twin_input.source_files:
            return

        if "5.11.1" not in ttn.nodes:
            ttn.add_node(
                "5.11.1",
                "Source File Registry",
                parent_id="5.11",
                kind=self._node_kind(dtk, "storage"),
            )

        manifests = [SourceFileManifest.from_mapping(x) for x in twin_input.source_files]

        registry_payload = {
            "source_file_count": len(manifests),
            "files": [_json_safe(m) for m in manifests],
            "created_at": _now_iso(),
        }

        self._safe_update(
            ttn,
            "5.11.1",
            registry_payload,
            note="source_file_registry",
        )

        for idx, manifest in enumerate(manifests):
            node_id = f"5.11.1.{idx + 1}"
            if node_id not in ttn.nodes:
                ttn.add_node(
                    node_id,
                    f"Source File: {manifest.filename or idx + 1}",
                    parent_id="5.11.1",
                    kind=self._node_kind(dtk, "data"),
                    metadata={
                        "filename": manifest.filename,
                        "sha256": manifest.sha256,
                        "mime_type": manifest.mime_type,
                        "role": manifest.role,
                    },
                )

            self._safe_update(
                ttn,
                node_id,
                manifest,
                note="source_file_manifest",
            )

    # -------------------------------------------------------------------------
    # Tensor-network profile
    # -------------------------------------------------------------------------

    def _install_tensor_network_profile_if_available(
        self,
        ttn: Any,
        twin_input: TwinInput,
        node_map: Mapping[str, str],
    ) -> None:
        if not bool(self.cfg.install_tensor_network_profile):
            return

        if "5.12" not in getattr(ttn, "nodes", {}):
            return

        builder = TensorNetworkProfileBuilder(
            self.cfg.tn_runtime,
            seed=int(self.cfg.seed),
            vector_dim=int(self.cfg.vector_dim),
        )
        profile = builder.build_profile()
        profile["thing_type"] = twin_input.thing_type
        profile["thing_name"] = twin_input.thing_name
        profile["factory_version"] = self.VERSION

        self._safe_update(
            ttn,
            "5.12",
            profile,
            note="tensor_network_runtime_profile",
        )

        try:
            ttn.nodes["5.12"].metadata["tn_available"] = bool(profile.get("available", False))
            ttn.nodes["5.12"].metadata["tn_module"] = profile.get("status", {}).get("module", "")
            ttn.nodes["5.12"].metadata["tt_rank"] = int(self.cfg.tn_runtime.tt_rank)
            ttn.nodes["5.12"].metadata["tucker_rank"] = int(self.cfg.tn_runtime.tucker_rank)
            ttn.bump_tick()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Defaults
    # -------------------------------------------------------------------------

    def _default_environment(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {
            "context": "unspecified",
            "assumption": "Environment should be updated from observations.",
            "thing_type": twin_input.thing_type,
        }

    def _infer_components(self, twin_input: TwinInput) -> Dict[str, Any]:
        if twin_input.components:
            return dict(twin_input.components)

        if twin_input.process_steps:
            return {
                "process_step_count": len(twin_input.process_steps),
                "process_steps": list(twin_input.process_steps),
            }

        if twin_input.source_files:
            return {
                "source_file_count": len(twin_input.source_files),
                "source_files": _json_safe(twin_input.source_files),
            }

        return {
            "components": [],
            "note": "No explicit components supplied.",
        }

    def _default_observables(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {
            "observable_status": "initial",
            "required": [
                "state",
                "drift",
                "constraint_satisfaction",
                "resource_readiness",
                "success_probability",
                "tensor_profile_health",
            ],
        }

    def _default_process_steps(self, twin_input: TwinInput) -> List[str]:
        thing_type = str(twin_input.thing_type).lower()

        if "plan" in thing_type:
            return [
                "define objective",
                "verify resources",
                "simulate plan",
                "approve execution",
                "monitor result",
                "rollback if needed",
            ]

        if "software" in thing_type or "runtime" in thing_type or "pipeline" in thing_type:
            return [
                "ingest input",
                "process state",
                "validate output",
                "estimate success",
                "emit telemetry",
                "persist snapshot",
            ]

        if "machine" in thing_type or "device" in thing_type or "robot" in thing_type or "cnc" in thing_type:
            return [
                "observe sensors",
                "estimate health",
                "compare against baseline",
                "propose control action",
                "simulate",
                "execute if approved",
            ]

        if "curriculum" in thing_type or "lk20" in thing_type or "education" in thing_type:
            return [
                "ingest canonical curriculum",
                "ingest local overlay",
                "bind local artefact to canonical nodes",
                "score coverage and alignment",
                "approve or quarantine",
                "persist verifiable snapshot",
            ]

        if "document" in thing_type or "knowledge" in thing_type or "corpus" in thing_type:
            return [
                "register source",
                "extract content",
                "parse structure",
                "project into latent state",
                "bind to target taxonomy",
                "persist snapshot",
            ]

        return [
            "observe",
            "model",
            "estimate",
            "act if approved",
            "learn from outcome",
        ]

    def _default_resources(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {
            "required": [
                "operator_or_agent_attention",
                "data",
                "governance_policy",
                "digital_twin_kernel",
                "tn_runtime_when_available",
            ],
            "status": "unknown_until_observed",
        }

    def _default_constraints(self, twin_input: TwinInput) -> List[str]:
        return [
            "Actions must pass governed simulation.",
            "Success probability must meet threshold before execution.",
            "Rollback or recovery path should be known for high-risk actions.",
            "All material state changes must be auditable.",
            "Low-level tensor operations remain owned by tn.py.",
            "World-model topology and governance remain owned by digital_twin_kernel.py.",
        ]

    def _default_success_criteria(self, twin_input: TwinInput) -> Dict[str, Any]:
        return {
            "success_definition": (
                "Stable state, satisfied constraints, ready resources, acceptable risk, "
                "and success odds above threshold."
            ),
            "minimum_probability": 0.80,
            "requires_governance_approval": True,
            "requires_reversibility_or_mitigation": True,
            "requires_audit_trail": True,
        }

    def _default_risks(self, twin_input: TwinInput) -> List[Dict[str, Any]]:
        return [
            {
                "name": "insufficient_observability",
                "probability": 0.20,
                "impact": 0.40,
                "mitigation": "Add telemetry and update observables.",
            },
            {
                "name": "unmodeled_dependency",
                "probability": 0.15,
                "impact": 0.50,
                "mitigation": "Expose dependencies as components/resources.",
            },
            {
                "name": "tensor_runtime_unavailable",
                "probability": 0.10 if tn_status().get("available", False) else 0.60,
                "impact": 0.30,
                "mitigation": "Fallback to deterministic hashing projection in digital_twin_kernel.",
            },
        ]

    def _default_actions(self, twin_input: TwinInput) -> List[Dict[str, Any]]:
        return [
            {"kind": "observe", "description": "Collect or update telemetry."},
            {"kind": "simulate", "description": "Run governed sandbox simulation."},
            {"kind": "estimate_success", "description": "Estimate success odds and constraint satisfaction."},
            {"kind": "approve", "description": "Approve only if success threshold is met."},
            {"kind": "execute", "description": "Execute bounded governed action."},
            {"kind": "snapshot", "description": "Persist Merkle-verifiable state."},
        ]

    # -------------------------------------------------------------------------
    # Updating existing twins
    # -------------------------------------------------------------------------

    def update_observation(
        self,
        network: Any,
        *,
        observation: Mapping[str, Any],
        note: str = "thing_observation_update",
    ) -> Dict[str, Any]:
        try:
            payload = {
                "observation": dict(observation),
                "ts": _now_iso(),
                "source": "TwinAnythingFactory.update_observation",
            }

            network.update_node_data("5.4", payload, note=note)

            try:
                network.update_node_data(
                    "5.11",
                    {
                        "last_observation": dict(observation),
                        "ts": _now_iso(),
                    },
                    note="thing_memory_observation",
                )
            except Exception:
                pass

            self._finalize_histories(network)

            return {"ok": True, "node_id": "5.4", "ts": _now_iso()}

        except Exception as exc:
            return {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    def append_event(
        self,
        network: Any,
        *,
        event: Mapping[str, Any],
        note: str = "thing_event_append",
    ) -> Dict[str, Any]:
        try:
            payload = {
                "event": dict(event),
                "ts": _now_iso(),
                "source": "TwinAnythingFactory.append_event",
            }

            network.update_node_data("5.11", payload, note=note)
            self._finalize_histories(network)

            return {"ok": True, "node_id": "5.11", "ts": _now_iso()}

        except Exception as exc:
            return {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    def attach_source_file(
        self,
        network: Any,
        *,
        source_file: Mapping[str, Any],
        note: str = "thing_source_file_attach",
    ) -> Dict[str, Any]:
        try:
            dtk = importlib.import_module("digital_twin_kernel")
            manifest = SourceFileManifest.from_mapping(source_file)

            if "5.11.1" not in network.nodes:
                network.add_node(
                    "5.11.1",
                    "Source File Registry",
                    parent_id="5.11",
                    kind=self._node_kind(dtk, "storage"),
                )

            key = manifest.sha256 or _payload_hash(asdict(manifest))
            node_id = f"5.11.1.{_slug(key, 'source', max_len=32)}"

            if node_id not in network.nodes:
                network.add_node(
                    node_id,
                    f"Source File: {manifest.filename or key[:8]}",
                    parent_id="5.11.1",
                    kind=self._node_kind(dtk, "data"),
                    metadata={
                        "filename": manifest.filename,
                        "sha256": manifest.sha256,
                        "mime_type": manifest.mime_type,
                        "role": manifest.role,
                    },
                )

            network.update_node_data(node_id, manifest, note=note)
            self._finalize_histories(network)

            return {
                "ok": True,
                "node_id": node_id,
                "manifest": _json_safe(manifest),
                "ts": _now_iso(),
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    def refresh_tensor_network_profile(
        self,
        network: Any,
        *,
        note: str = "tensor_network_profile_refresh",
    ) -> Dict[str, Any]:
        try:
            if "5.12" not in network.nodes:
                dtk = importlib.import_module("digital_twin_kernel")
                network.add_node(
                    "5.12",
                    "Tensor-Network Runtime Profile",
                    parent_id=str(self.cfg.thing_root_id),
                    kind=self._node_kind(dtk, "model"),
                )

            builder = TensorNetworkProfileBuilder(
                self.cfg.tn_runtime,
                seed=int(self.cfg.seed),
                vector_dim=int(self.cfg.vector_dim),
            )
            profile = builder.build_profile()
            network.update_node_data("5.12", profile, note=note)
            self._finalize_histories(network)

            return {
                "ok": True,
                "node_id": "5.12",
                "profile": _json_safe(profile),
                "ts": _now_iso(),
            }

        except Exception as exc:
            return {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    # -------------------------------------------------------------------------
    # Success defaults
    # -------------------------------------------------------------------------

    def _install_success_defaults_if_available(
        self,
        ttn: Any,
        twin_input: TwinInput,
        node_map: Mapping[str, str],
    ) -> None:
        if not bool(self.cfg.install_success_defaults):
            return

        try:
            if "5.8" not in ttn.nodes:
                return

            payload = {
                "minimum_probability": _safe_float(
                    twin_input.success_criteria.get("minimum_probability", 0.80)
                    if isinstance(twin_input.success_criteria, Mapping)
                    else 0.80,
                    0.80,
                ),
                "requires_governance_approval": True,
                "requires_audit_trail": True,
                "requires_reversibility_or_mitigation": True,
                "default_threshold_source": "TwinAnythingFactory",
                "factory_version": self.VERSION,
            }
            ttn.nodes["5.8"].metadata["success_defaults"] = _json_safe(payload)
            ttn.bump_tick()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Finalization / snapshot
    # -------------------------------------------------------------------------

    def _finalize_histories(self, ttn: Any) -> None:
        try:
            ttn.push_global_histories()
            return
        except Exception:
            pass

        try:
            for nid in sorted(getattr(ttn, "nodes", {}).keys()):
                if hasattr(ttn, "_push_path_histories"):
                    ttn._push_path_histories(nid)
        except Exception:
            pass

    def _snapshot(self, dtk: Any, ttn: Any) -> Dict[str, Any]:
        try:
            root_hash = dtk.MerkleHasher.hex64(ttn.merkle_root())
        except Exception:
            root_hash = None

        try:
            root_measure = ttn.measure(str(self.cfg.thing_root_id))
        except Exception as exc:
            root_measure = {"error": repr(exc)}

        try:
            tn_measure = ttn.measure("5.12") if "5.12" in getattr(ttn, "nodes", {}) else None
        except Exception as exc:
            tn_measure = {"error": repr(exc)}

        return {
            "ts": _now_iso(),
            "node_count": len(getattr(ttn, "nodes", {})),
            "thing_root": str(self.cfg.thing_root_id),
            "merkle_root": root_hash,
            "thing_measure": _json_safe(root_measure),
            "tensor_network_measure": _json_safe(tn_measure),
        }


# =============================================================================
# Convenience functions
# =============================================================================

def build_anything_twin(
    *,
    thing_name: str,
    thing_type: str = "generic",
    description: str = "",
    observables: Optional[Mapping[str, Any]] = None,
    process_steps: Optional[Sequence[str]] = None,
    resources: Optional[Mapping[str, Any]] = None,
    constraints: Optional[Sequence[str]] = None,
    components: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, Any]] = None,
    success_criteria: Optional[Mapping[str, Any]] = None,
    risks: Optional[Sequence[Any]] = None,
    actions: Optional[Sequence[Any]] = None,
    source_files: Optional[Sequence[Mapping[str, Any]]] = None,
    links: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[TwinAnythingConfig] = None,
    **factory_kwargs: Any,
) -> Dict[str, Any]:
    factory = TwinAnythingFactory(config=config, **factory_kwargs)
    return factory.create_twin(
        thing_name=thing_name,
        thing_type=thing_type,
        description=description,
        observables=observables,
        process_steps=process_steps,
        resources=resources,
        constraints=constraints,
        components=components,
        environment=environment,
        success_criteria=success_criteria,
        risks=risks,
        actions=actions,
        source_files=source_files,
        links=links,
        tags=tags,
        metadata=metadata,
    )


def create_twin(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return build_anything_twin(*args, **kwargs)


def sample_twin_input() -> Dict[str, Any]:
    return {
        "thing_name": "Example Governed Process",
        "thing_type": "plan",
        "description": "A sample governed plan twin.",
        "process_steps": [
            "define objective",
            "verify resources",
            "simulate",
            "approve",
            "execute",
            "monitor",
        ],
        "observables": {
            "state": "initial",
            "readiness": 0.0,
            "success_probability": None,
        },
        "resources": {
            "operator": "required",
            "data": "required",
            "runtime": "digital_twin_kernel + tn.py",
        },
        "constraints": [
            "Must pass sandbox simulation.",
            "Must preserve audit trail.",
        ],
        "success_criteria": {
            "minimum_probability": 0.80,
            "requires_governance_approval": True,
        },
        "risks": [
            {
                "name": "unknown_dependency",
                "probability": 0.2,
                "impact": 0.5,
            }
        ],
        "actions": [
            {"kind": "observe"},
            {"kind": "simulate"},
            {"kind": "approve"},
        ],
        "tags": ["sample", "governed", "tensor-aware"],
        "metadata": {
            "created_by": "TwinAnythingFactory.sample_twin_input",
        },
    }


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akkurat Twin Anything Factory")

    p.add_argument(
        "--mode",
        choices=["status", "tn-status", "create", "sample-input"],
        default="status",
    )

    p.add_argument("--thing-name", default="Unnamed Thing")
    p.add_argument("--thing-type", default="generic")
    p.add_argument("--description", default="")
    p.add_argument("--input-json", default="")
    p.add_argument("--output", default="")
    p.add_argument("--project-root", default="")

    p.add_argument("--vector-dim", type=int, default=256)
    p.add_argument("--sketch-dim", type=int, default=96)
    p.add_argument("--history-capacity", type=int, default=64)
    p.add_argument("--seed", type=int, default=2027)

    p.add_argument("--disable-tn-projection", action="store_true")
    p.add_argument("--disable-platform-taxonomy", action="store_true")
    p.add_argument("--disable-success-defaults", action="store_true")
    p.add_argument("--disable-tn-profile", action="store_true")

    p.add_argument("--latent-geometry", choices=["euclidean", "hyperbolic"], default="euclidean")

    p.add_argument("--tt-rank", type=int, default=8)
    p.add_argument("--tucker-rank", type=int, default=16)
    p.add_argument("--tt-num-modes", type=int, default=2)
    p.add_argument("--tn-dtype", choices=["float16", "float32", "float64"], default="float32")
    p.add_argument("--tn-device", choices=["cpu", "gpu"], default="cpu")

    return p


def _config_from_args(args: argparse.Namespace) -> TwinAnythingConfig:
    tn_cfg = TensorNetworkRuntimeConfig(
        enable_tn_profile=not bool(args.disable_tn_profile),
        prefer_tn_projection=not bool(args.disable_tn_projection),
        prefer_tucker_fusion=True,
        tt_rank=int(args.tt_rank),
        tucker_rank=int(args.tucker_rank),
        tt_num_modes=int(args.tt_num_modes),
        dtype=str(args.tn_dtype),
        device=str(args.tn_device),
        check_finite=True,
    ).normalized()

    return TwinAnythingConfig(
        vector_dim=int(args.vector_dim),
        sketch_dim=int(args.sketch_dim),
        history_capacity=int(args.history_capacity),
        seed=int(args.seed),
        use_tn_projection=not bool(args.disable_tn_projection),
        latent_geometry=str(args.latent_geometry),
        install_platform_taxonomy=not bool(args.disable_platform_taxonomy),
        install_success_defaults=not bool(args.disable_success_defaults),
        install_tensor_network_profile=not bool(args.disable_tn_profile),
        project_root=str(args.project_root or ""),
        tn_runtime=tn_cfg,
    ).normalized()


def _create_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_json:
        payload = _read_json(args.input_json, {})
        if not isinstance(payload, Mapping):
            raise ValueError("--input-json must contain a JSON object")
        twin_input = TwinInput.from_mapping(payload)
    else:
        twin_input = TwinInput(
            thing_name=str(args.thing_name),
            thing_type=str(args.thing_type),
            description=str(args.description),
        )

    cfg = _config_from_args(args)
    factory = TwinAnythingFactory(config=cfg)
    result = factory.create_twin_from_input(twin_input)

    out = result.to_dict()

    if args.output:
        try:
            result.network.save_json(args.output, include_histories=True)
            out["saved_network_json"] = str(Path(args.output).resolve())
        except Exception:
            _write_json(args.output, out)
            out["saved_summary_json"] = str(Path(args.output).resolve())

    return out


def _sample_input_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    payload = sample_twin_input()

    if args.output:
        _write_json(args.output, payload)
        payload["saved_sample_input_json"] = str(Path(args.output).resolve())

    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        configure_paths(project_root=args.project_root or None)

        if args.mode == "status":
            out = status_report()

        elif args.mode == "tn-status":
            out = {
                "ok": True,
                "ts": _now_iso(),
                "tn": tn_status(),
            }

        elif args.mode == "create":
            out = _create_from_cli(args)

        elif args.mode == "sample-input":
            out = _sample_input_from_cli(args)

        else:
            raise ValueError(f"unknown mode: {args.mode}")

        print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
        return 0 if bool(out.get("ok", True)) else 2

    except Exception as exc:
        err = {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())