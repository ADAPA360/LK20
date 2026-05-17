#!/usr/bin/env python3
# digital_twin_kernel.py
r"""
Project Chimera / Akkurat - Governed Digital Twin Kernel
========================================================

Production CPU-first digital-twin substrate for the Akkurat cognitive stack and
AtomTN hybrid runtime.

Core capabilities
-----------------
- Fixed-width latent-state tree for physical, virtual, data, cognitive, and
  AtomTN physics-reservoir nodes.
- Deterministic heterogeneous projection for numeric arrays, JSON, text, and
  optional external embeddings.
- Optional TensorTrain random projection via tn.py with signed-hashing fallback.
- Euclidean or hyperbolic/Poincare-style latent geometry.
- Bounded hierarchical fusion: mean/sum, attention, Tucker-like, optional
  tn.TuckerFusionLayer pairwise fusion.
- Ring-buffer sketch histories, adaptive baselines, and local/cross-twin
  Granger-lite causal probing.
- Merkle-style stable state verification for local and federated twins.
- TwinContract trust state machine with peer identity, payload integrity, and
  age checks.
- Governed action execution: propose -> sandbox simulate -> approve -> execute.
- JSON persistence for topology, node state, metadata, histories, baselines, and
  projection/geometry configuration.

Relationship to cognitive_lobe_runtime.py and atom_adapter_runtime.py
---------------------------------------------------------------------
This module deliberately does not own NCP/CfC/AtomTN runtime loops. It is the
high-level governed world model. Cognitive and AtomTN adapters submit state
updates through AkkuratInterface.governed_actions(...), or directly through the
TreeTensorNetwork update methods in controlled demos.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Path setup and optional tn.py import
# =============================================================================


def _add_project_paths() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent]
    for p in candidates:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return here.parent.parent if len(here.parents) >= 2 else here.parent


_AKKURAT_ROOT = _add_project_paths()

_TN_OK = False
_TN_ERR = ""
tn = None
try:
    import tn as tn  # type: ignore
    _TN_OK = True
except Exception as e1:  # pragma: no cover
    try:
        from .. import tn as tn  # type: ignore
        _TN_OK = True
    except Exception as e2:  # pragma: no cover
        _TN_OK = False
        _TN_ERR = f"tn.py import failed: {repr(e1)} | {repr(e2)}"


# =============================================================================
# Optional integration protocol
# =============================================================================


class CortexLike(Protocol):
    def embed(self, text: str) -> np.ndarray: ...

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str: ...


# =============================================================================
# Errors
# =============================================================================


class TensorNetworkError(Exception):
    """Raised for invalid topology, state, or tensor operations."""


class PolicyError(Exception):
    """Raised when governance rejects an action set in strict mode."""


class FederationError(Exception):
    """Raised for federation snapshot / trust contract errors."""


# =============================================================================
# Utilities
# =============================================================================


_EPS = 1e-9


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _parse_iso_epoch(ts: str) -> Optional[float]:
    if not ts:
        return None
    try:
        return float(time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")))
    except Exception:
        return None


def _sanitize_array(x: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=dtype).reshape(-1)
    except Exception:
        arr = np.zeros(0, dtype=dtype)
    if arr.size == 0:
        return arr.astype(dtype, copy=False)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)


def _is_finite_vec(v: Any) -> bool:
    x = np.asarray(v)
    return bool(x.size > 0 and np.all(np.isfinite(x)))


def _l2_normalize(v: Any, eps: float = _EPS) -> np.ndarray:
    x = _sanitize_array(v, np.float32)
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32, copy=False)


def _poincare_project(v: Any, eps: float = _EPS, radius: float = 0.95) -> np.ndarray:
    """Project to the open Poincare ball. This is a stable runtime embedding,
    not a full Riemannian optimizer."""
    x = _sanitize_array(v, np.float32)
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(x, dtype=np.float32)
    # Map arbitrary norm to a bounded radius while preserving direction.
    r = float(radius) * math.tanh(n)
    return (x / n * r).astype(np.float32, copy=False)


def _coerce_geometry(v: Any, *, geometry: str = "euclidean", eps: float = _EPS) -> np.ndarray:
    if str(geometry).lower().strip() in {"hyperbolic", "poincare", "poincaré"}:
        return _poincare_project(v, eps=eps)
    return _l2_normalize(v, eps=eps)


def _cosine(a: Any, b: Any, eps: float = _EPS) -> float:
    aa = _sanitize_array(a, np.float32)
    bb = _sanitize_array(b, np.float32)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if not np.isfinite(na) or not np.isfinite(nb) or na < eps or nb < eps:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def _softmax_stable(x: Any) -> np.ndarray:
    z = _sanitize_array(x, np.float32)
    if z.size == 0:
        return z
    m = float(np.max(z))
    ez = np.exp(np.clip(z - m, -40.0, 40.0)).astype(np.float32)
    s = float(np.sum(ez))
    if not np.isfinite(s) or s <= 0.0:
        return np.full((z.size,), 1.0 / float(max(1, z.size)), dtype=np.float32)
    return (ez / s).astype(np.float32, copy=False)


def _stable_hash_u32(s: str) -> int:
    h = 2166136261
    for b in (s or "").encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _bytes_hash_u64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h ^= int(b)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h)


def _rng_from_key(seed: int, key: str) -> np.random.Generator:
    s = (int(seed) & 0xFFFFFFFF) ^ _stable_hash_u32(str(key))
    return np.random.default_rng(int(s))


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(x))))


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return _sanitize_array(obj, np.float32).astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    b = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
    return f"{_bytes_hash_u64(b):016x}"


# =============================================================================
# Enums and configs
# =============================================================================


class NodeKind(str, Enum):
    ROOT = "root"
    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    DATA = "data"
    ENTITY = "entity"
    RESOURCE = "resource"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    PROPERTY = "property"
    MODEL = "model"
    PIPELINE = "pipeline"
    STORAGE = "storage"
    VISUAL = "visual"
    COGNITIVE = "cognitive"
    ATOMTN = "atomtn"
    QUANTUM = "quantum"
    FLOW = "flow"
    OBSERVABLE = "observable"
    GOVERNANCE = "governance"
    GENERIC = "generic"


class FusionMode(str, Enum):
    SUM = "sum"
    ATTENTION = "attention"
    TUCKER = "tucker"
    TN_TUCKER = "tn_tucker"


class LatentGeometry(str, Enum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"


@dataclass
class FusionConfig:
    mode: FusionMode = FusionMode.ATTENTION
    attention_beta: float = 2.0
    residual_mix: float = 0.25
    tucker_rank: int = 16
    auto_tune_beta: bool = True
    learnable_bias: float = 0.0
    eps: float = 1e-9


@dataclass
class ProjectionConfig:
    vector_dim: int
    seed: int = 2027
    use_tn_projection: bool = True
    tt_rank: int = 8
    max_in_for_tn: int = 8192
    latent_geometry: str = LatentGeometry.EUCLIDEAN.value
    eps: float = 1e-9


@dataclass
class SketchConfig:
    sketch_dim: int = 96
    capacity: int = 64
    use_float16: bool = True


@dataclass
class BaselineConfig:
    base_beta: float = 0.02
    beta_min: float = 0.005
    beta_max: float = 0.20
    z_open: float = 2.0
    z_harden: float = 0.5
    eps: float = 1e-9


@dataclass
class CausalConfig:
    lag: int = 1
    k: int = 32
    min_samples: int = 12
    threshold: float = 0.08
    max_suggestions: int = 10


@dataclass
class PolicyConfig:
    allowed_action_kinds: Tuple[str, ...] = (
        "raise_alert",
        "attach_metadata",
        "update_node_data",
        "update_node_latent",
    )
    max_latent_l2_delta: float = 0.60
    max_actions_per_execute: int = 25
    strict: bool = False
    dreaming_trials: int = 3
    dreaming_horizon_steps: int = 2
    dreaming_fail_threshold: int = 2
    eps: float = 1e-9


@dataclass
class SimulationReport:
    ok: bool
    predicted_alerts: int
    max_delta: float
    notes: List[str] = field(default_factory=list)


# =============================================================================
# Projection engine
# =============================================================================


class ProjectionEngine:
    """Deterministic embeddings for heterogeneous payloads."""

    def __init__(self, cfg: ProjectionConfig):
        self.cfg = cfg
        self._tt_cache: Dict[Tuple[str, int, int], Any] = {}

    @staticmethod
    def _get_factors(n: int) -> Tuple[int, int]:
        n = int(n)
        if n <= 0:
            return (1, 1)
        a = int(math.isqrt(n))
        while a > 0:
            if n % a == 0:
                return (a, n // a)
            a -= 1
        return (1, n)

    def _finish(self, v: Any) -> np.ndarray:
        return _coerce_geometry(v, geometry=self.cfg.latent_geometry, eps=self.cfg.eps)

    def _hash_project(self, x: Any, *, key: str) -> np.ndarray:
        D = int(self.cfg.vector_dim)
        out = np.zeros(D, dtype=np.float32)
        arr = _sanitize_array(x, np.float32)
        if arr.size == 0:
            return out
        base = _stable_hash_u32(f"hproj::{self.cfg.seed}::{key}::{arr.size}::{D}")
        for i, val in enumerate(arr):
            if not np.isfinite(float(val)):
                continue
            h = _stable_hash_u32(f"{base}:{i}")
            out[int(h % D)] += (-1.0 if (h & 1) else 1.0) * np.float32(val)
        return self._finish(out)

    def _tn_project(self, x: Any, *, key: str) -> np.ndarray:
        if not (_TN_OK and self.cfg.use_tn_projection and tn is not None):
            return self._hash_project(x, key=key)
        arr = _sanitize_array(x, np.float32)
        if arr.size == 0:
            return np.zeros(int(self.cfg.vector_dim), dtype=np.float32)
        if int(arr.size) > int(self.cfg.max_in_for_tn):
            return self._hash_project(arr, key=key)
        D = int(self.cfg.vector_dim)
        L = int(arr.size)
        cache_key = (key, L, D)
        op = self._tt_cache.get(cache_key)
        if op is None:
            try:
                o1, o2 = self._get_factors(D)
                i1, i2 = self._get_factors(L)
                rng = _rng_from_key(self.cfg.seed, f"ttproj::{key}::{L}->{D}::{self.cfg.tt_rank}")
                cfg = tn.TTConfig(dtype=np.float32, device="cpu") if hasattr(tn, "TTConfig") else None
                kwargs = {"config": cfg} if cfg is not None else {}
                op = tn.TensorTrain(  # type: ignore[attr-defined]
                    output_dims=[o1, o2],
                    input_dims=[i1, i2],
                    bond_dims=[int(max(1, self.cfg.tt_rank))],
                    rng=rng,
                    init_scale=1e-2,
                    **kwargs,
                )
                self._tt_cache[cache_key] = op
            except Exception:
                return self._hash_project(arr, key=key)
        try:
            y = np.asarray(op.apply(arr), dtype=np.float32).reshape(-1)  # type: ignore[attr-defined]
            if y.size != D or not np.all(np.isfinite(y)):
                return self._hash_project(arr, key=key)
            return self._finish(y)
        except Exception:
            return self._hash_project(arr, key=key)

    def project(self, x: Any, *, key: str) -> np.ndarray:
        return self._tn_project(x, key=key)

    def encode_numbers(self, arr: Any, *, key: str) -> np.ndarray:
        x = _sanitize_array(arr, np.float32)
        stats = np.array(
            [
                float(np.mean(x)) if x.size else 0.0,
                float(np.std(x)) if x.size else 0.0,
                float(np.min(x)) if x.size else 0.0,
                float(np.max(x)) if x.size else 0.0,
                float(np.linalg.norm(x)) if x.size else 0.0,
            ],
            dtype=np.float32,
        )
        feat = np.concatenate([x[: min(256, x.size)], stats], axis=0)
        return self.project(feat, key=f"num::{key}")

    def encode_json(self, obj: Any, *, key: str) -> np.ndarray:
        try:
            s = json.dumps(_json_safe(obj), sort_keys=True, ensure_ascii=False)
        except Exception:
            s = str(obj)
        b = s.encode("utf-8", errors="ignore")
        buckets = np.zeros(256, dtype=np.float32)
        for byte in b[:8000]:
            buckets[int(byte)] += 1.0
        return self.project(_l2_normalize(buckets), key=f"json::{key}")

    def encode_text(self, text: str, *, key: str, cortex: Optional[CortexLike] = None) -> np.ndarray:
        if cortex is not None and hasattr(cortex, "embed"):
            try:
                emb = _sanitize_array(cortex.embed(text or ""), np.float32)
                if emb.size:
                    return self.project(emb, key=f"textemb::{key}")
            except Exception:
                pass
        return self.encode_json({"text": text or ""}, key=key)


# =============================================================================
# Histories and baselines
# =============================================================================


class RingSketch:
    def __init__(self, cfg: SketchConfig):
        self.cfg = cfg
        dtype = np.float16 if cfg.use_float16 else np.float32
        self._buf = np.zeros((int(cfg.capacity), int(cfg.sketch_dim)), dtype=dtype)
        self._ts = np.zeros((int(cfg.capacity),), dtype=np.int64)
        self._n = 0
        self._i = 0

    def push(self, sketch: Any, *, ts: Optional[int] = None) -> None:
        s = _sanitize_array(sketch, np.float32)
        if s.size != int(self.cfg.sketch_dim):
            s = np.zeros(int(self.cfg.sketch_dim), dtype=np.float32)
        self._buf[self._i] = s.astype(self._buf.dtype, copy=False)
        self._ts[self._i] = int(ts if ts is not None else time.time())
        self._i = (self._i + 1) % int(self.cfg.capacity)
        self._n = min(int(self.cfg.capacity), self._n + 1)

    def size(self) -> int:
        return int(self._n)

    def get_recent(self, k: int) -> np.ndarray:
        k = int(max(0, min(int(k), self._n)))
        if k <= 0:
            return np.zeros((0, int(self.cfg.sketch_dim)), dtype=np.float32)
        idxs = [((self._i - k + j) % int(self.cfg.capacity)) for j in range(k)]
        return self._buf[np.array(idxs, dtype=np.int64)].astype(np.float32, copy=False)

    def get_recent_pair(self, lag: int = 1, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        lag = int(max(1, lag))
        S = self.get_recent(k + lag)
        if S.shape[0] <= lag:
            return (np.zeros((0, int(self.cfg.sketch_dim)), dtype=np.float32), np.zeros((0, int(self.cfg.sketch_dim)), dtype=np.float32))
        return (S[:-lag], S[lag:])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cfg": _json_safe(asdict(self.cfg)),
            "buf": self._buf.astype(np.float32).tolist(),
            "ts": self._ts.astype(int).tolist(),
            "n": int(self._n),
            "i": int(self._i),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RingSketch":
        cfgd = dict(payload.get("cfg", {}))
        cfg = SketchConfig(**{k: cfgd.get(k, getattr(SketchConfig(), k)) for k in asdict(SketchConfig()).keys()})
        obj = cls(cfg)
        try:
            arr = np.asarray(payload.get("buf", []), dtype=np.float32)
            if arr.shape == obj._buf.shape:
                obj._buf = arr.astype(obj._buf.dtype, copy=False)
            ts = np.asarray(payload.get("ts", []), dtype=np.int64)
            if ts.shape == obj._ts.shape:
                obj._ts = ts
            obj._n = int(payload.get("n", 0))
            obj._i = int(payload.get("i", 0)) % int(cfg.capacity)
        except Exception:
            pass
        return obj


class AdaptiveBaselineTracker:
    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self.mu = 0.0
        self.var = 1.0
        self.n = 0

    def update(self, x: float) -> Dict[str, float]:
        x = float(x)
        if not np.isfinite(x):
            return self.snapshot()
        sigma = math.sqrt(max(float(self.cfg.eps), float(self.var)))
        z = 0.0 if sigma <= self.cfg.eps else (x - float(self.mu)) / sigma
        az = abs(z)
        beta = float(self.cfg.base_beta)
        if az >= self.cfg.z_open:
            beta = min(float(self.cfg.beta_max), beta * (1.0 + 0.25 * (az - self.cfg.z_open)))
        elif az <= self.cfg.z_harden:
            beta = max(float(self.cfg.beta_min), beta * 0.5)
        beta = _clamp(beta, self.cfg.beta_min, self.cfg.beta_max)
        if self.n == 0:
            self.mu = x
            self.var = 1.0
            self.n = 1
            out = self.snapshot()
            out.update({"z": float(z), "beta": float(beta)})
            return out
        mu_new = (1.0 - beta) * self.mu + beta * x
        dev = x - mu_new
        var_new = (1.0 - beta) * self.var + beta * dev * dev
        if np.isfinite(mu_new) and np.isfinite(var_new):
            self.mu = float(mu_new)
            self.var = float(max(float(self.cfg.eps), var_new))
            self.n += 1
        out = self.snapshot()
        out.update({"z": float(z), "beta": float(beta)})
        return out

    def snapshot(self) -> Dict[str, float]:
        return {"mu": float(self.mu), "sigma": float(math.sqrt(max(float(self.cfg.eps), float(self.var)))), "n": float(self.n), "z": 0.0, "beta": float(self.cfg.base_beta)}

    def to_dict(self) -> Dict[str, Any]:
        return {"cfg": _json_safe(asdict(self.cfg)), "mu": float(self.mu), "var": float(self.var), "n": int(self.n)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptiveBaselineTracker":
        cfgd = dict(payload.get("cfg", {}))
        cfg = BaselineConfig(**{k: cfgd.get(k, getattr(BaselineConfig(), k)) for k in asdict(BaselineConfig()).keys()})
        obj = cls(cfg)
        obj.mu = float(payload.get("mu", 0.0))
        obj.var = float(payload.get("var", 1.0))
        obj.n = int(payload.get("n", 0))
        return obj


# =============================================================================
# Merkle hashing and causal probes
# =============================================================================


class MerkleHasher:
    def __init__(self, *, seed: int = 2027):
        self.seed = int(seed)

    def hash_leaf(self, node_id: str, sketch: Any) -> int:
        sid = (node_id or "").encode("utf-8", errors="ignore")
        sk = _sanitize_array(sketch, np.float32).astype(np.float16).tobytes()
        return _bytes_hash_u64(sid + sk + int(self.seed).to_bytes(4, "little", signed=False))

    def hash_node(self, node_id: str, sketch: Any, child_hashes: Sequence[int]) -> int:
        base = self.hash_leaf(node_id, sketch)
        payload = base.to_bytes(8, "little", signed=False)
        for h in sorted(int(x) & 0xFFFFFFFFFFFFFFFF for x in child_hashes):
            payload += int(h).to_bytes(8, "little", signed=False)
        return _bytes_hash_u64(payload)

    @staticmethod
    def hex64(h: int) -> str:
        return f"{int(h) & 0xFFFFFFFFFFFFFFFF:016x}"


@dataclass
class SuggestEdge:
    src: str
    dst: str
    score: float
    note: str = ""


class CausalProbe:
    def __init__(self, cfg: CausalConfig):
        self.cfg = cfg

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return 0.0
        T = int(a.shape[0])
        if T <= 0:
            return 0.0
        return float(sum(_cosine(a[t], b[t]) for t in range(T)) / max(1, T))

    def suggest(self, histories: Mapping[str, RingSketch], candidates: Optional[List[Tuple[str, str]]] = None) -> List[SuggestEdge]:
        pairs: List[Tuple[str, str]] = []
        if candidates:
            pairs = list(candidates)
        else:
            top = [nid for nid, hist in sorted(histories.items()) if hist.size() >= self.cfg.min_samples and nid.count(".") <= 2]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    pairs.append((top[i], top[j]))
        out: List[SuggestEdge] = []
        for a, b in pairs:
            ha = histories.get(a)
            hb = histories.get(b)
            if ha is None or hb is None:
                continue
            A_p, A_n = ha.get_recent_pair(lag=self.cfg.lag, k=self.cfg.k)
            B_p, B_n = hb.get_recent_pair(lag=self.cfg.lag, k=self.cfg.k)
            T = min(A_p.shape[0], A_n.shape[0], B_p.shape[0], B_n.shape[0])
            if T < self.cfg.min_samples:
                continue
            A_p, A_n, B_p, B_n = A_p[-T:], A_n[-T:], B_p[-T:], B_n[-T:]
            c_ab = self._corr(A_p, B_n)
            c_ba = self._corr(B_p, A_n)
            score = float(c_ab - c_ba)
            if score >= self.cfg.threshold:
                out.append(SuggestEdge(src=a, dst=b, score=score, note="A_past predicts B_now"))
            elif score <= -self.cfg.threshold:
                out.append(SuggestEdge(src=b, dst=a, score=-score, note="B_past predicts A_now"))
        out.sort(key=lambda e: float(e.score), reverse=True)
        return out[: int(self.cfg.max_suggestions)]

    def suggest_cross_twin(self, local_histories: Mapping[str, RingSketch], peer_histories: Mapping[str, Any], *, peer_prefix: str = "peer") -> List[SuggestEdge]:
        candidates: List[SuggestEdge] = []
        local_keys = [k for k, h in local_histories.items() if h.size() >= self.cfg.min_samples and k.count(".") <= 3]
        peer_keys = sorted(peer_histories.keys())
        for lk in local_keys:
            L_p, L_n = local_histories[lk].get_recent_pair(lag=self.cfg.lag, k=self.cfg.k)
            for pk in peer_keys:
                P = np.asarray(peer_histories[pk], dtype=np.float32)
                if P.ndim != 2 or P.shape[0] <= self.cfg.lag:
                    continue
                P_p, P_n = P[:-self.cfg.lag], P[self.cfg.lag:]
                T = min(L_p.shape[0], L_n.shape[0], P_p.shape[0], P_n.shape[0])
                if T < self.cfg.min_samples or L_p.shape[1] != P_n.shape[1]:
                    continue
                c_lp = self._corr(L_p[-T:], P_n[-T:])
                c_pl = self._corr(P_p[-T:], L_n[-T:])
                score = float(c_lp - c_pl)
                if score >= self.cfg.threshold:
                    candidates.append(SuggestEdge(src=lk, dst=f"{peer_prefix}:{pk}", score=score, note="local_past predicts peer_now"))
                elif score <= -self.cfg.threshold:
                    candidates.append(SuggestEdge(src=f"{peer_prefix}:{pk}", dst=lk, score=-score, note="peer_past predicts local_now"))
        candidates.sort(key=lambda e: float(e.score), reverse=True)
        return candidates[: int(self.cfg.max_suggestions)]


# =============================================================================
# Nodes and fusion
# =============================================================================


@dataclass
class TensorNode:
    node_id: str
    name: str
    kind: NodeKind = NodeKind.GENERIC
    vector_dim: int = 1024
    state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    fusion: FusionConfig = field(default_factory=FusionConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)
    latent_geometry: str = LatentGeometry.EUCLIDEAN.value

    def __post_init__(self) -> None:
        if int(self.vector_dim) <= 0:
            raise ValueError("vector_dim must be positive")
        if np.asarray(self.state).size == 0:
            self.state = np.zeros(int(self.vector_dim), dtype=np.float32)
        self.state = self._coerce(self.state)

    def _coerce(self, v: Any) -> np.ndarray:
        x = _sanitize_array(v, np.float32)
        if x.size != int(self.vector_dim):
            # Size mismatch should not silently erase signal; project deterministically
            rng = _rng_from_key(0, f"node_resize::{self.node_id}::{x.size}->{self.vector_dim}")
            out = np.zeros(int(self.vector_dim), dtype=np.float32)
            if x.size:
                for i, val in enumerate(x):
                    h = _stable_hash_u32(f"{self.node_id}:{i}:{float(val):.8g}")
                    out[h % int(self.vector_dim)] += (-1.0 if (h & 1) else 1.0) * np.float32(val)
            x = out
        return _coerce_geometry(x, geometry=self.latent_geometry, eps=self.fusion.eps)

    def set_state(self, v: Any) -> None:
        self.state = self._coerce(v)
        self.updated_at = _now_iso()

    def entropy(self) -> float:
        x = _sanitize_array(self.state, np.float32)
        if x.size == 0:
            return 0.0
        p = (x * x).astype(np.float32)
        s = float(np.sum(p))
        if not np.isfinite(s) or s <= self.fusion.eps:
            return 0.0
        p = p / s
        return float(-np.sum(p * np.log(p + 1e-10)))

    def summary(self) -> Dict[str, Any]:
        x = _sanitize_array(self.state, np.float32)
        return {
            "id": self.node_id,
            "name": self.name,
            "kind": str(self.kind.value if isinstance(self.kind, Enum) else self.kind),
            "updated_at": self.updated_at,
            "norm": float(np.linalg.norm(x)),
            "entropy": float(self.entropy()),
            "mean": float(np.mean(x)) if x.size else 0.0,
            "std": float(np.std(x)) if x.size else 0.0,
            "latent_geometry": self.latent_geometry,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "kind": self.kind.value if isinstance(self.kind, Enum) else str(self.kind),
            "vector_dim": int(self.vector_dim),
            "state": self.state.astype(float).tolist(),
            "fusion": _json_safe(asdict(self.fusion)),
            "metadata": _json_safe(self.metadata),
            "connections": list(self.connections),
            "updated_at": self.updated_at,
            "latent_geometry": self.latent_geometry,
        }


class _TuckerFusionAdapter:
    def __init__(self, *, vector_dim: int, tucker_rank: int, seed: int, key: str, projector: ProjectionEngine):
        self.vector_dim = int(vector_dim)
        self.tucker_rank = int(max(4, tucker_rank))
        self.projector = projector
        self.key = str(key)
        self._layer = None
        if _TN_OK and tn is not None:
            try:
                rng = _rng_from_key(seed, f"tn_tucker::{key}::{vector_dim}->{tucker_rank}")
                self._layer = tn.TuckerFusionLayer(  # type: ignore[attr-defined]
                    input_dims=[self.vector_dim, self.vector_dim],
                    output_rank=self.tucker_rank,
                    rng=rng,
                    dtype=np.float32,
                )
            except Exception:
                self._layer = None

    def fuse_pair(self, a: Any, b: Any) -> np.ndarray:
        aa = _l2_normalize(a)
        bb = _l2_normalize(b)
        if self._layer is not None:
            try:
                z = self._layer.contract([aa, bb])  # type: ignore[union-attr]
                return self.projector.project(z, key=f"tn_tucker_out::{self.key}")
            except Exception:
                pass
        return self.projector.project(np.concatenate([aa, bb], axis=0), key=f"tn_tucker_fallback::{self.key}")


def _fuse_children(
    parent_state: np.ndarray,
    child_states: List[np.ndarray],
    cfg: FusionConfig,
    *,
    projector: ProjectionEngine,
    fusion_key: str,
    causal_bias: Optional[Dict[str, float]] = None,
    adapter_cache: Optional[MutableMapping[Tuple[int, int, str], _TuckerFusionAdapter]] = None,
) -> np.ndarray:
    D = int(np.asarray(parent_state).size)
    if D <= 0:
        return np.zeros(0, dtype=np.float32)
    if not child_states:
        return projector._finish(parent_state)
    C = np.stack([projector._finish(c) for c in child_states], axis=0).astype(np.float32, copy=False)
    M = int(C.shape[0])
    mode = FusionMode(str(cfg.mode)) if not isinstance(cfg.mode, FusionMode) else cfg.mode
    if mode == FusionMode.SUM:
        fused = np.mean(C, axis=0).astype(np.float32, copy=False)
    elif mode == FusionMode.ATTENTION:
        beta = float(cfg.attention_beta) + float(cfg.learnable_bias)
        if cfg.auto_tune_beta and causal_bias is not None:
            beta = _clamp(beta * (1.0 + 0.5 * float(causal_bias.get("gain", 0.0))), 0.5, 8.0)
        scores = np.array([_cosine(C[i], parent_state, eps=cfg.eps) for i in range(M)], dtype=np.float32)
        w = _softmax_stable(beta * scores).reshape(M, 1)
        fused = np.sum(C * w, axis=0).astype(np.float32, copy=False)
    elif mode == FusionMode.TN_TUCKER:
        ck = (D, int(cfg.tucker_rank), fusion_key)
        if adapter_cache is not None and ck in adapter_cache:
            adapter = adapter_cache[ck]
        else:
            adapter = _TuckerFusionAdapter(vector_dim=D, tucker_rank=int(cfg.tucker_rank), seed=projector.cfg.seed, key=fusion_key, projector=projector)
            if adapter_cache is not None:
                adapter_cache[ck] = adapter
        cur = C[0]
        for i in range(1, M):
            cur = adapter.fuse_pair(cur, C[i])
        fused = cur.astype(np.float32, copy=False)
    else:
        r = int(max(4, min(int(cfg.tucker_rank), D)))
        Z = []
        for i in range(M):
            z = projector.project(C[i], key=f"tucker_rank::{fusion_key}::{i}::{r}")[:r]
            Z.append(z)
        avg = np.mean(np.stack(Z, axis=0), axis=0).astype(np.float32, copy=False)
        fused = projector.project(avg, key=f"tucker_back::{fusion_key}::{r}->{D}")
    mix = _clamp(float(cfg.residual_mix), 0.0, 1.0)
    return projector._finish(mix * parent_state + (1.0 - mix) * fused)


# =============================================================================
# Federation data structures
# =============================================================================


class ContractStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    DISPUTED = "DISPUTED"
    QUARANTINED = "QUARANTINED"


@dataclass
class ContractPolicy:
    degrade_k: int = 2
    dispute_k: int = 3
    quarantine_k: int = 5
    min_coherence_ok: float = 0.35
    min_coherence_degraded: float = 0.20
    max_payload_age_s: int = 60
    trust_gamma: float = 0.85


@dataclass
class TwinSnapshot:
    peer_id: str
    ts: str
    root_hash: str
    node_id: str
    sketch: List[float]
    payload_hash: str


@dataclass
class TwinHistorySnapshot:
    peer_id: str
    ts: str
    root_hash: str
    node_ids: List[str]
    histories: Dict[str, List[List[float]]]
    payload_hash: str


class TwinContract:
    def __init__(self, peer_id: str, *, policy: ContractPolicy):
        self.peer_id = str(peer_id)
        self.policy = policy
        self.status: ContractStatus = ContractStatus.OK
        self._bad = 0
        self._good = 0
        self.trust = 1.0

    def ingest(self, local_sketch: Any, snapshot: TwinSnapshot) -> Dict[str, Any]:
        ok_peer = str(snapshot.peer_id) == self.peer_id
        snap_payload = {
            "peer_id": snapshot.peer_id,
            "ts": snapshot.ts,
            "root_hash": snapshot.root_hash,
            "node_id": snapshot.node_id,
            "sketch": snapshot.sketch,
        }
        ok_hash = _payload_hash(snap_payload) == str(snapshot.payload_hash)
        ts_epoch = _parse_iso_epoch(snapshot.ts)
        age = float(time.time() - ts_epoch) if ts_epoch is not None else float("inf")
        ok_age = age <= float(self.policy.max_payload_age_s)
        peer_sk = _sanitize_array(snapshot.sketch, np.float32)
        loc = _sanitize_array(local_sketch, np.float32)
        coherence = _cosine(loc, peer_sk) if loc.size == peer_sk.size else 0.0
        good = bool(ok_peer and ok_hash and ok_age and coherence >= self.policy.min_coherence_ok)
        degraded = bool(ok_peer and ok_hash and ok_age and coherence >= self.policy.min_coherence_degraded)
        if good:
            self._good += 1
            self._bad = max(0, self._bad - 1)
            self.trust = _clamp(self.trust + 0.02 * (1.0 - self.trust), 0.0, 1.0)
        else:
            self._bad += 1
            self._good = max(0, self._good - 1)
            self.trust = _clamp(self.policy.trust_gamma * self.trust, 0.0, 1.0)
        if self._bad >= self.policy.quarantine_k:
            self.status = ContractStatus.QUARANTINED
        elif self._bad >= self.policy.dispute_k:
            self.status = ContractStatus.DISPUTED
        elif (not good) and degraded:
            self.status = ContractStatus.DEGRADED
        elif good and self._bad == 0:
            self.status = ContractStatus.OK
        return {
            "peer_id": self.peer_id,
            "ok_peer": bool(ok_peer),
            "ok_hash": bool(ok_hash),
            "ok_age": bool(ok_age),
            "age_s": float(age) if np.isfinite(age) else None,
            "coherence": float(coherence),
            "good": bool(good),
            "status": str(self.status.value),
            "trust": float(self.trust),
            "bad_k": int(self._bad),
            "good_k": int(self._good),
        }


# =============================================================================
# Tree Tensor Network
# =============================================================================


class TreeTensorNetwork:
    def __init__(
        self,
        *,
        vector_dim: int,
        seed: int = 2027,
        root_name: str = "Digital Twin",
        sketch_dim: Optional[int] = None,
        history_capacity: Optional[int] = None,
        use_tn_projection: bool = True,
        latent_geometry: str = LatentGeometry.EUCLIDEAN.value,
    ):
        if int(vector_dim) <= 0:
            raise ValueError("vector_dim must be positive")
        self.vector_dim = int(vector_dim)
        self.seed = int(seed)
        self.latent_geometry = str(latent_geometry).lower().strip()
        if self.latent_geometry not in {LatentGeometry.EUCLIDEAN.value, LatentGeometry.HYPERBOLIC.value}:
            self.latent_geometry = LatentGeometry.EUCLIDEAN.value
        self.sketch_dim = int(sketch_dim) if sketch_dim is not None else min(96, self.vector_dim)
        self.sketch_dim = int(max(8, min(self.sketch_dim, self.vector_dim)))
        self.history_capacity = int(history_capacity) if history_capacity is not None else 64
        self.history_capacity = int(max(1, self.history_capacity))
        self.use_tn_projection = bool(use_tn_projection)

        self.projector = ProjectionEngine(ProjectionConfig(vector_dim=self.vector_dim, seed=self.seed, use_tn_projection=self.use_tn_projection, latent_geometry=self.latent_geometry))
        self.sketch_projector = ProjectionEngine(ProjectionConfig(vector_dim=self.sketch_dim, seed=self.seed, use_tn_projection=self.use_tn_projection, max_in_for_tn=4096, latent_geometry=LatentGeometry.EUCLIDEAN.value))

        self.nodes: Dict[str, TensorNode] = {}
        self.children: Dict[str, List[str]] = {}
        self.parent: Dict[str, str] = {}
        self.histories: Dict[str, RingSketch] = {}
        self.baselines: Dict[str, AdaptiveBaselineTracker] = {}
        self.merkle = MerkleHasher(seed=self.seed)
        self.causal = CausalProbe(CausalConfig())
        self._tick = 0
        self._agg_cache: Dict[Tuple[int, str], np.ndarray] = {}
        self._sketch_cache: Dict[Tuple[int, str], np.ndarray] = {}
        self._merkle_cache: Dict[Tuple[int, str], int] = {}
        self._tucker_cache: Dict[Tuple[int, int, str], _TuckerFusionAdapter] = {}
        self.control_plane: Optional["AkkuratInterface"] = None
        self.fusion_learning: Dict[str, float] = {}

        self.root_id = "0"
        self.add_node(self.root_id, root_name, kind=NodeKind.ROOT)

    # --------------------------
    # Graph ops
    # --------------------------

    def add_node(
        self,
        node_id: str,
        name: str,
        *,
        parent_id: Optional[str] = None,
        kind: Union[NodeKind, str] = NodeKind.GENERIC,
        fusion: Optional[FusionConfig] = None,
        initial_state: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        node_id = str(node_id)
        if node_id in self.nodes:
            return
        try:
            nk = kind if isinstance(kind, NodeKind) else NodeKind(str(kind))
        except Exception:
            nk = NodeKind.GENERIC
        node = TensorNode(
            node_id=node_id,
            name=str(name),
            kind=nk,
            vector_dim=self.vector_dim,
            state=np.zeros(self.vector_dim, dtype=np.float32) if initial_state is None else initial_state,
            fusion=fusion or FusionConfig(),
            metadata=dict(metadata or {}),
            latent_geometry=self.latent_geometry,
        )
        self.nodes[node_id] = node
        self.children.setdefault(node_id, [])
        if parent_id is not None:
            parent_id = str(parent_id)
            self.children.setdefault(parent_id, [])
            if node_id not in self.children[parent_id]:
                self.children[parent_id].append(node_id)
            self.parent[node_id] = parent_id
        self.histories[node_id] = RingSketch(SketchConfig(sketch_dim=self.sketch_dim, capacity=self.history_capacity, use_float16=True))
        self.baselines[node_id] = AdaptiveBaselineTracker(BaselineConfig())
        self.bump_tick()

    def get_path_to_root(self, node_id: str) -> List[str]:
        out = []
        cur = str(node_id)
        seen = set()
        while cur and cur not in seen:
            out.append(cur)
            seen.add(cur)
            if cur == self.root_id:
                break
            cur = self.parent.get(cur, "")
        return out

    def set_fusion(self, node_id: str, fusion: FusionConfig) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].fusion = fusion
            self.bump_tick()

    def bump_tick(self) -> None:
        self._tick += 1
        self._agg_cache.clear()
        self._sketch_cache.clear()
        self._merkle_cache.clear()

    # --------------------------
    # Update / encode / histories
    # --------------------------

    def _push_history(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        sk = self.sketch_projector.project(self.nodes[node_id].state, key=f"sk::{node_id}")
        self.histories[node_id].push(sk)
        info = self.baselines[node_id].update(float(np.linalg.norm(sk)))
        self.nodes[node_id].metadata["baseline"] = info

    def _push_path_histories(self, node_id: str) -> None:
        # Ensure aggregate histories on parents/root reflect child updates.
        for nid in self.get_path_to_root(node_id):
            if nid in self.nodes:
                # For ancestors, push aggregate sketch rather than raw node state.
                sk = self.sketch(nid)
                self.histories[nid].push(sk)
                info = self.baselines[nid].update(float(np.linalg.norm(sk)))
                self.nodes[nid].metadata["baseline"] = info

    def push_global_histories(self) -> None:
        for nid in sorted(self.nodes.keys()):
            sk = self.sketch(nid)
            self.histories[nid].push(sk)
            info = self.baselines[nid].update(float(np.linalg.norm(sk)))
            self.nodes[nid].metadata["baseline"] = info

    def update_node_latent(self, node_id: str, latent: Any, *, note: Optional[str] = None) -> None:
        if node_id not in self.nodes:
            raise TensorNetworkError(f"Unknown node_id: {node_id}")
        self.nodes[node_id].set_state(latent)
        if note:
            self.nodes[node_id].metadata["note"] = str(note)
        self.bump_tick()
        self._push_path_histories(node_id)
        if self.control_plane:
            self.control_plane.log_event(f"Node {node_id} latent updated.", meta={"note": note})

    def update_node_data(self, node_id: str, data: Any, *, cortex: Optional[CortexLike] = None, data_key: Optional[str] = None, note: Optional[str] = None) -> None:
        if node_id not in self.nodes:
            raise TensorNetworkError(f"Unknown node_id: {node_id}")
        key = data_key or f"{node_id}:{self.nodes[node_id].name}"
        if isinstance(data, str):
            v = self.projector.encode_text(data, key=key, cortex=cortex)
        elif isinstance(data, (list, tuple, np.ndarray)):
            v = self.projector.encode_numbers(np.asarray(data), key=key)
        else:
            v = self.projector.encode_json(data, key=key)
        self.nodes[node_id].set_state(v)
        if note:
            self.nodes[node_id].metadata["note"] = str(note)
        self.nodes[node_id].metadata["last_data_hash"] = _payload_hash({"data": data})
        self.bump_tick()
        self._push_path_histories(node_id)
        if self.control_plane:
            self.control_plane.log_event(f"Node {node_id} updated from raw data.", meta={"note": note})

    # --------------------------
    # Aggregation
    # --------------------------

    def aggregate(self, node_id: str) -> np.ndarray:
        if node_id not in self.nodes:
            raise TensorNetworkError(f"Unknown node_id: {node_id}")
        ck = (self._tick, node_id)
        if ck in self._agg_cache:
            return self._agg_cache[ck]
        node = self.nodes[node_id]
        kids = self.children.get(node_id, [])
        if not kids:
            out = node.state.copy()
            self._agg_cache[ck] = out
            return out
        child_vecs = [self.aggregate(k) for k in kids]
        base = node.metadata.get("baseline", {})
        z = float(base.get("z", 0.0)) if isinstance(base, Mapping) else 0.0
        gain = float(max(0.0, 1.0 - min(1.0, abs(z) / 3.0)))
        if node_id in self.fusion_learning:
            node.fusion.learnable_bias = float(self.fusion_learning[node_id])
        out = _fuse_children(
            node.state,
            child_vecs,
            node.fusion,
            projector=self.projector,
            fusion_key=f"fuse::{node_id}",
            causal_bias={"gain": gain},
            adapter_cache=self._tucker_cache,
        )
        self._agg_cache[ck] = out
        return out

    def global_state(self) -> np.ndarray:
        return self.aggregate(self.root_id)

    def sketch(self, node_id: str) -> np.ndarray:
        ck = (self._tick, node_id)
        if ck in self._sketch_cache:
            return self._sketch_cache[ck]
        v = self.aggregate(node_id)
        sk = self.sketch_projector.project(v, key=f"aggsk::{node_id}")
        self._sketch_cache[ck] = sk
        return sk

    def merkle_root(self, node_id: Optional[str] = None) -> int:
        nid = node_id or self.root_id
        if nid not in self.nodes:
            raise TensorNetworkError(f"Unknown node_id: {nid}")
        ck = (self._tick, nid)
        if ck in self._merkle_cache:
            return self._merkle_cache[ck]
        child_hashes = [self.merkle_root(k) for k in self.children.get(nid, [])]
        h = self.merkle.hash_node(nid, self.sketch(nid), child_hashes)
        self._merkle_cache[ck] = h
        return h

    # --------------------------
    # Measurements
    # --------------------------

    def correlation(self, node_a: str, node_b: str) -> float:
        return _cosine(self.aggregate(node_a), self.aggregate(node_b))

    def measure(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self.nodes:
            raise TensorNetworkError(f"Unknown node_id: {node_id}")
        node = self.nodes[node_id]
        agg = self.aggregate(node_id)
        sk = self.sketch(node_id)
        temp = TensorNode(node_id=node_id, name=node.name, kind=node.kind, vector_dim=self.vector_dim, state=agg, latent_geometry=self.latent_geometry)
        return {
            "node": node.summary(),
            "aggregate": {
                "norm": float(np.linalg.norm(agg)),
                "entropy": float(temp.entropy()),
                "mean": float(np.mean(agg)) if agg.size else 0.0,
                "std": float(np.std(agg)) if agg.size else 0.0,
            },
            "sketch": {
                "dim": int(self.sketch_dim),
                "norm": float(np.linalg.norm(sk)),
                "baseline": node.metadata.get("baseline", {}),
            },
            "children": list(self.children.get(node_id, [])),
        }

    def learn_fusion_bias(self, node_id: str, reward: float, *, lr: float = 0.05) -> None:
        if node_id not in self.nodes:
            return
        old = float(self.fusion_learning.get(node_id, 0.0))
        self.fusion_learning[node_id] = _clamp(old + float(lr) * float(reward), -2.0, 2.0)
        self.nodes[node_id].fusion.learnable_bias = self.fusion_learning[node_id]
        self.bump_tick()

    # --------------------------
    # Sandbox clone
    # --------------------------

    def sandbox_clone(self, suffix: str = "sandbox") -> "TreeTensorNetwork":
        clone = TreeTensorNetwork(
            vector_dim=self.vector_dim,
            seed=self.seed,
            root_name=f"{self.nodes[self.root_id].name}::{suffix}",
            sketch_dim=self.sketch_dim,
            history_capacity=self.history_capacity,
            use_tn_projection=self.use_tn_projection,
            latent_geometry=self.latent_geometry,
        )
        clone.nodes = {}
        clone.children = {k: list(v) for k, v in self.children.items()}
        clone.parent = dict(self.parent)
        clone.histories = {}
        clone.baselines = {}
        clone.fusion_learning = dict(self.fusion_learning)
        for nid, node in self.nodes.items():
            fusion = FusionConfig(**{**asdict(FusionConfig()), **_json_safe(asdict(node.fusion))})
            try:
                fusion.mode = FusionMode(str(fusion.mode))
            except Exception:
                fusion.mode = FusionMode.ATTENTION
            clone.nodes[nid] = TensorNode(
                node_id=nid,
                name=node.name,
                kind=node.kind,
                vector_dim=self.vector_dim,
                state=node.state.copy(),
                fusion=fusion,
                metadata=copy.deepcopy(node.metadata),
                connections=list(node.connections),
                updated_at=node.updated_at,
                latent_geometry=self.latent_geometry,
            )
            clone.histories[nid] = RingSketch(SketchConfig(sketch_dim=self.sketch_dim, capacity=self.history_capacity, use_float16=True))
            clone.baselines[nid] = AdaptiveBaselineTracker(BaselineConfig())
        clone.root_id = self.root_id
        clone._tick = self._tick
        return clone

    # --------------------------
    # Federation snapshots
    # --------------------------

    def export_snapshot(self, *, peer_id: str, node_id: Optional[str] = None) -> TwinSnapshot:
        nid = node_id or self.root_id
        sk = self.sketch(nid)
        root = self.merkle_root(self.root_id)
        payload = {
            "peer_id": str(peer_id),
            "ts": _now_iso(),
            "root_hash": MerkleHasher.hex64(root),
            "node_id": str(nid),
            "sketch": sk.astype(np.float32).tolist(),
        }
        return TwinSnapshot(payload["peer_id"], payload["ts"], payload["root_hash"], payload["node_id"], payload["sketch"], _payload_hash(payload))

    def export_history_snapshot(self, *, peer_id: str, node_ids: Optional[Sequence[str]] = None, k: int = 32) -> TwinHistorySnapshot:
        ids = list(node_ids) if node_ids is not None else sorted(self.histories.keys())
        histories = {nid: self.histories[nid].get_recent(k).astype(float).tolist() for nid in ids if nid in self.histories}
        payload = {
            "peer_id": str(peer_id),
            "ts": _now_iso(),
            "root_hash": MerkleHasher.hex64(self.merkle_root()),
            "node_ids": sorted(histories.keys()),
            "histories": histories,
        }
        return TwinHistorySnapshot(payload["peer_id"], payload["ts"], payload["root_hash"], payload["node_ids"], payload["histories"], _payload_hash(payload))

    @staticmethod
    def verify_history_snapshot(snapshot: TwinHistorySnapshot, *, max_age_s: int = 300) -> Dict[str, Any]:
        payload = {
            "peer_id": snapshot.peer_id,
            "ts": snapshot.ts,
            "root_hash": snapshot.root_hash,
            "node_ids": sorted(list(snapshot.histories.keys())),
            "histories": snapshot.histories,
        }
        ok_hash = _payload_hash(payload) == snapshot.payload_hash
        ts_epoch = _parse_iso_epoch(snapshot.ts)
        age = float(time.time() - ts_epoch) if ts_epoch is not None else float("inf")
        ok_age = bool(age <= float(max_age_s))
        return {"ok_hash": bool(ok_hash), "ok_age": bool(ok_age), "age_s": float(age) if np.isfinite(age) else None, "peer_id": snapshot.peer_id, "root_hash": snapshot.root_hash, "node_count": len(snapshot.histories)}

    # --------------------------
    # Persistence
    # --------------------------

    def to_dict(self, *, include_histories: bool = True) -> Dict[str, Any]:
        return {
            "version": "digital_twin_kernel_v3_atomtn",
            "vector_dim": int(self.vector_dim),
            "seed": int(self.seed),
            "sketch_dim": int(self.sketch_dim),
            "history_capacity": int(self.history_capacity),
            "use_tn_projection": bool(self.use_tn_projection),
            "latent_geometry": self.latent_geometry,
            "root_id": self.root_id,
            "tick": int(self._tick),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "children": {k: list(v) for k, v in self.children.items()},
            "parent": dict(self.parent),
            "fusion_learning": dict(self.fusion_learning),
            "histories": {nid: h.to_dict() for nid, h in self.histories.items()} if include_histories else {},
            "baselines": {nid: b.to_dict() for nid, b in self.baselines.items()},
        }

    def save_json(self, path: Union[str, os.PathLike], *, include_histories: bool = True) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(self.to_dict(include_histories=include_histories)), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreeTensorNetwork":
        ttn = cls(
            vector_dim=int(payload.get("vector_dim", 128)),
            seed=int(payload.get("seed", 2027)),
            root_name="Digital Twin",
            sketch_dim=int(payload.get("sketch_dim", min(96, int(payload.get("vector_dim", 128))))),
            history_capacity=int(payload.get("history_capacity", 64)),
            use_tn_projection=bool(payload.get("use_tn_projection", True)),
            latent_geometry=str(payload.get("latent_geometry", LatentGeometry.EUCLIDEAN.value)),
        )
        ttn.nodes = {}
        ttn.children = {str(k): [str(x) for x in v] for k, v in dict(payload.get("children", {})).items()}
        ttn.parent = {str(k): str(v) for k, v in dict(payload.get("parent", {})).items()}
        ttn.histories = {}
        ttn.baselines = {}
        for nid, nd in dict(payload.get("nodes", {})).items():
            fusion_payload = dict(nd.get("fusion", {}))
            mode = fusion_payload.get("mode", FusionMode.ATTENTION.value)
            try:
                mode = FusionMode(str(mode))
            except Exception:
                mode = FusionMode.ATTENTION
            fusion = FusionConfig(
                mode=mode,
                attention_beta=float(fusion_payload.get("attention_beta", 2.0)),
                residual_mix=float(fusion_payload.get("residual_mix", 0.25)),
                tucker_rank=int(fusion_payload.get("tucker_rank", 16)),
                auto_tune_beta=bool(fusion_payload.get("auto_tune_beta", True)),
                learnable_bias=float(fusion_payload.get("learnable_bias", 0.0)),
                eps=float(fusion_payload.get("eps", 1e-9)),
            )
            try:
                kind = NodeKind(str(nd.get("kind", NodeKind.GENERIC.value)))
            except Exception:
                kind = NodeKind.GENERIC
            ttn.nodes[str(nid)] = TensorNode(
                node_id=str(nid),
                name=str(nd.get("name", nid)),
                kind=kind,
                vector_dim=int(payload.get("vector_dim", 128)),
                state=np.asarray(nd.get("state", []), dtype=np.float32),
                fusion=fusion,
                metadata=dict(nd.get("metadata", {})),
                connections=list(nd.get("connections", [])),
                updated_at=str(nd.get("updated_at", _now_iso())),
                latent_geometry=str(nd.get("latent_geometry", payload.get("latent_geometry", LatentGeometry.EUCLIDEAN.value))),
            )
        for nid in ttn.nodes:
            ttn.children.setdefault(nid, [])
            ttn.histories[nid] = RingSketch(SketchConfig(sketch_dim=ttn.sketch_dim, capacity=ttn.history_capacity, use_float16=True))
            ttn.baselines[nid] = AdaptiveBaselineTracker(BaselineConfig())
        for nid, hd in dict(payload.get("histories", {})).items():
            if nid in ttn.nodes:
                ttn.histories[nid] = RingSketch.from_dict(hd)
        for nid, bd in dict(payload.get("baselines", {})).items():
            if nid in ttn.nodes:
                ttn.baselines[nid] = AdaptiveBaselineTracker.from_dict(bd)
        ttn.root_id = str(payload.get("root_id", "0"))
        ttn._tick = int(payload.get("tick", 0))
        ttn.fusion_learning = {str(k): float(v) for k, v in dict(payload.get("fusion_learning", {})).items()}
        ttn.bump_tick()
        return ttn

    @classmethod
    def load_json(cls, path: Union[str, os.PathLike]) -> "TreeTensorNetwork":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# =============================================================================
# Governance
# =============================================================================


@dataclass
class Action:
    kind: str
    node_id: Optional[str] = None
    payload: Any = None
    note: str = ""


class PolicyEngine:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

    def simulate(self, network: TreeTensorNetwork, actions: List[Action], *, cortex: Optional[CortexLike] = None) -> SimulationReport:
        notes: List[str] = []
        if len(actions) > int(self.cfg.max_actions_per_execute):
            return SimulationReport(ok=False, predicted_alerts=999, max_delta=999.0, notes=["too_many_actions"])
        predicted_alerts = 0
        max_delta = 0.0
        trials = int(max(1, self.cfg.dreaming_trials))
        horizon = int(max(1, self.cfg.dreaming_horizon_steps))
        for t in range(trials):
            sandbox = network.sandbox_clone(suffix=f"sim{t}")
            before = sandbox.global_state().copy()
            cp = AkkuratInterface(sandbox, cortex=cortex, strict=False)
            for _ in range(horizon):
                cp.execute(actions)
                sandbox.push_global_histories()
            after = sandbox.global_state()
            delta = float(np.linalg.norm(after - before))
            max_delta = max(max_delta, delta)
            predicted_alerts = max(predicted_alerts, len(cp.alerts))
            if delta > float(self.cfg.max_latent_l2_delta):
                notes.append(f"delta_exceeds_budget(trial={t},delta={delta:.3f})")
                return SimulationReport(False, predicted_alerts, max_delta, notes)
        ok = bool(predicted_alerts <= int(self.cfg.dreaming_fail_threshold) and max_delta <= float(self.cfg.max_latent_l2_delta))
        if not ok:
            notes.append("simulation_failed")
        return SimulationReport(ok, int(predicted_alerts), float(max_delta), notes)

    def approve(self, report: SimulationReport) -> bool:
        if bool(report.ok):
            return True
        if self.cfg.strict:
            raise PolicyError(f"Policy reject: {report}")
        return False


class AkkuratInterface:
    def __init__(self, network: TreeTensorNetwork, *, cortex: Optional[CortexLike] = None, strict: bool = False):
        self.network = network
        self.network.control_plane = self
        self.cortex = cortex
        self.strict = bool(strict)
        self.audit_log: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.rejection_memory: List[Dict[str, Any]] = []
        self.policy = PolicyEngine(PolicyConfig(strict=self.strict))

    def log_event(self, message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        self.audit_log.append({"ts": _now_iso(), "msg": str(message), "meta": _json_safe(meta or {})})

    def raise_alert(self, message: str, *, severity: str = "warning", meta: Optional[Dict[str, Any]] = None) -> None:
        rec = {"ts": _now_iso(), "severity": str(severity), "msg": str(message), "meta": _json_safe(meta or {})}
        self.alerts.append(rec)
        self.log_event(f"ALERT({severity}): {message}", meta=meta)

    def observe(self, query: str) -> Dict[str, Any]:
        q = (query or "").lower().strip()
        if not q:
            return {"status": "empty_query"}
        candidates: List[str] = []
        terms = {
            "physical": "1.1",
            "virtual": "1.2",
            "data": "1.3",
            "pipeline": "1.3",
            "sensor": "2.3.2",
            "actuator": "2.3.3",
            "control": "2.3.3",
            "cognitive": "3.2.3.4",
            "atom": "3.2.3.5",
            "quantum": "3.2.3.5.1",
            "flow": "3.2.3.5.2",
            "observable": "3.2.3.5.3",
        }
        for term, nid in terms.items():
            if term in q:
                candidates.append(nid)
        if not candidates:
            m = re.search(r"\b(\d+(?:\.\d+)*)\b", q)
            if m and m.group(1) in self.network.nodes:
                candidates.append(m.group(1))
        out = {"query": query, "results": []}
        for nid in candidates[:8]:
            if nid in self.network.nodes:
                out["results"].append(self.network.measure(nid))
        out["status"] = "ok" if out["results"] else "no_match"
        self.log_event("observe()", meta={"query": query, "matches": candidates[:8]})
        return out

    def _rule_plan(self, intent: str) -> List[Action]:
        it = intent.lower()
        if any(k in it for k in ["stabilize", "resync", "synchronize", "reduce drift"]):
            return [Action("raise_alert", payload="Requested stabilization workflow.", note="operator_attention"), Action("attach_metadata", node_id="1.2", payload={"requested": "recalibration"}, note="virtual_space")]
        if "atom" in it or "quantum" in it:
            return [Action("attach_metadata", node_id="3.2.3.5", payload={"requested": intent}, note="atomtn")]
        if "sensor" in it or "telemetry" in it:
            return [Action("attach_metadata", node_id="2.3.2", payload={"requested": "sensor_health_check"}, note="sensors")]
        return [Action("raise_alert", payload=f"Unrecognized intent: {intent}", note="needs_mapping")]

    def plan(self, intent: str, *, allow_llm: bool = True) -> List[Action]:
        it = (intent or "").strip()
        if not it:
            return []
        fallback = self._rule_plan(it)
        if not allow_llm or self.cortex is None or not hasattr(self.cortex, "chat"):
            self.log_event("plan() using fallback", meta={"intent": it})
            return fallback
        system = (
            "You are an operations planner for a governed digital twin. Return ONLY JSON: "
            "{\"actions\":[{\"kind\":...,\"node_id\":...,\"payload\":...,\"note\":...}]}"
        )
        user = f"INTENT: {it}\nKNOWN_NODE_IDS: {', '.join(sorted(self.network.nodes.keys()))[:1800]}\nPlan safe actions."
        try:
            raw = self.cortex.chat(system=system, user=user, temperature=0.0, max_tokens=450)
            obj = self._extract_first_json(raw)
            actions = self._parse_actions(obj)
            if actions:
                self.log_event("plan() using LLM", meta={"intent": it, "actions": len(actions)})
                return actions
        except Exception as e:
            self.log_event("plan() LLM failed", meta={"intent": it, "error": repr(e)})
        return fallback

    @staticmethod
    def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
        t = (text or "").strip()
        if not t:
            return None
        if t.startswith("{") and t.endswith("}"):
            try:
                return json.loads(t)
            except Exception:
                pass
        start = t.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except Exception:
                        return None
        return None

    @staticmethod
    def _parse_actions(obj: Optional[Mapping[str, Any]]) -> List[Action]:
        if not isinstance(obj, Mapping):
            return []
        arr = obj.get("actions", [])
        if not isinstance(arr, list):
            return []
        out = []
        for a in arr[:25]:
            if isinstance(a, Mapping):
                out.append(Action(kind=str(a.get("kind", "")).strip(), node_id=(str(a.get("node_id")).strip() if a.get("node_id") is not None else None), payload=a.get("payload", None), note=str(a.get("note", "")).strip()))
        return [x for x in out if x.kind]

    def execute(self, actions: List[Action]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"applied": 0, "skipped": 0, "errors": 0, "error_details": []}
        for act in actions[: int(self.policy.cfg.max_actions_per_execute)]:
            try:
                if act.kind == "raise_alert":
                    self.raise_alert(str(act.payload), severity="warning", meta={"note": act.note})
                elif act.kind == "attach_metadata":
                    if act.node_id and act.node_id in self.network.nodes:
                        payload = act.payload if isinstance(act.payload, Mapping) else {"payload": act.payload}
                        self.network.nodes[act.node_id].metadata.update(dict(payload))
                        self.network.bump_tick()
                        self.log_event(f"metadata attached to {act.node_id}", meta={"note": act.note})
                    else:
                        stats["skipped"] += 1
                        continue
                elif act.kind == "update_node_data":
                    if not act.node_id:
                        stats["skipped"] += 1
                        continue
                    self.network.update_node_data(act.node_id, act.payload, cortex=self.cortex, note=act.note)
                elif act.kind == "update_node_latent":
                    if not act.node_id:
                        stats["skipped"] += 1
                        continue
                    self.network.update_node_latent(act.node_id, act.payload, note=act.note)
                else:
                    if self.strict:
                        raise TensorNetworkError(f"Unknown action kind: {act.kind}")
                    stats["skipped"] += 1
                    continue
                stats["applied"] += 1
            except Exception as e:
                stats["errors"] += 1
                detail = {"kind": act.kind, "node_id": act.node_id, "error": repr(e)}
                stats["error_details"].append(detail)
                self.log_event("execute() action error", meta=detail)
                if self.strict:
                    raise
        self.log_event("execute()", meta={k: v for k, v in stats.items() if k != "error_details"})
        return stats

    def governed_actions(self, actions: List[Action], *, intent: str = "direct_actions") -> Dict[str, Any]:
        allowed = set(self.policy.cfg.allowed_action_kinds)
        filtered = [a for a in actions if a.kind in allowed]
        if len(filtered) != len(actions):
            self.log_event("policy filter removed actions", meta={"removed": len(actions) - len(filtered)})
        report = self.policy.simulate(self.network, filtered, cortex=self.cortex)
        try:
            approved = bool(self.policy.approve(report))
        except PolicyError as e:
            approved = False
            self.rejection_memory.append({"ts": _now_iso(), "intent": intent, "reason": str(e), "actions": _json_safe(filtered)})
            self.raise_alert("Policy rejected plan.", severity="warning", meta={"intent": intent, "report": _json_safe(report)})
            if self.strict:
                raise
        exec_stats: Dict[str, Any] = {"applied": 0, "skipped": 0, "errors": 0, "error_details": []}
        if approved:
            exec_stats = self.execute(filtered)
        else:
            self.rejection_memory.append({"ts": _now_iso(), "intent": intent, "reason": "simulation_failed", "actions": _json_safe(filtered)})
            self.log_event("governed_actions rejected", meta={"intent": intent, "report": _json_safe(report)})
        return {
            "intent": intent,
            "actions": _json_safe(filtered),
            "simulation": _json_safe(report),
            "approved": bool(approved),
            "execute": exec_stats,
            "executed": exec_stats,
            "submitted": len(filtered),
        }

    def governed_execute(self, intent: str, *, allow_llm: bool = False) -> Dict[str, Any]:
        return self.governed_actions(self.plan(intent, allow_llm=allow_llm), intent=intent)


# =============================================================================
# Builder taxonomy
# =============================================================================


class DigitalTwinsBuilder:
    @staticmethod
    def build_platform(
        *,
        vector_dim: int,
        seed: int = 2027,
        sketch_dim: Optional[int] = None,
        history_capacity: int = 64,
        use_tn_projection: bool = True,
        latent_geometry: str = LatentGeometry.EUCLIDEAN.value,
    ) -> TreeTensorNetwork:
        ttn = TreeTensorNetwork(vector_dim=vector_dim, seed=seed, root_name="Digital Twin", sketch_dim=sketch_dim, history_capacity=history_capacity, use_tn_projection=use_tn_projection, latent_geometry=latent_geometry)

        # Core elements
        ttn.add_node("1.1", "Physical space (Physical Twin)", parent_id="0", kind=NodeKind.PHYSICAL)
        ttn.add_node("1.2", "Virtual space (Virtual Twin)", parent_id="0", kind=NodeKind.VIRTUAL)
        ttn.add_node("1.3", "Data processing (Twin Data)", parent_id="0", kind=NodeKind.DATA)

        # Physical space subtree
        ttn.add_node("2.1", "Study objects / entities", parent_id="1.1", kind=NodeKind.ENTITY)
        ttn.add_node("2.1.1", "Products / services", parent_id="2.1", kind=NodeKind.ENTITY)
        ttn.add_node("2.1.2", "Machine / devices", parent_id="2.1", kind=NodeKind.ENTITY)
        ttn.add_node("2.1.3", "Infrastructure", parent_id="2.1", kind=NodeKind.ENTITY)
        ttn.add_node("2.1.4", "Operator / people", parent_id="2.1", kind=NodeKind.ENTITY)
        ttn.add_node("2.2", "Physical properties", parent_id="1.1", kind=NodeKind.PROPERTY)
        ttn.add_node("2.3", "Resources", parent_id="1.1", kind=NodeKind.RESOURCE)
        ttn.add_node("2.3.1", "Workshop / lab / shop floor", parent_id="2.3", kind=NodeKind.RESOURCE)
        ttn.add_node("2.3.2", "Sensors", parent_id="2.3", kind=NodeKind.SENSOR)
        ttn.add_node("2.3.3", "Controllers / Actuators", parent_id="2.3", kind=NodeKind.ACTUATOR)

        # Virtual space subtree
        ttn.add_node("3.1", "Physics-based models", parent_id="1.2", kind=NodeKind.MODEL)
        ttn.add_node("3.1.1", "Geometric model", parent_id="3.1", kind=NodeKind.MODEL)
        ttn.add_node("3.1.2", "Analytical / empirical models", parent_id="3.1", kind=NodeKind.MODEL)
        ttn.add_node("3.1.3", "0D-3D computational models", parent_id="3.1", kind=NodeKind.MODEL)
        ttn.add_node("3.1.4", "Statistical models", parent_id="3.1", kind=NodeKind.MODEL)
        ttn.add_node("3.1.5", "System dynamics models", parent_id="3.1", kind=NodeKind.MODEL)
        ttn.add_node("3.2", "Data-driven models", parent_id="1.2", kind=NodeKind.MODEL)
        ttn.add_node("3.2.1", "Behavior model", parent_id="3.2", kind=NodeKind.MODEL)
        ttn.add_node("3.2.2", "Rule Model", parent_id="3.2", kind=NodeKind.MODEL)
        ttn.add_node("3.2.3", "AI (ML / SciML)", parent_id="3.2", kind=NodeKind.MODEL)
        ttn.add_node("3.2.3.1", "Unsupervised Learning", parent_id="3.2.3", kind=NodeKind.MODEL)
        ttn.add_node("3.2.3.2", "Supervised Learning", parent_id="3.2.3", kind=NodeKind.MODEL)
        ttn.add_node("3.2.3.3", "Reinforcement Learning", parent_id="3.2.3", kind=NodeKind.MODEL)

        # Akkurat cognitive lobe runtime branch
        ttn.add_node("3.2.3.4", "Cognitive lobe runtime", parent_id="3.2.3", kind=NodeKind.COGNITIVE)
        ttn.add_node("3.2.3.4.1", "Sensory lobe pair", parent_id="3.2.3.4", kind=NodeKind.COGNITIVE)
        ttn.add_node("3.2.3.4.2", "Memory lobe pair", parent_id="3.2.3.4", kind=NodeKind.COGNITIVE)
        ttn.add_node("3.2.3.4.3", "Semantic lobe pair", parent_id="3.2.3.4", kind=NodeKind.COGNITIVE)
        ttn.add_node("3.2.3.4.4", "Planning lobe pair", parent_id="3.2.3.4", kind=NodeKind.COGNITIVE)
        ttn.add_node("3.2.3.4.5", "Regulation lobe pair", parent_id="3.2.3.4", kind=NodeKind.COGNITIVE)

        # AtomTN physics-reservoir branch
        ttn.add_node("3.2.3.5", "AtomTN Physics Reservoir", parent_id="3.2.3", kind=NodeKind.ATOMTN)
        ttn.add_node("3.2.3.5.1", "AtomTN Quantum State", parent_id="3.2.3.5", kind=NodeKind.QUANTUM)
        ttn.add_node("3.2.3.5.2", "AtomTN Flow Field", parent_id="3.2.3.5", kind=NodeKind.FLOW)
        ttn.add_node("3.2.3.5.3", "AtomTN Holographic Observables", parent_id="3.2.3.5", kind=NodeKind.OBSERVABLE)
        ttn.add_node("3.2.3.5.4", "AtomTN Hamiltonian Diagnostics", parent_id="3.2.3.5", kind=NodeKind.MODEL)
        ttn.add_node("3.2.3.5.5", "AtomTN Physics Governance Simulator", parent_id="3.2.3.5", kind=NodeKind.GOVERNANCE)

        # Data processing subtree
        ttn.add_node("4.1", "Data processing", parent_id="1.3", kind=NodeKind.PIPELINE)
        ttn.add_node("4.1.1", "Data acquisition", parent_id="4.1", kind=NodeKind.PIPELINE)
        ttn.add_node("4.1.2", "Data preprocessing", parent_id="4.1", kind=NodeKind.PIPELINE)
        ttn.add_node("4.1.3", "Analysis and mining", parent_id="4.1", kind=NodeKind.PIPELINE)
        ttn.add_node("4.1.4", "Data fusion", parent_id="4.1", kind=NodeKind.PIPELINE)
        ttn.add_node("4.2", "Data mapping", parent_id="1.3", kind=NodeKind.PIPELINE)
        ttn.add_node("4.2.1", "Time-sequence analysis", parent_id="4.2", kind=NodeKind.PIPELINE)
        ttn.add_node("4.2.2", "Data correlation", parent_id="4.2", kind=NodeKind.PIPELINE)
        ttn.add_node("4.2.3", "Data synchronization", parent_id="4.2", kind=NodeKind.PIPELINE)
        ttn.add_node("4.2.4", "Data visualization & integration in VR/AR", parent_id="4.2", kind=NodeKind.VISUAL)
        ttn.add_node("4.3", "Data storage", parent_id="1.3", kind=NodeKind.STORAGE)
        ttn.add_node("4.3.1", "Physical space data", parent_id="4.3", kind=NodeKind.STORAGE)
        ttn.add_node("4.3.2", "Virtual space data", parent_id="4.3", kind=NodeKind.STORAGE)

        # Fusion tuning
        ttn.set_fusion("0", FusionConfig(mode=FusionMode.ATTENTION, attention_beta=2.0, residual_mix=0.20, auto_tune_beta=True))
        ttn.set_fusion("1.1", FusionConfig(mode=FusionMode.ATTENTION, attention_beta=2.2, residual_mix=0.25, auto_tune_beta=True))
        ttn.set_fusion("1.2", FusionConfig(mode=FusionMode.ATTENTION, attention_beta=2.2, residual_mix=0.25, auto_tune_beta=True))
        ttn.set_fusion("1.3", FusionConfig(mode=FusionMode.ATTENTION, attention_beta=2.0, residual_mix=0.25, auto_tune_beta=True))
        ttn.set_fusion("3.2.3.4", FusionConfig(mode=FusionMode.ATTENTION, attention_beta=2.4, residual_mix=0.15, auto_tune_beta=True))
        ttn.set_fusion("3.2.3.5", FusionConfig(mode=FusionMode.ATTENTION, attention_beta=2.4, residual_mix=0.15, auto_tune_beta=True))
        ttn.set_fusion("4.1.4", FusionConfig(mode=(FusionMode.TN_TUCKER if _TN_OK else FusionMode.TUCKER), tucker_rank=16, residual_mix=0.15, auto_tune_beta=False))

        ttn.push_global_histories()
        return ttn


# =============================================================================
# Demo
# =============================================================================


def run_automation_demo(*, vector_dim: int = 256, seed: int = 2027, cortex: Optional[CortexLike] = None, latent_geometry: str = LatentGeometry.EUCLIDEAN.value) -> None:
    print("=" * 65)
    print(" AKKURAT GOVERNED DIGITAL TWIN KERNEL")
    print("=" * 65)
    print(f"[SYSTEM] Project root: {_AKKURAT_ROOT}")
    print(f"[SYSTEM] tn.py available: {_TN_OK}" + ("" if _TN_OK else f" ({_TN_ERR})"))
    ttn = DigitalTwinsBuilder.build_platform(vector_dim=vector_dim, seed=seed, sketch_dim=min(96, vector_dim), history_capacity=64, latent_geometry=latent_geometry)
    cp = AkkuratInterface(ttn, cortex=cortex, strict=False)
    print(f"[SYSTEM] Nodes: {len(ttn.nodes)} | vector_dim={vector_dim} | sketch_dim={ttn.sketch_dim} | geometry={ttn.latent_geometry}")
    print(f"[SYSTEM] Initial Merkle root: {MerkleHasher.hex64(ttn.merkle_root())}")

    sensor_readings = np.array([22.5, 101.3, 2400.0, 55.2], dtype=np.float32)
    ttn.update_node_data("2.3.2", sensor_readings, cortex=cortex, note="telemetry")
    ttn.update_node_data("4.1.1", sensor_readings, cortex=cortex, note="acquired")
    ttn.update_node_data("3.1.1", np.array([0.12, 0.33, 0.07, 0.91], dtype=np.float32), cortex=cortex, note="geom_state")
    ttn.update_node_data("4.1.4", {"sensor": sensor_readings.tolist(), "ts": _now_iso(), "note": "bounded fusion artifact"}, cortex=cortex, note="fusion")
    ttn.update_node_data("3.2.3.5.1", {"quantum_norm_squared": 1.0, "stable": True}, cortex=cortex, note="AtomTN placeholder")
    ttn.update_node_data("3.2.3.5.3", {"Z_mean": 0.0, "Z_std": 0.25}, cortex=cortex, note="AtomTN observables placeholder")

    corr = ttn.correlation("1.1", "1.2")
    cp.log_event("coherence_check", meta={"corr": corr})
    print(f"[SYSTEM] Physical-Virtual coherence: {corr:.4f}")
    suggestions = ttn.causal.suggest(ttn.histories)
    print(f"[SYSTEM] Causal suggestions: {len(suggestions)}")
    gov = cp.governed_execute("stabilize and resync the virtual model with physical data", allow_llm=False)
    print(json.dumps(_json_safe(gov), indent=2))
    snap = ttn.export_snapshot(peer_id="peer_demo", node_id="0")
    contract = TwinContract("peer_demo", policy=ContractPolicy())
    print(json.dumps(_json_safe(contract.ingest(ttn.sketch("0"), snap)), indent=2))
    print(f"[SYSTEM] Final Merkle root: {MerkleHasher.hex64(ttn.merkle_root())}")
    print("=" * 65)


if __name__ == "__main__":
    run_automation_demo(vector_dim=256, seed=2027, cortex=None)
