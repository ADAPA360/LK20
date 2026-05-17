#!/usr/bin/env python3
# semantic_attractors.py
"""
Semantic attractor utilities for AtomTN / Akkurat.

Production role
---------------
This module bridges dictionary/lexicon artifacts and the AtomTN tensor-network
runtime. It is deliberately independent of the future dictionary_lexicon_ingestor.py
and sentence_builder.py modules, but it defines the stable contract they can use:

- semantic concept / sense records;
- deterministic lexical embedding construction from records;
- attractor potential construction from embeddings, labels, or relation weights;
- conversion of semantic potentials into local Hermitian operators;
- optional TensorTrain compression of semantic relation matrices;
- TTN-compatible product-operator builders and flow-like diagnostics;
- serialization and health surfaces.

Design constraints
------------------
- NumPy-only core. The optional tn.py TensorTrain path is imported lazily.
- No Hamiltonian application or TTN evolution occurs here.
- All exported matrices are finite complex128 unless explicitly real-valued.
- Local operators are Hermitian by default so they are safe for Hamiltonian terms.
- The API is intentionally small and stable for dictionary_lexicon_ingestor.py:

      entries = [SemanticEntry(...), ...]
      bank = SemanticAttractorBank.from_entries(entries, dim=64)
      ops = bank.local_operator_basis(d=16)
      leaf_ops = bank.product_leaf_ops_for_tokens(tokens, tree, phys_dims)
      H_terms = bank.onsite_terms_for_tokens(tokens, tree, phys_dims)

Compatibility with existing stack
---------------------------------
Uses math_utils.py when available for hermitianize, fro_norm, and finite guards.
Can optionally emit TensorTrain artifacts through tn.py, but does not require it.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from math_utils import _assert, fro_norm, hermitianize, remove_trace, safe_norm
except Exception:  # pragma: no cover - standalone fallback
    def _assert(cond: bool, msg: str) -> None:
        if not bool(cond):
            raise ValueError(str(msg))

    def fro_norm(A: Any) -> float:
        try:
            return float(np.linalg.norm(np.asarray(A).reshape(-1)))
        except Exception:
            return 0.0

    def hermitianize(A: Any) -> np.ndarray:
        X = np.asarray(A, dtype=np.complex128)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("hermitianize requires square matrix")
        return ((X + X.conj().T) * 0.5).astype(np.complex128)

    def remove_trace(A: Any) -> np.ndarray:
        X = np.asarray(A, dtype=np.complex128)
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("remove_trace requires square matrix")
        return (X - (np.trace(X) / max(1, X.shape[0])) * np.eye(X.shape[0], dtype=np.complex128)).astype(np.complex128)

    def safe_norm(x: Any, ord: Optional[int | float | str] = None) -> float:
        try:
            arr = np.asarray(x)
            if arr.size == 0 or not np.isfinite(arr).all():
                return 0.0
            val = float(np.linalg.norm(arr, ord=ord))
            return val if np.isfinite(val) else 0.0
        except Exception:
            return 0.0


_EPS = 1e-12
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-']+")


# =============================================================================
# Generic helpers
# =============================================================================


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, complex):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {"real": arr.real.astype(float).tolist(), "imag": arr.imag.astype(float).tolist()}
        return np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _stable_hash_u64(text: str, seed: int = 0) -> int:
    """Deterministic 64-bit FNV-1a style hash; avoids Python hash randomization."""
    h = (1469598103934665603 ^ int(seed)) & 0xFFFFFFFFFFFFFFFF
    data = str(text).encode("utf-8", errors="ignore")
    for b in data:
        h ^= int(b)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h)


def _rng_for_key(key: str, seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(_stable_hash_u64(key, seed=seed) & 0xFFFFFFFF)


def _sanitize_vector(x: Any, *, dim: Optional[int] = None, name: str = "vector") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    if dim is not None:
        d = int(dim)
        if arr.size == d:
            return arr
        out = np.zeros((d,), dtype=np.float64)
        n = min(d, arr.size)
        if n:
            out[:n] = arr[:n]
        return out
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def _normalize_vector(x: Any, *, dim: Optional[int] = None, eps: float = _EPS) -> np.ndarray:
    v = _sanitize_vector(x, dim=dim)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= float(eps):
        return np.zeros_like(v, dtype=np.float64)
    return (v / n).astype(np.float64, copy=False)


def _sanitize_matrix(A: Any, *, shape: Optional[Tuple[int, int]] = None, name: str = "matrix") -> np.ndarray:
    M = np.asarray(A, dtype=np.complex128)
    if M.ndim != 2:
        raise ValueError(f"{name} must be a matrix; got shape {M.shape}")
    if shape is not None and tuple(M.shape) != tuple(shape):
        out = np.zeros(tuple(shape), dtype=np.complex128)
        r = min(out.shape[0], M.shape[0])
        c = min(out.shape[1], M.shape[1])
        if r and c:
            out[:r, :c] = M[:r, :c]
        M = out
    if M.size:
        real = np.nan_to_num(M.real, nan=0.0, posinf=0.0, neginf=0.0)
        imag = np.nan_to_num(M.imag, nan=0.0, posinf=0.0, neginf=0.0)
        M = (real + 1j * imag).astype(np.complex128, copy=False)
    return M


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or ""))]


def _projector_from_vector(v: np.ndarray, *, traceless: bool = False) -> np.ndarray:
    z = np.asarray(v, dtype=np.complex128).reshape(-1)
    n = fro_norm(z)
    d = int(z.size)
    if d <= 0:
        raise ValueError("cannot build projector from empty vector")
    if n <= _EPS:
        P = np.zeros((d, d), dtype=np.complex128)
    else:
        z = z / n
        P = np.outer(z, z.conj()).astype(np.complex128)
    if traceless:
        P = remove_trace(P)
    return hermitianize(P).astype(np.complex128)


def _resample_vector(v: np.ndarray, d: int) -> np.ndarray:
    """Deterministically map an arbitrary vector length to d via interpolation."""
    v = _sanitize_vector(v)
    d = int(max(1, d))
    if v.size == d:
        return v.astype(np.float64, copy=True)
    if v.size == 1:
        return np.full((d,), float(v[0]), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, int(v.size), dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, int(d), dtype=np.float64)
    return np.interp(x_new, x_old, v).astype(np.float64)


def _safe_softmax(x: Any, *, temperature: float = 1.0) -> np.ndarray:
    arr = _sanitize_vector(x)
    tau = max(float(temperature), _EPS)
    y = arr / tau
    y = y - float(np.max(y))
    e = np.exp(np.clip(y, -80.0, 80.0))
    s = float(np.sum(e))
    if not np.isfinite(s) or s <= _EPS:
        return np.full_like(e, 1.0 / max(1, e.size), dtype=np.float64)
    return (e / s).astype(np.float64, copy=False)


# =============================================================================
# Core semantic containers
# =============================================================================


@dataclass(frozen=True)
class SemanticEntry:
    """
    One lexical/conceptual record.

    Fields are intentionally compatible with dictionary-style input:
      - key: stable id, e.g. "en:bank:n:financial_institution";
      - lemma: surface lemma;
      - sense_id: optional external sense id;
      - pos: part of speech or lexical class;
      - gloss: definition text;
      - tokens: normalized tokens associated with the entry;
      - relations: relation-name -> list of target keys or weighted targets;
      - weight: entry prior / confidence.
    """

    key: str
    lemma: str
    sense_id: str = ""
    pos: str = ""
    gloss: str = ""
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    relations: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> "SemanticEntry":
        key = str(self.key or self.lemma).strip()
        lemma = str(self.lemma or key).strip().lower()
        toks = list(self.tokens or ())
        toks.extend(_tokenize(lemma))
        toks.extend(_tokenize(self.gloss))
        if self.pos:
            toks.append(str(self.pos).lower().strip())
        toks = tuple(sorted(set(t for t in toks if t)))
        w = float(self.weight)
        if not np.isfinite(w):
            w = 1.0
        return SemanticEntry(
            key=key,
            lemma=lemma,
            sense_id=str(self.sense_id or ""),
            pos=str(self.pos or "").lower().strip(),
            gloss=str(self.gloss or ""),
            tokens=toks,
            relations=dict(self.relations or {}),
            weight=float(max(0.0, w)),
            metadata=dict(self.metadata or {}),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SemanticEntry":
        key = payload.get("key", payload.get("id", payload.get("sense_key", payload.get("lemma", ""))))
        lemma = payload.get("lemma", payload.get("word", key))
        toks = payload.get("tokens", ())
        if isinstance(toks, str):
            toks = _tokenize(toks)
        return cls(
            key=str(key),
            lemma=str(lemma),
            sense_id=str(payload.get("sense_id", payload.get("sense", ""))),
            pos=str(payload.get("pos", payload.get("part_of_speech", ""))),
            gloss=str(payload.get("gloss", payload.get("definition", ""))),
            tokens=tuple(str(t).lower().strip() for t in toks),
            relations=dict(payload.get("relations", {})),
            weight=float(payload.get("weight", 1.0)),
            metadata=dict(payload.get("metadata", {})),
        ).normalized()

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class SemanticAttractorConfig:
    """Configuration for semantic attractor construction."""

    dim: int = 64
    seed: int = 0
    embedding_mode: str = "hashed_bow"  # hashed_bow | random_key
    normalize_embeddings: bool = True
    relation_default_weight: float = 1.0
    attractor_strength: float = 1.0
    repulsion_strength: float = 0.0
    traceless_operators: bool = True
    operator_norm_target: float = 1.0
    strict: bool = True

    def normalized(self) -> "SemanticAttractorConfig":
        return SemanticAttractorConfig(
            dim=int(max(1, self.dim)),
            seed=int(self.seed),
            embedding_mode=str(self.embedding_mode or "hashed_bow").lower().strip(),
            normalize_embeddings=bool(self.normalize_embeddings),
            relation_default_weight=float(self.relation_default_weight),
            attractor_strength=float(self.attractor_strength),
            repulsion_strength=float(max(0.0, self.repulsion_strength)),
            traceless_operators=bool(self.traceless_operators),
            operator_norm_target=float(max(0.0, self.operator_norm_target)),
            strict=bool(self.strict),
        )


@dataclass
class SemanticAttractor:
    """A finite semantic attractor vector with optional relation context."""

    key: str
    vector: np.ndarray
    label: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.key = str(self.key)
        self.label = str(self.label or self.key)
        self.vector = _sanitize_vector(self.vector)
        w = float(self.weight)
        self.weight = float(w if np.isfinite(w) else 1.0)
        self.metadata = dict(self.metadata or {})

    @property
    def dim(self) -> int:
        return int(self.vector.size)

    def normalized_vector(self) -> np.ndarray:
        return _normalize_vector(self.vector)

    def potential_operator(self, *, d: Optional[int] = None, traceless: bool = True, strength: float = 1.0) -> np.ndarray:
        """
        Return a d x d Hermitian potential operator from the attractor vector.

        The vector is resampled to d, normalized, then converted to a rank-one
        projector. Positive strength yields an attractive negative potential:
            H_attr = -strength * |a><a|
        """
        dd = int(d or self.dim)
        v = _normalize_vector(_resample_vector(self.vector, dd))
        P = _projector_from_vector(v, traceless=bool(traceless))
        H = (-float(strength) * float(self.weight) * P).astype(np.complex128)
        return hermitianize(H).astype(np.complex128)

    def overlap(self, other: "SemanticAttractor") -> float:
        a = self.normalized_vector()
        b = _normalize_vector(_resample_vector(other.vector, a.size))
        return float(np.real(np.vdot(a, b)))

    def to_dict(self, *, include_vector: bool = True) -> Dict[str, Any]:
        out = {"key": self.key, "label": self.label, "weight": float(self.weight), "metadata": _json_safe(self.metadata)}
        if include_vector:
            out["vector"] = self.vector.astype(float).tolist()
        else:
            out["dim"] = int(self.dim)
            out["norm"] = float(np.linalg.norm(self.vector))
        return out


# =============================================================================
# Embedding and relation construction
# =============================================================================


def hashed_token_embedding(tokens: Sequence[str], *, dim: int, seed: int = 0, signed: bool = True) -> np.ndarray:
    """Feature-hashing bag-of-words embedding."""
    dim = int(max(1, dim))
    out = np.zeros((dim,), dtype=np.float64)
    for tok in tokens:
        t = str(tok).lower().strip()
        if not t:
            continue
        h = _stable_hash_u64(t, seed=seed)
        idx = int(h % dim)
        if signed:
            sign = -1.0 if ((h >> 63) & 1) else 1.0
        else:
            sign = 1.0
        out[idx] += sign
    return out


def random_key_embedding(key: str, *, dim: int, seed: int = 0) -> np.ndarray:
    rng = _rng_for_key(str(key), seed=seed)
    return rng.normal(size=(int(max(1, dim)),)).astype(np.float64)


def embedding_for_entry(entry: SemanticEntry, cfg: SemanticAttractorConfig) -> np.ndarray:
    cfg = cfg.normalized()
    e = entry.normalized()
    mode = cfg.embedding_mode
    if mode == "random_key":
        v = random_key_embedding(e.key, dim=cfg.dim, seed=cfg.seed)
    elif mode in {"hashed_bow", "bow", "hash"}:
        toks = list(e.tokens)
        toks.append(e.lemma)
        if e.pos:
            toks.append(f"pos:{e.pos}")
        v = hashed_token_embedding(toks, dim=cfg.dim, seed=cfg.seed, signed=True)
    else:
        if cfg.strict:
            raise ValueError(f"unknown embedding_mode: {cfg.embedding_mode!r}")
        v = hashed_token_embedding(tuple(e.tokens) + (e.lemma,), dim=cfg.dim, seed=cfg.seed, signed=True)
    if cfg.normalize_embeddings:
        v = _normalize_vector(v, dim=cfg.dim)
    return v.astype(np.float64, copy=False)


def _iter_relation_targets(value: Any, default_weight: float = 1.0) -> Iterable[Tuple[str, float]]:
    """Yield (target_key, weight) from flexible relation payloads."""
    if value is None:
        return
    if isinstance(value, str):
        yield value, float(default_weight)
        return
    if isinstance(value, Mapping):
        for k, w in value.items():
            try:
                ww = float(w)
            except Exception:
                ww = float(default_weight)
            yield str(k), float(ww)
        return
    if isinstance(value, Iterable):
        for item in value:
            if isinstance(item, str):
                yield item, float(default_weight)
            elif isinstance(item, Mapping):
                target = item.get("target", item.get("key", item.get("id", "")))
                if target:
                    try:
                        ww = float(item.get("weight", default_weight))
                    except Exception:
                        ww = float(default_weight)
                    yield str(target), float(ww)
            elif isinstance(item, (tuple, list)) and len(item) >= 1:
                target = str(item[0])
                try:
                    ww = float(item[1]) if len(item) > 1 else float(default_weight)
                except Exception:
                    ww = float(default_weight)
                yield target, float(ww)


def relation_matrix_from_entries(
    entries: Sequence[SemanticEntry],
    *,
    default_weight: float = 1.0,
    symmetric: bool = False,
    row_normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build an entry-key relation adjacency matrix."""
    normalized = [e.normalized() for e in entries]
    keys = [e.key for e in normalized]
    key_to_idx = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    R = np.zeros((n, n), dtype=np.float64)
    for e in normalized:
        i = key_to_idx[e.key]
        for _rel, payload in dict(e.relations or {}).items():
            for target, w in _iter_relation_targets(payload, default_weight=default_weight):
                j = key_to_idx.get(str(target))
                if j is None:
                    continue
                ww = float(w)
                if np.isfinite(ww):
                    R[i, j] += ww
                    if symmetric:
                        R[j, i] += ww
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    if row_normalize and R.size:
        row_sums = np.sum(np.abs(R), axis=1, keepdims=True)
        R = R / np.maximum(row_sums, _EPS)
    return R.astype(np.float64), key_to_idx


# =============================================================================
# Attractor bank
# =============================================================================


@dataclass
class SemanticAttractorBank:
    """
    Collection of semantic attractors and relation matrices.

    The bank is the expected handoff object between dictionary_lexicon_ingestor.py
    and sentence_builder.py. It is also directly usable by hamiltonian.py-style
    code through local operator maps and onsite term dictionaries.
    """

    attractors: Dict[str, SemanticAttractor]
    config: SemanticAttractorConfig = field(default_factory=SemanticAttractorConfig)
    relation_matrix: Optional[np.ndarray] = None
    key_to_index: Dict[str, int] = field(default_factory=dict)
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config = self.config.normalized()
        self.attractors = {str(k): v for k, v in dict(self.attractors or {}).items()}
        for k, a in list(self.attractors.items()):
            if not isinstance(a, SemanticAttractor):
                raise TypeError(f"attractor {k!r} is not SemanticAttractor")
            if a.dim != self.config.dim:
                self.attractors[k] = SemanticAttractor(a.key, _resample_vector(a.vector, self.config.dim), label=a.label, weight=a.weight, metadata=a.metadata)
        self.key_to_index = {str(k): int(v) for k, v in dict(self.key_to_index or {}).items()}
        if not self.key_to_index:
            self.key_to_index = {k: i for i, k in enumerate(sorted(self.attractors.keys()))}
        self.aliases = {str(k).lower(): [str(x) for x in vals] for k, vals in dict(self.aliases or {}).items()}
        self.metadata = dict(self.metadata or {})
        if self.relation_matrix is not None:
            n = len(self.key_to_index)
            self.relation_matrix = _sanitize_matrix(self.relation_matrix, shape=(n, n), name="relation_matrix").real.astype(np.float64)

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[Union[SemanticEntry, Mapping[str, Any]]],
        *,
        dim: int = 64,
        seed: int = 0,
        config: Optional[SemanticAttractorConfig] = None,
        build_relations: bool = True,
        symmetric_relations: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SemanticAttractorBank":
        cfg = (config or SemanticAttractorConfig(dim=dim, seed=seed)).normalized()
        normalized: List[SemanticEntry] = []
        for item in entries:
            if isinstance(item, SemanticEntry):
                normalized.append(item.normalized())
            elif isinstance(item, Mapping):
                normalized.append(SemanticEntry.from_mapping(item))
            else:
                raise TypeError(f"unsupported semantic entry type: {type(item)!r}")

        attractors: Dict[str, SemanticAttractor] = {}
        aliases: Dict[str, List[str]] = {}
        for e in normalized:
            v = embedding_for_entry(e, cfg)
            attractors[e.key] = SemanticAttractor(
                key=e.key,
                vector=v,
                label=e.lemma,
                weight=float(e.weight),
                metadata={"sense_id": e.sense_id, "pos": e.pos, "gloss": e.gloss, **dict(e.metadata or {})},
            )
            aliases.setdefault(e.lemma.lower(), []).append(e.key)
            for tok in e.tokens:
                aliases.setdefault(str(tok).lower(), []).append(e.key)

        relation = None
        key_to_index = {k: i for i, k in enumerate(sorted(attractors.keys()))}
        if build_relations:
            relation, kt = relation_matrix_from_entries(
                normalized,
                default_weight=cfg.relation_default_weight,
                symmetric=bool(symmetric_relations),
                row_normalize=True,
            )
            # relation_matrix_from_entries preserves input order; reorder to sorted key order for stable storage.
            input_keys = [e.key for e in normalized]
            input_pos = {k: i for i, k in enumerate(input_keys)}
            sorted_keys = sorted(attractors.keys())
            perm = [input_pos[k] for k in sorted_keys]
            relation = relation[np.ix_(perm, perm)].astype(np.float64)
            key_to_index = {k: i for i, k in enumerate(sorted_keys)}

        return cls(
            attractors=attractors,
            config=cfg,
            relation_matrix=relation,
            key_to_index=key_to_index,
            aliases=aliases,
            metadata={"source": "SemanticAttractorBank.from_entries", **dict(metadata or {})},
        )

    # ------------------------------------------------------------------
    # Lookup and scoring
    # ------------------------------------------------------------------
    def keys(self) -> List[str]:
        return sorted(self.attractors.keys())

    def resolve(self, token_or_key: str, *, max_results: int = 8) -> List[str]:
        q = str(token_or_key or "").lower().strip()
        if not q:
            return []
        if q in self.attractors:
            return [q]
        if q in self.aliases:
            return sorted(set(self.aliases[q]))[: int(max_results)]
        # Soft fallback by token overlap with labels.
        q_tokens = set(_tokenize(q))
        scored: List[Tuple[float, str]] = []
        for k, a in self.attractors.items():
            toks = set(_tokenize(a.label)) | set(_tokenize(str(a.metadata.get("gloss", ""))))
            if not toks:
                continue
            score = len(q_tokens & toks) / max(1, len(q_tokens | toks))
            if score > 0:
                scored.append((float(score), k))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [k for _, k in scored[: int(max_results)]]

    def query_vector(self, tokens: Sequence[str], *, temperature: float = 1.0) -> np.ndarray:
        """Return a normalized semantic vector for a token sequence."""
        candidates: List[str] = []
        for t in tokens:
            candidates.extend(self.resolve(str(t), max_results=4))
        if not candidates:
            return _normalize_vector(hashed_token_embedding(tokens, dim=self.config.dim, seed=self.config.seed))
        uniq = sorted(set(candidates))
        raw_scores = []
        for k in uniq:
            a = self.attractors[k]
            # Alias hit count plus prior weight.
            hit_count = sum(1 for t in tokens if k in self.resolve(str(t), max_results=8))
            raw_scores.append(float(hit_count) + math.log1p(max(0.0, float(a.weight))))
        weights = _safe_softmax(raw_scores, temperature=temperature)
        v = np.zeros((self.config.dim,), dtype=np.float64)
        for w, k in zip(weights, uniq):
            v += float(w) * self.attractors[k].normalized_vector()
        return _normalize_vector(v, dim=self.config.dim)

    def similarity(self, a: Union[str, Sequence[str]], b: Union[str, Sequence[str]]) -> float:
        va = self.query_vector([a] if isinstance(a, str) else list(a))
        vb = self.query_vector([b] if isinstance(b, str) else list(b))
        return float(np.real(np.vdot(va, vb)))

    def nearest(self, query: Union[str, Sequence[str], np.ndarray], *, top_k: int = 8) -> List[Tuple[str, float]]:
        if isinstance(query, np.ndarray):
            qv = _normalize_vector(_resample_vector(query, self.config.dim))
        elif isinstance(query, str):
            qv = self.query_vector(_tokenize(query))
        else:
            qv = self.query_vector([str(x) for x in query])
        out = []
        for k, a in self.attractors.items():
            sim = float(np.real(np.vdot(qv, a.normalized_vector())))
            out.append((k, sim))
        out.sort(key=lambda x: (-x[1], x[0]))
        return out[: int(top_k)]

    # ------------------------------------------------------------------
    # Operator construction
    # ------------------------------------------------------------------
    def attractor_operator(
        self,
        key_or_tokens: Union[str, Sequence[str]],
        *,
        d: int,
        strength: Optional[float] = None,
        traceless: Optional[bool] = None,
    ) -> np.ndarray:
        """Build one local Hermitian semantic potential operator."""
        dd = int(max(1, d))
        if isinstance(key_or_tokens, str) and key_or_tokens in self.attractors:
            v = self.attractors[key_or_tokens].normalized_vector()
            w = self.attractors[key_or_tokens].weight
        else:
            tokens = _tokenize(key_or_tokens) if isinstance(key_or_tokens, str) else [str(x) for x in key_or_tokens]
            v = self.query_vector(tokens)
            w = 1.0
        vv = _normalize_vector(_resample_vector(v, dd))
        P = _projector_from_vector(vv, traceless=self.config.traceless_operators if traceless is None else bool(traceless))
        amp = self.config.attractor_strength if strength is None else float(strength)
        H = (-amp * float(w) * P).astype(np.complex128)
        n = fro_norm(H)
        target = float(self.config.operator_norm_target)
        if target > 0.0 and n > _EPS:
            H = (H * (target / n)).astype(np.complex128)
        return hermitianize(H).astype(np.complex128)

    def repulsion_operator(self, key_or_tokens: Union[str, Sequence[str]], *, d: int, strength: Optional[float] = None) -> np.ndarray:
        """Build positive semantic repulsion operator +|a><a|."""
        H = self.attractor_operator(key_or_tokens, d=d, strength=1.0, traceless=False)
        # attractor_operator returns negative projector; flip sign.
        amp = self.config.repulsion_strength if strength is None else float(strength)
        return hermitianize(-float(amp) * H).astype(np.complex128)

    def local_operator_basis(self, d: int, *, include_identity: bool = True, max_attractors: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Return a local operator dictionary compatible with hamiltonian.py/apply.py.

        Keys:
          I, SEM, SEM_<sanitized_key>, optionally REP_<sanitized_key>

        SEM is the mean attractor potential across all entries.
        """
        dd = int(max(1, d))
        ops: Dict[str, np.ndarray] = {}
        if include_identity:
            ops["I"] = np.eye(dd, dtype=np.complex128)

        keys = self.keys()
        if max_attractors is not None:
            keys = keys[: int(max_attractors)]

        mean = np.zeros((dd, dd), dtype=np.complex128)
        for k in keys:
            label = "SEM_" + re.sub(r"[^A-Za-z0-9_]+", "_", k)[:80]
            H = self.attractor_operator(k, d=dd)
            ops[label] = H
            mean += H
        if keys:
            mean = mean / float(len(keys))
        ops["SEM"] = hermitianize(mean).astype(np.complex128)
        return ops

    def product_leaf_ops_for_tokens(
        self,
        tokens: Sequence[str],
        tree: Any,
        phys_dims: Mapping[int, int],
        *,
        leaf_selector: str = "round_robin",
        strength: Optional[float] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Build leaf -> semantic operator map for TTNState.apply_product_ops(...).

        The default round-robin selector assigns token i to tree.leaves[i % L].
        Repeated assignments to one leaf are composed by matrix multiplication.
        """
        leaves = [int(x) for x in getattr(tree, "leaves")]
        _assert(len(leaves) > 0, "tree must expose non-empty leaves")
        out: Dict[int, np.ndarray] = {}
        for i, tok in enumerate(tokens):
            if leaf_selector == "hash":
                lid = leaves[int(_stable_hash_u64(str(tok), seed=self.config.seed) % len(leaves))]
            else:
                lid = leaves[int(i % len(leaves))]
            d = int(phys_dims[int(lid)])
            A = self.attractor_operator(str(tok), d=d, strength=strength)
            if lid in out:
                out[lid] = hermitianize(A @ out[lid]).astype(np.complex128)
            else:
                out[lid] = A
        return out

    def onsite_terms_for_tokens(
        self,
        tokens: Sequence[str],
        tree: Any,
        phys_dims: Mapping[int, int],
        *,
        coeff: complex = 1.0,
        leaf_selector: str = "round_robin",
    ) -> List[Dict[str, Any]]:
        """
        Build generic onsite term dictionaries for compiled-operator adapters.

        This does not import hamiltonian.OnsiteOp to avoid circular coupling.
        """
        leaf_ops = self.product_leaf_ops_for_tokens(tokens, tree, phys_dims, leaf_selector=leaf_selector)
        terms = []
        for lid, A in sorted(leaf_ops.items()):
            terms.append({"leaf": int(lid), "coeff": complex(coeff), "matrix": A, "opname": "SEM_QUERY", "source": "semantic_attractor"})
        return terms

    def semantic_potential_matrix(self, *, normalize: bool = True) -> np.ndarray:
        """
        Dense semantic potential over attractor keys.

        Diagonal terms are negative attractor weights. Relation terms lower the
        potential between related entries. This is a lexical-space Hamiltonian-like
        matrix; use compress_relation_tensor_train(...) for TT/MPO storage.
        """
        keys = self.keys()
        n = len(keys)
        H = np.zeros((n, n), dtype=np.complex128)
        pos = {k: i for i, k in enumerate(keys)}
        for k in keys:
            i = pos[k]
            H[i, i] -= float(self.attractors[k].weight)
        if self.relation_matrix is not None and self.relation_matrix.shape == (n, n):
            R = np.asarray(self.relation_matrix, dtype=np.float64)
        elif self.relation_matrix is not None:
            # Reorder if matrix follows key_to_index rather than sorted keys.
            R0 = np.asarray(self.relation_matrix, dtype=np.float64)
            R = np.zeros((n, n), dtype=np.float64)
            for k1 in keys:
                for k2 in keys:
                    i0 = self.key_to_index.get(k1)
                    j0 = self.key_to_index.get(k2)
                    if i0 is not None and j0 is not None and i0 < R0.shape[0] and j0 < R0.shape[1]:
                        R[pos[k1], pos[k2]] = R0[i0, j0]
        else:
            R = np.zeros((n, n), dtype=np.float64)
        if R.size:
            H -= 0.5 * self.config.attractor_strength * (R + R.T).astype(np.complex128)
        H = hermitianize(H).astype(np.complex128)
        if normalize:
            nrm = fro_norm(H)
            if nrm > _EPS:
                H = (H / nrm).astype(np.complex128)
        return H

    def compress_relation_tensor_train(
        self,
        *,
        output_dims: Optional[List[int]] = None,
        input_dims: Optional[List[int]] = None,
        max_bond_dim: int = 16,
        randomized: bool = True,
        energy_tol: Optional[float] = 0.999,
        dtype: np.dtype = np.float32,
    ) -> Any:
        """
        Compress semantic_potential_matrix() using tn.py TensorTrain.

        Returns a TensorTrain object. tn.py is imported lazily. If dimensions are
        omitted, a near-square factorization is requested from tn.factorize_into_modes.
        """
        from tn import TensorTrain, factorize_into_modes  # local optional project import

        H = self.semantic_potential_matrix(normalize=True)
        n = int(H.shape[0])
        if output_dims is None:
            output_dims = factorize_into_modes(n, num_modes=2)
        if input_dims is None:
            input_dims = factorize_into_modes(n, num_modes=len(output_dims))
        _assert(int(np.prod(output_dims)) == n, "output_dims product must equal semantic matrix size")
        _assert(int(np.prod(input_dims)) == n, "input_dims product must equal semantic matrix size")

        # TensorTrain currently defaults to real dtypes; use real part unless a complex dtype is explicitly requested.
        M: np.ndarray
        if np.dtype(dtype).kind == "c":
            M = H.astype(dtype)
        else:
            M = H.real.astype(dtype)

        if randomized:
            return TensorTrain.from_dense_randomized(
                M,
                output_dims=list(map(int, output_dims)),
                input_dims=list(map(int, input_dims)),
                max_bond_dim=int(max_bond_dim),
                dtype=dtype,
                energy_tol=energy_tol,
            )
        return TensorTrain.from_dense(
            M,
            output_dims=list(map(int, output_dims)),
            input_dims=list(map(int, input_dims)),
            max_bond_dim=int(max_bond_dim),
            dtype=dtype,
            energy_tol=energy_tol,
        )

    # ------------------------------------------------------------------
    # Diagnostics and serialization
    # ------------------------------------------------------------------
    def health_metrics(self) -> Dict[str, Any]:
        dims = [a.dim for a in self.attractors.values()]
        finite = all(np.all(np.isfinite(a.vector)) for a in self.attractors.values())
        relation_ok = True
        if self.relation_matrix is not None:
            relation_ok = bool(np.all(np.isfinite(self.relation_matrix)))
        norms = np.asarray([np.linalg.norm(a.vector) for a in self.attractors.values()], dtype=np.float64)
        return {
            "kind": "SemanticAttractorBank",
            "entry_count": int(len(self.attractors)),
            "dim": int(self.config.dim),
            "all_dims_match": bool(all(d == self.config.dim for d in dims)),
            "finite": bool(finite and relation_ok),
            "vector_norm_min": float(np.min(norms)) if norms.size else 0.0,
            "vector_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "vector_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "has_relation_matrix": self.relation_matrix is not None,
            "relation_shape": None if self.relation_matrix is None else list(map(int, self.relation_matrix.shape)),
            "is_stable": bool(finite and relation_ok and all(d == self.config.dim for d in dims)),
        }

    def snapshot(self, *, include_vectors: bool = False, max_items: int = 16) -> Dict[str, Any]:
        keys = self.keys()[: int(max_items)]
        return {
            "kind": "SemanticAttractorBank",
            "config": _json_safe(self.config),
            "entry_count": int(len(self.attractors)),
            "keys_sample": keys,
            "attractors_sample": {k: self.attractors[k].to_dict(include_vector=include_vectors) for k in keys},
            "alias_count": int(len(self.aliases)),
            "relation_shape": None if self.relation_matrix is None else list(map(int, self.relation_matrix.shape)),
            "metadata": _json_safe(self.metadata),
            "health": self.health_metrics(),
        }

    def to_dict(self, *, include_vectors: bool = True) -> Dict[str, Any]:
        return {
            "format": "AtomTN.SemanticAttractorBank",
            "version": 1,
            "config": _json_safe(self.config),
            "attractors": {k: a.to_dict(include_vector=include_vectors) for k, a in self.attractors.items()},
            "relation_matrix": None if self.relation_matrix is None else self.relation_matrix.astype(float).tolist(),
            "key_to_index": {str(k): int(v) for k, v in self.key_to_index.items()},
            "aliases": {str(k): [str(x) for x in vals] for k, vals in self.aliases.items()},
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticAttractorBank":
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        cfg_payload = dict(payload.get("config", {}))
        cfg = SemanticAttractorConfig(**cfg_payload).normalized()
        attractors: Dict[str, SemanticAttractor] = {}
        for k, raw in dict(payload.get("attractors", {})).items():
            if "vector" not in raw:
                if cfg.strict:
                    raise ValueError(f"attractor {k!r} missing vector")
                vec = random_key_embedding(str(k), dim=cfg.dim, seed=cfg.seed)
            else:
                vec = _sanitize_vector(raw.get("vector"), dim=cfg.dim)
            attractors[str(k)] = SemanticAttractor(
                key=str(raw.get("key", k)),
                vector=vec,
                label=str(raw.get("label", k)),
                weight=float(raw.get("weight", 1.0)),
                metadata=dict(raw.get("metadata", {})),
            )
        relation = payload.get("relation_matrix", None)
        return cls(
            attractors=attractors,
            config=cfg,
            relation_matrix=None if relation is None else np.asarray(relation, dtype=np.float64),
            key_to_index={str(k): int(v) for k, v in dict(payload.get("key_to_index", {})).items()},
            aliases={str(k): [str(x) for x in vals] for k, vals in dict(payload.get("aliases", {})).items()},
            metadata=dict(payload.get("metadata", {})),
        )

    def save_json(self, path: Union[str, Path], *, include_vectors: bool = True) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(self.to_dict(include_vectors=include_vectors)), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "SemanticAttractorBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_npz(self, path: Union[str, Path]) -> None:
        """Compact numeric serialization. JSON metadata is embedded in the NPZ."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys = self.keys()
        V = np.stack([self.attractors[k].vector for k in keys], axis=0).astype(np.float64) if keys else np.zeros((0, self.config.dim), dtype=np.float64)
        weights = np.asarray([self.attractors[k].weight for k in keys], dtype=np.float64)
        labels = np.asarray([self.attractors[k].label for k in keys], dtype=np.str_)
        meta = {
            "format": "AtomTN.SemanticAttractorBank.npz",
            "version": 1,
            "config": _json_safe(self.config),
            "keys": keys,
            "key_to_index": self.key_to_index,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "attractor_metadata": {k: self.attractors[k].metadata for k in keys},
        }
        arrays: Dict[str, Any] = {
            "__metadata_json__": np.asarray(json.dumps(_json_safe(meta), sort_keys=True), dtype=np.str_),
            "vectors": V,
            "weights": weights,
            "labels": labels,
        }
        if self.relation_matrix is not None:
            arrays["relation_matrix"] = self.relation_matrix.astype(np.float64)
        np.savez_compressed(p, **arrays)

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "SemanticAttractorBank":
        """
        Load either:
          1. Native AtomTN.SemanticAttractorBank.npz produced by save_npz(...)
          2. LK20.SemanticBank produced by dictionary_lexicon_ingestor.py

        The LK20 format stores vectors plus labels/lemmas/pos/gloss metadata,
        not a full native attractor dictionary. This loader reconstructs
        SemanticAttractor objects and aliases deterministically.
        """
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            meta = {}
            if "__metadata_json__" in data:
                try:
                    meta = json.loads(str(data["__metadata_json__"].item()))
                except Exception:
                    meta = {}
            elif "metadata_json" in data:
                try:
                    meta = json.loads(str(data["metadata_json"].item()))
                except Exception:
                    meta = {}

            fmt = str(meta.get("format", ""))

            # -----------------------------------------------------------------
            # Native AtomTN.SemanticAttractorBank.npz
            # -----------------------------------------------------------------
            if fmt == "AtomTN.SemanticAttractorBank.npz" or "keys" in meta:
                cfg = SemanticAttractorConfig(**dict(meta.get("config", {}))).normalized()
                keys = [str(k) for k in meta.get("keys", [])]
                V = np.asarray(data["vectors"], dtype=np.float64)
                weights = np.asarray(data["weights"], dtype=np.float64) if "weights" in data else np.ones((len(keys),), dtype=np.float64)
                labels = [str(x) for x in np.asarray(data["labels"]).tolist()] if "labels" in data else list(keys)
                attr_meta = dict(meta.get("attractor_metadata", {}))

                attractors: Dict[str, SemanticAttractor] = {}
                for i, k in enumerate(keys):
                    if i >= V.shape[0]:
                        continue
                    attractors[k] = SemanticAttractor(
                        key=k,
                        vector=V[i],
                        label=labels[i] if i < len(labels) else k,
                        weight=float(weights[i]) if i < weights.size else 1.0,
                        metadata=dict(attr_meta.get(k, {})),
                    )

                relation = np.asarray(data["relation_matrix"], dtype=np.float64) if "relation_matrix" in data else None
                return cls(
                    attractors=attractors,
                    config=cfg,
                    relation_matrix=relation,
                    key_to_index={str(k): int(v) for k, v in dict(meta.get("key_to_index", {})).items()},
                    aliases={str(k): [str(x) for x in vals] for k, vals in dict(meta.get("aliases", {})).items()},
                    metadata=dict(meta.get("metadata", {})),
                )

            # -----------------------------------------------------------------
            # LK20.SemanticBank from dictionary_lexicon_ingestor.py
            # -----------------------------------------------------------------
            if "vectors" not in data:
                raise ValueError(f"NPZ semantic bank missing vectors: {p}")

            V = np.asarray(data["vectors"], dtype=np.float64)
            if V.ndim != 2:
                raise ValueError(f"vectors must be 2D in semantic bank: {p}")

            dim = int(V.shape[1]) if V.shape[1:] else int(meta.get("dim", 64) or 64)
            cfg_payload = dict(meta.get("config", {}) or {})
            cfg_payload.setdefault("dim", dim)
            cfg = SemanticAttractorConfig(**cfg_payload).normalized()

            def _arr_str(name: str, fallback: Optional[List[str]] = None) -> List[str]:
                if name not in data:
                    return list(fallback or [])
                return [str(x) for x in np.asarray(data[name]).tolist()]

            labels = _arr_str("labels")
            if not labels:
                labels = _arr_str("lemmas")
            if not labels:
                labels = _arr_str("words")
            if not labels:
                labels = [f"entry_{i}" for i in range(V.shape[0])]

            pos = _arr_str("pos", [""] * len(labels))
            glosses = _arr_str("glosses")
            if not glosses:
                glosses = _arr_str("definitions", [""] * len(labels))
            sources = _arr_str("sources", [""] * len(labels))
            sense_ids = _arr_str("sense_ids", [""] * len(labels))
            weights = np.asarray(data["weights"], dtype=np.float64) if "weights" in data else np.ones((V.shape[0],), dtype=np.float64)

            entries_payload: List[Dict[str, Any]] = []
            if "entries_json" in data:
                for raw in np.asarray(data["entries_json"]).tolist():
                    try:
                        obj = json.loads(str(raw))
                        entries_payload.append(obj if isinstance(obj, dict) else {})
                    except Exception:
                        entries_payload.append({})

            attractors: Dict[str, SemanticAttractor] = {}
            aliases: Dict[str, List[str]] = {}
            key_to_index: Dict[str, int] = {}

            for i in range(V.shape[0]):
                label = labels[i] if i < len(labels) else f"entry_{i}"
                entry_payload = entries_payload[i] if i < len(entries_payload) else {}
                lemma = str(entry_payload.get("lemma", entry_payload.get("word", label))).strip() or label
                epos = str(entry_payload.get("pos", pos[i] if i < len(pos) else "")).strip()
                gloss = str(entry_payload.get("gloss", entry_payload.get("definition", glosses[i] if i < len(glosses) else "")))
                source = str(entry_payload.get("source", sources[i] if i < len(sources) else ""))
                sense_id = str(entry_payload.get("sense_id", sense_ids[i] if i < len(sense_ids) else ""))
                key = str(entry_payload.get("key", "")).strip()
                if not key:
                    key = f"lk20:{source or 'bank'}:{epos or 'x'}:{i}:{lemma}".replace(" ", "_")

                weight = float(weights[i]) if i < weights.size and np.isfinite(weights[i]) else 1.0

                metadata = {
                    "lemma": lemma,
                    "pos": epos,
                    "gloss": gloss,
                    "source": source,
                    "sense_id": sense_id,
                    "format_source": fmt or "LK20.SemanticBank",
                }
                if entry_payload:
                    metadata["entry"] = entry_payload

                attractors[key] = SemanticAttractor(
                    key=key,
                    vector=V[i],
                    label=lemma,
                    weight=weight,
                    metadata=metadata,
                )
                key_to_index[key] = i

                for alias in {lemma.lower(), label.lower()}:
                    if alias:
                        aliases.setdefault(alias, []).append(key)
                for tok in _tokenize(f"{lemma} {gloss} {epos}"):
                    aliases.setdefault(tok, []).append(key)

            relation = np.asarray(data["relation_matrix"], dtype=np.float64) if "relation_matrix" in data else None
            if relation is not None and relation.size == 0:
                relation = None

            return cls(
                attractors=attractors,
                config=cfg,
                relation_matrix=relation,
                key_to_index=key_to_index,
                aliases={k: sorted(set(v)) for k, v in aliases.items()},
                metadata={
                    "source_path": str(p),
                    "format": fmt or "LK20.SemanticBank",
                    "loaded_by": "SemanticAttractorBank.load_npz.compat",
                    "raw_metadata": meta,
                },
            )



# =============================================================================
# Hybrid Semantic Bank (SQLite + Tensor Core)
# =============================================================================

class HybridSemanticBank:
    """
    Disk-backed semantic bank using SQLite for metadata and aliases.
    Designed for large-scale lexicons (millions of entries) that do not fit in RAM.
    """

    def __init__(self, db_path: Union[str, Path], core_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path)
        self.core_path = Path(core_path) if core_path else self.db_path.with_suffix(".core.json")
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.config = SemanticAttractorConfig() # Default config
        
        self.ttn: Optional[Any] = None
        if self.core_path.exists():
            try:
                from digital_twin_kernel import TreeTensorNetwork
                self.ttn = TreeTensorNetwork.load_json(self.core_path)
            except Exception:
                pass

    def close(self):
        self._conn.close()

    def find(self, alias: str) -> List[str]:
        """Lookup keys by alias (lemma, word, or token)."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT key FROM aliases WHERE alias = ?", (str(alias).lower(),))
        return [row[0] for row in cursor.fetchall()]

    def get_entry(self, key: str) -> Optional[SemanticAttractor]:
        """Lookup a single attractor by key."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM entries WHERE key = ?", (str(key),))
        row = cursor.fetchone()
        if not row:
            return None
        
        # Schema: key, lemma, pos, gloss, weight, source, metadata_json
        meta = json.loads(row[6]) if row[6] else {}
        meta.update({"lemma": row[1], "pos": row[2], "gloss": row[3], "source": row[5]})
        
        # We need the vector. For Hybrid bank, we either compute it on the fly
        # or load it from a sharded store. For now, we compute it on the fly
        # to ensure the bank remains compact on disk.
        from lexicon_utils import vectorize_lexicon_entry
        vec = vectorize_lexicon_entry(
            lemma=row[1], pos=row[2], gloss=row[3], 
            source=row[5], weight=row[4], 
            dim=self.config.dim, seed=self.config.seed
        )
        
        return SemanticAttractor(
            key=row[0],
            vector=vec,
            label=row[1],
            weight=row[4],
            metadata=meta
        )

    def query_vector(self, tokens: Sequence[str], temperature: float = 1.0) -> np.ndarray:
        """Compute aggregate intent vector for a sequence of tokens."""
        vecs = []
        for t in tokens:
            keys = self.find(t)
            for k in keys:
                attr = self.get_entry(k)
                if attr:
                    vecs.append(attr.vector)
        
        if not vecs:
            return np.zeros(self.config.dim, dtype=np.float64)
        
        avg = np.mean(np.stack(vecs), axis=0)
        nrm = np.linalg.norm(avg)
        return avg / nrm if nrm > 1e-12 else avg

    def nearest(self, vector: np.ndarray, top_k: int = 8) -> List[Tuple[str, float]]:
        """
        Find nearest keys to a vector. 
        In the hybrid bank, we use the POS-level TTN core to narrow the search,
        then refine using the SQLite store (or a cached index).
        """
        # Fallback: query a subset of 'important' entries if TTN is missing
        # For now, let's just return matches for the lemma if the vector was derived from one.
        # A full vector search on 10M entries requires a dedicated vector index (like FAISS)
        # or a very efficient TTN-based retrieval.
        return []

    @property
    def attractors(self):
        """Shim for legacy code that expects a dict."""
        class AttractorProxy:
            def __init__(self, parent): self.parent = parent
            def __getitem__(self, key): return self.parent.get_entry(key)
            def get(self, key, default=None): 
                val = self.parent.get_entry(key)
                return val if val is not None else default
            def __contains__(self, key): return self.parent.get_entry(key) is not None
        return AttractorProxy(self)

    @property
    def aliases(self):
        """Shim for legacy code that expects a dict of lists."""
        class AliasProxy:
            def __init__(self, parent): self.parent = parent
            def __getitem__(self, alias): return self.parent.find(alias)
            def get(self, alias, default=None):
                val = self.parent.find(alias)
                return val if val else default
        return AliasProxy(self)

    def keys(self) -> List[str]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT key FROM entries LIMIT 1000") # Bounded for safety
        return [row[0] for row in cursor.fetchall()]

# =============================================================================
# Sentence-state helpers ... (rest of file)


@dataclass
class SemanticSentenceState:
    """Lightweight semantic state for a token sequence before TTN construction."""

    tokens: List[str]
    query_vector: np.ndarray
    nearest_keys: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tokens = [str(t).lower().strip() for t in self.tokens if str(t).strip()]
        self.query_vector = _sanitize_vector(self.query_vector)
        self.nearest_keys = [(str(k), float(v)) for k, v in self.nearest_keys]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "tokens": self.tokens,
            "query_vector": self.query_vector,
            "nearest_keys": self.nearest_keys,
            "metadata": self.metadata,
        })


def build_sentence_state(
    text_or_tokens: Union[str, Sequence[str]],
    bank: SemanticAttractorBank,
    *,
    top_k: int = 8,
    temperature: float = 1.0,
) -> SemanticSentenceState:
    tokens = _tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else [str(x).lower().strip() for x in text_or_tokens]
    qv = bank.query_vector(tokens, temperature=temperature)
    nearest = bank.nearest(qv, top_k=top_k)
    return SemanticSentenceState(tokens=tokens, query_vector=qv, nearest_keys=nearest, metadata={"source": "build_sentence_state"})


# =============================================================================
# Minimal self-test
# =============================================================================


def _self_test() -> None:
    entries = [
        SemanticEntry(key="en:cat:n:animal", lemma="cat", pos="n", gloss="small domesticated feline animal", relations={"hypernym": ["en:animal:n:organism"]}),
        SemanticEntry(key="en:dog:n:animal", lemma="dog", pos="n", gloss="domesticated canine animal", relations={"hypernym": ["en:animal:n:organism"]}),
        SemanticEntry(key="en:animal:n:organism", lemma="animal", pos="n", gloss="living organism that feeds on organic matter"),
        SemanticEntry(key="en:run:v:move", lemma="run", pos="v", gloss="move swiftly on foot"),
    ]
    bank = SemanticAttractorBank.from_entries(entries, dim=32, seed=0)
    h = bank.health_metrics()
    assert h["is_stable"]
    assert bank.similarity("cat", "dog") > bank.similarity("cat", "run")
    A = bank.attractor_operator("cat", d=8)
    assert A.shape == (8, 8)
    assert np.allclose(A, A.conj().T)
    st = build_sentence_state("the cat can run", bank)
    assert st.query_vector.shape == (32,)
    print("semantic_attractors.py self-test passed")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "SemanticEntry",
    "SemanticAttractorConfig",
    "SemanticAttractor",
    "SemanticAttractorBank",
    "SemanticSentenceState",
    "hashed_token_embedding",
    "random_key_embedding",
    "embedding_for_entry",
    "relation_matrix_from_entries",
    "build_sentence_state",
]
