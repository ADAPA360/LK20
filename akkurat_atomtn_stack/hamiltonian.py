#!/usr/bin/env python3
# hamiltonian.py
"""
Hamiltonian / operator builder for AtomTN.

Production role
---------------
This module converts AtomTN flow, vibration, projection, and local-fiber context
into Hamiltonian term containers consumable by both legacy and Phase-4 apply
paths:

1. TreeMPO
   Backward-compatible scaffold representation used by apply_tree_mpo_fast(...).

2. CompiledTreeOperator
   Correct-apply-oriented representation with concrete onsite and pairwise leaf
   operators, routing metadata, LCA grouping, deterministic term labels, and
   compatibility views for earlier apply_compiled_operator_zipup(...).

The module intentionally does not apply operators to a TTN state.  Operator
application belongs in apply.py and evolution orchestration belongs in evolve.py.

Compatibility
-------------
Existing AtomTN callers may continue to use:

    builder = TreeMPOBuilder(calc, cfg, decomp=..., projection=..., holonomy=...)
    H, local_ops = builder.build(state, fiber, X, vib, diag, step_id)

Phase-4 callers may use:

    op, local_ops = builder.build_compiled(state, fiber, X, vib, diag, step_id)
    op, local_ops = builder.build_compiled_operator(state=..., fiber=..., X=..., ...)
    pieces = builder.build_split_operators(...)

Dependencies
------------
- numpy
- math_utils.py
- projection.py optionally, for ProjectionLayer type compatibility
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from math_utils import _assert, fro_norm, hermitianize

try:  # type-only / optional runtime support
    from projection import ProjectionLayer  # noqa: F401
except Exception:  # pragma: no cover
    ProjectionLayer = Any  # type: ignore


_EPS = 1e-12
_DEFAULT_NUM_LEAVES = 64


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
        out = np.asarray(arr, dtype=np.float64)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _sanitize_complex_matrix(A: Any, *, name: str = "operator", strict: bool = True) -> np.ndarray:
    M = np.asarray(A, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        if strict:
            raise ValueError(f"{name} must be a square matrix; got shape {M.shape}")
        d = int(M.shape[0]) if M.ndim >= 1 and M.shape[0] > 0 else 1
        return np.eye(d, dtype=np.complex128)
    if not np.all(np.isfinite(M)):
        if strict:
            raise FloatingPointError(f"{name} contains non-finite values")
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex128)
    return M.astype(np.complex128, copy=False)


def _sanitize_weight(w: Any, *, max_abs: Optional[float] = None, strict: bool = True) -> complex:
    try:
        z = complex(w)
    except Exception:
        if strict:
            raise
        z = 0.0 + 0.0j
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        if strict:
            raise FloatingPointError("Hamiltonian weight contains non-finite values")
        z = 0.0 + 0.0j
    if max_abs is not None and float(max_abs) > 0:
        mag = abs(z)
        if mag > float(max_abs) and mag > 0:
            z *= float(max_abs) / mag
    return np.complex128(z).item()


def _maybe_hermitianize(A: np.ndarray, *, enabled: bool = True) -> np.ndarray:
    if not enabled:
        return np.asarray(A, dtype=np.complex128)
    return hermitianize(A).astype(np.complex128)


def _op_label(prefix: str, *parts: Any) -> str:
    safe = [str(prefix)]
    for p in parts:
        s = str(p).replace(".", "_").replace("-", "m").replace(" ", "_")
        safe.append(s)
    return "__" + "_".join(safe)


def _get_children(tree: Any, nid: int) -> List[int]:
    node = tree.nodes[int(nid)]
    if hasattr(node, "children"):
        return [int(x) for x in getattr(node, "children")]
    if hasattr(node, "child_ids"):
        return [int(x) for x in getattr(node, "child_ids")]
    return []


def _parent_of(tree: Any, nid: int) -> Optional[int]:
    node = tree.nodes[int(nid)]
    p = getattr(node, "parent", None)
    return None if p is None else int(p)


def _depth_of(tree: Any, nid: int) -> int:
    depth = 0
    cur = int(nid)
    seen = set()
    while cur not in seen:
        seen.add(cur)
        p = _parent_of(tree, cur)
        if p is None:
            return int(depth)
        cur = int(p)
        depth += 1
    raise ValueError("tree contains a parent cycle")


def _path_to_root(tree: Any, nid: int) -> List[int]:
    path = [int(nid)]
    cur = int(nid)
    seen = {cur}
    while True:
        p = _parent_of(tree, cur)
        if p is None:
            break
        p = int(p)
        if p in seen:
            raise ValueError("tree contains a parent cycle")
        path.append(p)
        seen.add(p)
        cur = p
    return path


def _lca_and_paths(tree: Any, u: int, v: int) -> Tuple[int, List[int], List[int]]:
    pu = _path_to_root(tree, int(u))
    pv = _path_to_root(tree, int(v))
    pos_u = {nid: i for i, nid in enumerate(pu)}
    lca: Optional[int] = None
    for nid in pv:
        if nid in pos_u:
            lca = int(nid)
            break
    _assert(lca is not None, "LCA not found; tree is disconnected or malformed")
    return lca, pu[: pos_u[lca] + 1], pv[: pv.index(lca) + 1]


def _leaf_index_map(tree: Any) -> Dict[int, int]:
    return {int(lid): i for i, lid in enumerate(list(tree.leaves))}


def _scalar_from_edge_value(value: Any) -> float:
    try:
        if isinstance(value, (int, float, complex, np.integer, np.floating, np.complexfloating)):
            val = float(np.real(value))
        else:
            val = float(fro_norm(np.asarray(value)))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _flow_lookup(X: Any, u: int, v: int, default: Any = 0.0) -> Any:
    try:
        return getattr(X, "edge_values", {}).get((int(u), int(v)), default)
    except Exception:
        return default


def _has_matrix_flow(X: Any) -> bool:
    return bool(getattr(X, "matrix_valued", False))


def _safe_frequency_mean(vib: Any) -> float:
    if vib is None:
        return 0.0
    try:
        freq = np.asarray(getattr(vib, "frequencies", []), dtype=np.float64).reshape(-1)
        if freq.size == 0:
            return 0.0
        freq = np.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)
        val = float(np.mean(freq))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _basis_op(ops: Mapping[str, np.ndarray], name: str, *, d: int, strict: bool = True) -> np.ndarray:
    key = str(name)
    if key in ops:
        return _sanitize_complex_matrix(ops[key], name=f"operator {key}", strict=strict)
    if not strict:
        return np.eye(int(d), dtype=np.complex128)
    raise ValueError(f"missing local operator '{key}' for d={d}")


def _ensure_basic_ops(ops: MutableMapping[str, np.ndarray], d: int) -> None:
    d = int(max(1, d))
    ops.setdefault("I", np.eye(d, dtype=np.complex128))
    if d >= 2:
        X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        Y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        for name, P in (("X", X2), ("Y", Y2), ("Z", Z2)):
            if name not in ops:
                A = np.eye(d, dtype=np.complex128)
                A[:2, :2] = P
                ops[name] = A
    else:
        ops.setdefault("X", np.eye(d, dtype=np.complex128))
        ops.setdefault("Y", np.zeros((d, d), dtype=np.complex128))
        ops.setdefault("Z", np.eye(d, dtype=np.complex128))


def _ensure_su2_aliases(ops: MutableMapping[str, np.ndarray], d: int) -> None:
    """Ensure Lx/Ly/Lz exist, falling back to Pauli-like X/Y/Z when needed."""
    _ensure_basic_ops(ops, d)
    if "Lx" not in ops:
        ops["Lx"] = np.asarray(ops["X"], dtype=np.complex128)
    if "Ly" not in ops:
        ops["Ly"] = np.asarray(ops["Y"], dtype=np.complex128)
    if "Lz" not in ops:
        ops["Lz"] = np.asarray(ops["Z"], dtype=np.complex128)


def _validate_local_ops(ops: MutableMapping[str, np.ndarray], d: int, *, strict: bool) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, A in list(ops.items()):
        M = _sanitize_complex_matrix(A, name=f"local_ops[{name}]", strict=strict)
        if M.shape != (int(d), int(d)):
            if strict:
                raise ValueError(f"local operator '{name}' shape {M.shape} != ({d},{d})")
            M2 = np.zeros((int(d), int(d)), dtype=np.complex128)
            r = min(int(d), M.shape[0])
            c = min(int(d), M.shape[1])
            if r > 0 and c > 0:
                M2[:r, :c] = M[:r, :c]
            M = M2
        out[str(name)] = M.astype(np.complex128, copy=False)
    _ensure_basic_ops(out, int(d))
    return out


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HamiltonianBuildConfig:
    """Hamiltonian term construction knobs."""

    onsite_scale: float = 0.5
    onsite_mode: str = "zfield"  # "zfield" | "holographic_su2" | "none"

    vib_scale: float = 0.02
    vib_op: str = "I"

    edge_scale: float = 0.25
    edge_mode: str = "zz"  # "zz" | "heisenberg_su2" | "holonomy_su2" | "none"
    hop_scale: float = 1.0

    extra_scale: float = 0.0
    extra_op: str = "G"

    # Production guards.
    drop_abs_below: float = 0.0
    max_abs_weight: Optional[float] = None
    symmetrize_local_ops: bool = True
    strict: bool = True

    def normalized(self) -> "HamiltonianBuildConfig":
        return HamiltonianBuildConfig(
            onsite_scale=float(self.onsite_scale),
            onsite_mode=str(self.onsite_mode).lower().strip(),
            vib_scale=float(self.vib_scale),
            vib_op=str(self.vib_op),
            edge_scale=float(self.edge_scale),
            edge_mode=str(self.edge_mode).lower().strip(),
            hop_scale=float(self.hop_scale),
            extra_scale=float(self.extra_scale),
            extra_op=str(self.extra_op),
            drop_abs_below=max(0.0, float(self.drop_abs_below)),
            max_abs_weight=(None if self.max_abs_weight is None else float(max(0.0, self.max_abs_weight))),
            symmetrize_local_ops=bool(self.symmetrize_local_ops),
            strict=bool(self.strict),
        )


# =============================================================================
# Legacy scaffold MPO container
# =============================================================================

@dataclass
class TreeMPO:
    """Backward-compatible scaffold term representation."""

    tree: Any
    W: int
    weights: np.ndarray

    leaf_term_leaf: List[int]
    leaf_term_opname: List[str]

    edge_term_uv: List[Tuple[int, int]]
    edge_term_op_u: List[str]
    edge_term_op_v: List[Optional[str]]
    edge_term_Bv: List[Optional[np.ndarray]]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.complex128).reshape(-1)
        _assert(self.weights.ndim == 1, "TreeMPO.weights must be 1D")
        _assert(int(self.W) == int(self.weights.size), "TreeMPO.W must equal len(weights)")
        _assert(len(self.leaf_term_leaf) == len(self.leaf_term_opname), "leaf term list mismatch")
        _assert(
            len(self.edge_term_uv) == len(self.edge_term_op_u) == len(self.edge_term_op_v) == len(self.edge_term_Bv),
            "edge term list mismatch",
        )
        if self.weights.size:
            _assert(np.all(np.isfinite(self.weights)), "TreeMPO.weights contain non-finite values")

    @property
    def n_leaf_terms(self) -> int:
        return int(len(self.leaf_term_leaf))

    @property
    def n_edge_terms(self) -> int:
        return int(len(self.edge_term_uv))

    def global_id_for_leaf_term(self, k: int) -> int:
        return int(1 + int(k))

    def global_id_for_edge_term(self, e: int) -> int:
        return int(1 + self.n_leaf_terms + int(e))

    def term_count(self) -> int:
        return int(self.n_leaf_terms + self.n_edge_terms)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "TreeMPO",
            "W": int(self.W),
            "n_leaf_terms": int(self.n_leaf_terms),
            "n_edge_terms": int(self.n_edge_terms),
            "weight_l2": float(np.linalg.norm(self.weights)) if self.weights.size else 0.0,
            "metadata": _json_safe(self.metadata),
        }


# =============================================================================
# Compiled operator format
# =============================================================================

@dataclass
class OnsiteOp:
    leaf_id: int
    weight: complex
    op: np.ndarray
    opname: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.leaf_id = int(self.leaf_id)
        self.weight = np.complex128(self.weight).item()
        self.op = _sanitize_complex_matrix(self.op, name=f"OnsiteOp[{self.leaf_id}:{self.opname}]", strict=True)
        self.opname = str(self.opname or "A")
        self.source = str(self.source or "onsite")


@dataclass
class PairOp:
    u: int
    v: int
    weight: complex
    Au: np.ndarray
    Bv: np.ndarray
    opname_u: str = ""
    opname_v: str = ""
    source: str = ""
    lca: Optional[int] = None
    path_u_to_lca: List[int] = field(default_factory=list)
    path_v_to_lca: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.u = int(self.u)
        self.v = int(self.v)
        if self.u > self.v:
            self.u, self.v = self.v, self.u
            self.Au, self.Bv = self.Bv, self.Au
            self.opname_u, self.opname_v = self.opname_v, self.opname_u
            self.path_u_to_lca, self.path_v_to_lca = self.path_v_to_lca, self.path_u_to_lca
        self.weight = np.complex128(self.weight).item()
        self.Au = _sanitize_complex_matrix(self.Au, name=f"PairOp[{self.u},{self.v}].Au", strict=True)
        self.Bv = _sanitize_complex_matrix(self.Bv, name=f"PairOp[{self.u},{self.v}].Bv", strict=True)
        self.opname_u = str(self.opname_u or "Au")
        self.opname_v = str(self.opname_v or "Bv")
        self.source = str(self.source or "pair")
        self.lca = None if self.lca is None else int(self.lca)
        self.path_u_to_lca = [int(x) for x in self.path_u_to_lca]
        self.path_v_to_lca = [int(x) for x in self.path_v_to_lca]


@dataclass
class CompiledTreeOperator:
    """
    Correct-apply-oriented Hamiltonian representation.

    The concrete matrices in OnsiteOp and PairOp are authoritative.  The
    leaf_terms and pair_terms properties intentionally expose a legacy-compatible
    view for existing apply.py implementations that still resolve operators by
    name through local_ops.
    """

    tree: Any
    onsite_by_leaf: Dict[int, List[OnsiteOp]] = field(default_factory=dict)
    pair_by_edge: Dict[Tuple[int, int], List[PairOp]] = field(default_factory=dict)
    pair_by_lca: Dict[int, List[PairOp]] = field(default_factory=dict)
    lca_order: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_onsite(self, term: OnsiteOp) -> None:
        self.onsite_by_leaf.setdefault(int(term.leaf_id), []).append(term)

    def add_pair(self, term: PairOp) -> None:
        key = (min(int(term.u), int(term.v)), max(int(term.u), int(term.v)))
        self.pair_by_edge.setdefault(key, []).append(term)
        if term.lca is not None:
            self.pair_by_lca.setdefault(int(term.lca), []).append(term)

    @property
    def leaf_terms(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for lid in sorted(self.onsite_by_leaf.keys()):
            for t in self.onsite_by_leaf[lid]:
                out.append({"leaf": int(t.leaf_id), "opname": str(t.opname), "coeff": complex(t.weight), "matrix": t.op, "source": t.source})
        return out

    @property
    def pair_terms(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for key in sorted(self.pair_by_edge.keys()):
            for t in self.pair_by_edge[key]:
                out.append(
                    {
                        "u": int(t.u),
                        "v": int(t.v),
                        "op_u": str(t.opname_u),
                        "op_v": str(t.opname_v),
                        "coeff": complex(t.weight),
                        "Au": t.Au,
                        "Bv": t.Bv,
                        "lca": t.lca,
                        "path_u_to_lca": list(t.path_u_to_lca),
                        "path_v_to_lca": list(t.path_v_to_lca),
                        "source": t.source,
                    }
                )
        return out

    def validate(self) -> None:
        leaves = set(int(x) for x in getattr(self.tree, "leaves", []))
        for lid, terms in self.onsite_by_leaf.items():
            _assert(int(lid) in leaves, f"CompiledTreeOperator: onsite leaf {lid} not in tree.leaves")
            for t in terms:
                _assert(t.op.ndim == 2 and t.op.shape[0] == t.op.shape[1], "onsite op must be square")
                _assert(np.all(np.isfinite(t.op)), "onsite op contains non-finite values")
                _assert(np.isfinite(np.real(t.weight)) and np.isfinite(np.imag(t.weight)), "onsite weight non-finite")
        for (u, v), terms in self.pair_by_edge.items():
            _assert(int(u) < int(v), "pair_by_edge keys must be ordered (u<v)")
            _assert(int(u) in leaves and int(v) in leaves, f"edge ({u},{v}) endpoint not in tree.leaves")
            for t in terms:
                _assert(t.Au.ndim == 2 and t.Au.shape[0] == t.Au.shape[1], "Au must be square")
                _assert(t.Bv.ndim == 2 and t.Bv.shape[0] == t.Bv.shape[1], "Bv must be square")
                _assert(np.all(np.isfinite(t.Au)) and np.all(np.isfinite(t.Bv)), "pair op contains non-finite values")
                _assert(np.isfinite(np.real(t.weight)) and np.isfinite(np.imag(t.weight)), "pair weight non-finite")

    def rebuild_lca_order(self) -> None:
        if not self.pair_by_lca:
            self.lca_order = []
            return
        lcas = list(self.pair_by_lca.keys())
        lcas.sort(key=lambda nid: _depth_of(self.tree, int(nid)), reverse=True)
        self.lca_order = [int(x) for x in lcas]

    def term_count(self) -> int:
        return int(sum(len(v) for v in self.onsite_by_leaf.values()) + sum(len(v) for v in self.pair_by_edge.values()))

    def is_empty(self) -> bool:
        return self.term_count() == 0

    def split(self, *, onsite: bool = True, pairs: bool = True, source_filter: Optional[str] = None) -> "CompiledTreeOperator":
        out = CompiledTreeOperator(tree=self.tree, metadata=dict(self.metadata))
        if onsite:
            for lid, terms in self.onsite_by_leaf.items():
                for t in terms:
                    if source_filter is None or t.source == source_filter:
                        out.add_onsite(t)
        if pairs:
            for terms in self.pair_by_edge.values():
                for t in terms:
                    if source_filter is None or t.source == source_filter:
                        out.add_pair(t)
        out.rebuild_lca_order()
        return out

    def split_by_lca(self) -> List["CompiledTreeOperator"]:
        pieces: List[CompiledTreeOperator] = []
        for lca in self.lca_order:
            piece = CompiledTreeOperator(tree=self.tree, metadata={**dict(self.metadata), "split": f"lca:{lca}"})
            for t in self.pair_by_lca.get(int(lca), []):
                piece.add_pair(t)
            piece.rebuild_lca_order()
            if not piece.is_empty():
                pieces.append(piece)
        return pieces

    def snapshot(self) -> Dict[str, Any]:
        weights = [complex(t.weight) for terms in self.onsite_by_leaf.values() for t in terms]
        weights += [complex(t.weight) for terms in self.pair_by_edge.values() for t in terms]
        abs_w = np.asarray([abs(w) for w in weights], dtype=np.float64)
        return {
            "kind": "CompiledTreeOperator",
            "n_onsite_terms": int(sum(len(v) for v in self.onsite_by_leaf.values())),
            "n_pair_terms": int(sum(len(v) for v in self.pair_by_edge.values())),
            "term_count": int(self.term_count()),
            "weight_abs_min": float(np.min(abs_w)) if abs_w.size else 0.0,
            "weight_abs_mean": float(np.mean(abs_w)) if abs_w.size else 0.0,
            "weight_abs_max": float(np.max(abs_w)) if abs_w.size else 0.0,
            "lca_count": int(len(self.pair_by_lca)),
            "metadata": _json_safe(self.metadata),
        }


# =============================================================================
# Builder
# =============================================================================

class TreeMPOBuilder:
    """Build legacy and compiled AtomTN Hamiltonian operators."""

    def __init__(
        self,
        calc: Any,
        cfg: HamiltonianBuildConfig,
        decomp: Optional[Any] = None,
        projection: Optional[Any] = None,
        holonomy: Optional[Any] = None,
        cache_bucket: int = 1,
    ):
        self.calc = calc
        self.cfg = cfg.normalized() if isinstance(cfg, HamiltonianBuildConfig) else HamiltonianBuildConfig(**dict(cfg)).normalized()
        self.decomp = decomp
        self.projection = projection
        self.holonomy = holonomy
        self.cache_bucket = max(int(cache_bucket), 1)

        try:
            self._edges = [(int(u), int(v)) for (u, v) in list(calc.oriented_edges())]
        except Exception:
            self._edges = []

        self._local_ops_cache: Dict[Tuple[int, int, int], Dict[str, np.ndarray]] = {}
        self._R_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._last_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Cache / diagnostics
    # ------------------------------------------------------------------

    def clear_caches(self) -> None:
        self._local_ops_cache.clear()
        self._R_cache.clear()

    def _bucket(self, step_id: int) -> int:
        return int(step_id) // int(max(1, self.cache_bucket))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "TreeMPOBuilder",
            "edge_count": int(len(self._edges)),
            "cache_bucket": int(self.cache_bucket),
            "local_ops_cache_size": int(len(self._local_ops_cache)),
            "rotation_cache_size": int(len(self._R_cache)),
            "cfg": _json_safe(self.cfg),
            "last_summary": _json_safe(self._last_summary),
        }

    # ------------------------------------------------------------------
    # Local operator construction
    # ------------------------------------------------------------------

    def _projected_ops_for_leaf(self, *, lid: int, d: int, diag: Any, step_id: int) -> Dict[str, np.ndarray]:
        if self.projection is None:
            return {}

        kappa = None
        try:
            Kmap = getattr(diag, "kappa_node_mats_su2", None)
            if Kmap is not None:
                kappa = Kmap[int(lid)]
            else:
                Kmap = getattr(diag, "kappa_node_mats_raw", None)
                if Kmap is not None:
                    kappa = Kmap[int(lid)]
        except Exception:
            kappa = None

        try:
            fn = getattr(self.projection, "projected_ops")
            return dict(fn(leaf_id=int(lid), d=int(d), kappa=kappa, step_id=int(step_id)))
        except TypeError:
            try:
                return dict(self.projection.projected_ops(int(lid), int(d), kappa, step_id=int(step_id)))
            except Exception:
                if self.cfg.strict:
                    raise
                return {}
        except Exception:
            if self.cfg.strict:
                raise
            return {}

    def _local_ops_for_leaf(
        self,
        *,
        state: Any,
        fiber: Any,
        lid: int,
        diag: Any,
        step_id: int,
    ) -> Dict[str, np.ndarray]:
        d = int(state.phys_dims[int(lid)])
        key = (self._bucket(step_id), int(lid), int(d))
        cached = self._local_ops_cache.get(key)
        if cached is not None:
            return cached

        try:
            base = dict(fiber.base_operator_basis(d))
        except Exception:
            if self.cfg.strict:
                raise
            base = {}

        _ensure_basic_ops(base, d)
        proj_ops = self._projected_ops_for_leaf(lid=lid, d=d, diag=diag, step_id=step_id)
        base.update(proj_ops)
        _ensure_su2_aliases(base, d)

        ops = _validate_local_ops(base, d, strict=self.cfg.strict)
        self._local_ops_cache[key] = ops
        return ops

    def _local_ops_all(
        self,
        *,
        state: Any,
        fiber: Any,
        diag: Any,
        step_id: int,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for lid in list(state.tree.leaves):
            out[int(lid)] = self._local_ops_for_leaf(state=state, fiber=fiber, lid=int(lid), diag=diag, step_id=step_id)
        return out

    # ------------------------------------------------------------------
    # Flow / coefficient helpers
    # ------------------------------------------------------------------

    def _div_proxy(self, diag: Any, tree: Any) -> np.ndarray:
        leaf_count = len(list(tree.leaves))
        try:
            arr = np.asarray(getattr(diag, "divX_scalar"), dtype=np.float64).reshape(-1)
        except Exception:
            arr = np.zeros((_DEFAULT_NUM_LEAVES,), dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size >= leaf_count:
            return arr
        out = np.zeros((leaf_count,), dtype=np.float64)
        out[: arr.size] = arr
        return out

    def _kappa_map(self, diag: Any) -> Optional[Mapping[int, np.ndarray]]:
        Kmap = getattr(diag, "kappa_node_mats_su2", None)
        if Kmap is not None:
            return Kmap
        return getattr(diag, "kappa_node_mats_raw", None)

    def _weight(self, value: Any) -> complex:
        z = _sanitize_weight(value, max_abs=self.cfg.max_abs_weight, strict=self.cfg.strict)
        if abs(z) <= float(self.cfg.drop_abs_below):
            return 0.0 + 0.0j
        return z

    def _edge_coupling(self, X: Any, u: int, v: int) -> float:
        scale = float(self.cfg.edge_scale) * float(self.cfg.hop_scale)
        if scale == 0.0:
            return 0.0

        u, v = int(u), int(v)
        if not _has_matrix_flow(X):
            tuv = _scalar_from_edge_value(_flow_lookup(X, u, v, 0.0))
            tvu = _scalar_from_edge_value(_flow_lookup(X, v, u, 0.0))
            val = 0.5 * (tuv - tvu)
            out = scale * val
            return float(out) if np.isfinite(out) else 0.0

        Muv = _flow_lookup(X, u, v, None)
        Mvu = _flow_lookup(X, v, u, None)
        if Muv is None and Mvu is None:
            return 0.0
        try:
            if Muv is None:
                D = -np.asarray(Mvu, dtype=np.complex128)
            elif Mvu is None:
                D = np.asarray(Muv, dtype=np.complex128)
            else:
                D = np.asarray(Muv, dtype=np.complex128) - np.asarray(Mvu, dtype=np.complex128)
            if self.decomp is not None and hasattr(self.decomp, "magnitude"):
                mag = float(self.decomp.magnitude(D))
            else:
                mag = float(fro_norm(D))
            out = 0.5 * scale * mag
            return float(out) if np.isfinite(out) else 0.0
        except Exception:
            if self.cfg.strict:
                raise
            return 0.0

    def _rotation_uv(self, step_id: int, u: int, v: int, X: Any) -> np.ndarray:
        b = self._bucket(step_id)
        key = (b, int(u), int(v))
        cached = self._R_cache.get(key)
        if cached is not None:
            return cached

        if self.holonomy is None:
            raise ValueError("holonomy_su2 requires a HolonomyBuilder or HolonomyCache")

        Muv = _flow_lookup(X, int(u), int(v), None)
        _assert(Muv is not None, f"missing matrix flow X({u}->{v}) for holonomy")
        Muv_arr = np.asarray(Muv, dtype=np.complex128)

        try:
            if hasattr(self.holonomy, "rotation_uv"):
                R = self.holonomy.rotation_uv(int(u), int(v), X_uv=Muv_arr, step_id=int(step_id))
            elif hasattr(self.holonomy, "rotation_from_X"):
                R = self.holonomy.rotation_from_X(Muv_arr)
            else:
                raise AttributeError("holonomy object lacks rotation_uv/rotation_from_X")
        except TypeError:
            R = self.holonomy.rotation_from_X(Muv_arr)

        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            if self.cfg.strict:
                raise FloatingPointError("invalid holonomy rotation matrix")
            R = np.eye(3, dtype=np.float64)
        self._R_cache[key] = R
        return R

    # ------------------------------------------------------------------
    # Term construction
    # ------------------------------------------------------------------

    def _add_onsite_terms(
        self,
        *,
        compiled: CompiledTreeOperator,
        local_ops: Dict[int, Dict[str, np.ndarray]],
        state: Any,
        diag: Any,
    ) -> None:
        tree = state.tree
        leaves = list(tree.leaves)
        leaf_pos = _leaf_index_map(tree)
        mode = str(self.cfg.onsite_mode).lower().strip()

        if mode in ("", "none", "off") or float(self.cfg.onsite_scale) == 0.0:
            return

        if mode == "holographic_su2":
            if self.decomp is None:
                if self.cfg.strict:
                    raise ValueError("holographic_su2 requires GeneratorDecomposition")
                return
            Kmap = self._kappa_map(diag)
            if Kmap is None:
                if self.cfg.strict:
                    raise ValueError("holographic_su2 requires kappa matrices in diagnostics")
                return

            for lid in leaves:
                lid = int(lid)
                try:
                    coeffs = np.asarray(self.decomp.decompose_vec3(Kmap[lid]), dtype=np.float64).reshape(3)
                except Exception:
                    if self.cfg.strict:
                        raise
                    coeffs = np.zeros((3,), dtype=np.float64)
                for opname, coeff in zip(("Lx", "Ly", "Lz"), coeffs):
                    w = self._weight(float(self.cfg.onsite_scale) * float(coeff))
                    if w == 0:
                        continue
                    A = _basis_op(local_ops[lid], opname, d=int(state.phys_dims[lid]), strict=self.cfg.strict)
                    A = _maybe_hermitianize(A, enabled=self.cfg.symmetrize_local_ops)
                    compiled.add_onsite(OnsiteOp(lid, w, A, opname=opname, source="onsite:holographic_su2"))
            return

        # Default scalar z-field from divergence proxy.
        zfield = self._div_proxy(diag, tree)
        for lid in leaves:
            lid = int(lid)
            idx = int(leaf_pos.get(lid, lid))
            coeff = 0.5 * float(zfield[idx]) if 0 <= idx < zfield.size else 0.0
            w = self._weight(float(self.cfg.onsite_scale) * coeff)
            if w == 0:
                continue
            A = _basis_op(local_ops[lid], "Z", d=int(state.phys_dims[lid]), strict=self.cfg.strict)
            A = _maybe_hermitianize(A, enabled=self.cfg.symmetrize_local_ops)
            compiled.add_onsite(OnsiteOp(lid, w, A, opname="Z", source="onsite:zfield"))

    def _add_vibration_terms(
        self,
        *,
        compiled: CompiledTreeOperator,
        local_ops: Dict[int, Dict[str, np.ndarray]],
        state: Any,
        vib: Any,
    ) -> None:
        if float(self.cfg.vib_scale) == 0.0:
            return
        w_eff = _safe_frequency_mean(vib)
        if w_eff == 0.0:
            return
        opname = str(self.cfg.vib_op)
        for lid in list(state.tree.leaves):
            lid = int(lid)
            w = self._weight(float(self.cfg.vib_scale) * w_eff)
            if w == 0:
                continue
            A = _basis_op(local_ops[lid], opname, d=int(state.phys_dims[lid]), strict=self.cfg.strict)
            A = _maybe_hermitianize(A, enabled=self.cfg.symmetrize_local_ops)
            compiled.add_onsite(OnsiteOp(lid, w, A, opname=opname, source="onsite:vibration"))

    def _add_extra_terms(
        self,
        *,
        compiled: CompiledTreeOperator,
        local_ops: Dict[int, Dict[str, np.ndarray]],
        state: Any,
    ) -> None:
        if float(self.cfg.extra_scale) == 0.0:
            return
        opname = str(self.cfg.extra_op)
        for lid in list(state.tree.leaves):
            lid = int(lid)
            w = self._weight(float(self.cfg.extra_scale))
            if w == 0:
                continue
            A = _basis_op(local_ops[lid], opname, d=int(state.phys_dims[lid]), strict=self.cfg.strict)
            A = _maybe_hermitianize(A, enabled=self.cfg.symmetrize_local_ops)
            compiled.add_onsite(OnsiteOp(lid, w, A, opname=opname, source="onsite:extra"))

    def _register_custom_local_op(self, local_ops: Dict[int, Dict[str, np.ndarray]], lid: int, label: str, A: np.ndarray) -> str:
        local_ops[int(lid)][str(label)] = _sanitize_complex_matrix(A, name=label, strict=self.cfg.strict)
        return str(label)

    def _add_pair_terms(
        self,
        *,
        compiled: CompiledTreeOperator,
        local_ops: Dict[int, Dict[str, np.ndarray]],
        state: Any,
        X: Any,
        step_id: int,
    ) -> None:
        mode = str(self.cfg.edge_mode).lower().strip()
        if mode in ("", "none", "off") or float(self.cfg.edge_scale) == 0.0:
            return

        leaves = set(int(x) for x in list(state.tree.leaves))
        for u0, v0 in self._edges:
            if int(u0) not in leaves or int(v0) not in leaves:
                continue
            u, v = (int(u0), int(v0)) if int(u0) < int(v0) else (int(v0), int(u0))
            Juv = self._edge_coupling(X, u, v)
            w = self._weight(Juv)
            if w == 0:
                continue

            try:
                lca, path_u, path_v = _lca_and_paths(state.tree, u, v)
            except Exception:
                if self.cfg.strict:
                    raise
                lca, path_u, path_v = None, [], []  # type: ignore[assignment]

            if mode == "holonomy_su2":
                if not _has_matrix_flow(X):
                    if self.cfg.strict:
                        raise ValueError("holonomy_su2 requires matrix-valued flow")
                    # Safe degradation: fall through to heisenberg_su2.
                    mode_eff = "heisenberg_su2"
                else:
                    mode_eff = mode
            else:
                mode_eff = mode

            if mode_eff == "holonomy_su2":
                R = self._rotation_uv(step_id, u, v, X)
                ops_u = local_ops[u]
                ops_v = local_ops[v]
                Lu = [_basis_op(ops_u, op, d=int(state.phys_dims[u]), strict=self.cfg.strict) for op in ("Lx", "Ly", "Lz")]
                Lv = [_basis_op(ops_v, op, d=int(state.phys_dims[v]), strict=self.cfg.strict) for op in ("Lx", "Ly", "Lz")]
                Lu = [_maybe_hermitianize(A, enabled=self.cfg.symmetrize_local_ops) for A in Lu]
                Lv = [_maybe_hermitianize(A, enabled=self.cfg.symmetrize_local_ops) for A in Lv]
                for i, opname_u in enumerate(("Lx", "Ly", "Lz")):
                    Bv = (R[i, 0] * Lv[0] + R[i, 1] * Lv[1] + R[i, 2] * Lv[2]).astype(np.complex128)
                    label_v = self._register_custom_local_op(local_ops, v, _op_label("hol", step_id, u, v, i), Bv)
                    compiled.add_pair(
                        PairOp(
                            u=u,
                            v=v,
                            weight=w,
                            Au=Lu[i],
                            Bv=Bv,
                            opname_u=opname_u,
                            opname_v=label_v,
                            source="edge:holonomy_su2",
                            lca=lca,
                            path_u_to_lca=path_u,
                            path_v_to_lca=path_v,
                        )
                    )
                continue

            if mode_eff == "heisenberg_su2":
                for op in ("Lx", "Ly", "Lz"):
                    Au = _basis_op(local_ops[u], op, d=int(state.phys_dims[u]), strict=self.cfg.strict)
                    Bv = _basis_op(local_ops[v], op, d=int(state.phys_dims[v]), strict=self.cfg.strict)
                    Au = _maybe_hermitianize(Au, enabled=self.cfg.symmetrize_local_ops)
                    Bv = _maybe_hermitianize(Bv, enabled=self.cfg.symmetrize_local_ops)
                    compiled.add_pair(
                        PairOp(
                            u=u,
                            v=v,
                            weight=w,
                            Au=Au,
                            Bv=Bv,
                            opname_u=op,
                            opname_v=op,
                            source="edge:heisenberg_su2",
                            lca=lca,
                            path_u_to_lca=path_u,
                            path_v_to_lca=path_v,
                        )
                    )
                continue

            # Default ZZ coupling.
            Au = _basis_op(local_ops[u], "Z", d=int(state.phys_dims[u]), strict=self.cfg.strict)
            Bv = _basis_op(local_ops[v], "Z", d=int(state.phys_dims[v]), strict=self.cfg.strict)
            Au = _maybe_hermitianize(Au, enabled=self.cfg.symmetrize_local_ops)
            Bv = _maybe_hermitianize(Bv, enabled=self.cfg.symmetrize_local_ops)
            compiled.add_pair(
                PairOp(
                    u=u,
                    v=v,
                    weight=w,
                    Au=Au,
                    Bv=Bv,
                    opname_u="Z",
                    opname_v="Z",
                    source="edge:zz",
                    lca=lca,
                    path_u_to_lca=path_u,
                    path_v_to_lca=path_v,
                )
            )

    # ------------------------------------------------------------------
    # Public build APIs
    # ------------------------------------------------------------------

    def build_compiled(
        self,
        state: Any,
        fiber: Any,
        X: Any,
        vib: Optional[Any],
        diag: Any,
        step_id: int,
    ) -> Tuple[CompiledTreeOperator, Dict[int, Dict[str, np.ndarray]]]:
        """Build the Phase-4 compiled operator representation."""
        _assert(state is not None and hasattr(state, "tree"), "state must be a TTNState-like object")
        _assert(hasattr(state.tree, "leaves"), "state.tree must expose leaves")

        local_ops = self._local_ops_all(state=state, fiber=fiber, diag=diag, step_id=int(step_id))
        compiled = CompiledTreeOperator(
            tree=state.tree,
            metadata={
                "builder": "TreeMPOBuilder",
                "step_id": int(step_id),
                "cfg": _json_safe(self.cfg),
                "edge_count": int(len(self._edges)),
            },
        )

        self._add_onsite_terms(compiled=compiled, local_ops=local_ops, state=state, diag=diag)
        self._add_vibration_terms(compiled=compiled, local_ops=local_ops, state=state, vib=vib)
        self._add_extra_terms(compiled=compiled, local_ops=local_ops, state=state)
        self._add_pair_terms(compiled=compiled, local_ops=local_ops, state=state, X=X, step_id=int(step_id))

        compiled.rebuild_lca_order()
        compiled.validate()
        self._last_summary = compiled.snapshot()
        return compiled, local_ops

    def build_compiled_operator(
        self,
        *,
        state: Any,
        fiber: Any,
        X: Any,
        vib: Optional[Any],
        diag: Any,
        step_id: int,
        **_: Any,
    ) -> Tuple[CompiledTreeOperator, Dict[int, Dict[str, np.ndarray]]]:
        """Keyword-compatible alias used by evolve.py."""
        return self.build_compiled(state, fiber, X, vib, diag, int(step_id))

    def build(
        self,
        state: Any,
        fiber: Any,
        X: Any,
        vib: Optional[Any],
        diag: Any,
        step_id: int,
    ) -> Tuple[TreeMPO, Dict[int, Dict[str, np.ndarray]]]:
        """
        Build the backward-compatible TreeMPO scaffold representation.

        The returned local_ops includes any generated concrete holonomy operators
        under deterministic private labels.  Legacy apply paths can therefore
        resolve both named basis operators and generated Bv overrides.
        """
        compiled, local_ops = self.build_compiled(state, fiber, X, vib, diag, int(step_id))

        leaf_term_leaf: List[int] = []
        leaf_term_opname: List[str] = []
        leaf_weights: List[complex] = []

        for lid in sorted(compiled.onsite_by_leaf.keys()):
            for t in compiled.onsite_by_leaf[lid]:
                # Ensure concrete operator is resolvable by legacy apply even if it
                # came from a generated/adaptive source.
                if t.opname not in local_ops[int(lid)]:
                    local_ops[int(lid)][t.opname] = t.op
                leaf_term_leaf.append(int(t.leaf_id))
                leaf_term_opname.append(str(t.opname))
                leaf_weights.append(complex(t.weight))

        edge_term_uv: List[Tuple[int, int]] = []
        edge_term_op_u: List[str] = []
        edge_term_op_v: List[Optional[str]] = []
        edge_term_Bv: List[Optional[np.ndarray]] = []
        edge_weights: List[complex] = []

        for key in sorted(compiled.pair_by_edge.keys()):
            for t in compiled.pair_by_edge[key]:
                if t.opname_u not in local_ops[int(t.u)]:
                    local_ops[int(t.u)][t.opname_u] = t.Au
                edge_term_uv.append((int(t.u), int(t.v)))
                edge_term_op_u.append(str(t.opname_u))

                # Preserve direct concrete Bv for holonomy or generated operators.
                edge_term_op_v.append(None)
                edge_term_Bv.append(t.Bv.astype(np.complex128, copy=False))
                edge_weights.append(complex(t.weight))

        W = int(1 + len(leaf_weights) + len(edge_weights))
        weights = np.zeros((W,), dtype=np.complex128)
        for k, w in enumerate(leaf_weights):
            weights[1 + k] = np.complex128(w)
        offset = 1 + len(leaf_weights)
        for e, w in enumerate(edge_weights):
            weights[offset + e] = np.complex128(w)

        H = TreeMPO(
            tree=state.tree,
            W=W,
            weights=weights,
            leaf_term_leaf=leaf_term_leaf,
            leaf_term_opname=leaf_term_opname,
            edge_term_uv=edge_term_uv,
            edge_term_op_u=edge_term_op_u,
            edge_term_op_v=edge_term_op_v,
            edge_term_Bv=edge_term_Bv,
            metadata={"compiled_snapshot": compiled.snapshot(), "source": "TreeMPOBuilder.build"},
        )
        H.validate()
        self._last_summary = H.snapshot()
        return H, local_ops

    def build_split_operators(
        self,
        *,
        state: Any,
        fiber: Any,
        X: Any,
        vib: Optional[Any],
        diag: Any,
        step_id: int,
        grouping: str = "lca_routed",
        **_: Any,
    ) -> List[Tuple[CompiledTreeOperator, Dict[int, Dict[str, np.ndarray]]]]:
        """
        Build operator pieces for splitting methods.

        grouping:
            - "onsite_first": onsite piece then all pair terms.
            - "edge_grouped": onsite piece then one piece per graph edge group.
            - "lca_routed": onsite piece then one piece per LCA, deeper first.
            - anything else: a single all-in-one compiled operator.
        """
        compiled, local_ops = self.build_compiled(state, fiber, X, vib, diag, int(step_id))
        mode = str(grouping or "lca_routed").lower().strip()

        pieces: List[CompiledTreeOperator] = []
        onsite_piece = compiled.split(onsite=True, pairs=False)
        if not onsite_piece.is_empty():
            onsite_piece.metadata["split"] = "onsite"
            pieces.append(onsite_piece)

        if mode == "onsite_first":
            pair_piece = compiled.split(onsite=False, pairs=True)
            if not pair_piece.is_empty():
                pair_piece.metadata["split"] = "pairs"
                pieces.append(pair_piece)
        elif mode == "edge_grouped":
            for key in sorted(compiled.pair_by_edge.keys()):
                piece = CompiledTreeOperator(tree=compiled.tree, metadata={**dict(compiled.metadata), "split": f"edge:{key[0]}-{key[1]}"})
                for t in compiled.pair_by_edge[key]:
                    piece.add_pair(t)
                piece.rebuild_lca_order()
                if not piece.is_empty():
                    pieces.append(piece)
        elif mode == "lca_routed":
            pieces.extend(compiled.split_by_lca())
        else:
            return [(compiled, local_ops)]

        if not pieces:
            return [(compiled, local_ops)]
        return [(p, local_ops) for p in pieces]


__all__ = [
    "HamiltonianBuildConfig",
    "TreeMPO",
    "OnsiteOp",
    "PairOp",
    "CompiledTreeOperator",
    "TreeMPOBuilder",
]
