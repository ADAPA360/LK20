#!/usr/bin/env python3
# apply.py
"""
Apply AtomTN Hamiltonian/operator containers to a TTNState.

Production role
---------------
This module provides two application paths used by the AtomTN runtime family:

1. apply_tree_mpo_fast(...)
   Backward-compatible scaffold path.  This is intentionally approximate and is
   retained for speed probes, legacy demos, and quick debugging.

2. apply_compiled_operator_zipup(...)
   Correctness-oriented Phase-4 baseline.  It applies each Hamiltonian term as a
   product of concrete leaf operators, accumulates the term-states by exact TTN
   direct-sum, and truncates/canonicalizes bonds to keep the representation
   bounded.

Tensor convention
-----------------
The implementation follows TTNState's storage convention:

- leaf tensor:     (d_leaf, b_parent)
- internal tensor: (b_child0, ..., b_child{m-1}, b_parent)
- root tensor:     last parent-bond dimension is 1

The zip-up path is deliberately conservative: if callers do not request a
truncation rank/tolerance, a safe default rank is selected.  Unbounded direct-sum
accumulation is available only by explicitly passing a cfg object with
``allow_unbounded_direct_sum=True``.

Dependencies
------------
- numpy
- math_utils.py
- ttn_state.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from math_utils import _assert, fro_norm
from ttn_state import TTNState


_EPS = 1e-12


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


def _sanitize_complex_matrix(A: Any, *, name: str = "operator", strict: bool = True) -> np.ndarray:
    try:
        M = np.asarray(A, dtype=np.complex128)
    except Exception as exc:
        if strict:
            raise TypeError(f"{name} could not be converted to complex matrix: {exc!r}") from exc
        return np.eye(1, dtype=np.complex128)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        if strict:
            raise ValueError(f"{name} must be square; got shape {M.shape}")
        d = int(M.shape[0]) if M.ndim >= 1 and int(M.shape[0]) > 0 else 1
        return np.eye(d, dtype=np.complex128)

    if M.size and not np.all(np.isfinite(M)):
        if strict:
            raise FloatingPointError(f"{name} contains non-finite values")
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    return M.astype(np.complex128, copy=False)


def _sanitize_coeff(c: Any, *, strict: bool = True) -> complex:
    try:
        z = complex(c)
    except Exception as exc:
        if strict:
            raise TypeError(f"coefficient could not be converted to complex: {exc!r}") from exc
        z = 0.0 + 0.0j
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        if strict:
            raise FloatingPointError("coefficient contains non-finite values")
        z = 0.0 + 0.0j
    return np.complex128(z).item()


def _node_children(state: TTNState, nid: int) -> List[int]:
    node = state.tree.nodes[int(nid)]
    if hasattr(node, "children"):
        return [int(x) for x in getattr(node, "children")]
    if hasattr(node, "child_ids"):
        return [int(x) for x in getattr(node, "child_ids")]
    return []


def _node_parent(state: TTNState, nid: int) -> Optional[int]:
    node = state.tree.nodes[int(nid)]
    p = getattr(node, "parent", None)
    return None if p is None else int(p)


def _node_is_leaf(state: TTNState, nid: int) -> bool:
    return bool(getattr(state.tree.nodes[int(nid)], "is_leaf", False))


def _post_order(state: TTNState) -> List[int]:
    tree = state.tree
    if hasattr(tree, "post_order") and callable(getattr(tree, "post_order")):
        return [int(x) for x in tree.post_order()]

    root = int(tree.root)
    out: List[int] = []
    seen = set()

    def dfs(u: int) -> None:
        if u in seen:
            return
        seen.add(u)
        for c in _node_children(state, u):
            dfs(c)
        out.append(u)

    dfs(root)
    return out


def _depth_of_tree(tree: Any, nid: int) -> int:
    depth = 0
    cur = int(nid)
    seen = set()
    while cur not in seen:
        seen.add(cur)
        node = tree.nodes[cur]
        p = getattr(node, "parent", None)
        if p is None:
            return int(depth)
        cur = int(p)
        depth += 1
    return int(depth)


def _resolve_local_operator(
    local_ops: Mapping[int, Mapping[str, np.ndarray]],
    lid: int,
    opname: Optional[str],
    matrix_override: Optional[Any] = None,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Resolve a concrete local operator, preferring matrix_override."""
    lid = int(lid)
    if matrix_override is not None:
        return _sanitize_complex_matrix(matrix_override, name=f"operator override for leaf {lid}", strict=strict)

    if opname is None:
        if strict:
            raise ValueError(f"missing operator name for leaf {lid}")
        # Use identity if possible.
        d = 1
        try:
            if lid in local_ops and "I" in local_ops[lid]:
                d = int(np.asarray(local_ops[lid]["I"]).shape[0])
        except Exception:
            pass
        return np.eye(d, dtype=np.complex128)

    ops_l = local_ops.get(lid, None)
    if ops_l is None:
        if strict:
            raise ValueError(f"missing local operator dictionary for leaf {lid}")
        return np.eye(1, dtype=np.complex128)

    A = ops_l.get(str(opname), None)
    if A is None:
        if strict:
            raise ValueError(f"missing local operator '{opname}' for leaf {lid}")
        try:
            d = int(np.asarray(next(iter(ops_l.values()))).shape[0])
        except Exception:
            d = 1
        return np.eye(d, dtype=np.complex128)

    return _sanitize_complex_matrix(A, name=f"local_ops[{lid!r}][{opname!r}]", strict=strict)


def _extract_matrix_from_mapping(t: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in t and t[key] is not None:
            return t[key]
    return None


# =============================================================================
# Legacy scaffold apply
# =============================================================================


def _make_scaffold_output_state(state: TTNState) -> TTNState:
    if hasattr(state, "clone_with_internal_copied_once_and_leaf_zero"):
        return state.clone_with_internal_copied_once_and_leaf_zero()  # type: ignore[attr-defined]
    out = state.clone()
    for lid in out.tree.leaves:
        out.tensors[int(lid)] = np.zeros_like(out.tensors[int(lid)], dtype=np.complex128)
    return out


def apply_tree_mpo_fast(
    H: Any,
    state: TTNState,
    local_ops: Dict[int, Dict[str, np.ndarray]],
) -> TTNState:
    """
    Fast legacy scaffold application.

    This is not a mathematically exact TTN/MPO apply.  It is retained for legacy
    AtomTN scripts and motor-control-style speed probes.  For correctness, use
    apply_compiled_operator_zipup(...).
    """
    state.validate()
    if hasattr(H, "validate") and callable(getattr(H, "validate")):
        H.validate()

    out = _make_scaffold_output_state(state)
    tensors_in = state.tensors
    tensors_out = out.tensors
    weights = np.asarray(getattr(H, "weights", np.zeros((0,), dtype=np.complex128)), dtype=np.complex128).reshape(-1)

    total_coeff = np.complex128(0.0)
    if weights.size > 1:
        total_coeff = np.sum(weights[1:]).astype(np.complex128)

    # Internal tensors receive the global scaffold coefficient once.
    for nid, node in state.tree.nodes.items():
        nid = int(nid)
        if bool(getattr(node, "is_leaf", False)):
            continue
        tensors_out[nid] = (total_coeff * tensors_in[nid]).astype(np.complex128)

    # Onsite leaf terms.
    n_leaf_terms = int(getattr(H, "n_leaf_terms", 0))
    for k in range(n_leaf_terms):
        widx = int(H.global_id_for_leaf_term(k))
        if widx >= weights.size:
            continue
        coeff = _sanitize_coeff(weights[widx], strict=False)
        if coeff == 0:
            continue
        lid = int(H.leaf_term_leaf[k])
        opname = str(H.leaf_term_opname[k])
        A = _resolve_local_operator(local_ops, lid, opname, strict=True)
        tensors_out[lid] = (tensors_out[lid] + coeff * (A @ tensors_in[lid])).astype(np.complex128)

    # Edge terms: legacy endpoint-add scaffold, not a true two-site apply.
    n_edge_terms = int(getattr(H, "n_edge_terms", 0))
    for e in range(n_edge_terms):
        widx = int(H.global_id_for_edge_term(e))
        if widx >= weights.size:
            continue
        coeff = _sanitize_coeff(weights[widx], strict=False)
        if coeff == 0:
            continue

        u, v = getattr(H, "edge_term_uv")[e]
        u, v = int(u), int(v)
        opu = getattr(H, "edge_term_op_u")[e]
        opv = getattr(H, "edge_term_op_v")[e]
        Bv_override = getattr(H, "edge_term_Bv")[e]

        Au = _resolve_local_operator(local_ops, u, str(opu) if opu is not None else None, strict=True)
        Bv = _resolve_local_operator(local_ops, v, str(opv) if opv is not None else None, Bv_override, strict=True)

        tensors_out[u] = (tensors_out[u] + coeff * (Au @ tensors_in[u])).astype(np.complex128)
        tensors_out[v] = (tensors_out[v] + coeff * (Bv @ tensors_in[v])).astype(np.complex128)

    out.validate()
    return out


# =============================================================================
# Correctness-oriented term extraction
# =============================================================================


@dataclass(frozen=True)
class _TermSpec:
    coeff: complex
    leaf_ops: Dict[int, np.ndarray]
    label: str = ""
    source: str = ""
    lca: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finite(self) -> bool:
        if not (np.isfinite(np.real(self.coeff)) and np.isfinite(np.imag(self.coeff))):
            return False
        return all(np.all(np.isfinite(A)) for A in self.leaf_ops.values())


def _term_from_same_leaf_pair(u: int, Au: np.ndarray, v: int, Bv: np.ndarray) -> Dict[int, np.ndarray]:
    """Return product leaf_ops for a pair term, handling degenerate u == v."""
    u, v = int(u), int(v)
    if u == v:
        # Apply Au then Bv on same physical index: Bv @ Au @ T.
        return {u: (Bv @ Au).astype(np.complex128)}
    return {u: Au.astype(np.complex128), v: Bv.astype(np.complex128)}


def _extract_terms_from_tree_mpo(
    H: Any,
    local_ops: Mapping[int, Mapping[str, np.ndarray]],
    *,
    strict: bool = True,
) -> List[_TermSpec]:
    terms: List[_TermSpec] = []
    weights = np.asarray(getattr(H, "weights", np.zeros((0,), dtype=np.complex128)), dtype=np.complex128).reshape(-1)

    n_leaf_terms = int(getattr(H, "n_leaf_terms", 0))
    for k in range(n_leaf_terms):
        widx = int(H.global_id_for_leaf_term(k))
        if widx >= weights.size:
            continue
        coeff = _sanitize_coeff(weights[widx], strict=strict)
        if coeff == 0:
            continue
        lid = int(H.leaf_term_leaf[k])
        opname = str(H.leaf_term_opname[k])
        A = _resolve_local_operator(local_ops, lid, opname, strict=strict)
        terms.append(_TermSpec(coeff=coeff, leaf_ops={lid: A}, label=f"leaf:{lid}:{opname}", source="TreeMPO.onsite"))

    n_edge_terms = int(getattr(H, "n_edge_terms", 0))
    for e in range(n_edge_terms):
        widx = int(H.global_id_for_edge_term(e))
        if widx >= weights.size:
            continue
        coeff = _sanitize_coeff(weights[widx], strict=strict)
        if coeff == 0:
            continue
        u, v = H.edge_term_uv[e]
        u, v = int(u), int(v)
        opu = H.edge_term_op_u[e]
        opv = H.edge_term_op_v[e]
        Bv_override = H.edge_term_Bv[e]
        Au = _resolve_local_operator(local_ops, u, str(opu) if opu is not None else None, strict=strict)
        Bv = _resolve_local_operator(local_ops, v, str(opv) if opv is not None else None, Bv_override, strict=strict)
        leaf_ops = _term_from_same_leaf_pair(u, Au, v, Bv)
        terms.append(_TermSpec(coeff=coeff, leaf_ops=leaf_ops, label=f"edge:{u}:{v}", source="TreeMPO.pair"))

    return terms


def _extract_terms_from_compiled_dataclass(op: Any, *, strict: bool = True) -> List[_TermSpec]:
    terms: List[_TermSpec] = []

    onsite_by_leaf = getattr(op, "onsite_by_leaf", None)
    if isinstance(onsite_by_leaf, Mapping):
        for lid, arr in onsite_by_leaf.items():
            for t in list(arr):
                leaf_id = int(getattr(t, "leaf_id", lid))
                coeff = _sanitize_coeff(getattr(t, "weight", 1.0), strict=strict)
                if coeff == 0:
                    continue
                A = _sanitize_complex_matrix(getattr(t, "op"), name=f"compiled onsite {leaf_id}", strict=strict)
                terms.append(
                    _TermSpec(
                        coeff=coeff,
                        leaf_ops={leaf_id: A},
                        label=f"compiled:onsite:{leaf_id}:{getattr(t, 'opname', 'A')}",
                        source=str(getattr(t, "source", "compiled.onsite")),
                    )
                )

    pair_by_edge = getattr(op, "pair_by_edge", None)
    if isinstance(pair_by_edge, Mapping):
        for key, arr in pair_by_edge.items():
            for t in list(arr):
                u = int(getattr(t, "u", key[0]))
                v = int(getattr(t, "v", key[1]))
                coeff = _sanitize_coeff(getattr(t, "weight", 1.0), strict=strict)
                if coeff == 0:
                    continue
                Au = _sanitize_complex_matrix(getattr(t, "Au"), name=f"compiled pair {u},{v}.Au", strict=strict)
                Bv = _sanitize_complex_matrix(getattr(t, "Bv"), name=f"compiled pair {u},{v}.Bv", strict=strict)
                leaf_ops = _term_from_same_leaf_pair(u, Au, v, Bv)
                terms.append(
                    _TermSpec(
                        coeff=coeff,
                        leaf_ops=leaf_ops,
                        label=f"compiled:pair:{u}:{v}",
                        source=str(getattr(t, "source", "compiled.pair")),
                        lca=(None if getattr(t, "lca", None) is None else int(getattr(t, "lca"))),
                        metadata={
                            "opname_u": str(getattr(t, "opname_u", "Au")),
                            "opname_v": str(getattr(t, "opname_v", "Bv")),
                            "path_u_to_lca": list(getattr(t, "path_u_to_lca", [])),
                            "path_v_to_lca": list(getattr(t, "path_v_to_lca", [])),
                        },
                    )
                )

    return terms


def _extract_terms_from_compiled_views(
    op: Any,
    local_ops: Mapping[int, Mapping[str, np.ndarray]],
    *,
    strict: bool = True,
) -> List[_TermSpec]:
    """Extract from generic leaf_terms / pair_terms compatibility views."""
    terms: List[_TermSpec] = []

    leaf_terms = getattr(op, "leaf_terms", None)
    if leaf_terms is not None:
        for t in list(leaf_terms):
            if isinstance(t, Mapping):
                lid = int(t.get("leaf", t.get("lid", t.get("leaf_id"))))
                coeff = _sanitize_coeff(t.get("coeff", t.get("weight", 1.0)), strict=strict)
                opname = t.get("opname", t.get("op", t.get("op_name", None)))
                mat = _extract_matrix_from_mapping(t, ("matrix", "op_matrix", "A", "op_mat"))
                source = str(t.get("source", "compiled.leaf_terms"))
            else:
                # tuple-style: (leaf, opname, coeff) or (leaf, matrix, coeff)
                vals = tuple(t)  # type: ignore[arg-type]
                _assert(len(vals) >= 3, "leaf term tuple must contain at least 3 items")
                lid = int(vals[0])
                coeff = _sanitize_coeff(vals[2], strict=strict)
                opname = vals[1] if isinstance(vals[1], str) else None
                mat = vals[1] if not isinstance(vals[1], str) else None
                source = "compiled.leaf_terms"
            if coeff == 0:
                continue
            A = _resolve_local_operator(local_ops, lid, str(opname) if opname is not None else None, mat, strict=strict)
            terms.append(_TermSpec(coeff=coeff, leaf_ops={lid: A}, label=f"leaf:{lid}:{opname or 'matrix'}", source=source))

    pair_terms = getattr(op, "pair_terms", None)
    if pair_terms is not None:
        for t in list(pair_terms):
            if isinstance(t, Mapping):
                u = int(t.get("u"))
                v = int(t.get("v"))
                coeff = _sanitize_coeff(t.get("coeff", t.get("weight", 1.0)), strict=strict)
                opu = t.get("op_u", t.get("opname_u", t.get("opu", None)))
                opv = t.get("op_v", t.get("opname_v", t.get("opv", None)))
                Au_override = _extract_matrix_from_mapping(t, ("Au", "A_u", "matrix_u", "op_u_matrix"))
                Bv_override = _extract_matrix_from_mapping(t, ("Bv", "B_v", "matrix_v", "op_v_matrix"))
                lca = t.get("lca", None)
                source = str(t.get("source", "compiled.pair_terms"))
            else:
                vals = tuple(t)  # type: ignore[arg-type]
                _assert(len(vals) >= 5, "pair term tuple must contain at least 5 items")
                u, v = int(vals[0]), int(vals[1])
                opu = vals[2]
                opv = vals[3]
                coeff = _sanitize_coeff(vals[4], strict=strict)
                Au_override = opu if not isinstance(opu, str) else None
                Bv_override = opv if not isinstance(opv, str) else None
                lca = None
                source = "compiled.pair_terms"
            if coeff == 0:
                continue
            Au = _resolve_local_operator(local_ops, u, str(opu) if isinstance(opu, str) else None, Au_override, strict=strict)
            Bv = _resolve_local_operator(local_ops, v, str(opv) if isinstance(opv, str) else None, Bv_override, strict=strict)
            leaf_ops = _term_from_same_leaf_pair(u, Au, v, Bv)
            terms.append(
                _TermSpec(
                    coeff=coeff,
                    leaf_ops=leaf_ops,
                    label=f"pair:{u}:{v}",
                    source=source,
                    lca=(None if lca is None else int(lca)),
                )
            )

    return terms


def _extract_terms(
    op: Any,
    local_ops: Mapping[int, Mapping[str, np.ndarray]],
    *,
    strict: bool = True,
) -> List[_TermSpec]:
    if hasattr(op, "n_leaf_terms") and hasattr(op, "n_edge_terms") and hasattr(op, "weights"):
        return _extract_terms_from_tree_mpo(op, local_ops, strict=strict)

    terms = _extract_terms_from_compiled_dataclass(op, strict=strict)
    if terms:
        return terms

    terms = _extract_terms_from_compiled_views(op, local_ops, strict=strict)
    return terms


def _filter_and_order_terms(terms: List[_TermSpec], state: TTNState, cfg: Any) -> List[_TermSpec]:
    drop = float(getattr(cfg, "drop_abs_below", 0.0) or 0.0)
    max_terms = getattr(cfg, "max_terms", None)
    strict = bool(getattr(cfg, "strict", True))

    out: List[_TermSpec] = []
    leaves = set(int(x) for x in state.tree.leaves)
    for term in terms:
        if abs(term.coeff) <= drop:
            continue
        if not term.finite():
            if strict:
                raise FloatingPointError(f"non-finite term encountered: {term.label}")
            continue
        unknown = [lid for lid in term.leaf_ops if int(lid) not in leaves]
        if unknown:
            if strict:
                raise ValueError(f"term references unknown leaves: {unknown}")
            continue
        out.append(term)

    grouping = str(getattr(cfg, "apply_grouping", "lca_routed") or "lca_routed").lower().strip()
    if grouping == "onsite_first":
        out.sort(key=lambda t: (len(t.leaf_ops), min(t.leaf_ops.keys()) if t.leaf_ops else -1, t.label))
    elif grouping == "edge_grouped":
        out.sort(key=lambda t: (tuple(sorted(t.leaf_ops.keys())), t.label))
    elif grouping == "lca_routed":
        # Deeper LCA first for two-site terms when metadata is present.
        def key(t: _TermSpec) -> Tuple[int, int, Tuple[int, ...], str]:
            if t.lca is None:
                depth = 10**9 if len(t.leaf_ops) == 1 else 0
            else:
                depth = _depth_of_tree(state.tree, int(t.lca))
            return (-depth, len(t.leaf_ops), tuple(sorted(int(x) for x in t.leaf_ops.keys())), t.label)

        out.sort(key=key)
    else:
        out.sort(key=lambda t: (tuple(sorted(t.leaf_ops.keys())), t.label))

    if max_terms is not None:
        mt = int(max_terms)
        if mt >= 0 and len(out) > mt:
            if strict:
                raise RuntimeError(f"operator contains {len(out)} terms, exceeding max_terms={mt}")
            out = out[:mt]

    return out


# =============================================================================
# Direct-sum zip-up substrate
# =============================================================================


def _apply_leaf_ops_to_state(
    state: TTNState,
    leaf_ops: Mapping[int, np.ndarray],
    *,
    coeff: complex = 1.0,
    coeff_carrier_leaf: Optional[int] = None,
) -> TTNState:
    """Build coeff * (⊗ leaf_ops) |state> by modifying only leaves."""
    if hasattr(state, "apply_product_ops") and callable(getattr(state, "apply_product_ops")):
        return state.apply_product_ops({int(k): v for k, v in leaf_ops.items()}, coeff=coeff)  # type: ignore[return-value]

    out = state.clone()
    for lid, A in leaf_ops.items():
        lid = int(lid)
        out.tensors[lid] = (np.asarray(A, dtype=np.complex128) @ out.tensors[lid]).astype(np.complex128)

    if coeff != 1.0:
        if coeff_carrier_leaf is None:
            keys = sorted(int(k) for k in leaf_ops.keys())
            coeff_carrier_leaf = keys[0] if keys else int(out.tree.root)
        out.tensors[int(coeff_carrier_leaf)] = (complex(coeff) * out.tensors[int(coeff_carrier_leaf)]).astype(np.complex128)
    return out


def _direct_sum_two_states(a: TTNState, b: TTNState) -> TTNState:
    """
    Exact TTN direct-sum representation of a + b.

    Bond dimensions of corresponding non-root nodes are added.  The root parent
    bond remains 1.  Both states must share tree and physical dimensions.
    """
    a.validate()
    b.validate()
    _assert(a.tree is b.tree, "direct_sum: states must share the same tree object")
    _assert(a.phys_dims == b.phys_dims, "direct_sum: physical dimensions mismatch")

    tree = a.tree
    root = int(tree.root)
    out = a.clone()

    out.parent_bond_dims = dict(a.parent_bond_dims)
    for nid in tree.nodes:
        nid = int(nid)
        if nid == root:
            out.parent_bond_dims[nid] = 1
        else:
            out.parent_bond_dims[nid] = int(a.parent_bond_dims[nid]) + int(b.parent_bond_dims[nid])

    for nid, node in tree.nodes.items():
        nid = int(nid)
        Ta = np.asarray(a.tensors[nid], dtype=np.complex128)
        Tb = np.asarray(b.tensors[nid], dtype=np.complex128)

        if bool(getattr(node, "is_leaf", False)):
            _assert(Ta.ndim == 2 and Tb.ndim == 2, "direct_sum: leaf tensors must be rank-2")
            _assert(Ta.shape[0] == Tb.shape[0], "direct_sum: leaf physical dimension mismatch")
            ba, bb = int(Ta.shape[1]), int(Tb.shape[1])
            Tout = np.zeros((int(Ta.shape[0]), ba + bb), dtype=np.complex128)
            Tout[:, :ba] = Ta
            Tout[:, ba:ba + bb] = Tb
            out.tensors[nid] = Tout
            continue

        children = _node_children(a, nid)
        m = len(children)
        _assert(Ta.ndim == m + 1 and Tb.ndim == m + 1, "direct_sum: internal tensor rank mismatch")

        if nid == root:
            _assert(Ta.shape[-1] == 1 and Tb.shape[-1] == 1, "direct_sum: root parent bond must be 1")
            new_shape = [int(Ta.shape[j]) + int(Tb.shape[j]) for j in range(m)] + [1]
            Tout = np.zeros(tuple(new_shape), dtype=np.complex128)
            slic_a = [slice(0, int(Ta.shape[j])) for j in range(m)] + [slice(0, 1)]
            slic_b = [slice(int(Ta.shape[j]), int(Ta.shape[j]) + int(Tb.shape[j])) for j in range(m)] + [slice(0, 1)]
            Tout[tuple(slic_a)] = Ta
            Tout[tuple(slic_b)] = Tb
            out.tensors[nid] = Tout
            continue

        new_shape = [int(Ta.shape[j]) + int(Tb.shape[j]) for j in range(m)] + [int(Ta.shape[-1]) + int(Tb.shape[-1])]
        Tout = np.zeros(tuple(new_shape), dtype=np.complex128)
        slic_a = [slice(0, int(Ta.shape[j])) for j in range(m)] + [slice(0, int(Ta.shape[-1]))]
        slic_b = [slice(int(Ta.shape[j]), int(Ta.shape[j]) + int(Tb.shape[j])) for j in range(m)] + [
            slice(int(Ta.shape[-1]), int(Ta.shape[-1]) + int(Tb.shape[-1]))
        ]
        Tout[tuple(slic_a)] = Ta
        Tout[tuple(slic_b)] = Tb
        out.tensors[nid] = Tout

    out.parent_bond_dims[root] = 1
    out.validate()
    return out


def _choose_safe_default_rank(state: TTNState, cfg: Any = None) -> int:
    """Conservative default rank for direct-sum zip-up."""
    try:
        cur = int(max(int(x) for x in state.parent_bond_dims.values()))
    except Exception:
        cur = 8
    cap = int(getattr(cfg, "safe_default_rank_cap", 32) or 32)
    floor = int(getattr(cfg, "safe_default_rank_floor", 4) or 4)
    # Direct-sum apply is expensive on arity-4 trees; avoid uncontrolled growth.
    return int(min(max(floor, 2 * max(1, cur)), max(floor, cap)))


def _rank_tol_from_cfg(state: TTNState, cfg: Any) -> Tuple[Optional[int], Optional[float], bool]:
    rank = getattr(cfg, "apply_truncate_rank", None)
    tol = getattr(cfg, "apply_truncate_tol", None)
    allow_unbounded = bool(getattr(cfg, "allow_unbounded_direct_sum", False))

    rank_i = None if rank is None else int(max(1, int(rank)))
    tol_f = None if tol is None else float(tol)

    used_default = False
    if rank_i is None and tol_f is None and not allow_unbounded:
        rank_i = _choose_safe_default_rank(state, cfg)
        used_default = True
    return rank_i, tol_f, used_default


def _truncate_all_parent_bonds(state: TTNState, rank: Optional[int], tol: Optional[float]) -> None:
    if rank is None and tol is None:
        return
    if not hasattr(state, "truncate_parent_bond_svd"):
        return
    root = int(state.tree.root)
    for nid in _post_order(state):
        nid = int(nid)
        if nid == root:
            continue
        state.truncate_parent_bond_svd(nid, rank=rank, tol=tol)  # type: ignore[attr-defined]
    state.validate()


def _maybe_canonicalize(state: TTNState, *, canonicalize_every: int, step_id: int, term_index: Optional[int] = None) -> None:
    every = int(canonicalize_every or 0)
    if every <= 0:
        return
    # Preserve older semantics: step cadence.  If term_index is provided, also
    # canonicalize periodically inside long applications.
    do_step = ((int(step_id) + 1) % every) == 0
    do_term = term_index is not None and ((int(term_index) + 1) % max(4, every)) == 0
    if do_step or do_term:
        fn = getattr(state, "canonicalize_upward_qr", None)
        if callable(fn):
            try:
                fn()
            except TypeError:
                fn(min_rank=1)
            state.validate()


def _accumulate_terms_zipup(
    state: TTNState,
    terms: Sequence[_TermSpec],
    *,
    rank: Optional[int],
    tol: Optional[float],
    cfg: Any,
    step_id: int,
) -> TTNState:
    if len(terms) == 0:
        return state.zero_like() if hasattr(state, "zero_like") else _zero_like_state(state)

    canonicalize_every = int(getattr(cfg, "canonicalize_every", 0) or 0)
    truncate_every_terms = int(getattr(cfg, "truncate_every_terms", 1) or 1)
    normalize_intermediate = bool(getattr(cfg, "normalize_intermediate", False))

    acc: Optional[TTNState] = None
    for idx, term in enumerate(terms):
        carrier = int(sorted(term.leaf_ops.keys())[0]) if term.leaf_ops else None
        term_state = _apply_leaf_ops_to_state(state, term.leaf_ops, coeff=term.coeff, coeff_carrier_leaf=carrier)

        if acc is None:
            acc = term_state
            if rank is not None or tol is not None:
                _truncate_all_parent_bonds(acc, rank=rank, tol=tol)
        else:
            acc = _direct_sum_two_states(acc, term_state)
            if truncate_every_terms <= 1 or ((idx + 1) % truncate_every_terms == 0):
                _truncate_all_parent_bonds(acc, rank=rank, tol=tol)

        _maybe_canonicalize(acc, canonicalize_every=canonicalize_every, step_id=int(step_id), term_index=idx)

        if normalize_intermediate and hasattr(acc, "normalize_in_place"):
            acc.normalize_in_place()

    _assert(acc is not None, "internal error: accumulator is None after non-empty terms")
    _truncate_all_parent_bonds(acc, rank=rank, tol=tol)
    _maybe_canonicalize(acc, canonicalize_every=canonicalize_every, step_id=int(step_id), term_index=None)
    acc.validate()
    return acc


def _zero_like_state(state: TTNState) -> TTNState:
    out = state.clone()
    for nid in out.tensors:
        out.tensors[nid] = np.zeros_like(out.tensors[nid], dtype=np.complex128)
    return out


# =============================================================================
# Public Phase-4 apply API
# =============================================================================


def apply_compiled_operator_zipup(
    *,
    state: TTNState,
    op: Any,
    local_ops: Dict[int, Dict[str, np.ndarray]],
    cfg: Any,
    step_id: int,
) -> TTNState:
    """
    Apply a TreeMPO or CompiledTreeOperator to state using direct-sum zip-up.

    Parameters
    ----------
    state:
        Input TTNState.
    op:
        TreeMPO-like or CompiledTreeOperator-like object.
    local_ops:
        leaf -> operator-name -> matrix map. Used for legacy TreeMPO and as a
        fallback for compatibility views.
    cfg:
        ApplyConfig-like object with optional fields:
        apply_truncate_rank, apply_truncate_tol, canonicalize_every,
        apply_grouping, allow_unbounded_direct_sum, max_terms, drop_abs_below.
    step_id:
        Integrator step or cache bucket id.

    Returns
    -------
    TTNState
        Approximate TTN representation of H|psi>.
    """
    state.validate()
    if hasattr(op, "validate") and callable(getattr(op, "validate")):
        op.validate()

    strict = bool(getattr(cfg, "strict", True))
    rank, tol, used_default_rank = _rank_tol_from_cfg(state, cfg)
    raw_terms = _extract_terms(op, local_ops, strict=strict)
    terms = _filter_and_order_terms(raw_terms, state, cfg)

    out = _accumulate_terms_zipup(state, terms, rank=rank, tol=tol, cfg=cfg, step_id=int(step_id))

    # Attach lightweight diagnostics for downstream health inspection without
    # changing the TTNState public API.  ttn_state.py keeps metadata in the
    # production version, but this guard preserves compatibility with older ones.
    try:
        meta = getattr(out, "metadata", None)
        if isinstance(meta, MutableMapping):
            meta["last_apply"] = {
                "mode": "zipup_direct_sum",
                "term_count": int(len(terms)),
                "raw_term_count": int(len(raw_terms)),
                "rank": None if rank is None else int(rank),
                "tol": None if tol is None else float(tol),
                "used_default_rank": bool(used_default_rank),
                "step_id": int(step_id),
                "grouping": str(getattr(cfg, "apply_grouping", "lca_routed")),
            }
    except Exception:
        pass

    out.validate()
    return out


def apply_operator(
    *,
    state: TTNState,
    op: Any,
    local_ops: Dict[int, Dict[str, np.ndarray]],
    cfg: Any,
    step_id: int = 0,
    prefer_zipup: bool = True,
) -> TTNState:
    """Convenience dispatcher used by tests and external callers."""
    if prefer_zipup:
        return apply_compiled_operator_zipup(state=state, op=op, local_ops=local_ops, cfg=cfg, step_id=step_id)
    return apply_tree_mpo_fast(op, state, local_ops)


# =============================================================================
# Introspection helpers
# =============================================================================


def inspect_operator_terms(op: Any, local_ops: Dict[int, Dict[str, np.ndarray]], *, strict: bool = False) -> Dict[str, Any]:
    """Return term-count and weight diagnostics without applying the operator."""
    terms = _extract_terms(op, local_ops, strict=strict)
    weights = np.asarray([abs(complex(t.coeff)) for t in terms], dtype=np.float64)
    supports = [len(t.leaf_ops) for t in terms]
    return {
        "term_count": int(len(terms)),
        "onsite_like_terms": int(sum(1 for s in supports if s <= 1)),
        "pair_like_terms": int(sum(1 for s in supports if s == 2)),
        "higher_support_terms": int(sum(1 for s in supports if s > 2)),
        "weight_abs_min": float(np.min(weights)) if weights.size else 0.0,
        "weight_abs_mean": float(np.mean(weights)) if weights.size else 0.0,
        "weight_abs_max": float(np.max(weights)) if weights.size else 0.0,
        "sources": sorted(set(str(t.source) for t in terms)),
    }


def estimate_zipup_growth(state: TTNState, term_count: int, *, rank: Optional[int] = None) -> Dict[str, Any]:
    """Rough direct-sum growth estimate for planning/troubleshooting."""
    cur = int(max(state.parent_bond_dims.values())) if state.parent_bond_dims else 1
    unbounded = int(cur * max(1, int(term_count)))
    capped = int(rank) if rank is not None else _choose_safe_default_rank(state)
    return {
        "current_max_bond": int(cur),
        "term_count": int(term_count),
        "unbounded_max_bond_after_sum": int(unbounded),
        "recommended_rank": int(capped),
        "direct_sum_requires_truncation": bool(term_count > 1),
    }


__all__ = [
    "apply_tree_mpo_fast",
    "apply_compiled_operator_zipup",
    "apply_operator",
    "inspect_operator_terms",
    "estimate_zipup_growth",
]
