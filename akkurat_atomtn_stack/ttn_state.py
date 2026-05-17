#!/usr/bin/env python3
# ttn_state.py
"""
TTNState — production TTN state container and contraction substrate for AtomTN.

This module provides the low-level tensor-state operations used by the AtomTN
runtime family:

- validated Tree Tensor Network state storage
- deterministic random initialization
- bottom-up bond reduced density matrices
- top-down bond environments
- local leaf-operator application
- upward QR canonicalization
- parent-bond SVD truncation / fixed-rank projection
- leaf physical-dimension adaptation
- cloning, zero-structure creation, diagnostics, and lightweight serialization

Tensor convention
-----------------
Leaf tensor:
    T_leaf.shape == (d_leaf, b_parent)

Internal tensor:
    T_node.shape == (b_child0, b_child1, ..., b_child{m-1}, b_parent)

Root tensor:
    parent bond dimension is always 1.

This module is NumPy-only and intentionally contains no Hamiltonian, flow, or
policy logic. Those responsibilities live in hamiltonian.py, apply.py, evolve.py,
atom.py, and the higher-level runtime adapters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from math_utils import _assert, fro_norm, svd_truncate


# =============================================================================
# Generic helpers
# =============================================================================

_EPS = 1e-12


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return int(p)


def _node_children(tree: Any, nid: int) -> List[int]:
    node = tree.nodes[int(nid)]
    if hasattr(node, "children"):
        return [int(x) for x in getattr(node, "children")]
    if hasattr(node, "child_ids"):
        return [int(x) for x in getattr(node, "child_ids")]
    return []


def _node_parent(tree: Any, nid: int) -> Optional[int]:
    node = tree.nodes[int(nid)]
    p = getattr(node, "parent", None)
    return None if p is None else int(p)


def _node_is_leaf(tree: Any, nid: int) -> bool:
    return bool(getattr(tree.nodes[int(nid)], "is_leaf", False))


def _post_order(tree: Any) -> List[int]:
    if hasattr(tree, "post_order") and callable(getattr(tree, "post_order")):
        return [int(x) for x in tree.post_order()]

    root = int(tree.root)
    out: List[int] = []
    seen = set()

    def dfs(u: int) -> None:
        if u in seen:
            return
        seen.add(u)
        for c in _node_children(tree, u):
            dfs(c)
        out.append(u)

    dfs(root)
    return out


def _topological_order(tree: Any) -> List[int]:
    root = int(tree.root)
    out: List[int] = []
    q = [root]
    seen = set()
    while q:
        u = int(q.pop(0))
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        q.extend(_node_children(tree, u))
    return out


def _sanitize_tensor(x: Any, *, dtype: np.dtype = np.complex128, copy: bool = True) -> np.ndarray:
    arr = np.asarray(x, dtype=np.dtype(dtype))
    if copy:
        arr = arr.copy()
    if arr.size:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.dtype(dtype), copy=False)
    return arr


def _complex_to_json(arr: np.ndarray) -> Dict[str, Any]:
    z = np.asarray(arr, dtype=np.complex128)
    return {
        "shape": list(map(int, z.shape)),
        "real": np.real(z).astype(float).reshape(-1).tolist(),
        "imag": np.imag(z).astype(float).reshape(-1).tolist(),
    }


def _complex_from_json(payload: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(x) for x in payload.get("shape", []))
    real = np.asarray(payload.get("real", []), dtype=np.float64)
    imag = np.asarray(payload.get("imag", []), dtype=np.float64)
    if real.size != imag.size:
        raise ValueError("complex JSON payload real/imag size mismatch")
    arr = (real + 1j * imag).astype(np.complex128)
    return arr.reshape(shape)


# =============================================================================
# Reshape / fuse utilities
# =============================================================================


def reshape_as_matrix(
    T: np.ndarray,
    left_axes: Sequence[int],
    right_axes: Sequence[int],
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    """
    Permute T so left_axes come first and right_axes last, then reshape to 2D.

    Returns:
        M, left_shape, right_shape
    """
    A = np.asarray(T)
    la = tuple(int(a) for a in left_axes)
    ra = tuple(int(a) for a in right_axes)
    _assert(len(set(la + ra)) == len(la) + len(ra), "reshape_as_matrix: overlapping axes")
    _assert(set(la + ra) == set(range(A.ndim)), "reshape_as_matrix: axes must cover all tensor axes")

    perm = list(la) + list(ra)
    Tp = np.transpose(A, perm)
    left_shape = tuple(int(A.shape[a]) for a in la)
    right_shape = tuple(int(A.shape[a]) for a in ra)
    M = Tp.reshape(_prod(left_shape), _prod(right_shape))
    return M, left_shape, right_shape


def unreshape_from_matrix(
    M: np.ndarray,
    left_shape: Tuple[int, ...],
    right_shape: Tuple[int, ...],
    left_axes: Sequence[int],
    right_axes: Sequence[int],
) -> np.ndarray:
    """Inverse of reshape_as_matrix."""
    la = tuple(int(a) for a in left_axes)
    ra = tuple(int(a) for a in right_axes)
    perm = list(la) + list(ra)
    inv = np.argsort(perm)
    Tp = np.asarray(M).reshape(*left_shape, *right_shape)
    return np.transpose(Tp, inv)


def fuse_axes(T: np.ndarray, axes: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    """
    Fuse selected axes into one trailing axis.

    Returns:
        fused_tensor, original_shape, axes_tuple

    This helper is primarily for debugging and experimental contractions; core
    TTN routines use reshape_as_matrix for clearer bipartitions.
    """
    A = np.asarray(T)
    original_shape = tuple(int(x) for x in A.shape)
    axes_tuple = tuple(int(a) for a in axes)
    _assert(len(set(axes_tuple)) == len(axes_tuple), "fuse_axes: duplicate axes")
    _assert(all(0 <= a < A.ndim for a in axes_tuple), "fuse_axes: axis out of range")
    keep_axes = tuple(i for i in range(A.ndim) if i not in axes_tuple)
    perm = list(keep_axes) + list(axes_tuple)
    fused_shape = tuple(original_shape[i] for i in keep_axes) + (_prod([original_shape[i] for i in axes_tuple]),)
    return np.transpose(A, perm).reshape(fused_shape), original_shape, axes_tuple


# =============================================================================
# TTNState
# =============================================================================

@dataclass
class TTNState:
    tree: Any
    tensors: Dict[int, np.ndarray]
    phys_dims: Dict[int, int]
    parent_bond_dims: Dict[int, int]

    metadata: Dict[str, Any] = field(default_factory=dict)
    strict_finite: bool = True

    # ------------------------------------------------------------------
    # Construction and validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.tensors = {int(k): _sanitize_tensor(v, dtype=np.complex128, copy=False) for k, v in dict(self.tensors).items()}
        self.phys_dims = {int(k): int(v) for k, v in dict(self.phys_dims).items()}
        self.parent_bond_dims = {int(k): int(v) for k, v in dict(self.parent_bond_dims).items()}
        self.metadata = dict(self.metadata or {})

    @staticmethod
    def random(
        tree: Any,
        phys_dims_leaf: Mapping[int, int],
        bond_dim: Union[int, Mapping[int, int]] = 4,
        seed: int = 0,
        *,
        dtype: np.dtype = np.complex128,
        normalize: bool = True,
        init_scale: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TTNState":
        """Create a deterministic random TTNState for the given tree."""
        if hasattr(tree, "validate") and callable(getattr(tree, "validate")):
            tree.validate()

        rng = np.random.default_rng(int(seed))
        dt = np.dtype(dtype)
        root = int(tree.root)

        if isinstance(bond_dim, Mapping):
            parent_bond_dims = {int(nid): int(bond_dim.get(nid, 4)) for nid in tree.nodes}
        else:
            parent_bond_dims = {int(nid): int(bond_dim) for nid in tree.nodes}
        parent_bond_dims[root] = 1
        for nid, b in list(parent_bond_dims.items()):
            parent_bond_dims[nid] = int(max(1, b))

        tensors: Dict[int, np.ndarray] = {}
        phys_dims = {int(k): int(v) for k, v in dict(phys_dims_leaf).items()}

        # Leaves.
        for lid in tree.leaves:
            lid = int(lid)
            _assert(lid in phys_dims, f"TTNState.random: missing physical dimension for leaf {lid}")
            d = int(max(1, phys_dims[lid]))
            b = int(parent_bond_dims[lid])
            scale = float(init_scale) / np.sqrt(max(1, d * b))
            T = (rng.normal(size=(d, b)) + 1j * rng.normal(size=(d, b))) * scale
            tensors[lid] = T.astype(dt, copy=False).astype(np.complex128, copy=False)

        # Internal nodes bottom-up.
        for nid in _post_order(tree):
            if _node_is_leaf(tree, nid):
                continue
            children = _node_children(tree, nid)
            child_bonds = [int(parent_bond_dims[c]) for c in children]
            bp = int(parent_bond_dims[nid])
            shape = tuple(child_bonds + [bp])
            scale = float(init_scale) / np.sqrt(max(1, _prod(shape)))
            T = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * scale
            tensors[nid] = T.astype(dt, copy=False).astype(np.complex128, copy=False)

        st = TTNState(
            tree=tree,
            tensors=tensors,
            phys_dims=phys_dims,
            parent_bond_dims=parent_bond_dims,
            metadata={"seed": int(seed), **dict(metadata or {})},
        )
        st.validate()
        if normalize:
            st.normalize_in_place()
        return st

    def validate(self) -> None:
        """Validate tree/tensor/bond consistency."""
        _assert(self.tree is not None, "TTNState: tree missing")
        _assert(hasattr(self.tree, "nodes") and hasattr(self.tree, "root") and hasattr(self.tree, "leaves"),
                "TTNState: tree must expose nodes/root/leaves")

        if hasattr(self.tree, "validate") and callable(getattr(self.tree, "validate")):
            self.tree.validate()

        root = int(self.tree.root)
        _assert(int(self.parent_bond_dims.get(root, -1)) == 1, "TTNState: root parent bond must be 1")

        # All nodes must have tensors and parent bond dims.
        for nid in self.tree.nodes:
            nid = int(nid)
            _assert(nid in self.tensors, f"TTNState: missing tensor for node {nid}")
            _assert(nid in self.parent_bond_dims, f"TTNState: missing parent bond dim for node {nid}")
            _assert(int(self.parent_bond_dims[nid]) >= 1, f"TTNState: invalid parent bond dim at node {nid}")
            T = self.tensors[nid]
            _assert(T.dtype == np.complex128, f"TTNState: tensor {nid} must be complex128")
            if self.strict_finite:
                _assert(np.all(np.isfinite(T)), f"TTNState: non-finite tensor values at node {nid}")

        # Leaves.
        for lid in self.tree.leaves:
            lid = int(lid)
            node = self.tree.nodes[lid]
            _assert(bool(getattr(node, "is_leaf", False)), f"TTNState: listed leaf {lid} is not marked as leaf")
            _assert(lid in self.phys_dims, f"TTNState: missing phys dim for leaf {lid}")
            T = self.tensors[lid]
            _assert(T.ndim == 2, f"TTNState: leaf {lid} tensor must have shape (d,b), got {T.shape}")
            _assert(int(T.shape[0]) == int(self.phys_dims[lid]), f"TTNState: leaf {lid} physical dim mismatch")
            _assert(int(T.shape[1]) == int(self.parent_bond_dims[lid]), f"TTNState: leaf {lid} parent bond mismatch")

        # Internal nodes.
        for nid, node in self.tree.nodes.items():
            nid = int(nid)
            if bool(getattr(node, "is_leaf", False)):
                continue
            children = _node_children(self.tree, nid)
            T = self.tensors[nid]
            _assert(T.ndim == len(children) + 1,
                    f"TTNState: internal node {nid} tensor rank mismatch: expected {len(children)+1}, got {T.ndim}")
            for ax, cid in enumerate(children):
                _assert(int(T.shape[ax]) == int(self.parent_bond_dims[cid]),
                        f"TTNState: node {nid} child-axis {ax} bond mismatch for child {cid}")
            _assert(int(T.shape[-1]) == int(self.parent_bond_dims[nid]),
                    f"TTNState: node {nid} parent bond mismatch")

    # ------------------------------------------------------------------
    # Cloning and structure helpers
    # ------------------------------------------------------------------

    def clone(self) -> "TTNState":
        return TTNState(
            tree=self.tree,
            tensors={k: v.copy() for k, v in self.tensors.items()},
            phys_dims=dict(self.phys_dims),
            parent_bond_dims=dict(self.parent_bond_dims),
            metadata=dict(self.metadata),
            strict_finite=bool(self.strict_finite),
        )

    def zero_like(self) -> "TTNState":
        out = self.clone()
        for nid in out.tensors:
            out.tensors[nid] = np.zeros_like(out.tensors[nid], dtype=np.complex128)
        return out

    def clone_with_internal_copied_once_and_leaf_zero(self) -> "TTNState":
        """
        Backward-compatible helper used by legacy scaffold apply.

        Internal tensors are copied from this state; leaf tensors are zeroed.
        """
        out = self.clone()
        for lid in out.tree.leaves:
            out.tensors[int(lid)] = np.zeros_like(out.tensors[int(lid)], dtype=np.complex128)
        return out

    def copy_structure_with_random_leaf_noise(self, scale: float = 1e-6, seed: int = 0) -> "TTNState":
        """Return a clone with tiny deterministic leaf noise; useful for degeneracy probes."""
        out = self.clone()
        rng = np.random.default_rng(int(seed))
        for lid in out.tree.leaves:
            lid = int(lid)
            noise = (rng.normal(size=out.tensors[lid].shape) + 1j * rng.normal(size=out.tensors[lid].shape))
            out.tensors[lid] = (out.tensors[lid] + float(scale) * noise).astype(np.complex128)
        return out

    # ------------------------------------------------------------------
    # Norms and diagnostics
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        return int(sum(int(T.size) for T in self.tensors.values()))

    def memory_bytes(self) -> int:
        return int(sum(int(T.nbytes) for T in self.tensors.values()))

    def max_parent_bond_dim(self) -> int:
        return int(max(self.parent_bond_dims.values())) if self.parent_bond_dims else 0

    def bond_summary(self) -> Dict[str, Any]:
        vals = np.asarray(list(self.parent_bond_dims.values()), dtype=np.int64)
        if vals.size == 0:
            return {"min": 0, "mean": 0.0, "max": 0, "root": int(self.tree.root)}
        return {
            "min": int(np.min(vals)),
            "mean": float(np.mean(vals)),
            "max": int(np.max(vals)),
            "root": int(self.tree.root),
            "num_bonds": int(vals.size),
        }

    def amplitude_norm_squared(self) -> float:
        """Compute ||psi||² via exact bottom-up bond reduced density matrices."""
        self.validate()
        rho = self.bottom_up_bond_rdms()
        root = int(self.tree.root)
        root_rho = rho[root]
        _assert(root_rho.shape == (1, 1), f"TTNState: root RDM must be (1,1), got {root_rho.shape}")
        val = float(np.real(root_rho[0, 0]))
        return val if np.isfinite(val) else 0.0

    def normalize_in_place(self, target_norm: float = 1.0) -> float:
        """
        Normalize the TTN to ``target_norm`` and return the previous norm².

        The initial 64-leaf random TTN can have a very small but valid norm²
        because every tensor is scaled locally. Scaling only the root by
        ``sqrt(target² / norm²)`` can then create an oversized root tensor.
        Instead, the scale is distributed over all tensors. Since a TTN state is
        multilinear in its tensors, multiplying every tensor by ``s`` multiplies
        norm² by ``s ** (2 * num_tensors)``.
        """
        n2 = float(self.amplitude_norm_squared())
        target = float(target_norm)
        if n2 > np.finfo(np.float64).tiny and np.isfinite(n2) and np.isfinite(target):
            num_tensors = max(1, int(len(self.tensors)))
            factor = (max(0.0, target * target) / n2) ** (1.0 / (2.0 * float(num_tensors)))
            if np.isfinite(factor) and factor > 0.0:
                for nid in list(self.tensors.keys()):
                    self.tensors[nid] = (self.tensors[nid] * factor).astype(np.complex128, copy=False)
        return n2

    def health_metrics(self) -> Dict[str, Any]:
        nonfinite = 0
        max_abs = 0.0
        for T in self.tensors.values():
            nonfinite += int(T.size - np.count_nonzero(np.isfinite(T)))
            if T.size:
                max_abs = max(max_abs, float(np.max(np.abs(np.nan_to_num(T)))))
        try:
            n2 = float(self.amplitude_norm_squared())
        except Exception:
            n2 = float("nan")
        stable = bool(np.isfinite(n2) and 1e-8 <= n2 <= 1e8 and nonfinite == 0 and np.isfinite(max_abs))
        return {
            "kind": "TTNState",
            "num_nodes": int(len(self.tree.nodes)),
            "num_leaves": int(len(self.tree.leaves)),
            "parameter_count": int(self.parameter_count()),
            "memory_bytes": int(self.memory_bytes()),
            "bond_summary": self.bond_summary(),
            "norm_squared": n2 if np.isfinite(n2) else None,
            "max_abs_tensor_value": float(max_abs),
            "nonfinite_count": int(nonfinite),
            "is_stable": stable,
        }

    def snapshot(self, *, include_tensor_shapes: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": "TTNState",
            "num_nodes": int(len(self.tree.nodes)),
            "num_leaves": int(len(self.tree.leaves)),
            "root": int(self.tree.root),
            "parameter_count": int(self.parameter_count()),
            "memory_bytes": int(self.memory_bytes()),
            "bond_summary": self.bond_summary(),
            "norm_squared": float(self.amplitude_norm_squared()),
            "metadata": dict(self.metadata),
        }
        if include_tensor_shapes:
            out["tensor_shapes"] = {str(k): list(map(int, v.shape)) for k, v in self.tensors.items()}
            out["phys_dims"] = {str(k): int(v) for k, v in self.phys_dims.items()}
            out["parent_bond_dims"] = {str(k): int(v) for k, v in self.parent_bond_dims.items()}
        return out

    # ------------------------------------------------------------------
    # Leaf operations
    # ------------------------------------------------------------------

    def leaf_tensor(self, lid: int) -> np.ndarray:
        lid = int(lid)
        self.validate()
        _assert(_node_is_leaf(self.tree, lid), f"leaf_tensor: node {lid} is not a leaf")
        return self.tensors[lid]

    def internal_tensor(self, nid: int) -> np.ndarray:
        nid = int(nid)
        self.validate()
        _assert(not _node_is_leaf(self.tree, nid), f"internal_tensor: node {nid} is a leaf")
        return self.tensors[nid]

    def contract_leaf_with_op(self, leaf_id: int, A: np.ndarray) -> np.ndarray:
        """Return T' = A @ T_leaf without modifying this state."""
        leaf_id = int(leaf_id)
        T = self.leaf_tensor(leaf_id)
        Op = np.asarray(A, dtype=np.complex128)
        _assert(Op.shape == (T.shape[0], T.shape[0]),
                f"contract_leaf_with_op: operator shape {Op.shape} incompatible with leaf {leaf_id} dim {T.shape[0]}")
        return (Op @ T).astype(np.complex128)

    def contract_leaf_with_op_in_place(self, leaf_id: int, A: np.ndarray) -> None:
        self.tensors[int(leaf_id)] = self.contract_leaf_with_op(int(leaf_id), A)

    def apply_product_ops(self, leaf_ops: Mapping[int, np.ndarray], *, coeff: complex = 1.0) -> "TTNState":
        """
        Return a new state with product operators applied to specified leaves.

        The scalar coeff is placed on the first affected leaf, or on the root if
        leaf_ops is empty.
        """
        out = self.clone()
        for lid, A in leaf_ops.items():
            out.contract_leaf_with_op_in_place(int(lid), A)
        if coeff != 1.0:
            keys = sorted(int(k) for k in leaf_ops.keys())
            carrier = keys[0] if keys else int(out.tree.root)
            out.tensors[carrier] = (complex(coeff) * out.tensors[carrier]).astype(np.complex128)
        return out

    # ------------------------------------------------------------------
    # Bottom-up and top-down environments
    # ------------------------------------------------------------------

    def bottom_up_bond_rdms(self) -> Dict[int, np.ndarray]:
        """
        Compute reduced density matrix on every node's parent bond.

        rho[nid].shape == (b_parent(nid), b_parent(nid)).
        """
        self.validate()
        rho: Dict[int, np.ndarray] = {}

        # Leaves: rho = T†T.
        for lid in self.tree.leaves:
            lid = int(lid)
            T = self.tensors[lid]
            rho[lid] = (T.conj().T @ T).astype(np.complex128)

        # Internals: generic contraction.
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for nid in _post_order(self.tree):
            if _node_is_leaf(self.tree, nid):
                continue
            children = _node_children(self.tree, nid)
            m = len(children)
            T = self.tensors[nid]
            _assert(2 * (m + 1) <= len(letters), "bottom_up_bond_rdms: too many child axes for einsum labels")

            a = [letters[i] for i in range(m)]
            ap = letters[m]
            b = [letters[m + 1 + i] for i in range(m)]
            bp = letters[m + 1 + m]

            expr = "".join(a) + ap + "," + "".join(b) + bp
            operands: List[np.ndarray] = [T, np.conj(T)]
            for i, cid in enumerate(children):
                expr += "," + a[i] + b[i]
                operands.append(rho[int(cid)])
            expr += "->" + ap + bp

            rho_n = np.einsum(expr, *operands, optimize=True).astype(np.complex128)
            if self.strict_finite:
                _assert(np.all(np.isfinite(rho_n)), f"bottom_up_bond_rdms: non-finite RDM at node {nid}")
            rho[nid] = rho_n

        return rho

    def top_down_bond_envs(self, bond_rdms: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
        """
        Compute environment matrix outside each node's subtree.

        E[root] == [[1]]. For child c of parent p, E[c] is the contraction of
        the whole network except c's subtree, open on c's parent bond.
        """
        self.validate()
        if bond_rdms is None:
            bond_rdms = self.bottom_up_bond_rdms()

        root = int(self.tree.root)
        E: Dict[int, np.ndarray] = {root: np.eye(1, dtype=np.complex128)}
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for p in _topological_order(self.tree):
            children = _node_children(self.tree, p)
            if not children:
                continue

            T = self.tensors[p]
            m = len(children)
            Ep = E[p]
            _assert(2 * (m + 1) <= len(letters), "top_down_bond_envs: too many child axes for einsum labels")

            a = [letters[i] for i in range(m)]
            ap = letters[m]
            b = [letters[m + 1 + i] for i in range(m)]
            bp = letters[m + 1 + m]

            for cidx, c in enumerate(children):
                expr = "".join(a) + ap + "," + "".join(b) + bp + "," + ap + bp
                operands: List[np.ndarray] = [T, np.conj(T), Ep]
                for sidx, sib in enumerate(children):
                    if sidx == cidx:
                        continue
                    expr += "," + a[sidx] + b[sidx]
                    operands.append(bond_rdms[int(sib)])
                expr += "->" + a[cidx] + b[cidx]
                Ec = np.einsum(expr, *operands, optimize=True).astype(np.complex128)
                if self.strict_finite:
                    _assert(np.all(np.isfinite(Ec)), f"top_down_bond_envs: non-finite environment at node {c}")
                E[int(c)] = Ec

        return E

    # ------------------------------------------------------------------
    # Canonicalization and truncation
    # ------------------------------------------------------------------

    def canonicalize_upward_qr(self, *, min_rank: int = 1) -> None:
        """
        Upward QR canonicalization with R absorption into the parent tensor.

        If a node tensor has fewer rows than its parent-bond dimension, this may
        reduce that parent bond. The parent tensor is updated consistently.
        """
        self.validate()
        root = int(self.tree.root)
        min_rank = int(max(1, min_rank))

        for nid in _post_order(self.tree):
            nid = int(nid)
            if nid == root:
                continue
            parent = _node_parent(self.tree, nid)
            if parent is None:
                continue

            T = self.tensors[nid]
            b_old = int(T.shape[-1])
            M = T.reshape((-1, b_old))
            if M.size == 0:
                continue

            Q, R = np.linalg.qr(M, mode="reduced")
            r = int(Q.shape[1])
            r = int(max(min_rank, r))

            # np.linalg.qr cannot produce more columns than min(M.shape). If an
            # exact larger min_rank is requested, pad with zero columns/rows.
            if r > Q.shape[1]:
                q_pad = np.zeros((Q.shape[0], r - Q.shape[1]), dtype=np.complex128)
                r_pad = np.zeros((r - R.shape[0], R.shape[1]), dtype=np.complex128)
                Q = np.concatenate([Q.astype(np.complex128), q_pad], axis=1)
                R = np.concatenate([R.astype(np.complex128), r_pad], axis=0)
            else:
                Q = Q.astype(np.complex128)
                R = R.astype(np.complex128)

            self.tensors[nid] = Q.reshape(*T.shape[:-1], r).astype(np.complex128)
            self.parent_bond_dims[nid] = int(r)

            pnode = self.tree.nodes[parent]
            children = _node_children(self.tree, parent)
            _assert(nid in children, "canonicalize_upward_qr: parent does not list child")
            ax = children.index(nid)
            Tp = self.tensors[parent]
            Tnew = np.tensordot(R, Tp, axes=(1, ax))
            Tnew = np.moveaxis(Tnew, 0, ax)
            self.tensors[parent] = Tnew.astype(np.complex128)

        self.validate()

    def _pad_parent_bond_to(self, nid: int, target_rank: int) -> None:
        """Expand the parent bond of nid to target_rank by zero padding."""
        nid = int(nid)
        target_rank = int(max(1, target_rank))
        cur = int(self.parent_bond_dims[nid])
        if target_rank <= cur:
            return

        parent = _node_parent(self.tree, nid)
        T = self.tensors[nid]
        pad_shape = list(T.shape)
        pad_shape[-1] = target_rank - cur
        self.tensors[nid] = np.concatenate([T, np.zeros(tuple(pad_shape), dtype=np.complex128)], axis=-1)
        self.parent_bond_dims[nid] = target_rank

        if parent is not None:
            children = _node_children(self.tree, parent)
            _assert(nid in children, "_pad_parent_bond_to: parent does not list child")
            ax = children.index(nid)
            Tp = self.tensors[parent]
            pshape = list(Tp.shape)
            pshape[ax] = target_rank - cur
            self.tensors[parent] = np.concatenate([Tp, np.zeros(tuple(pshape), dtype=np.complex128)], axis=ax)

    def truncate_parent_bond_svd(
        self,
        nid: int,
        rank: Optional[int] = None,
        tol: Optional[float] = None,
        *,
        exact_rank: Optional[bool] = None,
    ) -> int:
        """
        Truncate or project the bond between nid and its parent using SVD.

        If rank is provided and tol is None, rank is treated as an exact target
        rank by default. This is important for RK stage shape compatibility in
        evolve.py. If tol is provided, rank is treated as a cap unless
        exact_rank=True is explicitly passed.

        Returns the resulting parent-bond dimension for nid.
        """
        self.validate()
        nid = int(nid)
        parent = _node_parent(self.tree, nid)
        if parent is None:
            return int(self.parent_bond_dims[nid])

        if rank is None and tol is None:
            return int(self.parent_bond_dims[nid])

        if exact_rank is None:
            exact = bool(rank is not None and tol is None)
        else:
            exact = bool(exact_rank)

        T = self.tensors[nid]
        b_old = int(T.shape[-1])
        M = T.reshape((-1, b_old))

        # Compute SVD truncation. svd_truncate returns a cap; we pad later when
        # exact rank is requested and SVD rank is smaller than target.
        U, S, Vh = svd_truncate(M, rank=rank, tol=tol)[:3]
        U = np.asarray(U, dtype=np.complex128)
        S = np.asarray(S, dtype=np.float64)
        Vh = np.asarray(Vh, dtype=np.complex128)

        if S.size == 0:
            # Degenerate; create one zero channel.
            U = np.zeros((M.shape[0], 1), dtype=np.complex128)
            S = np.zeros((1,), dtype=np.float64)
            Vh = np.zeros((1, b_old), dtype=np.complex128)

        target = int(rank) if (rank is not None and exact) else int(S.size)
        target = int(max(1, target))

        if target > S.size:
            add = target - int(S.size)
            U = np.concatenate([U, np.zeros((U.shape[0], add), dtype=np.complex128)], axis=1)
            S = np.concatenate([S, np.zeros((add,), dtype=np.float64)], axis=0)
            Vh = np.concatenate([Vh, np.zeros((add, b_old), dtype=np.complex128)], axis=0)
        elif target < S.size:
            U = U[:, :target]
            S = S[:target]
            Vh = Vh[:target, :]

        r = int(target)
        US = (U * S.reshape(1, -1)).astype(np.complex128)
        self.tensors[nid] = US.reshape(*T.shape[:-1], r).astype(np.complex128)
        self.parent_bond_dims[nid] = int(r)

        children = _node_children(self.tree, parent)
        _assert(nid in children, "truncate_parent_bond_svd: parent does not list child")
        ax = children.index(nid)
        Tp = self.tensors[parent]
        Tnew = np.tensordot(Vh, Tp, axes=(1, ax))
        Tnew = np.moveaxis(Tnew, 0, ax)
        self.tensors[parent] = Tnew.astype(np.complex128)

        self.validate()
        return int(r)

    def project_to_bond_dims(self, target_bond_dims: Mapping[int, int]) -> None:
        """Project all non-root parent bonds to the supplied exact dimensions."""
        root = int(self.tree.root)
        for nid in _post_order(self.tree):
            nid = int(nid)
            if nid == root or nid not in target_bond_dims:
                continue
            target = int(max(1, target_bond_dims[nid]))
            cur = int(self.parent_bond_dims[nid])
            if cur < target:
                self._pad_parent_bond_to(nid, target)
            self.truncate_parent_bond_svd(nid, rank=target, tol=None, exact_rank=True)
        self.validate()

    # ------------------------------------------------------------------
    # Leaf dimension adaptation
    # ------------------------------------------------------------------

    def expand_leaf_dim(self, leaf_id: int, new_d: int, seed: int = 0, *, noise_scale: float = 1e-3) -> None:
        self.validate()
        leaf_id = int(leaf_id)
        new_d = int(new_d)
        _assert(_node_is_leaf(self.tree, leaf_id), f"expand_leaf_dim: node {leaf_id} is not a leaf")
        old_d = int(self.phys_dims[leaf_id])
        _assert(new_d >= old_d, "expand_leaf_dim: new_d must be >= current dimension")
        if new_d == old_d:
            return
        rng = np.random.default_rng(int(seed) + 997 * leaf_id + 13 * new_d)
        T = self.tensors[leaf_id]
        b = int(T.shape[1])
        pad = (rng.normal(size=(new_d - old_d, b)) + 1j * rng.normal(size=(new_d - old_d, b))).astype(np.complex128)
        pad *= float(noise_scale) / np.sqrt(max(1, new_d * b))
        self.tensors[leaf_id] = np.vstack([T, pad]).astype(np.complex128)
        self.phys_dims[leaf_id] = int(new_d)
        self.validate()

    def shrink_leaf_dim(self, leaf_id: int, new_d: int) -> None:
        self.validate()
        leaf_id = int(leaf_id)
        new_d = int(new_d)
        _assert(_node_is_leaf(self.tree, leaf_id), f"shrink_leaf_dim: node {leaf_id} is not a leaf")
        old_d = int(self.phys_dims[leaf_id])
        _assert(1 <= new_d <= old_d, "shrink_leaf_dim: new_d must be in [1,current]")
        if new_d == old_d:
            return
        self.tensors[leaf_id] = self.tensors[leaf_id][:new_d, :].copy().astype(np.complex128)
        self.phys_dims[leaf_id] = int(new_d)
        self.validate()

    def adapt_leaf_dims_in_place(self, target_d_leaf: Sequence[int], seed: int = 0) -> None:
        """Expand/shrink leaf physical dimensions to target values in tree.leaves order."""
        targets = np.asarray(target_d_leaf, dtype=np.int64).reshape(-1)
        leaves = [int(x) for x in self.tree.leaves]
        _assert(targets.size == len(leaves),
                f"adapt_leaf_dims_in_place: expected {len(leaves)} targets, got {targets.size}")
        for i, lid in enumerate(leaves):
            tgt = int(max(1, targets[i]))
            cur = int(self.phys_dims[lid])
            if tgt > cur:
                self.expand_leaf_dim(lid, tgt, seed=int(seed) + i)
            elif tgt < cur:
                self.shrink_leaf_dim(lid, tgt)
        self.validate()

    # ------------------------------------------------------------------
    # Arithmetic utilities used by integrators/tests
    # ------------------------------------------------------------------

    def scale_in_place(self, alpha: complex) -> None:
        for nid in self.tensors:
            self.tensors[nid] = (complex(alpha) * self.tensors[nid]).astype(np.complex128)

    def add_scaled_in_place(self, src: "TTNState", alpha: complex = 1.0) -> None:
        self.validate()
        src.validate()
        _assert(self.tree is src.tree, "add_scaled_in_place: states must share the same tree object")
        _assert(self.phys_dims == src.phys_dims, "add_scaled_in_place: physical dimensions mismatch")
        _assert(self.parent_bond_dims == src.parent_bond_dims, "add_scaled_in_place: bond dimensions mismatch")
        for nid in self.tensors:
            self.tensors[nid] = (self.tensors[nid] + complex(alpha) * src.tensors[nid]).astype(np.complex128)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, *, include_tensors: bool = True) -> Dict[str, Any]:
        """
        Serialize state metadata and optionally tensors to a JSON-safe dict.

        The tree object itself is not serialized; use from_dict(..., tree=tree)
        to reconstruct the state with an existing compatible tree.
        """
        payload: Dict[str, Any] = {
            "format": "AtomTN.TTNState",
            "version": 2,
            "root": int(self.tree.root),
            "leaves": [int(x) for x in self.tree.leaves],
            "phys_dims": {str(k): int(v) for k, v in self.phys_dims.items()},
            "parent_bond_dims": {str(k): int(v) for k, v in self.parent_bond_dims.items()},
            "metadata": dict(self.metadata),
            "strict_finite": bool(self.strict_finite),
            "snapshot": self.snapshot(include_tensor_shapes=True),
        }
        if include_tensors:
            payload["tensors"] = {str(k): _complex_to_json(v) for k, v in self.tensors.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, tree: Any) -> "TTNState":
        if not isinstance(payload, Mapping):
            raise TypeError("TTNState.from_dict: payload must be a mapping")
        tensors_payload = payload.get("tensors", {})
        if not isinstance(tensors_payload, Mapping):
            raise ValueError("TTNState.from_dict: payload missing tensor mapping")
        tensors = {int(k): _complex_from_json(v) for k, v in tensors_payload.items()}  # type: ignore[arg-type]
        phys_dims = {int(k): int(v) for k, v in dict(payload.get("phys_dims", {})).items()}
        parent_bond_dims = {int(k): int(v) for k, v in dict(payload.get("parent_bond_dims", {})).items()}
        obj = cls(
            tree=tree,
            tensors=tensors,
            phys_dims=phys_dims,
            parent_bond_dims=parent_bond_dims,
            metadata=dict(payload.get("metadata", {})),
            strict_finite=bool(payload.get("strict_finite", True)),
        )
        obj.validate()
        return obj

    def save_npz(self, path: Union[str, Path], *, compressed: bool = True) -> None:
        """
        Save tensors and metadata to NPZ. The tree is not serialized.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": "AtomTN.TTNState.npz",
            "version": 2,
            "root": int(self.tree.root),
            "leaves": [int(x) for x in self.tree.leaves],
            "phys_dims": {str(k): int(v) for k, v in self.phys_dims.items()},
            "parent_bond_dims": {str(k): int(v) for k, v in self.parent_bond_dims.items()},
            "metadata": dict(self.metadata),
            "strict_finite": bool(self.strict_finite),
            "tensor_keys": [str(k) for k in sorted(self.tensors.keys())],
        }
        arrays: Dict[str, Any] = {"__metadata_json__": np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_)}
        for k, v in self.tensors.items():
            arrays[f"tensor_{int(k)}"] = np.asarray(v, dtype=np.complex128)
        if compressed:
            np.savez_compressed(p, **arrays)
        else:
            np.savez(p, **arrays)

    @classmethod
    def load_npz(cls, path: Union[str, Path], *, tree: Any, strict_finite: Optional[bool] = None) -> "TTNState":
        p = Path(path)
        with np.load(p, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata_json__"].item()))
            tensors: Dict[int, np.ndarray] = {}
            for key_s in meta.get("tensor_keys", []):
                k = int(key_s)
                tensors[k] = np.asarray(data[f"tensor_{k}"], dtype=np.complex128)
            obj = cls(
                tree=tree,
                tensors=tensors,
                phys_dims={int(k): int(v) for k, v in dict(meta.get("phys_dims", {})).items()},
                parent_bond_dims={int(k): int(v) for k, v in dict(meta.get("parent_bond_dims", {})).items()},
                metadata=dict(meta.get("metadata", {})),
                strict_finite=bool(meta.get("strict_finite", True) if strict_finite is None else strict_finite),
            )
            obj.validate()
            return obj


__all__ = [
    "TTNState",
    "fuse_axes",
    "reshape_as_matrix",
    "unreshape_from_matrix",
]
