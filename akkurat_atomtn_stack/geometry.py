#!/usr/bin/env python3
# geometry.py
"""
Geometry + tree compilation for AtomTN.

This module is the deliberately small, deterministic geometry substrate used by
AtomTN's flow, TTN-state, Hamiltonian, and runtime wrappers. It owns only:

- a generic rooted tree representation over physical leaves;
- deterministic k-ary tree compilation over the 64 geometry leaves;
- a compact 4x4x4 graph scaffold, exposed as TetraMesh64;
- commutative graph-calculus helpers used by the flow solvers.

No Hamiltonian, evolution, projection, fiber, or governance logic belongs here.

Compatibility guarantees
------------------------
The public names expected by the current AtomTN stack are preserved:

    TreeNode, Tree, make_balanced_kary_tree, GraphCalculus, TetraMesh64

Important invariant
-------------------
For the default TetraMesh64, physical leaf ids are the graph node ids 0..63.
Tree.leaves is kept in natural graph-node order, even if future tree-grouping
modes use a non-natural grouping order internally. Several downstream schedules
index curvature arrays by graph node id and rely on that stable mapping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from math_utils import _assert
except Exception:  # pragma: no cover - import fallback for standalone linting
    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)


# =============================================================================
# Small helpers
# =============================================================================


def _coerce_int(x: Any, *, name: str) -> int:
    try:
        v = int(x)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer-like value") from exc
    return v


def _unique_preserve_order(xs: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for x in xs:
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


def _coerce_children(value: Any) -> List[int]:
    """Materialize a child container into a stable ``list[int]``.

    Some tensor operations, serialization layers, or third-party wrappers can
    accidentally leave ``TreeNode.children`` as an iterator. Validation and
    traversal must never call ``len()`` or ``reversed()`` on such objects
    directly, because iterators have no length and are single-use. This helper
    normalizes lists, tuples, arrays, generators, and list_iterators into a
    de-duplicated list while preserving order.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        raw = [value]
    else:
        try:
            raw = list(value)
        except TypeError:
            raw = [value]
    return _unique_preserve_order(int(c) for c in raw)


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return int(p)


# =============================================================================
# Tree representation
# =============================================================================


@dataclass
class TreeNode:
    """A node in a rooted TTN tree.

    Tensors are stored in ``ttn_state.py``; this class is topology only.
    """

    node_id: int
    children: List[int] = field(default_factory=list)
    parent: Optional[int] = None
    is_leaf: bool = False
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_id = int(self.node_id)
        self.children = _coerce_children(self.children)
        self.parent = None if self.parent is None else int(self.parent)
        self.is_leaf = bool(self.is_leaf)
        self.label = str(self.label)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": int(self.node_id),
            "children": _coerce_children(self.children),
            "parent": None if self.parent is None else int(self.parent),
            "is_leaf": bool(self.is_leaf),
            "label": str(self.label),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreeNode":
        return cls(
            node_id=int(payload.get("node_id")),
            children=[int(c) for c in payload.get("children", [])],
            parent=(None if payload.get("parent", None) is None else int(payload.get("parent"))),
            is_leaf=bool(payload.get("is_leaf", False)),
            label=str(payload.get("label", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class Tree:
    """Rooted tree used by TTNState.

    Required downstream attributes are ``nodes``, ``root``, and ``leaves``.
    Additional methods provide validation, traversal, LCA, and serialization.
    """

    nodes: Dict[int, TreeNode]
    root: int
    leaves: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = {int(k): v for k, v in dict(self.nodes).items()}
        # Ensure node_id fields match dict keys.
        for k, node in list(self.nodes.items()):
            if not isinstance(node, TreeNode):
                node = TreeNode.from_dict(node)  # type: ignore[arg-type]
                self.nodes[k] = node
            node.node_id = int(k)
            node.children = _coerce_children(node.children)
            node.parent = None if node.parent is None else int(node.parent)
        self.root = int(self.root)
        self.leaves = [int(l) for l in self.leaves]
        self.metadata = dict(self.metadata or {})
        self.validate()

    # ------------------------------------------------------------------
    # Validation and topology properties
    # ------------------------------------------------------------------

    def validate(self) -> None:
        _assert(self.root in self.nodes, "Tree: root missing")
        _assert(len(self.nodes) > 0, "Tree: empty node dictionary")

        root_node = self.nodes[self.root]
        _assert(root_node.parent is None, "Tree: root parent must be None")
        _assert(not root_node.is_leaf or len(self.nodes) == 1, "Tree: root may be leaf only for a one-node tree")

        # Node-level consistency.
        for nid, node in self.nodes.items():
            _assert(node.node_id == nid, f"Tree: node_id mismatch at key {nid}")
            node.children = _coerce_children(node.children)
            if node.is_leaf:
                _assert(len(node.children) == 0, f"Tree: leaf {nid} cannot have children")
            else:
                _assert(len(node.children) > 0, f"Tree: internal node {nid} has no children")
            for c in node.children:
                _assert(c in self.nodes, f"Tree: missing child {c}")
                _assert(self.nodes[c].parent == nid, f"Tree: parent pointer inconsistent for child {c}")

        # Parent pointers must reference existing nodes and contain no self-links.
        for nid, node in self.nodes.items():
            if nid == self.root:
                continue
            _assert(node.parent in self.nodes, f"Tree: node {nid} has missing parent {node.parent}")
            _assert(node.parent != nid, f"Tree: node {nid} cannot parent itself")

        # Reachability and acyclicity from root.
        seen: Set[int] = set()
        stack = [self.root]
        while stack:
            u = stack.pop()
            _assert(u not in seen, f"Tree: cycle detected at node {u}")
            seen.add(u)
            self.nodes[u].children = _coerce_children(self.nodes[u].children)
            stack.extend(reversed(self.nodes[u].children))
        _assert(seen == set(self.nodes.keys()), "Tree: not all nodes are reachable from root")

        true_leaves = sorted([nid for nid, node in self.nodes.items() if node.is_leaf])
        _assert(sorted(self.leaves) == true_leaves, "Tree: leaves inconsistent")
        for lid in self.leaves:
            _assert(lid in self.nodes and self.nodes[lid].is_leaf, f"Tree: invalid leaf {lid}")

    @property
    def num_nodes(self) -> int:
        return int(len(self.nodes))

    @property
    def num_leaves(self) -> int:
        return int(len(self.leaves))

    @property
    def max_depth(self) -> int:
        return int(max(self.depth(nid) for nid in self.nodes))

    def __len__(self) -> int:
        return self.num_nodes

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def post_order(self) -> List[int]:
        order: List[int] = []
        visited: Set[int] = set()

        def dfs(u: int) -> None:
            _assert(u not in visited, f"Tree.post_order: cycle at {u}")
            visited.add(u)
            for v in self.nodes[u].children:
                dfs(int(v))
            order.append(int(u))

        dfs(self.root)
        return order

    def pre_order(self) -> List[int]:
        order: List[int] = []

        def dfs(u: int) -> None:
            order.append(int(u))
            for v in self.nodes[u].children:
                dfs(int(v))

        dfs(self.root)
        return order

    def breadth_first(self) -> List[int]:
        q: Deque[int] = deque([self.root])
        order: List[int] = []
        while q:
            u = int(q.popleft())
            order.append(u)
            q.extend(int(c) for c in self.nodes[u].children)
        return order

    def levels(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        q: Deque[Tuple[int, int]] = deque([(self.root, 0)])
        while q:
            u, d = q.popleft()
            out.setdefault(int(d), []).append(int(u))
            for c in self.nodes[u].children:
                q.append((int(c), int(d) + 1))
        return out

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def parent_of(self, nid: int) -> Optional[int]:
        nid = int(nid)
        _assert(nid in self.nodes, f"Tree.parent_of: unknown node {nid}")
        return self.nodes[nid].parent

    def children_of(self, nid: int) -> List[int]:
        nid = int(nid)
        _assert(nid in self.nodes, f"Tree.children_of: unknown node {nid}")
        self.nodes[nid].children = _coerce_children(self.nodes[nid].children)
        return list(self.nodes[nid].children)

    def depth(self, nid: int) -> int:
        nid = int(nid)
        _assert(nid in self.nodes, f"Tree.depth: unknown node {nid}")
        d = 0
        cur = nid
        seen: Set[int] = set()
        while cur != self.root:
            _assert(cur not in seen, "Tree.depth: cycle detected")
            seen.add(cur)
            p = self.nodes[cur].parent
            _assert(p is not None, "Tree.depth: broken parent chain")
            cur = int(p)
            d += 1
        return int(d)

    def path_to_root(self, nid: int) -> List[int]:
        nid = int(nid)
        _assert(nid in self.nodes, f"Tree.path_to_root: unknown node {nid}")
        path = [nid]
        cur = nid
        seen: Set[int] = set()
        while cur != self.root:
            _assert(cur not in seen, "Tree.path_to_root: cycle detected")
            seen.add(cur)
            p = self.nodes[cur].parent
            _assert(p is not None, "Tree.path_to_root: broken parent chain")
            cur = int(p)
            path.append(cur)
        return path

    def lca(self, a: int, b: int) -> int:
        pa = self.path_to_root(int(a))
        pb = self.path_to_root(int(b))
        pos_a = {nid: i for i, nid in enumerate(pa)}
        for nid in pb:
            if nid in pos_a:
                return int(nid)
        raise ValueError("Tree.lca: no common ancestor; invalid tree")

    def path_between(self, a: int, b: int) -> List[int]:
        """Return node path a -> ... -> b, inclusive."""
        a = int(a)
        b = int(b)
        c = self.lca(a, b)
        pa = self.path_to_root(a)
        pb = self.path_to_root(b)
        ia = pa.index(c)
        ib = pb.index(c)
        return pa[: ia + 1] + list(reversed(pb[:ib]))

    def subtree_leaves(self, nid: int) -> List[int]:
        nid = int(nid)
        _assert(nid in self.nodes, f"Tree.subtree_leaves: unknown node {nid}")
        node = self.nodes[nid]
        if node.is_leaf:
            return [nid]
        out: List[int] = []
        for c in node.children:
            out.extend(self.subtree_leaves(c))
        return out

    def leaf_index(self) -> Dict[int, int]:
        return {int(lid): i for i, lid in enumerate(self.leaves)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": int(self.root),
            "leaves": [int(l) for l in self.leaves],
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Tree":
        nodes_payload = dict(payload.get("nodes", {}))
        nodes = {int(k): TreeNode.from_dict(v) for k, v in nodes_payload.items()}
        return cls(
            nodes=nodes,
            root=int(payload.get("root")),
            leaves=[int(x) for x in payload.get("leaves", [])],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


# =============================================================================
# Tree compilation
# =============================================================================


def make_balanced_kary_tree(
    num_leaves: int,
    k: int = 4,
    start_id: int = 0,
    leaf_labels: Optional[Sequence[str]] = None,
    *,
    grouping_order: Optional[Sequence[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tree:
    """Create a deterministic balanced-ish k-ary tree.

    Parameters
    ----------
    num_leaves:
        Number of physical leaves.
    k:
        Maximum arity of internal nodes.
    start_id:
        First leaf id. For AtomTN/TetraMesh64 this is 0.
    leaf_labels:
        Optional labels indexed by natural leaf index.
    grouping_order:
        Optional permutation of 0..num_leaves-1 that controls how leaves are
        grouped into parents. ``Tree.leaves`` remains natural leaf-id order to
        preserve curvature-score and graph-node indexing invariants.
    """

    num_leaves = _coerce_int(num_leaves, name="num_leaves")
    k = _coerce_int(k, name="k")
    start_id = _coerce_int(start_id, name="start_id")
    _assert(num_leaves >= 1 and k >= 2, "make_balanced_kary_tree: num_leaves>=1 and k>=2 required")

    natural_leaf_ids = [start_id + i for i in range(num_leaves)]

    if grouping_order is None:
        grouping = list(range(num_leaves))
    else:
        grouping = [int(i) for i in grouping_order]
        _assert(sorted(grouping) == list(range(num_leaves)), "grouping_order must be a permutation of 0..num_leaves-1")

    nodes: Dict[int, TreeNode] = {}
    for i, nid in enumerate(natural_leaf_ids):
        lbl = leaf_labels[i] if (leaf_labels is not None and i < len(leaf_labels)) else f"leaf{i}"
        nodes[nid] = TreeNode(node_id=nid, is_leaf=True, label=str(lbl))

    next_id = start_id + num_leaves
    current = [natural_leaf_ids[i] for i in grouping]
    level = 0

    while len(current) > 1:
        nxt: List[int] = []
        for group_idx, pos in enumerate(range(0, len(current), k)):
            group = [int(x) for x in current[pos: pos + k]]
            nid = next_id
            next_id += 1
            nodes[nid] = TreeNode(
                node_id=nid,
                children=group,
                is_leaf=False,
                label=f"intL{level}_{group_idx}",
            )
            for c in group:
                nodes[c].parent = nid
            nxt.append(nid)
        current = nxt
        level += 1

    root = int(current[0])
    nodes[root].parent = None

    tree = Tree(
        nodes=nodes,
        root=root,
        leaves=natural_leaf_ids,
        metadata={
            "compiler": "make_balanced_kary_tree",
            "num_leaves": int(num_leaves),
            "arity": int(k),
            "start_id": int(start_id),
            **dict(metadata or {}),
        },
    )
    return tree


# =============================================================================
# Graph calculus (commutative)
# =============================================================================


@dataclass
class GraphCalculus:
    """Small commutative graph-calculus helper over an undirected graph.

    Flow fields are represented as dictionaries keyed by directed edges. The
    canonical undirected edge list returned by ``oriented_edges`` uses u < v.
    """

    adjacency: Dict[int, List[int]]
    num_nodes: int = 64
    _edge_cache: Optional[List[Tuple[int, int]]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.num_nodes = int(self.num_nodes)
        self.adjacency = {int(u): _unique_preserve_order(int(v) for v in nbrs) for u, nbrs in dict(self.adjacency).items()}
        self.validate()

    def validate(self) -> None:
        _assert(self.num_nodes > 0, "GraphCalculus: num_nodes must be positive")
        _assert(set(self.adjacency.keys()) == set(range(self.num_nodes)), "GraphCalculus: adjacency keys must be 0..num_nodes-1")
        for u, nbrs in self.adjacency.items():
            _assert(u not in nbrs, f"GraphCalculus: self-loop at node {u}")
            for v in nbrs:
                _assert(0 <= int(v) < self.num_nodes, f"GraphCalculus: neighbor {v} out of range")
                _assert(u in self.adjacency[int(v)], f"GraphCalculus: asymmetric edge {u}-{v}")

    def neighbors(self, u: int) -> List[int]:
        return list(self.adjacency[int(u)])

    def oriented_edges(self) -> List[Tuple[int, int]]:
        if self._edge_cache is None:
            edges: List[Tuple[int, int]] = []
            for u in range(self.num_nodes):
                for v in self.adjacency[u]:
                    if int(u) < int(v):
                        edges.append((int(u), int(v)))
            self._edge_cache = edges
        return list(self._edge_cache)

    def directed_edges(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for u, v in self.oriented_edges():
            out.append((u, v))
            out.append((v, u))
        return out

    def degree_vector(self) -> np.ndarray:
        return np.asarray([len(self.adjacency[i]) for i in range(self.num_nodes)], dtype=np.float64)

    def incidence_matrix(self, *, oriented: bool = True) -> np.ndarray:
        """Return node-edge incidence matrix B with columns e=(u,v): B[u,e]=-1, B[v,e]=+1."""
        edges = self.oriented_edges() if oriented else self.directed_edges()
        B = np.zeros((self.num_nodes, len(edges)), dtype=np.float64)
        for j, (u, v) in enumerate(edges):
            B[int(u), j] = -1.0
            B[int(v), j] = 1.0
        return B

    def laplacian_matrix(self) -> np.ndarray:
        L = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for u in range(self.num_nodes):
            L[u, u] = len(self.adjacency[u])
            for v in self.adjacency[u]:
                L[u, int(v)] -= 1.0
        return L

    def div(self, F: Mapping[Tuple[int, int], Any]) -> np.ndarray:
        """Discrete divergence using outgoing directed entries.

        Missing entries are treated as zero. Values must be scalar-like for this
        commutative helper; matrix-valued divergence lives in fuzzy_backend.py.
        """
        div = np.zeros((self.num_nodes,), dtype=np.float64)
        for key, val in dict(F).items():
            if len(key) != 2:
                continue
            u, _v = int(key[0]), int(key[1])
            if 0 <= u < self.num_nodes:
                try:
                    vf = float(val)
                except Exception:
                    vf = float(np.asarray(val, dtype=np.float64).reshape(-1)[0])
                if np.isfinite(vf):
                    div[u] += vf
        return div

    def gradient(self, a: Any) -> Dict[Tuple[int, int], float]:
        """Return scalar edge gradient grad(a)_{u->v}=a[v]-a[u] for canonical u<v and reverse."""
        x = np.asarray(a, dtype=np.float64).reshape(-1)
        _assert(x.size == self.num_nodes, f"GraphCalculus.gradient: expected size {self.num_nodes}, got {x.size}")
        out: Dict[Tuple[int, int], float] = {}
        for u, v in self.oriented_edges():
            val = float(x[int(v)] - x[int(u)])
            out[(int(u), int(v))] = val
            out[(int(v), int(u))] = -val
        return out

    def laplacian(self, a: Any) -> np.ndarray:
        x = np.asarray(a, dtype=np.float64).reshape(-1)
        _assert(x.size == self.num_nodes, f"GraphCalculus.laplacian: expected size {self.num_nodes}, got {x.size}")
        out = np.zeros_like(x, dtype=np.float64)
        for u, nbrs in self.adjacency.items():
            uu = int(u)
            out[uu] = sum(float(x[int(v)] - x[uu]) for v in nbrs)
        return out

    def edge_field_norm(self, F: Mapping[Tuple[int, int], Any]) -> float:
        vals: List[float] = []
        for u, v in self.oriented_edges():
            val = F.get((u, v), 0.0)
            try:
                vals.append(float(val))
            except Exception:
                vals.append(float(np.linalg.norm(np.asarray(val))))
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.linalg.norm(arr)) if arr.size else 0.0


# =============================================================================
# Geometry scaffold: 64 tetrahedra / 4x4x4 graph
# =============================================================================


@dataclass
class TetraMesh64:
    """Minimal 64-node spatial graph scaffold for AtomTN.

    The name reflects the prototype's tetrahedral labeling, but the current
    connectivity is a regular 4x4x4 nearest-neighbor grid. This class is kept
    deliberately stable because downstream runtime components depend on exactly
    64 leaves / graph nodes.
    """

    num_tetra: int = 64
    adjacency: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.num_tetra = int(self.num_tetra)
        _assert(self.num_tetra == 64, "TetraMesh64: this scaffold is fixed to 64 nodes")
        if not self.adjacency:
            self.adjacency = self._default_adjacency()
        else:
            self.adjacency = {int(u): _unique_preserve_order(int(v) for v in nbrs) for u, nbrs in dict(self.adjacency).items()}
        self._validate_adjacency()

    @staticmethod
    def idx(x: int, y: int, z: int) -> int:
        _assert(0 <= int(x) < 4 and 0 <= int(y) < 4 and 0 <= int(z) < 4, "TetraMesh64.idx: coordinate out of range")
        return int(x + 4 * y + 16 * z)

    @staticmethod
    def coord(i: int) -> Tuple[int, int, int]:
        i = int(i)
        _assert(0 <= i < 64, "TetraMesh64.coord: index out of range")
        z = i // 16
        r = i % 16
        y = r // 4
        x = r % 4
        return int(x), int(y), int(z)

    @staticmethod
    def coordinates() -> np.ndarray:
        coords = [TetraMesh64.coord(i) for i in range(64)]
        return np.asarray(coords, dtype=np.float64)

    def _default_adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(64)}
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    i = self.idx(x, y, z)
                    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if 0 <= xx < 4 and 0 <= yy < 4 and 0 <= zz < 4:
                            adj[i].append(self.idx(xx, yy, zz))
        return adj

    def _validate_adjacency(self) -> None:
        _assert(set(self.adjacency.keys()) == set(range(64)), "TetraMesh64: adjacency keys must be 0..63")
        for u, nbrs in self.adjacency.items():
            _assert(int(u) not in [int(v) for v in nbrs], f"TetraMesh64: self-loop at {u}")
            for v in nbrs:
                _assert(0 <= int(v) < 64, f"TetraMesh64: neighbor {v} out of range")
                _assert(int(u) in self.adjacency[int(v)], f"TetraMesh64: adjacency is not symmetric on edge {u}-{v}")

    def tetra_labels(self) -> List[str]:
        return [f"tetra{i}" for i in range(self.num_tetra)]

    def oriented_edges(self) -> List[Tuple[int, int]]:
        return self.compile_calculus().oriented_edges()

    def _grouping_order(self, mode: str, seed: int) -> List[int]:
        """Return a grouping permutation while preserving Tree.leaves natural order."""
        m = str(mode or "balanced").lower().strip()
        if m in {"balanced", "natural", "default"}:
            return list(range(64))

        if m in {"fractal", "morton", "zorder", "spatial"}:
            # Morton/Z-order for locality-aware grouping; leaf ids remain natural.
            def spread2(n: int) -> int:
                n &= 0b11
                return ((n & 1) << 0) | ((n & 2) << 2)

            pairs: List[Tuple[int, int]] = []
            for i in range(64):
                x, y, z = self.coord(i)
                code = spread2(x) | (spread2(y) << 1) | (spread2(z) << 2)
                pairs.append((code, i))
            pairs.sort(key=lambda t: (t[0], t[1]))
            return [i for _, i in pairs]

        if m in {"fibonacci", "golden"}:
            # Deterministic low-discrepancy permutation over 64 ids.
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            vals = [((i * (phi - 1.0)) % 1.0, i) for i in range(64)]
            vals.sort(key=lambda t: (t[0], t[1]))
            return [i for _, i in vals]

        if m in {"random", "shuffle", "seeded"}:
            rng = np.random.default_rng(int(seed))
            return [int(x) for x in rng.permutation(64)]

        raise ValueError(f"unknown tree mode: {mode}")

    def compile_tree(self, mode: str = "balanced", arity: int = 4, seed: int = 0) -> Tree:
        """Compile a rooted k-ary tree over the 64 graph leaves.

        Accepted modes:
        - ``balanced`` / ``natural``: group leaves in graph-node order;
        - ``fractal`` / ``morton`` / ``spatial``: locality-aware Z-order grouping;
        - ``fibonacci`` / ``golden``: deterministic low-discrepancy grouping;
        - ``random`` / ``shuffle``: seeded grouping.

        ``Tree.leaves`` always remains [0, ..., 63].
        """
        arity = int(arity)
        _assert(arity >= 2, "TetraMesh64.compile_tree: arity must be >= 2")
        grouping = self._grouping_order(str(mode), int(seed))
        return make_balanced_kary_tree(
            64,
            k=arity,
            start_id=0,
            leaf_labels=self.tetra_labels(),
            grouping_order=grouping,
            metadata={"geometry": "TetraMesh64", "tree_mode": str(mode), "seed": int(seed)},
        )

    def compile_calculus(self, num_nodes: int = 64) -> GraphCalculus:
        _assert(int(num_nodes) == 64, "TetraMesh64.compile_calculus: num_nodes must be 64 for this scaffold")
        return GraphCalculus(self.adjacency, num_nodes=64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "TetraMesh64",
            "num_tetra": int(self.num_tetra),
            "adjacency": {str(k): [int(v) for v in vals] for k, vals in self.adjacency.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TetraMesh64":
        adj_raw = dict(payload.get("adjacency", {}))
        adj = {int(k): [int(v) for v in vals] for k, vals in adj_raw.items()} if adj_raw else {}
        return cls(num_tetra=int(payload.get("num_tetra", 64)), adjacency=adj)


__all__ = [
    "TreeNode",
    "Tree",
    "make_balanced_kary_tree",
    "GraphCalculus",
    "TetraMesh64",
]
