#!/usr/bin/env python3
# environments.py
"""
environments.py — cached TTN environments / messages for correct apply (Phase 4 substrate).

Goal
----
Phase 4 (correctness beyond the linear scaffold) needs *environments*:
- bottom-up messages (subtree contractions) so we can route/contract operators correctly
- optional top-down messages (full environments) for LCA-routed / two-site operations
- caching and invalidation so apply doesn't become O(#terms * tree_size) every step

This module is intentionally generic:
- It does NOT decide physics or operator structure.
- It provides a cacheable message-passing backbone for TTNState operations.

Design choices
--------------
1) Bottom-up "norm messages" (unconditional) are cheap and useful:
   msg_up[u] : vector on the parent bond of node u, representing contraction of the whole subtree at u.

2) "Operator-inserted messages" are the workhorse for correct apply:
   you can compute messages that represent a subtree with an operator applied on some leaf(s),
   and then "zip-up" those messages to the root or to an LCA.

3) Caching with "dirty stamps":
   - When leaf tensors change (after integration/truncation), mark affected path to root dirty.
   - Then update only what's needed.

4) The shapes are deliberately minimal:
   - Node u has parent bond dimension b_u = state.parent_bond_dims[u]
   - Each message is length b_u (complex128)
   - Internal contraction uses the node tensor and children messages.

This module assumes:
- TTNState stores tensors and the Tree with children/parent structure
- TTNState tensors:
  leaf:    T[d, b_u]
  internal T[b_c1, b_c2, ..., b_u]
- Messages are "diagonal-ish" in the sense used by your amplitude_norm_squared:
  they are vectors that get broadcast-multiplied into |T|^2 contractions.

This is consistent with your current norm calculation and is the fastest foundation to build from.

When Phase 4 lands, the correct apply will likely use:
- up messages for baseline environments
- "operator-inserted up messages" for onsite terms
- LCA routing for two-site terms (compute up-messages from u and v to their LCA, then contract)

NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Iterable, Tuple

import numpy as np

from math_utils import _assert
from ttn_state import TTNState


# -----------------------------------------------------------------------------
# Core message contraction primitives (mirror TTNState's norm message passing)
# -----------------------------------------------------------------------------

def _leaf_up_message(T_leaf: np.ndarray) -> np.ndarray:
    """
    Leaf message: m[b] = Σ_d |T[d,b]|^2
    """
    # (d,b) -> (b,)
    return np.sum(np.conj(T_leaf) * T_leaf, axis=0).astype(np.complex128)

def _internal_up_message(T_int: np.ndarray, child_msgs: Tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Internal node message:
      - Start from S = |T|^2 (same shape as T)
      - Multiply in each child message along that child axis (broadcast)
      - Sum over all child axes -> vector on parent bond axis

    T_int shape: (b_c1, b_c2, ..., b_parent)
    Returns: (b_parent,)
    """
    _assert(T_int.ndim >= 2, "internal tensor must have >=2 dims")
    n_children = T_int.ndim - 1
    _assert(len(child_msgs) == n_children, "child msg count mismatch")

    S = (np.conj(T_int) * T_int).astype(np.complex128, copy=False)

    # Multiply each child message along its axis
    for ax, m in enumerate(child_msgs):
        _assert(m.ndim == 1 and m.shape[0] == T_int.shape[ax], "child msg shape mismatch")
        shp = [1] * S.ndim
        shp[ax] = m.shape[0]
        S = S * m.reshape(shp)

    # Sum over child axes -> parent axis remains
    axes = tuple(range(n_children))
    out = np.sum(S, axis=axes).astype(np.complex128, copy=False)
    _assert(out.ndim == 1, "internal up message must be vector")
    return out


# -----------------------------------------------------------------------------
# Environment cache
# -----------------------------------------------------------------------------

@dataclass
class TTNEnvironments:
    """
    Cache of TTN messages/environments.

    Stored caches:
      - up_norm[nid] : subtree contraction message for node nid (vector on parent bond)

    Dirty tracking:
      - dirty_nodes: nodes whose up_norm must be recomputed
      - when a leaf changes, mark leaf and all ancestors dirty

    Optional extensions (Phase 4+):
      - down_norm: top-down environment messages
      - operator-inserted message caches keyed by (kind, leaf_id, op_id, step_bucket)
    """
    state: TTNState

    up_norm: Dict[int, np.ndarray] = field(default_factory=dict)
    dirty_nodes: Set[int] = field(default_factory=set)

    # simple stamp system (increments each time we "commit" updates)
    stamp: int = 0

    def __post_init__(self) -> None:
        self.state.validate()
        # Start dirty everywhere so first update computes all messages
        self.mark_all_dirty()

    # -------------------------------------------------------------------------
    # Dirty marking
    # -------------------------------------------------------------------------

    def mark_all_dirty(self) -> None:
        self.dirty_nodes = set(self.state.tree.nodes.keys())

    def mark_path_to_root_dirty(self, nid: int) -> None:
        tree = self.state.tree
        _assert(nid in tree.nodes, "unknown node id")
        u = nid
        while u is not None:
            self.dirty_nodes.add(u)
            u = tree.nodes[u].parent

    def mark_leaves_dirty(self, leaf_ids: Iterable[int]) -> None:
        for lid in leaf_ids:
            self.mark_path_to_root_dirty(lid)

    # -------------------------------------------------------------------------
    # Update / recompute caches
    # -------------------------------------------------------------------------

    def update_up_norm(self) -> None:
        """
        Recompute up_norm for all dirty nodes, bottom-up.
        Complexity: O(#dirty_subtree).
        """
        state = self.state
        tree = state.tree
        tensors = state.tensors

        # Compute in post-order so children are available
        for nid in tree.post_order():
            if nid not in self.dirty_nodes:
                continue

            node = tree.nodes[nid]
            if node.is_leaf:
                self.up_norm[nid] = _leaf_up_message(tensors[nid])
            else:
                child_msgs = tuple(self.up_norm[c] for c in node.children)
                self.up_norm[nid] = _internal_up_message(tensors[nid], child_msgs)

        # root message exists but is on bond dim 1
        root_msg = self.up_norm.get(tree.root, None)
        _assert(root_msg is not None and root_msg.shape == (1,), "root up_norm must be (1,)")

        self.dirty_nodes.clear()
        self.stamp += 1

    # -------------------------------------------------------------------------
    # Convenience getters
    # -------------------------------------------------------------------------

    def get_up_norm(self, nid: int) -> np.ndarray:
        """
        Returns up_norm[nid], updating caches if needed.
        """
        if nid in self.dirty_nodes or nid not in self.up_norm:
            self.update_up_norm()
        m = self.up_norm.get(nid, None)
        _assert(m is not None, "up_norm missing after update")
        return m

    def norm_squared_from_up(self) -> float:
        """
        Compute ||psi||^2 from cached root message.
        """
        root = self.state.tree.root
        m = self.get_up_norm(root)
        _assert(m.shape == (1,), "root message must be scalar")
        return float(np.real(m[0]))

    # -------------------------------------------------------------------------
    # Operator-inserted messages (minimal API, Phase 4 uses this)
    # -------------------------------------------------------------------------

    def compute_up_with_leaf_op(
        self,
        leaf_id: int,
        A: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Compute an "operator-inserted" up message map for a SINGLE onsite leaf operator.

        This returns up_op[nid] for all nodes on the path leaf->root, where:
          - at the leaf: use T' = (A @ T_leaf) and build message from |T'|^2
          - for ancestors: same internal contraction rule, but using the modified child message
            for the branch containing leaf_id, and cached up_norm for other children.

        Why this is useful:
          - You can route onsite contributions correctly without recomputing the whole tree.
          - For two-site terms you will generalize this to two modified branches + LCA merge.

        Notes:
          - This is still "norm-style message passing" (|T|^2). It’s the right backbone
            for stability/truncation bookkeeping and environment scaffolding.
          - For full correct Hamiltonian apply you will likely move from |T|^2 messages to
            linear-amplitude environments. This function is still valuable as a cache pattern.

        Returns:
          Dict[nid] -> message vector on parent bond of nid, for nid along path to root.
        """
        state = self.state
        tree = state.tree
        tensors = state.tensors

        _assert(leaf_id in tree.nodes and tree.nodes[leaf_id].is_leaf, "leaf_id must be a leaf")
        Tleaf = tensors[leaf_id]
        d, b = Tleaf.shape
        _assert(A.shape == (d, d), f"A shape mismatch: {A.shape} vs {(d, d)}")

        # Ensure baseline up_norm is fresh so we can reuse it for unaffected branches
        if self.dirty_nodes:
            self.update_up_norm()

        # Leaf modified message
        Tmod = (A @ Tleaf).astype(np.complex128, copy=False)
        up_op: Dict[int, np.ndarray] = {}
        up_op[leaf_id] = _leaf_up_message(Tmod)

        # Walk upward, recomputing only nodes on the path
        u = tree.nodes[leaf_id].parent
        child_on_path = leaf_id

        while u is not None:
            node = tree.nodes[u]
            _assert(not node.is_leaf, "ancestor of a leaf must be internal")

            # Build child messages tuple in correct order:
            # - use modified message for the child on the path
            # - reuse cached up_norm for other children
            msgs = []
            for c in node.children:
                if c == child_on_path:
                    msgs.append(up_op[c])
                else:
                    msgs.append(self.up_norm[c])
            T = tensors[u]
            up_op[u] = _internal_up_message(T, tuple(msgs))

            child_on_path = u
            u = node.parent

        return up_op


__all__ = ["TTNEnvironments"]
