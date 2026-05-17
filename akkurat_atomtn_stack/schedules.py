#!/usr/bin/env python3
# schedules.py
"""
Generic scheduling utilities for AtomTN.

This module is deliberately score-agnostic. It turns already-computed scalar
scores into bounded integer schedules used by the AtomTN runtime:

- per-leaf physical dimensions for LocalFiberBuilder / TTNState adaptation;
- per-node parent-bond caps for TTN truncation;
- simple cadence and cache-bucket helpers shared by integrators/builders;
- small temporal smoothing utilities for slowly varying schedules.

Complementary role with curvature.py
------------------------------------
curvature.py should own physics-specific score construction from FlowDiagnostics
or geometry/flow fields. schedules.py should own generic score normalization and
schedule mapping. A small curvature_scores_from_flow compatibility shim is kept
at the bottom for older scripts; new code should import that function from
curvature.py once the upgraded file is in place.

NumPy-only. No flow, TTN, projection, or Hamiltonian imports are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from math_utils import _assert, clip_int
except Exception:  # pragma: no cover - standalone fallback
    def _assert(cond: bool, msg: str) -> None:
        if not bool(cond):
            raise ValueError(str(msg))

    def clip_int(x: int, lo: int, hi: int) -> int:
        return int(np.clip(int(x), int(lo), int(hi)))


_EPS = 1e-12


# =============================================================================
# Configuration containers
# =============================================================================

@dataclass(frozen=True)
class ScoreNormConfig:
    """Configuration for score normalization.

    method:
        "minmax" maps min->0 and max->1.
        "zscore" maps through tanh(z / z_scale) into [0, 1].
        "rank" maps ordinal rank to [0, 1], useful for outlier-heavy scores.
        "none" returns finite scores unchanged.

    clip_percentiles:
        Optional pair (lo, hi), e.g. (2, 98), applied before normalization.
    """

    method: str = "minmax"
    eps: float = _EPS
    clip_percentiles: Optional[Tuple[float, float]] = None
    z_scale: float = 3.0


@dataclass(frozen=True)
class IntegerScheduleConfig:
    """Generic integer schedule mapping from normalized scores."""

    base: int
    minimum: int
    maximum: int
    strength: float = 1.0
    quantize_to: int = 1
    root_value: int = 1


# =============================================================================
# Small guards / coercion
# =============================================================================

def _as_1d_float(x: Any, *, name: str = "scores", expected_len: Optional[int] = None) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like") from exc

    if expected_len is not None and int(arr.size) != int(expected_len):
        raise ValueError(f"{name} length mismatch: expected {expected_len}, got {arr.size}")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def _children_of(tree: Any, nid: int) -> List[int]:
    node = tree.nodes[int(nid)]
    if hasattr(node, "children"):
        return [int(c) for c in getattr(node, "children")]
    if hasattr(node, "child_ids"):
        return [int(c) for c in getattr(node, "child_ids")]
    return []


def _is_leaf(tree: Any, nid: int) -> bool:
    node = tree.nodes[int(nid)]
    return bool(getattr(node, "is_leaf", len(_children_of(tree, nid)) == 0))


def _post_order(tree: Any) -> List[int]:
    if hasattr(tree, "post_order") and callable(getattr(tree, "post_order")):
        return [int(x) for x in tree.post_order()]

    root = int(tree.root)
    out: List[int] = []

    def dfs(u: int) -> None:
        for c in _children_of(tree, u):
            dfs(c)
        out.append(int(u))

    dfs(root)
    return out


def _depth_of(tree: Any, nid: int) -> int:
    d = 0
    cur = int(nid)
    seen = set()
    while cur not in seen:
        seen.add(cur)
        parent = getattr(tree.nodes[cur], "parent", None)
        if parent is None:
            break
        cur = int(parent)
        d += 1
    return int(d)


def _round_to_multiple(x: np.ndarray, q: int) -> np.ndarray:
    q = int(max(1, q))
    if q == 1:
        return np.rint(x).astype(int)
    return (np.rint(x / float(q)) * q).astype(int)


# =============================================================================
# Cadence helpers
# =============================================================================

def bucket_id(step_id: int, bucket_every: int = 1) -> int:
    """Return a deterministic cache bucket id for an outer integrator step."""
    return int(int(step_id) // max(int(bucket_every), 1))


def cadence_due(step_id: int, every: Optional[int], *, include_zero: bool = True) -> bool:
    """Return True when a periodic schedule should fire at step_id.

    This helper treats ``every <= 0`` or ``None`` as disabled.
    """
    if every is None:
        return False
    e = int(every)
    if e <= 0:
        return False
    k = int(step_id)
    if k == 0 and include_zero:
        return True
    return bool((k + 1) % e == 0)


# =============================================================================
# Score normalization / smoothing
# =============================================================================

def sanitize_scores(scores: Any, *, expected_len: Optional[int] = None, name: str = "scores") -> np.ndarray:
    """Return a finite 1D float64 score vector."""
    return _as_1d_float(scores, name=name, expected_len=expected_len)


def normalize_scores(
    scores: Any,
    *,
    method: str = "minmax",
    eps: float = _EPS,
    clip_percentiles: Optional[Tuple[float, float]] = None,
    z_scale: float = 3.0,
) -> np.ndarray:
    """Normalize a score vector.

    The default reproduces the original AtomTN behavior: finite min-max scaling
    into [0, 1], returning all zeros when scores are constant.
    """
    s = sanitize_scores(scores)
    method_l = str(method or "minmax").lower().strip()
    eps_f = max(float(eps), _EPS)

    if clip_percentiles is not None:
        lo_p, hi_p = map(float, clip_percentiles)
        lo_p = float(np.clip(lo_p, 0.0, 100.0))
        hi_p = float(np.clip(hi_p, 0.0, 100.0))
        if hi_p < lo_p:
            lo_p, hi_p = hi_p, lo_p
        lo_v, hi_v = np.percentile(s, [lo_p, hi_p])
        s = np.clip(s, float(lo_v), float(hi_v))

    if method_l in {"none", "raw", "identity"}:
        return s.astype(np.float64, copy=False)

    if method_l == "minmax":
        lo = float(np.min(s))
        hi = float(np.max(s))
        denom = hi - lo
        if not np.isfinite(denom) or denom <= eps_f:
            return np.zeros_like(s, dtype=np.float64)
        return ((s - lo) / denom).astype(np.float64, copy=False)

    if method_l == "zscore":
        mu = float(np.mean(s))
        sd = float(np.std(s))
        if not np.isfinite(sd) or sd <= eps_f:
            return np.zeros_like(s, dtype=np.float64)
        z = (s - mu) / sd
        scale = max(float(z_scale), eps_f)
        # maps roughly z=-scale -> 0.119, z=0 -> 0.5, z=scale -> 0.881
        return (0.5 * (np.tanh(z / scale) + 1.0)).astype(np.float64, copy=False)

    if method_l == "rank":
        if s.size == 1:
            return np.zeros_like(s, dtype=np.float64)
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(s.size, dtype=np.float64)
        return (ranks / float(max(1, s.size - 1))).astype(np.float64, copy=False)

    raise ValueError(f"unknown score normalization method: {method}")


def normalize_scores_with_config(scores: Any, cfg: ScoreNormConfig = ScoreNormConfig()) -> np.ndarray:
    """Normalize scores using a ScoreNormConfig dataclass."""
    return normalize_scores(
        scores,
        method=cfg.method,
        eps=cfg.eps,
        clip_percentiles=cfg.clip_percentiles,
        z_scale=cfg.z_scale,
    )


def smooth_scores_ema(
    previous: Optional[Any],
    current: Any,
    *,
    beta: float = 0.2,
) -> np.ndarray:
    """Exponential moving average for score vectors.

    beta=1.0 returns current; beta=0.0 returns previous.
    """
    cur = sanitize_scores(current, name="current")
    if previous is None:
        return cur.astype(np.float64, copy=True)
    prev = sanitize_scores(previous, name="previous", expected_len=cur.size)
    b = float(np.clip(beta, 0.0, 1.0))
    return ((1.0 - b) * prev + b * cur).astype(np.float64, copy=False)


# =============================================================================
# Generic integer schedule mapping
# =============================================================================

def integer_schedule_from_scores(
    scores: Any,
    *,
    base: int,
    minimum: int,
    maximum: int,
    strength: float = 1.0,
    quantize_to: int = 1,
    norm: ScoreNormConfig | None = None,
) -> np.ndarray:
    """Map scalar scores to bounded integer values.

    Formula:
        value = base * (1 + strength * normalized_score)

    Then values are rounded, optionally quantized to a multiple, and clipped.
    """
    mn = int(minimum)
    mx = int(maximum)
    _assert(mn >= 1, "minimum must be >= 1")
    _assert(mx >= mn, "maximum must be >= minimum")
    _assert(int(base) >= 1, "base must be >= 1")

    cfg = norm or ScoreNormConfig()
    ns = normalize_scores_with_config(scores, cfg)
    raw = float(base) * (1.0 + float(strength) * ns)
    vals = _round_to_multiple(raw, int(max(1, quantize_to)))
    vals = np.clip(vals, mn, mx).astype(int, copy=False)
    return vals


# =============================================================================
# Fiber physical-dimension schedules
# =============================================================================

def adaptive_fiber_dims(
    scores_leaf: Any,
    d_uniform: int,
    d_min: int,
    d_max: int,
    strength: float = 1.0,
    *,
    norm: ScoreNormConfig | None = None,
    quantize_to: int = 1,
    cap: Optional[int] = None,
) -> np.ndarray:
    """Turn per-leaf scores into per-leaf physical dimensions.

    Existing callers can use the original five positional arguments. Optional
    keyword arguments add production controls without breaking compatibility.
    """
    max_dim = int(d_max)
    if cap is not None:
        max_dim = min(max_dim, int(cap))
    min_dim = min(int(d_min), max_dim)
    base = min(max(int(d_uniform), min_dim), max_dim)

    return integer_schedule_from_scores(
        scores_leaf,
        base=base,
        minimum=min_dim,
        maximum=max_dim,
        strength=float(strength),
        quantize_to=int(max(1, quantize_to)),
        norm=norm or ScoreNormConfig(method="minmax"),
    )


# =============================================================================
# Tree bond schedules
# =============================================================================

def subtree_leaf_map(tree: Any) -> Dict[int, List[int]]:
    """Return node_id -> leaf ids under that node.

    The returned lists preserve the tree's child ordering and are deterministic.
    """
    _assert(hasattr(tree, "nodes") and hasattr(tree, "root") and hasattr(tree, "leaves"), "invalid tree object")

    out: Dict[int, List[int]] = {}

    def collect(nid: int) -> List[int]:
        nid = int(nid)
        if nid in out:
            return out[nid]
        if _is_leaf(tree, nid):
            out[nid] = [nid]
            return out[nid]
        acc: List[int] = []
        for c in _children_of(tree, nid):
            acc.extend(collect(c))
        out[nid] = acc
        return acc

    collect(int(tree.root))
    # Include any disconnected entries only to fail deterministically later; a
    # valid Tree should already be connected.
    for nid in tree.nodes:
        collect(int(nid))
    return out


def node_scores_from_leaf_scores(tree: Any, scores_leaf: Any) -> Dict[int, float]:
    """Aggregate per-leaf scores to every tree node by subtree mean."""
    leaves = [int(l) for l in getattr(tree, "leaves")]
    scores = sanitize_scores(scores_leaf, expected_len=len(leaves), name="scores_leaf")
    leaf_pos = {lid: i for i, lid in enumerate(leaves)}
    sub = subtree_leaf_map(tree)

    node_score: Dict[int, float] = {}
    for nid, leaf_ids in sub.items():
        idxs = [leaf_pos[int(l)] for l in leaf_ids if int(l) in leaf_pos]
        node_score[int(nid)] = float(np.mean(scores[idxs])) if idxs else 0.0
    return node_score


def bond_schedule_from_scores(
    tree: Any,
    scores_leaf: Any,
    base_bond: int,
    min_bond: int,
    max_bond: int,
    strength: float = 1.0,
    *,
    norm: ScoreNormConfig | None = None,
    quantize_to: int = 1,
    root_bond: int = 1,
) -> Dict[int, int]:
    """Build a per-node parent-bond target schedule.

    The score of an internal node is the mean score of leaves in its subtree.
    The root parent bond is forced to ``root_bond``. For normal TTN states this
    must remain 1.
    """
    _assert(hasattr(tree, "nodes") and hasattr(tree, "root") and hasattr(tree, "leaves"), "invalid tree object")
    node_score = node_scores_from_leaf_scores(tree, scores_leaf)

    # Normalize over all node scores so high-curvature subtrees receive larger caps.
    ordered_nodes = [int(nid) for nid in tree.nodes.keys()]
    vals = np.asarray([node_score.get(nid, 0.0) for nid in ordered_nodes], dtype=np.float64)
    targets = integer_schedule_from_scores(
        vals,
        base=int(base_bond),
        minimum=int(min_bond),
        maximum=int(max_bond),
        strength=float(strength),
        quantize_to=int(max(1, quantize_to)),
        norm=norm or ScoreNormConfig(method="minmax"),
    )

    schedule = {int(nid): int(targets[i]) for i, nid in enumerate(ordered_nodes)}
    schedule[int(tree.root)] = int(root_bond)
    return schedule


def depth_weighted_bond_schedule(
    tree: Any,
    scores_leaf: Any,
    base_bond: int,
    min_bond: int,
    max_bond: int,
    strength: float = 1.0,
    *,
    depth_strength: float = 0.0,
    root_bond: int = 1,
) -> Dict[int, int]:
    """Bond schedule with optional depth bias.

    Positive depth_strength gives deeper nodes slightly larger scores before
    normalization. This is useful when leaf-local detail should be preserved.
    """
    node_score = node_scores_from_leaf_scores(tree, scores_leaf)
    if float(depth_strength) != 0.0:
        depths = {nid: _depth_of(tree, nid) for nid in tree.nodes}
        max_depth = max(depths.values()) if depths else 1
        for nid in list(node_score.keys()):
            node_score[nid] *= 1.0 + float(depth_strength) * (depths.get(nid, 0) / max(1, max_depth))

    ordered_nodes = [int(nid) for nid in tree.nodes.keys()]
    vals = np.asarray([node_score.get(nid, 0.0) for nid in ordered_nodes], dtype=np.float64)
    targets = integer_schedule_from_scores(
        vals,
        base=int(base_bond),
        minimum=int(min_bond),
        maximum=int(max_bond),
        strength=float(strength),
        norm=ScoreNormConfig(method="minmax"),
    )
    schedule = {int(nid): int(targets[i]) for i, nid in enumerate(ordered_nodes)}
    schedule[int(tree.root)] = int(root_bond)
    return schedule


# =============================================================================
# Schedule comparison / stabilization
# =============================================================================

def clamp_schedule_delta(
    previous: Mapping[int, int],
    proposed: Mapping[int, int],
    *,
    max_delta: int = 2,
    minimum: int = 1,
    maximum: Optional[int] = None,
    preserve_keys: Optional[Iterable[int]] = None,
) -> Dict[int, int]:
    """Limit how much an integer schedule may change in one update."""
    md = int(max(0, max_delta))
    mn = int(max(1, minimum))
    mx = None if maximum is None else int(maximum)
    preserve = {int(k) for k in (preserve_keys or [])}

    out: Dict[int, int] = {}
    keys = set(int(k) for k in previous.keys()) | set(int(k) for k in proposed.keys())
    for k in keys:
        p = int(previous.get(k, proposed.get(k, mn)))
        q = int(proposed.get(k, p))
        if k in preserve:
            val = q
        else:
            val = int(np.clip(q, p - md, p + md))
        if mx is not None:
            val = int(np.clip(val, mn, mx))
        else:
            val = max(mn, val)
        out[k] = val
    return out


def schedule_changed_fraction(a: Mapping[int, int], b: Mapping[int, int]) -> float:
    """Return fraction of keys whose values differ between two schedules."""
    keys = set(int(k) for k in a.keys()) | set(int(k) for k in b.keys())
    if not keys:
        return 0.0
    changed = sum(1 for k in keys if int(a.get(k, -1)) != int(b.get(k, -1)))
    return float(changed / len(keys))


# =============================================================================
# Backward compatibility shim
# =============================================================================

def curvature_scores_from_flow(diags: Sequence[Any]) -> np.ndarray:
    """Compatibility shim for older imports.

    New code should import curvature_scores_from_flow from curvature.py. This
    implementation is intentionally minimal and depends only on diagnostics-like
    objects with divX_scalar and optional DdivDt_scalar fields.
    """
    _assert(len(diags) > 0, "curvature_scores_from_flow: need diagnostics list")
    last = diags[-1]
    div = sanitize_scores(getattr(last, "divX_scalar"), name="divX_scalar")
    ddiv = getattr(last, "DdivDt_scalar", None)
    if ddiv is None:
        return np.abs(div).astype(np.float64, copy=False)
    d = sanitize_scores(ddiv, expected_len=div.size, name="DdivDt_scalar")
    return (np.abs(div) + 0.5 * np.abs(d)).astype(np.float64, copy=False)


__all__ = [
    "ScoreNormConfig",
    "IntegerScheduleConfig",
    "bucket_id",
    "cadence_due",
    "sanitize_scores",
    "normalize_scores",
    "normalize_scores_with_config",
    "smooth_scores_ema",
    "integer_schedule_from_scores",
    "adaptive_fiber_dims",
    "subtree_leaf_map",
    "node_scores_from_leaf_scores",
    "bond_schedule_from_scores",
    "depth_weighted_bond_schedule",
    "clamp_schedule_delta",
    "schedule_changed_fraction",
    "curvature_scores_from_flow",
]
