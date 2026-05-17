#!/usr/bin/env python3
# curvature.py
"""
Curvature and flow-derived score utilities for AtomTN.

Complementary role with schedules.py
------------------------------------
This module owns physics-specific score construction. It converts AtomTN flow
objects and FlowDiagnostics-like records into finite per-node or per-leaf scalar
curvature proxies.

schedules.py owns generic normalization and integer schedule mapping. For
backward compatibility, this module still re-exports small wrappers named
normalize_scores, adaptive_fiber_dims, and bond_schedule_from_scores.

Core scoring model
------------------
For commutative flow diagnostics, the default proxy is:

    score(u) = |div X(u)| + 0.5 |d/dt div X(u)|

For noncommutative diagnostics, this module also uses optional κ channels when
present:

    ||κ_raw(u)||_F, ||κ_su2(u)||_F, ||c_su2(u)||_2

All outputs are sanitized to finite float64 arrays. The historical default API
returns a vector of length 64 when the supplied diagnostics are length 64.

NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from math_utils import _assert, fro_norm
except Exception:  # pragma: no cover - standalone fallback
    def _assert(cond: bool, msg: str) -> None:
        if not bool(cond):
            raise ValueError(str(msg))

    def fro_norm(A: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(A).reshape(-1)))

try:
    from schedules import (
        ScoreNormConfig,
        sanitize_scores,
        normalize_scores as _schedule_normalize_scores,
        adaptive_fiber_dims as _schedule_adaptive_fiber_dims,
        bond_schedule_from_scores as _schedule_bond_schedule_from_scores,
        smooth_scores_ema,
    )
except Exception:  # pragma: no cover - minimal fallback if schedules.py is absent
    @dataclass(frozen=True)
    class ScoreNormConfig:  # type: ignore[no-redef]
        method: str = "minmax"
        eps: float = 1e-12
        clip_percentiles: Optional[Tuple[float, float]] = None
        z_scale: float = 3.0

    def sanitize_scores(scores: Any, *, expected_len: Optional[int] = None, name: str = "scores") -> np.ndarray:  # type: ignore[no-redef]
        arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if expected_len is not None and arr.size != int(expected_len):
            raise ValueError(f"{name} length mismatch: expected {expected_len}, got {arr.size}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    def _schedule_normalize_scores(scores: Any, **_: Any) -> np.ndarray:
        s = sanitize_scores(scores)
        lo, hi = float(np.min(s)), float(np.max(s))
        den = max(hi - lo, 1e-12)
        return (s - lo) / den if den > 1e-12 else np.zeros_like(s)

    def _schedule_adaptive_fiber_dims(scores_leaf: Any, d_uniform: int, d_min: int, d_max: int, strength: float = 1.0, **_: Any) -> np.ndarray:
        ns = _schedule_normalize_scores(scores_leaf)
        d = np.round(float(d_uniform) * (1.0 + float(strength) * ns)).astype(int)
        return np.clip(d, int(d_min), int(d_max)).astype(int)

    def _schedule_bond_schedule_from_scores(tree: Any, scores_leaf: Any, base_bond: int, min_bond: int, max_bond: int, strength: float = 1.0, **_: Any) -> Dict[int, int]:
        scores = sanitize_scores(scores_leaf, expected_len=len(tree.leaves), name="scores_leaf")
        ns = _schedule_normalize_scores(scores)
        leaf_pos = {int(l): i for i, l in enumerate(tree.leaves)}
        out: Dict[int, int] = {}
        for nid, node in tree.nodes.items():
            if getattr(node, "is_leaf", False):
                val = ns[leaf_pos[int(nid)]] if int(nid) in leaf_pos else 0.0
            else:
                vals: List[float] = []
                stack = list(getattr(node, "children", []))
                while stack:
                    x = int(stack.pop())
                    xn = tree.nodes[x]
                    if getattr(xn, "is_leaf", False) and x in leaf_pos:
                        vals.append(float(ns[leaf_pos[x]]))
                    else:
                        stack.extend(list(getattr(xn, "children", [])))
                val = float(np.mean(vals)) if vals else 0.0
            out[int(nid)] = int(np.clip(round(float(base_bond) * (1.0 + float(strength) * val)), int(min_bond), int(max_bond)))
        out[int(tree.root)] = 1
        return out

    def smooth_scores_ema(previous: Optional[Any], current: Any, *, beta: float = 0.2) -> np.ndarray:  # type: ignore[no-redef]
        cur = sanitize_scores(current)
        if previous is None:
            return cur
        prev = sanitize_scores(previous, expected_len=cur.size)
        b = float(np.clip(beta, 0.0, 1.0))
        return (1.0 - b) * prev + b * cur


_EPS = 1e-12


# =============================================================================
# Configuration containers
# =============================================================================

@dataclass(frozen=True)
class CurvatureChannelWeights:
    """Weights used to combine diagnostic channels into one score vector."""

    div: float = 1.0
    ddiv_dt: float = 0.5
    kappa_raw: float = 0.25
    kappa_su2: float = 0.25
    kappa_su2_coeff: float = 0.25
    alarm_bias: float = 0.0


@dataclass(frozen=True)
class CurvatureScoreConfig:
    """Controls construction and optional stabilization of curvature scores.

    expected_nodes:
        Expected score length when the source diagnostic length is ambiguous.
        AtomTN's default tetra mesh uses 64 nodes.

    use_absolute:
        If True, divergence-style channels use absolute values.

    normalize_channels:
        If True, each channel is min-max normalized before weighted summation.
        The default is False to preserve the original magnitude-sensitive score.

    history_ema_beta:
        Optional EMA beta for smoothing a full diagnostics history. None disables
        smoothing. Values closer to 1 track the latest diagnostic more strongly.
    """

    expected_nodes: int = 64
    weights: CurvatureChannelWeights = field(default_factory=CurvatureChannelWeights)
    use_absolute: bool = True
    normalize_channels: bool = False
    history_ema_beta: Optional[float] = None
    norm: ScoreNormConfig = field(default_factory=lambda: ScoreNormConfig(method="minmax"))


@dataclass(frozen=True)
class Hotspot:
    """A high-curvature node/leaf record."""

    node_id: int
    score: float
    rank: int


# =============================================================================
# Generic coercion and introspection helpers
# =============================================================================

def _as_1d_finite(x: Any, *, name: str = "array", expected_len: Optional[int] = None) -> np.ndarray:
    arr = sanitize_scores(x, expected_len=expected_len, name=name)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr.astype(np.float64, copy=False)


def _safe_channel(x: Any, *, n: int, name: str) -> np.ndarray:
    try:
        arr = _as_1d_finite(x, name=name)
    except Exception:
        return np.zeros((int(n),), dtype=np.float64)
    if arr.size == int(n):
        return arr
    if arr.size == 0:
        return np.zeros((int(n),), dtype=np.float64)
    if arr.size > int(n):
        return arr[: int(n)].astype(np.float64, copy=False)
    out = np.zeros((int(n),), dtype=np.float64)
    out[: arr.size] = arr
    return out


def _infer_num_nodes(diag: Any, cfg: CurvatureScoreConfig) -> int:
    for attr in ("divX_scalar", "DdivDt_scalar", "kappa_su2_coeffs"):
        val = getattr(diag, attr, None)
        if val is None:
            continue
        try:
            arr = np.asarray(val)
            if arr.ndim >= 1 and int(arr.shape[0]) > 0:
                return int(arr.shape[0])
        except Exception:
            pass

    for attr in ("kappa_node_mats_raw", "kappa_node_mats_su2"):
        val = getattr(diag, attr, None)
        if isinstance(val, Mapping) and val:
            try:
                return max(int(max(val.keys())) + 1, int(cfg.expected_nodes))
            except Exception:
                return int(cfg.expected_nodes)

    return int(cfg.expected_nodes)


def _mapping_matrix_norms(mats: Optional[Mapping[int, Any]], *, n: int) -> np.ndarray:
    out = np.zeros((int(n),), dtype=np.float64)
    if not isinstance(mats, Mapping):
        return out
    for k, M in mats.items():
        try:
            idx = int(k)
            if 0 <= idx < int(n):
                out[idx] = float(fro_norm(np.asarray(M, dtype=np.complex128)))
        except Exception:
            continue
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def _coeff_norms(coeffs: Any, *, n: int) -> np.ndarray:
    if coeffs is None:
        return np.zeros((int(n),), dtype=np.float64)
    try:
        arr = np.asarray(coeffs, dtype=np.float64)
        if arr.ndim == 1:
            return _safe_channel(np.abs(arr), n=n, name="kappa_su2_coeffs")
        if arr.ndim >= 2:
            vals = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
            return _safe_channel(vals, n=n, name="kappa_su2_coeff_norms")
    except Exception:
        pass
    return np.zeros((int(n),), dtype=np.float64)


def _maybe_normalize_channel(x: np.ndarray, cfg: CurvatureScoreConfig) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if not cfg.normalize_channels:
        return x
    return _schedule_normalize_scores(x, method=cfg.norm.method, eps=cfg.norm.eps, clip_percentiles=cfg.norm.clip_percentiles, z_scale=cfg.norm.z_scale)


def _children_of(tree: Any, nid: int) -> List[int]:
    node = tree.nodes[int(nid)]
    if hasattr(node, "children"):
        return [int(x) for x in getattr(node, "children")]
    if hasattr(node, "child_ids"):
        return [int(x) for x in getattr(node, "child_ids")]
    return []


def _leaf_ids(tree: Any) -> List[int]:
    return [int(x) for x in getattr(tree, "leaves")]


# =============================================================================
# Diagnostic channels and score construction
# =============================================================================

def diagnostic_channels(diag: Any, cfg: CurvatureScoreConfig = CurvatureScoreConfig()) -> Dict[str, np.ndarray]:
    """Extract finite diagnostic channels from a FlowDiagnostics-like object."""
    n = _infer_num_nodes(diag, cfg)
    div = _safe_channel(getattr(diag, "divX_scalar", np.zeros(n)), n=n, name="divX_scalar")
    ddiv_src = getattr(diag, "DdivDt_scalar", None)
    ddiv = np.zeros((n,), dtype=np.float64) if ddiv_src is None else _safe_channel(ddiv_src, n=n, name="DdivDt_scalar")

    if cfg.use_absolute:
        div = np.abs(div)
        ddiv = np.abs(ddiv)

    raw = _mapping_matrix_norms(getattr(diag, "kappa_node_mats_raw", None), n=n)
    su2 = _mapping_matrix_norms(getattr(diag, "kappa_node_mats_su2", None), n=n)
    coeff = _coeff_norms(getattr(diag, "kappa_su2_coeffs", None), n=n)

    alarm = float(getattr(diag, "alarm_score", 0.0) or 0.0)
    if not np.isfinite(alarm):
        alarm = 0.0
    alarm_arr = np.full((n,), float(abs(alarm)), dtype=np.float64)

    return {
        "div": div,
        "ddiv_dt": ddiv,
        "kappa_raw": raw,
        "kappa_su2": su2,
        "kappa_su2_coeff": coeff,
        "alarm": alarm_arr,
    }


def combine_curvature_channels(
    channels: Mapping[str, np.ndarray],
    cfg: CurvatureScoreConfig = CurvatureScoreConfig(),
) -> np.ndarray:
    """Combine named diagnostic channels into one finite score vector."""
    if not channels:
        return np.zeros((int(cfg.expected_nodes),), dtype=np.float64)

    # Determine common length from the first channel.
    first = next(iter(channels.values()))
    n = int(np.asarray(first).reshape(-1).size)
    w = cfg.weights

    weighted = [
        (float(w.div), "div"),
        (float(w.ddiv_dt), "ddiv_dt"),
        (float(w.kappa_raw), "kappa_raw"),
        (float(w.kappa_su2), "kappa_su2"),
        (float(w.kappa_su2_coeff), "kappa_su2_coeff"),
        (float(w.alarm_bias), "alarm"),
    ]

    score = np.zeros((n,), dtype=np.float64)
    for weight, key in weighted:
        if weight == 0.0 or key not in channels:
            continue
        ch = _safe_channel(channels[key], n=n, name=key)
        score += weight * _maybe_normalize_channel(ch, cfg)

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def curvature_scores_from_diagnostic(
    diag: Any,
    cfg: CurvatureScoreConfig = CurvatureScoreConfig(),
) -> np.ndarray:
    """Return per-node curvature scores from one FlowDiagnostics-like object."""
    return combine_curvature_channels(diagnostic_channels(diag, cfg), cfg)


def curvature_scores_from_flow(
    diags: Sequence[Any],
    cfg: Optional[CurvatureScoreConfig] = None,
) -> np.ndarray:
    """Return per-node curvature scores from a diagnostics history.

    Backward-compatible default:
        |divX| + 0.5|DdivDt|

    If cfg.history_ema_beta is set, all diagnostics in the supplied history are
    scored and smoothed by exponential moving average.
    """
    _assert(len(diags) > 0, "curvature_scores_from_flow: need diagnostics")
    cfg = cfg or CurvatureScoreConfig()

    if cfg.history_ema_beta is None:
        return curvature_scores_from_diagnostic(diags[-1], cfg)

    acc: Optional[np.ndarray] = None
    for d in diags:
        cur = curvature_scores_from_diagnostic(d, cfg)
        acc = smooth_scores_ema(acc, cur, beta=float(cfg.history_ema_beta))
    _assert(acc is not None, "curvature_scores_from_flow: internal empty history")
    return acc.astype(np.float64, copy=False)


# =============================================================================
# Flow-field direct scoring
# =============================================================================

def edge_energy_scores_from_flow(
    X: Any,
    *,
    num_nodes: int = 64,
    normalize_by_degree: bool = True,
) -> np.ndarray:
    """Compute node scores from a GraphVectorField-like edge field.

    Each directed edge contributes |x_uv|^2 for scalar fields or ||X_uv||_F^2
    for matrix-valued fields to its source node. If both orientations are stored,
    both are included as directed energy.
    """
    n = int(num_nodes)
    scores = np.zeros((n,), dtype=np.float64)
    deg = np.zeros((n,), dtype=np.float64)
    edge_values = getattr(X, "edge_values", {})
    if not isinstance(edge_values, Mapping):
        return scores

    for edge, val in edge_values.items():
        try:
            u, _v = edge
            uu = int(u)
            if not (0 <= uu < n):
                continue
            arr = np.asarray(val)
            if arr.ndim == 0:
                mag2 = float(abs(complex(arr.item())) ** 2)
            else:
                mag = fro_norm(arr.astype(np.complex128, copy=False))
                mag2 = float(mag * mag)
            if np.isfinite(mag2):
                scores[uu] += mag2
                deg[uu] += 1.0
        except Exception:
            continue

    if normalize_by_degree:
        scores = scores / np.maximum(deg, 1.0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def divergence_scores_from_edge_field(
    X: Any,
    *,
    num_nodes: int = 64,
) -> np.ndarray:
    """Compute a simple divergence-magnitude proxy directly from edge values."""
    n = int(num_nodes)
    edge_values = getattr(X, "edge_values", {})
    matrix_valued = bool(getattr(X, "matrix_valued", False))
    if not isinstance(edge_values, Mapping):
        return np.zeros((n,), dtype=np.float64)

    if not matrix_valued:
        div = np.zeros((n,), dtype=np.float64)
        for edge, val in edge_values.items():
            try:
                u, _v = edge
                uu = int(u)
                if 0 <= uu < n:
                    div[uu] += float(val)
            except Exception:
                continue
        return np.abs(np.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0))

    # Matrix-valued: accumulate matrices by node, then take Frobenius norms.
    shape: Optional[Tuple[int, int]] = None
    for val in edge_values.values():
        try:
            arr = np.asarray(val, dtype=np.complex128)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                shape = (int(arr.shape[0]), int(arr.shape[1]))
                break
        except Exception:
            continue
    if shape is None:
        return np.zeros((n,), dtype=np.float64)

    div_m = {i: np.zeros(shape, dtype=np.complex128) for i in range(n)}
    for edge, val in edge_values.items():
        try:
            u, _v = edge
            uu = int(u)
            if 0 <= uu < n:
                arr = np.asarray(val, dtype=np.complex128)
                if arr.shape == shape:
                    div_m[uu] = div_m[uu] + arr
        except Exception:
            continue
    return np.asarray([fro_norm(div_m[i]) for i in range(n)], dtype=np.float64)


# =============================================================================
# Tree and leaf mapping
# =============================================================================

def leaf_scores_from_node_scores(tree: Any, node_scores: Any) -> np.ndarray:
    """Map a node-indexed score vector to tree.leaves order."""
    leaves = _leaf_ids(tree)
    scores = _as_1d_finite(node_scores, name="node_scores")
    out = np.zeros((len(leaves),), dtype=np.float64)
    for i, lid in enumerate(leaves):
        if 0 <= int(lid) < scores.size:
            out[i] = scores[int(lid)]
    return out


def node_scores_from_leaf_scores(tree: Any, leaf_scores: Any) -> Dict[int, float]:
    """Aggregate leaf scores to all tree nodes by subtree mean."""
    leaves = _leaf_ids(tree)
    scores = _as_1d_finite(leaf_scores, name="leaf_scores", expected_len=len(leaves))
    leaf_pos = {lid: i for i, lid in enumerate(leaves)}
    out: Dict[int, float] = {}

    def collect(nid: int) -> List[int]:
        node = tree.nodes[int(nid)]
        if getattr(node, "is_leaf", False) or not _children_of(tree, nid):
            return [int(nid)]
        acc: List[int] = []
        for c in _children_of(tree, nid):
            acc.extend(collect(c))
        return acc

    for nid in tree.nodes:
        ids = [l for l in collect(int(nid)) if l in leaf_pos]
        out[int(nid)] = float(np.mean([scores[leaf_pos[l]] for l in ids])) if ids else 0.0
    return out


def curvature_scores_for_tree(
    tree: Any,
    diags: Sequence[Any],
    cfg: Optional[CurvatureScoreConfig] = None,
) -> np.ndarray:
    """Return curvature scores ordered by tree.leaves."""
    node_scores = curvature_scores_from_flow(diags, cfg=cfg)
    return leaf_scores_from_node_scores(tree, node_scores)


# =============================================================================
# Analysis utilities
# =============================================================================

def normalize_scores(scores: Any) -> np.ndarray:
    """Backward-compatible min-max normalization wrapper."""
    return _schedule_normalize_scores(scores, method="minmax")


def normalized_curvature_scores(
    scores: Any,
    norm: ScoreNormConfig = ScoreNormConfig(method="minmax"),
) -> np.ndarray:
    """Normalize curvature scores with an explicit ScoreNormConfig."""
    return _schedule_normalize_scores(
        scores,
        method=norm.method,
        eps=norm.eps,
        clip_percentiles=norm.clip_percentiles,
        z_scale=norm.z_scale,
    )


def curvature_hotspots(
    scores: Any,
    *,
    top_k: int = 8,
    node_ids: Optional[Sequence[int]] = None,
    min_score: Optional[float] = None,
) -> List[Hotspot]:
    """Return top-k high-score nodes/leaves as Hotspot records."""
    s = _as_1d_finite(scores, name="scores")
    ids = [int(x) for x in node_ids] if node_ids is not None else list(range(s.size))
    if len(ids) != s.size:
        raise ValueError(f"node_ids length mismatch: expected {s.size}, got {len(ids)}")
    order = np.argsort(-s, kind="mergesort")
    out: List[Hotspot] = []
    for rank, idx in enumerate(order, start=1):
        score = float(s[int(idx)])
        if min_score is not None and score < float(min_score):
            continue
        out.append(Hotspot(node_id=int(ids[int(idx)]), score=score, rank=rank))
        if len(out) >= int(top_k):
            break
    return out


def curvature_summary(scores: Any) -> Dict[str, float]:
    """Return compact descriptive statistics for a score vector."""
    s = _as_1d_finite(scores, name="scores")
    return {
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "l2": float(np.linalg.norm(s)),
        "p50": float(np.percentile(s, 50)),
        "p90": float(np.percentile(s, 90)),
        "p99": float(np.percentile(s, 99)),
    }


def curvature_delta(prev_scores: Any, next_scores: Any) -> Dict[str, float]:
    """Compare two score vectors."""
    a = _as_1d_finite(prev_scores, name="prev_scores")
    b = _as_1d_finite(next_scores, name="next_scores", expected_len=a.size)
    d = b - a
    denom = max(float(np.linalg.norm(a)), _EPS)
    return {
        "delta_l2": float(np.linalg.norm(d)),
        "relative_delta_l2": float(np.linalg.norm(d) / denom),
        "delta_mean": float(np.mean(d)),
        "delta_max_abs": float(np.max(np.abs(d))),
        "cosine": float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), _EPS)),
    }


def curvature_alarm_level(
    scores: Any,
    *,
    warn_quantile: float = 0.90,
    critical_quantile: float = 0.99,
    absolute_warn: Optional[float] = None,
    absolute_critical: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a coarse alarm classification from score distribution."""
    s = _as_1d_finite(scores, name="scores")
    q_warn = float(np.quantile(s, float(np.clip(warn_quantile, 0.0, 1.0))))
    q_crit = float(np.quantile(s, float(np.clip(critical_quantile, 0.0, 1.0))))
    max_s = float(np.max(s))

    warn_thr = q_warn if absolute_warn is None else max(q_warn, float(absolute_warn))
    crit_thr = q_crit if absolute_critical is None else max(q_crit, float(absolute_critical))

    if max_s >= crit_thr and crit_thr > 0:
        level = "critical"
    elif max_s >= warn_thr and warn_thr > 0:
        level = "warning"
    else:
        level = "ok"

    return {
        "level": level,
        "max_score": max_s,
        "warn_threshold": float(warn_thr),
        "critical_threshold": float(crit_thr),
    }


# =============================================================================
# Backward-compatible schedule wrappers
# =============================================================================

def adaptive_fiber_dims(
    scores_leaf: Any,
    d_uniform: int,
    d_min: int,
    d_max: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Compatibility wrapper; schedule logic lives in schedules.py."""
    return _schedule_adaptive_fiber_dims(scores_leaf, d_uniform, d_min, d_max, strength)


def bond_schedule_from_scores(
    tree: Any,
    scores_leaf: Any,
    base_bond: int,
    min_bond: int,
    max_bond: int,
    strength: float = 1.0,
) -> Dict[int, int]:
    """Compatibility wrapper; schedule logic lives in schedules.py."""
    return _schedule_bond_schedule_from_scores(tree, scores_leaf, base_bond, min_bond, max_bond, strength)


__all__ = [
    "CurvatureChannelWeights",
    "CurvatureScoreConfig",
    "Hotspot",
    "diagnostic_channels",
    "combine_curvature_channels",
    "curvature_scores_from_diagnostic",
    "curvature_scores_from_flow",
    "edge_energy_scores_from_flow",
    "divergence_scores_from_edge_field",
    "leaf_scores_from_node_scores",
    "node_scores_from_leaf_scores",
    "curvature_scores_for_tree",
    "normalize_scores",
    "normalized_curvature_scores",
    "curvature_hotspots",
    "curvature_summary",
    "curvature_delta",
    "curvature_alarm_level",
    "adaptive_fiber_dims",
    "bond_schedule_from_scores",
]
