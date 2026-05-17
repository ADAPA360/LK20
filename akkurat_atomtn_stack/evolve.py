#!/usr/bin/env python3
# evolve.py
"""
AtomTN TTN evolution runtime.

This module is the integration layer between:
- TTNState, the tree tensor-network state container;
- TreeMPOBuilder / CompiledTreeOperator, the Hamiltonian builders;
- apply.py, which supplies legacy scaffold application and Phase-4 zip-up apply.

Design goals
------------
1. Preserve the public AtomTN API used by the current runtime family:
      ApplyConfig
      TTNEvolveConfig
      TTNTimeEvolver
      TTNTimeEvolver.step_euler_legacy(...)
      TTNTimeEvolver.step_rk4_legacy(...)
      TTNTimeEvolver.step_rk4_end_truncate(...)
      TTNTimeEvolver.step_lie_trotter(...)
      TTNTimeEvolver.integrate_with_flow(...)

2. Support both execution paths:
      - legacy fast scaffold: apply_tree_mpo_fast(...)
      - correctness path: apply_compiled_operator_zipup(...)

3. Keep RK stage tensor layouts compatible. The zip-up/direct-sum baseline can
   grow bond dimensions during H|psi>; this evolver projects/pads derivatives
   back to the template state bond layout before RK additions.

4. Remain CPU/NumPy-only, deterministic, import-safe, and production-friendly.
"""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from math_utils import _assert
from ttn_state import TTNState

try:  # Avoid hard circularity during tooling/import checks.
    from hamiltonian import HamiltonianBuildConfig
except Exception:  # pragma: no cover
    HamiltonianBuildConfig = Any  # type: ignore


# =============================================================================
# Generic helpers
# =============================================================================

_EPS = 1e-12


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {"real": np.real(arr).astype(float).tolist(), "imag": np.imag(arr).astype(float).tolist()}
        return arr.astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _safe_norm_state(state: TTNState) -> float:
    try:
        v = float(state.amplitude_norm_squared())
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _node_children(tree: Any, nid: int) -> List[int]:
    node = tree.nodes[int(nid)]
    if hasattr(node, "children"):
        return [int(x) for x in getattr(node, "children")]
    if hasattr(node, "child_ids"):
        return [int(x) for x in getattr(node, "child_ids")]
    return []


def _post_order(tree: Any) -> List[int]:
    if hasattr(tree, "post_order") and callable(getattr(tree, "post_order")):
        return [int(x) for x in tree.post_order()]
    out: List[int] = []

    def dfs(u: int) -> None:
        for c in _node_children(tree, u):
            dfs(c)
        out.append(int(u))

    dfs(int(tree.root))
    return out


def _finite_state_or_raise(state: TTNState, context: str) -> None:
    for nid, T in state.tensors.items():
        if not np.all(np.isfinite(np.asarray(T))):
            raise FloatingPointError(f"Non-finite tensor encountered in {context} at node {nid}")


# =============================================================================
# Curvature / schedule helpers retained for legacy adaptive modes
# =============================================================================


def _curvature_scores_from_flow(diags: List[Any]) -> np.ndarray:
    _assert(len(diags) > 0, "_curvature_scores_from_flow: need diagnostics")
    last = diags[-1]
    divX = np.asarray(getattr(last, "divX_scalar"), dtype=float).reshape(-1)
    if divX.size != 64:
        # Keep legacy semantics for 64-node AtomTN, but degrade safely if a future
        # geometry changes size.
        out = np.zeros((64,), dtype=float)
        out[: min(64, divX.size)] = np.abs(divX[: min(64, divX.size)])
        divX = out
    if getattr(last, "DdivDt_scalar", None) is None:
        return np.abs(divX)
    Ddiv = np.asarray(getattr(last, "DdivDt_scalar"), dtype=float).reshape(-1)
    if Ddiv.size != 64:
        tmp = np.zeros((64,), dtype=float)
        tmp[: min(64, Ddiv.size)] = Ddiv[: min(64, Ddiv.size)]
        Ddiv = tmp
    return np.abs(divX) + 0.5 * np.abs(Ddiv)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    if s.size == 0:
        return s
    lo, hi = float(np.min(s)), float(np.max(s))
    denom = max(hi - lo, _EPS)
    return (s - lo) / denom


def _adaptive_fiber_dims(
    scores_leaf: np.ndarray,
    d_uniform: int,
    d_min: int,
    d_max: int,
    strength: float = 1.0,
) -> np.ndarray:
    ns = _normalize_scores(scores_leaf)
    d = np.round(float(d_uniform) * (1.0 + float(strength) * ns)).astype(int)
    return np.clip(d, int(d_min), int(d_max)).astype(int)


def _bond_schedule_from_scores(
    tree: Any,
    scores_leaf: np.ndarray,
    base_bond: int,
    min_bond: int,
    max_bond: int,
    strength: float = 1.0,
) -> Dict[int, int]:
    scores_leaf = np.asarray(scores_leaf, dtype=float).reshape(-1)
    _assert(scores_leaf.size == len(tree.leaves), "_bond_schedule_from_scores: score/leaf count mismatch")

    leaf_ids = [int(x) for x in tree.leaves]
    leaf_id_to_idx = {leaf_ids[i]: i for i in range(len(leaf_ids))}

    subtree_leaves: Dict[int, List[int]] = {}

    def collect(nid: int) -> List[int]:
        nid = int(nid)
        if nid in subtree_leaves:
            return subtree_leaves[nid]
        node = tree.nodes[nid]
        if getattr(node, "is_leaf", False):
            subtree_leaves[nid] = [nid]
            return subtree_leaves[nid]
        acc: List[int] = []
        for c in _node_children(tree, nid):
            acc.extend(collect(c))
        subtree_leaves[nid] = acc
        return acc

    for nid in tree.nodes:
        collect(int(nid))

    node_score: Dict[int, float] = {}
    for nid, leaves in subtree_leaves.items():
        idxs = [leaf_id_to_idx[int(l)] for l in leaves if int(l) in leaf_id_to_idx]
        node_score[int(nid)] = float(np.mean(scores_leaf[idxs])) if idxs else 0.0

    vals = np.asarray(list(node_score.values()), dtype=float)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    denom = max(hi - lo, _EPS)

    schedule: Dict[int, int] = {}
    for nid, score in node_score.items():
        ns = (float(score) - lo) / denom
        bd = int(round(float(base_bond) * (1.0 + float(strength) * ns)))
        schedule[int(nid)] = int(np.clip(bd, int(min_bond), int(max_bond)))

    schedule[int(tree.root)] = 1
    return schedule


def _adapt_leaf_dims_in_place(state: TTNState, target_d_leaf: np.ndarray, seed: int = 0) -> None:
    target_d_leaf = np.asarray(target_d_leaf, dtype=int).reshape(-1)
    leaves = [int(x) for x in state.tree.leaves]
    _assert(target_d_leaf.size == len(leaves), "_adapt_leaf_dims_in_place: target count mismatch")

    for i, lid in enumerate(leaves):
        target = int(max(1, target_d_leaf[i]))
        current = int(state.phys_dims[lid])
        if target == current:
            continue
        if target > current:
            _assert(hasattr(state, "expand_leaf_dim"), "TTNState missing expand_leaf_dim")
            state.expand_leaf_dim(lid, target, seed=seed)  # type: ignore[attr-defined]
        else:
            _assert(hasattr(state, "shrink_leaf_dim"), "TTNState missing shrink_leaf_dim")
            state.shrink_leaf_dim(lid, target)  # type: ignore[attr-defined]
    state.validate()


# =============================================================================
# Config dataclasses
# =============================================================================

@dataclass
class ApplyConfig:
    """
    Controls operator application during a single H|psi> call.

    apply_truncate_rank / apply_truncate_tol
        Truncation applied inside zip-up/direct-sum apply. Without some cap,
        direct-sum accumulation is not practical for realistic Hamiltonians.

    canonicalize_every
        Cadence for TTNState.canonicalize_upward_qr() inside apply/post-step.

    apply_grouping
        Forward-compatible hint for builders/apply implementations.

    force_safe_truncation
        If True, this evolver injects a conservative rank cap when zip-up is
        requested with no truncation configured.
    """
    apply_truncate_rank: Optional[int] = None
    apply_truncate_tol: Optional[float] = None
    canonicalize_every: int = 1
    apply_grouping: str = "lca_routed"
    force_safe_truncation: bool = True
    max_safe_rank: int = 12


@dataclass
class TTNEvolveConfig:
    """Time-evolution configuration with legacy and production knobs."""
    dt: float = 5e-3
    steps: int = 80
    method: str = "rk4_end_truncate"

    renormalize_every: int = 1
    step_bucket_every: int = 1

    apply_config: ApplyConfig = field(default_factory=ApplyConfig)

    post_step_truncate_rank: Optional[int] = None
    post_step_truncate_tol: Optional[float] = None

    # Legacy adaptive scheduling knobs.
    truncate_every: int = 10
    base_bond: int = 4
    min_bond: int = 2
    max_bond: int = 12
    bond_strength: float = 1.5

    adapt_fiber_every: int = 20
    fiber_strength: float = 1.25

    # Builder-side Hamiltonian defaults. Real class is available in normal AtomTN.
    H_cfg: Any = field(default_factory=lambda: HamiltonianBuildConfig())  # type: ignore[operator]

    strict_finite_checks: bool = True


# =============================================================================
# Lazy apply import and operator/application introspection
# =============================================================================


def _import_apply() -> Any:
    return importlib.import_module("apply")


def _call_with_supported_kwargs(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    """Call fn with only kwargs its signature accepts, falling back for C/builtins."""
    try:
        sig = inspect.signature(fn)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_kwargs:
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)
    except TypeError:
        # Last-resort for legacy positional-only-ish methods is handled by callers.
        raise


def _apply_has(apply_mod: Any, name: str) -> bool:
    return hasattr(apply_mod, name) and callable(getattr(apply_mod, name))


def _looks_like_tree_mpo(op: Any) -> bool:
    return (
        hasattr(op, "weights")
        and hasattr(op, "n_leaf_terms")
        and hasattr(op, "n_edge_terms")
        and hasattr(op, "leaf_term_leaf")
    )


def _looks_like_compiled_operator(op: Any) -> bool:
    return any(
        hasattr(op, attr)
        for attr in ("onsite_by_leaf", "pair_by_edge", "leaf_terms", "pair_terms")
    )


# =============================================================================
# Bond-shape stabilization helpers
# =============================================================================


def _safe_zipup_rank_for_state(template: TTNState, max_safe_rank: int = 12) -> int:
    try:
        max_b = int(max(int(x) for x in template.parent_bond_dims.values()))
    except Exception:
        max_b = 4
    return int(max(1, min(int(max_safe_rank), max_b)))


def _pad_axis(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    axis = int(axis)
    target = int(target)
    if arr.shape[axis] >= target:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - int(arr.shape[axis]))
    return np.pad(arr, pad_width, mode="constant").astype(arr.dtype, copy=False)


def _pad_parent_bond_to_dim(state: TTNState, nid: int, target: int) -> None:
    """Pad bond nid->parent with zeros to exact target dimension."""
    nid = int(nid)
    target = int(target)
    root = int(state.tree.root)
    if nid == root:
        state.parent_bond_dims[nid] = 1
        return

    cur = int(state.parent_bond_dims[nid])
    if cur >= target:
        return

    # Node tensor parent axis is always last.
    state.tensors[nid] = _pad_axis(np.asarray(state.tensors[nid]), -1, target).astype(np.complex128, copy=False)

    parent = getattr(state.tree.nodes[nid], "parent", None)
    if parent is not None:
        parent = int(parent)
        children = _node_children(state.tree, parent)
        _assert(nid in children, "_pad_parent_bond_to_dim: parent does not list child")
        ax = children.index(nid)
        state.tensors[parent] = _pad_axis(np.asarray(state.tensors[parent]), ax, target).astype(np.complex128, copy=False)

    state.parent_bond_dims[nid] = target


def _align_to_template_bonds(x: TTNState, template: TTNState) -> None:
    """
    Force x to share template's parent bond dimensions.

    For oversized bonds, truncate with TTNState.truncate_parent_bond_svd.
    For undersized bonds, pad zero components. Padding is important because SVD
    can return rank below the requested cap for numerically low-rank tensors.
    """
    _assert(x.tree is template.tree, "bond alignment requires identical tree object")
    _assert(x.phys_dims == template.phys_dims, "bond alignment requires identical physical dims")
    _assert(hasattr(x, "truncate_parent_bond_svd"), "TTNState missing truncate_parent_bond_svd")

    root = int(x.tree.root)
    for nid in _post_order(x.tree):
        if nid == root:
            x.parent_bond_dims[nid] = 1
            continue
        target = int(template.parent_bond_dims[nid])
        current = int(x.parent_bond_dims[nid])
        if current > target:
            x.truncate_parent_bond_svd(nid, rank=target, tol=None)  # type: ignore[attr-defined]
        if int(x.parent_bond_dims[nid]) < target:
            _pad_parent_bond_to_dim(x, nid, target)

    x.validate()


def _truncate_state_treewide(state: TTNState, *, rank: Optional[int], tol: Optional[float]) -> None:
    if rank is None and tol is None:
        return
    _assert(hasattr(state, "truncate_parent_bond_svd"), "TTNState missing truncate_parent_bond_svd")
    root = int(state.tree.root)
    for nid in _post_order(state.tree):
        if int(nid) == root:
            continue
        state.truncate_parent_bond_svd(int(nid), rank=rank, tol=tol)  # type: ignore[attr-defined]
    state.validate()


# =============================================================================
# TTN evolver
# =============================================================================

class TTNTimeEvolver:
    """Time evolution driver for AtomTN TTN states."""

    def __init__(self, calc: Any):
        self.calc = calc
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # State arithmetic
    # ------------------------------------------------------------------
    @staticmethod
    def _scale_in_place(state: TTNState, alpha: complex) -> None:
        for nid in state.tensors:
            state.tensors[nid] = (complex(alpha) * state.tensors[nid]).astype(np.complex128, copy=False)

    @staticmethod
    def _add_scaled(dst: TTNState, src: TTNState, alpha: complex) -> None:
        dst.validate()
        src.validate()
        _assert(dst.tree is src.tree, "TTN add: trees must be identical objects")
        _assert(dst.phys_dims == src.phys_dims, "TTN add: physical dimensions mismatch")
        _assert(dst.parent_bond_dims == src.parent_bond_dims, "TTN add: parent bond dimensions mismatch")
        for nid in dst.tensors:
            _assert(dst.tensors[nid].shape == src.tensors[nid].shape, "TTN add: tensor shape mismatch")
            dst.tensors[nid] = (dst.tensors[nid] + complex(alpha) * src.tensors[nid]).astype(np.complex128, copy=False)

    @staticmethod
    def _zero_like(state: TTNState) -> TTNState:
        z = state.clone()
        for nid in z.tensors:
            z.tensors[nid] = np.zeros_like(z.tensors[nid], dtype=np.complex128)
        return z

    # ------------------------------------------------------------------
    # Operator callbacks
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_build_result(result: Any) -> Tuple[Any, Dict[int, Dict[str, np.ndarray]]]:
        if isinstance(result, tuple) and len(result) == 2:
            op, local_ops = result
            return op, local_ops
        if isinstance(result, list) and len(result) == 2:
            op, local_ops = result
            return op, local_ops
        raise TypeError("builder/apply callback must return (op, local_ops)")

    def _call_build_op(
        self,
        build_op: Callable[..., Any],
        state: TTNState,
        step_id: int,
        *,
        stage: int = 0,
    ) -> Tuple[Any, Dict[int, Dict[str, np.ndarray]]]:
        """Call legacy or keyword-friendly build_op callbacks."""
        # Most existing AtomTN code uses build_op(s, sid).
        try:
            return self._normalize_build_result(build_op(state, int(step_id)))
        except TypeError:
            pass

        # Newer callbacks may accept keyword semantics.
        result = _call_with_supported_kwargs(
            build_op,
            state=state,
            s=state,
            step_id=int(step_id),
            sid=int(step_id),
            full_step=int(step_id),
            stage=int(stage),
        )
        return self._normalize_build_result(result)

    @staticmethod
    def _make_apply_cfg_for_zipup(apply_cfg: ApplyConfig, template: TTNState, truncate_during_apply: bool) -> ApplyConfig:
        rank = apply_cfg.apply_truncate_rank
        tol = apply_cfg.apply_truncate_tol
        if (not truncate_during_apply) and apply_cfg.force_safe_truncation and rank is None and tol is None:
            rank = _safe_zipup_rank_for_state(template, max_safe_rank=apply_cfg.max_safe_rank)
            tol = None
            truncate_during_apply = True
        return ApplyConfig(
            apply_truncate_rank=(rank if truncate_during_apply else None),
            apply_truncate_tol=(tol if truncate_during_apply else None),
            canonicalize_every=int(apply_cfg.canonicalize_every),
            apply_grouping=str(apply_cfg.apply_grouping),
            force_safe_truncation=bool(apply_cfg.force_safe_truncation),
            max_safe_rank=int(apply_cfg.max_safe_rank),
        )

    def _apply_H_auto(
        self,
        state: TTNState,
        build_op: Callable[..., Any],
        step_id: int,
        apply_cfg: ApplyConfig,
        truncate_during_apply: bool,
        *,
        prefer_zipup: bool,
        stage: int = 0,
    ) -> Tuple[TTNState, bool]:
        """Return (H|psi>, used_zipup)."""
        op, local_ops = self._call_build_op(build_op, state, step_id, stage=stage)
        apply_mod = _import_apply()

        has_fast = _apply_has(apply_mod, "apply_tree_mpo_fast")
        has_zipup = _apply_has(apply_mod, "apply_compiled_operator_zipup")
        is_tree_mpo = _looks_like_tree_mpo(op)

        if prefer_zipup and has_zipup:
            cfg_eff = self._make_apply_cfg_for_zipup(apply_cfg, state, truncate_during_apply)
            Hpsi = apply_mod.apply_compiled_operator_zipup(
                state=state,
                op=op,
                local_ops=local_ops,
                cfg=cfg_eff,
                step_id=int(step_id),
            )
            return Hpsi, True

        if (not prefer_zipup) and is_tree_mpo and has_fast:
            return apply_mod.apply_tree_mpo_fast(op, state, local_ops), False

        if has_zipup:
            cfg_eff = self._make_apply_cfg_for_zipup(apply_cfg, state, truncate_during_apply)
            Hpsi = apply_mod.apply_compiled_operator_zipup(
                state=state,
                op=op,
                local_ops=local_ops,
                cfg=cfg_eff,
                step_id=int(step_id),
            )
            return Hpsi, True

        _assert(has_fast and is_tree_mpo, "No applicable apply path found")
        return apply_mod.apply_tree_mpo_fast(op, state, local_ops), False

    def _f(
        self,
        state: TTNState,
        build_op: Callable[..., Any],
        step_id: int,
        apply_cfg: ApplyConfig,
        truncate_during_apply: bool,
        *,
        prefer_zipup: bool,
        stage: int = 0,
    ) -> TTNState:
        """Return -i H|psi>, projected to state's bond layout when needed."""
        Hpsi, used_zipup = self._apply_H_auto(
            state=state,
            build_op=build_op,
            step_id=int(step_id),
            apply_cfg=apply_cfg,
            truncate_during_apply=truncate_during_apply,
            prefer_zipup=prefer_zipup,
            stage=stage,
        )
        if used_zipup:
            _align_to_template_bonds(Hpsi, state)
        out = Hpsi.clone()
        self._scale_in_place(out, -1j)
        return out

    # ------------------------------------------------------------------
    # One-step methods
    # ------------------------------------------------------------------
    def step_euler_legacy(
        self,
        state: TTNState,
        build_op: Callable[[TTNState, int], Tuple[Any, Dict[int, Dict[str, np.ndarray]]]],
        dt: float,
        step_id: int,
        apply_cfg: ApplyConfig,
    ) -> TTNState:
        k1 = self._f(state, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=False, stage=0)
        out = state.clone()
        self._add_scaled(out, k1, complex(dt))
        return out

    def step_rk4_legacy(
        self,
        state: TTNState,
        build_op: Callable[[TTNState, int], Tuple[Any, Dict[int, Dict[str, np.ndarray]]]],
        dt: float,
        step_id: int,
        apply_cfg: ApplyConfig,
    ) -> TTNState:
        k1 = self._f(state, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=False, stage=0)

        s2 = state.clone()
        self._add_scaled(s2, k1, 0.5 * complex(dt))
        k2 = self._f(s2, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=False, stage=1)

        s3 = state.clone()
        self._add_scaled(s3, k2, 0.5 * complex(dt))
        k3 = self._f(s3, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=False, stage=2)

        s4 = state.clone()
        self._add_scaled(s4, k3, complex(dt))
        k4 = self._f(s4, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=False, stage=3)

        out = state.clone()
        self._add_scaled(out, k1, complex(dt) / 6.0)
        self._add_scaled(out, k2, complex(dt) / 3.0)
        self._add_scaled(out, k3, complex(dt) / 3.0)
        self._add_scaled(out, k4, complex(dt) / 6.0)
        return out

    def step_rk4_end_truncate(
        self,
        state: TTNState,
        build_op: Callable[[TTNState, int], Tuple[Any, Dict[int, Dict[str, np.ndarray]]]],
        dt: float,
        step_id: int,
        apply_cfg: ApplyConfig,
    ) -> TTNState:
        """RK4 with zip-up preference and derivative projection to template bonds."""
        k1 = self._f(state, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=True, stage=0)

        s2 = state.clone()
        self._add_scaled(s2, k1, 0.5 * complex(dt))
        k2 = self._f(s2, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=True, stage=1)

        s3 = state.clone()
        self._add_scaled(s3, k2, 0.5 * complex(dt))
        k3 = self._f(s3, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=True, stage=2)

        s4 = state.clone()
        self._add_scaled(s4, k3, complex(dt))
        k4 = self._f(s4, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=True, stage=3)

        out = state.clone()
        self._add_scaled(out, k1, complex(dt) / 6.0)
        self._add_scaled(out, k2, complex(dt) / 3.0)
        self._add_scaled(out, k3, complex(dt) / 3.0)
        self._add_scaled(out, k4, complex(dt) / 6.0)
        return out

    def step_rk2_mid_truncate(
        self,
        state: TTNState,
        build_op: Callable[[TTNState, int], Tuple[Any, Dict[int, Dict[str, np.ndarray]]]],
        dt: float,
        step_id: int,
        apply_cfg: ApplyConfig,
    ) -> TTNState:
        k1 = self._f(state, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=True, stage=0)
        mid = state.clone()
        self._add_scaled(mid, k1, 0.5 * complex(dt))
        k2 = self._f(mid, build_op, step_id, apply_cfg, truncate_during_apply=False, prefer_zipup=True, stage=1)
        out = state.clone()
        self._add_scaled(out, k2, complex(dt))
        return out

    def step_lie_trotter(
        self,
        state: TTNState,
        build_split_ops: Callable[[TTNState, int], List[Tuple[Any, Dict[int, Dict[str, np.ndarray]]]]],
        dt: float,
        step_id: int,
        apply_cfg: ApplyConfig,
    ) -> TTNState:
        """
        First-order splitting method over pieces returned by build_split_ops.

        Each piece must be (op, local_ops). Zip-up is preferred if present;
        legacy TreeMPO fast apply is used as fallback for scaffold pieces.
        """
        apply_mod = _import_apply()
        has_zipup = _apply_has(apply_mod, "apply_compiled_operator_zipup")
        has_fast = _apply_has(apply_mod, "apply_tree_mpo_fast")

        psi = state.clone()
        pieces = build_split_ops(psi, int(step_id))
        for piece_idx, (op_piece, local_ops) in enumerate(pieces):
            used_zipup = False
            if has_zipup:
                cfg_eff = self._make_apply_cfg_for_zipup(apply_cfg, psi, truncate_during_apply=True)
                Hpsi = apply_mod.apply_compiled_operator_zipup(
                    state=psi,
                    op=op_piece,
                    local_ops=local_ops,
                    cfg=cfg_eff,
                    step_id=int(step_id),
                )
                used_zipup = True
            elif has_fast and _looks_like_tree_mpo(op_piece):
                Hpsi = apply_mod.apply_tree_mpo_fast(op_piece, psi, local_ops)
            else:
                raise RuntimeError("step_lie_trotter: no compatible apply path")

            if used_zipup:
                _align_to_template_bonds(Hpsi, psi)
            incr = Hpsi.clone()
            self._scale_in_place(incr, -1j)
            self._add_scaled(psi, incr, complex(dt))

        return psi

    def step(
        self,
        state: TTNState,
        build_op: Callable[[TTNState, int], Tuple[Any, Dict[int, Dict[str, np.ndarray]]]],
        dt: float,
        step_id: int,
        cfg: Optional[TTNEvolveConfig] = None,
    ) -> TTNState:
        """
        Generic one-step dispatcher used by neuromorphic.py and adapter runtimes.
        """
        cfg = cfg or TTNEvolveConfig(dt=float(dt), steps=1)
        method = str(cfg.method).lower().strip()
        apply_cfg = cfg.apply_config

        if method in ("euler", "euler_legacy"):
            out = self.step_euler_legacy(state, build_op, dt, step_id, apply_cfg)
        elif method in ("rk4", "rk4_legacy"):
            out = self.step_rk4_legacy(state, build_op, dt, step_id, apply_cfg)
        elif method == "rk4_end_truncate":
            out = self.step_rk4_end_truncate(state, build_op, dt, step_id, apply_cfg)
        elif method == "rk2_mid_truncate":
            out = self.step_rk2_mid_truncate(state, build_op, dt, step_id, apply_cfg)
        else:
            raise ValueError(f"TTNTimeEvolver.step does not support method '{cfg.method}' without split operators")

        self.post_process_step(out, int(step_id), cfg)
        if cfg.renormalize_every and (int(step_id) + 1) % int(cfg.renormalize_every) == 0:
            out.normalize_in_place()
        if cfg.strict_finite_checks:
            _finite_state_or_raise(out, f"TTNTimeEvolver.step[{method}]")
        return out

    # ------------------------------------------------------------------
    # Post-step maintenance
    # ------------------------------------------------------------------
    def post_process_step(self, state: TTNState, step_id: int, cfg: TTNEvolveConfig) -> None:
        # Canonicalization cadence.
        ce = int(getattr(cfg.apply_config, "canonicalize_every", 0) or 0)
        if ce > 0 and (int(step_id) + 1) % ce == 0 and hasattr(state, "canonicalize_upward_qr"):
            state.canonicalize_upward_qr()  # type: ignore[attr-defined]

        # Optional hard cap after one full step.
        _truncate_state_treewide(
            state,
            rank=cfg.post_step_truncate_rank,
            tol=cfg.post_step_truncate_tol,
        )
        state.validate()

    # ------------------------------------------------------------------
    # Builder dispatch used by integrate_with_flow
    # ------------------------------------------------------------------
    @staticmethod
    def _builder_call(
        builder: Any,
        *,
        state: TTNState,
        fiber: Any,
        X: Any,
        vib: Optional[Any],
        diag: Any,
        step_id: int,
        prefer_compiled: bool,
        stage: int = 0,
    ) -> Tuple[Any, Dict[int, Dict[str, np.ndarray]]]:
        """Call builder with the best available API."""
        method_names: List[str]
        if prefer_compiled:
            method_names = ["build_compiled_operator", "build_compiled", "build"]
        else:
            method_names = ["build", "build_compiled_operator", "build_compiled"]

        last_err: Optional[BaseException] = None
        for name in method_names:
            fn = getattr(builder, name, None)
            if not callable(fn):
                continue
            try:
                result = _call_with_supported_kwargs(
                    fn,
                    state=state,
                    s=state,
                    fiber=fiber,
                    X=X,
                    vib=vib,
                    diag=diag,
                    step_id=int(step_id),
                    sid=int(step_id),
                    full_step=int(step_id),
                    stage=int(stage),
                )
                return TTNTimeEvolver._normalize_build_result(result)
            except TypeError as e:
                last_err = e
                # Positional fallback for current AtomTN TreeMPOBuilder.build(...).
                try:
                    result = fn(state, fiber, X, vib, diag, int(step_id))
                    return TTNTimeEvolver._normalize_build_result(result)
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e
        raise AttributeError(f"builder exposes no compatible build method; last error: {last_err!r}")

    # ------------------------------------------------------------------
    # Main integration loop
    # ------------------------------------------------------------------
    def integrate_with_flow(
        self,
        state0: TTNState,
        fiber: Any,
        builder: Any,
        X_traj: List[Any],
        vib: Optional[Any],
        flow_diags: List[Any],
        cfg: TTNEvolveConfig,
        seed: int = 0,
    ) -> List[TTNState]:
        state0.validate()
        _assert(len(X_traj) >= 2, "integrate_with_flow: need flow trajectory")
        _assert(len(flow_diags) == len(X_traj), "integrate_with_flow: diagnostics length mismatch")

        method = str(cfg.method).lower().strip()
        bucket_every = max(int(cfg.step_bucket_every), 1)
        n_steps = int(max(0, min(int(cfg.steps), len(X_traj) - 1)))
        out: List[TTNState] = []
        state = state0.clone()

        prefer_compiled = method in {"rk4_end_truncate", "rk2_mid_truncate", "lie_trotter"}

        for k in range(n_steps):
            X = X_traj[k]
            diag = flow_diags[k]
            step_bucket = int(k // bucket_every)

            # Legacy adaptive fiber dimensions.
            if cfg.adapt_fiber_every and (k == 0 or (k + 1) % int(cfg.adapt_fiber_every) == 0):
                scores_leaf = _curvature_scores_from_flow(flow_diags[: k + 1])
                if hasattr(fiber, "leaf_dims") and callable(getattr(fiber, "leaf_dims")):
                    d_leaf = fiber.leaf_dims(scores_leaf=scores_leaf, vib=vib)
                else:
                    d_leaf = _adaptive_fiber_dims(
                        scores_leaf=scores_leaf,
                        d_uniform=int(getattr(getattr(fiber, "cfg", object()), "d_uniform", 16)),
                        d_min=int(getattr(getattr(fiber, "cfg", object()), "d_min", 8)),
                        d_max=int(getattr(getattr(fiber, "cfg", object()), "d_max", 32)),
                        strength=float(cfg.fiber_strength),
                    )
                _adapt_leaf_dims_in_place(state, np.asarray(d_leaf, dtype=int), seed=int(seed) + 1000 + k)

            def build_op(s: TTNState, sid: int) -> Tuple[Any, Dict[int, Dict[str, np.ndarray]]]:
                return self._builder_call(
                    builder,
                    state=s,
                    fiber=fiber,
                    X=X,
                    vib=vib,
                    diag=diag,
                    step_id=int(sid),
                    prefer_compiled=prefer_compiled,
                    stage=0,
                )

            def build_split_ops(s: TTNState, sid: int) -> List[Tuple[Any, Dict[int, Dict[str, np.ndarray]]]]:
                fn = getattr(builder, "build_split_operators", None)
                if callable(fn):
                    try:
                        res = _call_with_supported_kwargs(
                            fn,
                            state=s,
                            fiber=fiber,
                            X=X,
                            vib=vib,
                            diag=diag,
                            step_id=int(sid),
                            grouping=cfg.apply_config.apply_grouping,
                        )
                        return list(res)
                    except TypeError:
                        res = fn(s, fiber, X, vib, diag, int(sid))
                        return list(res)
                op0, local_ops0 = build_op(s, sid)
                return [(op0, local_ops0)]

            if method in ("euler", "euler_legacy"):
                state = self.step_euler_legacy(state, build_op, float(cfg.dt), int(step_bucket), cfg.apply_config)
                self._legacy_adaptive_truncate_if_due(state, flow_diags[: k + 1], k, cfg)

            elif method in ("rk4", "rk4_legacy"):
                state = self.step_rk4_legacy(state, build_op, float(cfg.dt), int(step_bucket), cfg.apply_config)
                self._legacy_adaptive_truncate_if_due(state, flow_diags[: k + 1], k, cfg)

            elif method == "rk4_end_truncate":
                state = self.step_rk4_end_truncate(state, build_op, float(cfg.dt), int(step_bucket), cfg.apply_config)
                self.post_process_step(state, k, cfg)

            elif method == "rk2_mid_truncate":
                state = self.step_rk2_mid_truncate(state, build_op, float(cfg.dt), int(step_bucket), cfg.apply_config)
                self.post_process_step(state, k, cfg)

            elif method == "lie_trotter":
                state = self.step_lie_trotter(state, build_split_ops, float(cfg.dt), int(step_bucket), cfg.apply_config)
                self.post_process_step(state, k, cfg)

            else:
                raise ValueError(f"Unknown evolve method '{cfg.method}'")

            if cfg.renormalize_every and (k + 1) % int(cfg.renormalize_every) == 0:
                state.normalize_in_place()

            if cfg.strict_finite_checks:
                _finite_state_or_raise(state, f"integrate_with_flow[{method}] step={k}")

            out.append(state.clone())

        self.last_stats = {
            "method": method,
            "steps": int(len(out)),
            "final_norm_squared": _safe_norm_state(state),
            "post_step_truncate_rank": cfg.post_step_truncate_rank,
            "post_step_truncate_tol": cfg.post_step_truncate_tol,
        }
        return out

    def _legacy_adaptive_truncate_if_due(self, state: TTNState, flow_diags_prefix: List[Any], k: int, cfg: TTNEvolveConfig) -> None:
        if not cfg.truncate_every or not (k == 0 or (k + 1) % int(cfg.truncate_every) == 0):
            return
        scores_leaf = _curvature_scores_from_flow(flow_diags_prefix)
        sched = _bond_schedule_from_scores(
            tree=state.tree,
            scores_leaf=scores_leaf[: len(state.tree.leaves)],
            base_bond=int(cfg.base_bond),
            min_bond=int(cfg.min_bond),
            max_bond=int(cfg.max_bond),
            strength=float(cfg.bond_strength),
        )
        _assert(hasattr(state, "truncate_parent_bond_svd"), "TTNState missing truncate_parent_bond_svd")
        root = int(state.tree.root)
        for nid in _post_order(state.tree):
            if int(nid) == root:
                continue
            target = int(sched.get(int(nid), int(state.parent_bond_dims.get(int(nid), cfg.base_bond))))
            state.truncate_parent_bond_svd(int(nid), rank=target, tol=None)  # type: ignore[attr-defined]
        state.validate()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return _json_safe({"kind": "TTNTimeEvolver", "last_stats": self.last_stats})


__all__ = ["ApplyConfig", "TTNEvolveConfig", "TTNTimeEvolver"]
