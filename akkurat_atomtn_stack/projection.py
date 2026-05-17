#!/usr/bin/env python3
# projection.py
"""
ProjectionLayer: fuzzy k -> local d holographic compression.

Production role
---------------
This module owns deterministic projection of fuzzy/noncommutative SU(2)
operators into each TTN leaf's local physical space.  It is deliberately small,
NumPy-only, and compatible with AtomTN's Phase-4/Phase-5 runtime semantics.

Key properties
--------------
- Static and energy-gauge projectors.
- RK-stage-aware cache keys via StepKey(full_step, stage, bucket).
- Optional freeze-within-full-step behavior so RK stages reuse the same gauge.
- Cadence controls measured in full outer steps, not substeps.
- Overlap tracking / Procrustes alignment to suppress eigenbasis sign and basis
  flips between projector refreshes.
- Projected operator cache with projector-generation tracking to prevent stale
  operator reuse when projection cadence and bucket cadence differ.
- Backward-compatible public API:
      ProjectionLayer(...).projected_ops(leaf_id, d, kappa, step_id=...)

Dependencies
------------
- numpy
- math_utils.py
- fuzzy_backend.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from math_utils import (
    _assert,
    build_energy_observable_from_kappa,
    eigh_hermitian_stable,
    fro_norm,
    hermitianize,
)
from fuzzy_backend import FuzzySU2


_EPS = 1e-12


# =============================================================================
# Step keying
# =============================================================================

@dataclass(frozen=True)
class StepKey:
    """
    Descriptor for one point in an integration schedule.

    full_step:
        Outer integration step index.
    stage:
        RK/substep stage index.  Euler and Lie-Trotter callers normally use 0.
    bucket:
        Coarse cache bucket computed from full_step and cache_bucket_full_steps.
    """

    full_step: int
    stage: int = 0
    bucket: int = 0



def make_step_key(
    *,
    full_step: int,
    stage: int = 0,
    cache_bucket_full_steps: int = 1,
) -> StepKey:
    bucket_every = max(int(cache_bucket_full_steps), 1)
    return StepKey(full_step=int(full_step), stage=int(stage), bucket=int(full_step) // bucket_every)


# =============================================================================
# Low-level helpers
# =============================================================================


def _as_c128_matrix(A: Any, *, name: str = "matrix") -> np.ndarray:
    M = np.asarray(A, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix; got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return M



def _sanitize_real_vector(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)



def _row_orthonormalize(P: np.ndarray) -> np.ndarray:
    """Return a row-orthonormal version of P with shape (d,k)."""
    P = np.asarray(P, dtype=np.complex128)
    if P.ndim != 2:
        raise ValueError("projector must be a matrix")
    # Orthonormalize columns of P^H, then return Q^H.
    Q, _ = np.linalg.qr(P.conj().T, mode="reduced")
    return Q.conj().T.astype(np.complex128)



def _projector_error(P: np.ndarray) -> float:
    P = np.asarray(P, dtype=np.complex128)
    if P.ndim != 2:
        return float("inf")
    d = P.shape[0]
    return float(np.linalg.norm(P @ P.conj().T - np.eye(d, dtype=np.complex128)))



def _align_subspace(P_prev: np.ndarray, P_new: np.ndarray) -> np.ndarray:
    """
    Align P_new to P_prev with a complex Procrustes rotation.

    Projectors are represented as row projectors P(d,k).  Equivalently V=P^H is
    a k x d basis for the chosen subspace.  We choose a d x d unitary Q that
    maximizes alignment of V_new Q with V_prev.
    """
    P_prev = np.asarray(P_prev, dtype=np.complex128)
    P_new = np.asarray(P_new, dtype=np.complex128)
    if P_prev.shape != P_new.shape:
        return P_new

    V_prev = P_prev.conj().T  # (k,d)
    V_new = P_new.conj().T    # (k,d)

    try:
        overlap = V_prev.conj().T @ V_new
        U, _, Vh = np.linalg.svd(overlap, full_matrices=False)
        Q = Vh.conj().T @ U.conj().T
        aligned = V_new @ Q
        return aligned.conj().T.astype(np.complex128)
    except Exception:
        return P_new



def _ordered_eigvecs(KH: np.ndarray, *, selector: str = "lowest") -> np.ndarray:
    """Return eigenvectors of Hermitian KH ordered by selector."""
    evals, evecs = eigh_hermitian_stable(KH, sort=True, check_herm=False, symmetrize=True)
    selector = str(selector or "lowest").lower().strip()
    n = int(evals.size)
    if selector in {"highest", "top", "max"}:
        idx = np.argsort(evals)[::-1]
    elif selector in {"center", "middle", "midband"}:
        order = np.argsort(np.abs(evals - float(np.median(evals))))
        idx = order
    elif selector in {"abs_lowest", "smallest_abs", "near_zero"}:
        idx = np.argsort(np.abs(evals))
    else:
        idx = np.argsort(evals)
    return np.asarray(evecs[:, idx[:n]], dtype=np.complex128)


# =============================================================================
# Projection layer
# =============================================================================

@dataclass
class ProjectionLayer:
    """
    Holographic fuzzy-space projection layer.

    Parameters
    ----------
    fuzzy:
        FuzzySU2 object containing k and generators Lx, Ly, Lz.
    strategy:
        "energy_based", "center_band", "lowest_weight", "highest_weight".
    overlap_track:
        Align refreshed energy projectors against previous projectors.
    update_every_full_steps:
        Refresh cadence for energy projectors, measured in outer full steps.
    freeze_within_step:
        If True, RK stages >0 reuse the full-step stage-0 projector.
    cache_bucket_full_steps:
        Operator cache bucket size.
    energy_mode:
        How to convert kappa to Hermitian energy observable:
        "direct", "i_kappa", or "kdagk".
    energy_selector:
        Which eigen-subspace to keep: "lowest", "highest", "center",
        or "near_zero".
    strict:
        If True, invalid projector inputs raise.  If False, projection falls
        back to a static center-band projector where safe.
    """

    fuzzy: FuzzySU2
    strategy: str = "energy_based"
    overlap_track: bool = True

    update_every_full_steps: int = 1
    freeze_within_step: bool = True
    cache_bucket_full_steps: int = 1

    energy_mode: str = "direct"
    energy_selector: str = "lowest"
    orthonormalize_projectors: bool = True
    strict: bool = True

    # Projected operator cache.  Key includes projector generation to avoid stale
    # projected ops if projection refresh cadence and bucket cadence diverge.
    cache_ops: Dict[Tuple[int, int, int, int], Dict[str, np.ndarray]] = field(default_factory=dict)

    # Projector caches.
    cache_P: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)           # (leaf,d) -> P(d,k)
    cache_last_update: Dict[Tuple[int, int], int] = field(default_factory=dict)        # (leaf,d) -> last full_step refreshed
    cache_P_step: Dict[Tuple[int, int], int] = field(default_factory=dict)             # (leaf,d) -> full_step for freeze reuse
    cache_generation: Dict[Tuple[int, int], int] = field(default_factory=dict)         # (leaf,d) -> monotonic generation

    def __post_init__(self) -> None:
        if not hasattr(self.fuzzy, "k"):
            raise TypeError("ProjectionLayer.fuzzy must expose integer attribute 'k'")
        k = int(self.fuzzy.k)
        if k <= 0:
            raise ValueError("fuzzy.k must be positive")
        for name in ("Lx", "Ly", "Lz"):
            G = _as_c128_matrix(getattr(self.fuzzy, name), name=f"fuzzy.{name}")
            if G.shape != (k, k):
                raise ValueError(f"fuzzy.{name} shape {G.shape} does not match k={k}")

        self.update_every_full_steps = max(int(self.update_every_full_steps), 1)
        self.cache_bucket_full_steps = max(int(self.cache_bucket_full_steps), 1)
        self.strategy = str(self.strategy or "energy_based").lower().strip()
        self.energy_mode = str(self.energy_mode or "direct").lower().strip()
        self.energy_selector = str(self.energy_selector or "lowest").lower().strip()

    # ------------------------------------------------------------------
    # Projector construction
    # ------------------------------------------------------------------

    def _validate_d(self, d: int) -> int:
        d = int(d)
        k = int(self.fuzzy.k)
        if d <= 0:
            raise ValueError("projection dimension d must be positive")
        if d > k:
            raise ValueError(f"cannot project: d={d} exceeds fuzzy k={k}")
        return d

    def _select_indices_static(self, d: int, kind: str) -> np.ndarray:
        d = self._validate_d(d)
        k = int(self.fuzzy.k)
        kind = str(kind or "center_band").lower().strip()

        if kind in {"lowest_weight", "lowest", "bottom"}:
            idx = np.arange(k - d, k, dtype=int)
        elif kind in {"highest_weight", "highest", "top"}:
            idx = np.arange(0, d, dtype=int)
        elif kind in {"outer_band"}:
            left = d // 2
            right = d - left
            idx = np.concatenate([np.arange(0, left), np.arange(k - right, k)]).astype(int)
        else:
            start = (k - d) // 2
            idx = np.arange(start, start + d, dtype=int)
        return idx

    def _P_from_static(self, d: int, kind: str = "center_band") -> np.ndarray:
        d = self._validate_d(d)
        k = int(self.fuzzy.k)
        idx = self._select_indices_static(d, kind)
        P = np.zeros((d, k), dtype=np.complex128)
        P[np.arange(d), idx] = 1.0
        return P

    def _make_step(self, *, step_id: Optional[int], full_step: Optional[int], stage: int) -> StepKey:
        if full_step is None:
            full_step = int(step_id) if step_id is not None else 0
        return make_step_key(
            full_step=int(full_step),
            stage=int(stage),
            cache_bucket_full_steps=int(self.cache_bucket_full_steps),
        )

    def _should_refresh_P(self, key: Tuple[int, int], step: StepKey) -> bool:
        if self.freeze_within_step and int(step.stage) != 0:
            return False
        last = self.cache_last_update.get(key)
        if last is None:
            return True
        return (int(step.full_step) - int(last)) >= int(self.update_every_full_steps)

    def _bump_generation(self, key: Tuple[int, int]) -> None:
        self.cache_generation[key] = int(self.cache_generation.get(key, -1)) + 1

    def _store_projector(self, key: Tuple[int, int], P: np.ndarray, step: StepKey, *, refresh: bool) -> np.ndarray:
        if self.orthonormalize_projectors:
            P = _row_orthonormalize(P)
        err = _projector_error(P)
        if not np.isfinite(err) or err > 1e-6:
            if self.strict:
                raise FloatingPointError(f"invalid projector row-orthonormality error={err:.3e}")
            P = _row_orthonormalize(P)

        old = self.cache_P.get(key)
        changed = refresh or old is None or old.shape != P.shape or float(np.linalg.norm(old - P)) > 1e-10
        self.cache_P[key] = P.astype(np.complex128, copy=False)
        self.cache_last_update[key] = int(step.full_step)
        self.cache_P_step[key] = int(step.full_step)
        if changed:
            self._bump_generation(key)
        else:
            self.cache_generation.setdefault(key, 0)
        return self.cache_P[key]

    def _P_from_energy(self, leaf_id: int, d: int, kappa: Optional[np.ndarray], step: StepKey) -> np.ndarray:
        d = self._validate_d(d)
        k = int(self.fuzzy.k)
        key = (int(leaf_id), int(d))

        # Freeze semantics: if a projector was already chosen for this full step,
        # every subsequent stage reuses it exactly.
        if self.freeze_within_step:
            p_step = self.cache_P_step.get(key)
            if p_step is not None and int(p_step) == int(step.full_step):
                P_cached = self.cache_P.get(key)
                if P_cached is not None:
                    return P_cached

        # Cadence semantics.
        if not self._should_refresh_P(key, step):
            P_cached = self.cache_P.get(key)
            if P_cached is not None:
                if self.freeze_within_step:
                    self.cache_P_step[key] = int(step.full_step)
                return P_cached

        # If κ is unavailable, either fail loudly or fall back to center-band.
        if kappa is None:
            if self.strict:
                raise ValueError("energy_based projection requires kappa matrix")
            P = self._P_from_static(d, "center_band")
            return self._store_projector(key, P, step, refresh=True)

        try:
            Kraw = _as_c128_matrix(kappa, name="kappa")
            if Kraw.shape != (k, k):
                raise ValueError(f"kappa shape {Kraw.shape} does not match fuzzy k={k}")
            KH = build_energy_observable_from_kappa(Kraw, mode=self.energy_mode)
            KH = hermitianize(KH)
            evecs_ordered = _ordered_eigvecs(KH, selector=self.energy_selector)
            Vd = evecs_ordered[:, :d]                         # (k,d)
            P_new = Vd.conj().T.astype(np.complex128)          # (d,k)

            if self.overlap_track:
                P_prev = self.cache_P.get(key)
                if P_prev is not None:
                    P_new = _align_subspace(P_prev, P_new)

            return self._store_projector(key, P_new, step, refresh=True)

        except Exception:
            if self.strict:
                raise
            P = self._P_from_static(d, "center_band")
            return self._store_projector(key, P, step, refresh=True)

    def projector(
        self,
        leaf_id: int,
        d: int,
        kappa: Optional[np.ndarray] = None,
        *,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
    ) -> np.ndarray:
        """Return P(d,k) for the requested leaf and integration step."""
        d = self._validate_d(d)
        leaf_id = int(leaf_id)
        step = self._make_step(step_id=step_id, full_step=full_step, stage=stage)
        key = (leaf_id, d)
        strat = self.strategy

        if strat in {"center_band", "lowest_weight", "highest_weight", "lowest", "highest", "outer_band"}:
            P_cached = self.cache_P.get(key)
            if P_cached is not None:
                self.cache_P_step[key] = int(step.full_step)
                return P_cached
            P = self._P_from_static(d, strat)
            return self._store_projector(key, P, step, refresh=True)

        if strat == "energy_based":
            return self._P_from_energy(leaf_id, d, kappa, step)

        if self.strict:
            raise ValueError(f"unknown projection strategy: {self.strategy!r}")
        P = self._P_from_static(d, "center_band")
        return self._store_projector(key, P, step, refresh=True)

    # ------------------------------------------------------------------
    # Projection operations
    # ------------------------------------------------------------------

    def project_matrix(
        self,
        A: np.ndarray,
        *,
        leaf_id: int,
        d: int,
        kappa: Optional[np.ndarray] = None,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
        hermitian: bool = False,
    ) -> np.ndarray:
        """Project a k x k matrix into d x d local space as P A P†."""
        P = self.projector(leaf_id, d, kappa, step_id=step_id, full_step=full_step, stage=stage)
        A = _as_c128_matrix(A, name="A")
        if A.shape != (int(self.fuzzy.k), int(self.fuzzy.k)):
            raise ValueError(f"A shape {A.shape} does not match fuzzy k={self.fuzzy.k}")
        out = (P @ A @ P.conj().T).astype(np.complex128)
        return hermitianize(out) if hermitian else out

    def project_vector(
        self,
        x: Any,
        *,
        leaf_id: int,
        d: int,
        kappa: Optional[np.ndarray] = None,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
    ) -> np.ndarray:
        """Project a fuzzy-space vector of length k into length d."""
        P = self.projector(leaf_id, d, kappa, step_id=step_id, full_step=full_step, stage=stage)
        vec = np.asarray(x, dtype=np.complex128).reshape(-1)
        if vec.size != int(self.fuzzy.k):
            raise ValueError(f"vector length {vec.size} does not match fuzzy k={self.fuzzy.k}")
        return (P @ vec).astype(np.complex128)

    def lift_vector(
        self,
        y: Any,
        *,
        leaf_id: int,
        d: int,
        kappa: Optional[np.ndarray] = None,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
    ) -> np.ndarray:
        """Lift a d-vector back to fuzzy k-space by P† y."""
        P = self.projector(leaf_id, d, kappa, step_id=step_id, full_step=full_step, stage=stage)
        vec = np.asarray(y, dtype=np.complex128).reshape(-1)
        if vec.size != int(d):
            raise ValueError(f"vector length {vec.size} does not match d={d}")
        return (P.conj().T @ vec).astype(np.complex128)

    def projected_ops(
        self,
        leaf_id: int,
        d: int,
        kappa: Optional[np.ndarray],
        *,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Return projected local operators for a leaf.

        Backward compatible with existing callers that pass `step_id` only.
        Keys returned: I, Lx, Ly, Lz.
        """
        d = self._validate_d(d)
        leaf_id = int(leaf_id)
        step = self._make_step(step_id=step_id, full_step=full_step, stage=stage)

        # Ensure projector is current before computing generation-aware key.
        P = self.projector(leaf_id, d, kappa, step_id=step_id, full_step=full_step, stage=stage)
        gen = int(self.cache_generation.get((leaf_id, d), 0))
        key_ops = (leaf_id, d, int(step.bucket), gen)
        cached = self.cache_ops.get(key_ops)
        if cached is not None:
            return cached

        Lx = P @ np.asarray(self.fuzzy.Lx, dtype=np.complex128) @ P.conj().T
        Ly = P @ np.asarray(self.fuzzy.Ly, dtype=np.complex128) @ P.conj().T
        Lz = P @ np.asarray(self.fuzzy.Lz, dtype=np.complex128) @ P.conj().T

        ops = {
            "I": np.eye(d, dtype=np.complex128),
            "Lx": hermitianize(Lx.astype(np.complex128)),
            "Ly": hermitianize(Ly.astype(np.complex128)),
            "Lz": hermitianize(Lz.astype(np.complex128)),
        }
        self.cache_ops[key_ops] = ops
        return ops

    # ------------------------------------------------------------------
    # Diagnostics / cache control
    # ------------------------------------------------------------------

    def projector_diagnostics(self) -> Dict[str, Any]:
        errors = []
        dims = []
        for (leaf, d), P in self.cache_P.items():
            errors.append(_projector_error(P))
            dims.append((int(leaf), int(d)))
        return {
            "strategy": str(self.strategy),
            "energy_mode": str(self.energy_mode),
            "energy_selector": str(self.energy_selector),
            "update_every_full_steps": int(self.update_every_full_steps),
            "freeze_within_step": bool(self.freeze_within_step),
            "cache_bucket_full_steps": int(self.cache_bucket_full_steps),
            "num_projectors": int(len(self.cache_P)),
            "num_projected_op_entries": int(len(self.cache_ops)),
            "max_projector_orthogonality_error": float(max(errors)) if errors else 0.0,
            "mean_projector_orthogonality_error": float(np.mean(errors)) if errors else 0.0,
            "cached_leaf_dims": dims[:64],
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "ProjectionLayer",
            "fuzzy_k": int(self.fuzzy.k),
            "strategy": str(self.strategy),
            "overlap_track": bool(self.overlap_track),
            "update_every_full_steps": int(self.update_every_full_steps),
            "freeze_within_step": bool(self.freeze_within_step),
            "cache_bucket_full_steps": int(self.cache_bucket_full_steps),
            "energy_mode": str(self.energy_mode),
            "energy_selector": str(self.energy_selector),
            "orthonormalize_projectors": bool(self.orthonormalize_projectors),
            "strict": bool(self.strict),
            "cache": self.projector_diagnostics(),
        }

    def clear_ops_cache(self) -> None:
        """Clear projected operator cache while preserving projector history."""
        self.cache_ops.clear()

    def clear_projector_cache(self) -> None:
        """Clear projector history and dependent projected operator cache."""
        self.cache_ops.clear()
        self.cache_P.clear()
        self.cache_last_update.clear()
        self.cache_P_step.clear()
        self.cache_generation.clear()

    def clear_all_caches(self) -> None:
        self.clear_projector_cache()


__all__ = [
    "ProjectionLayer",
    "StepKey",
    "make_step_key",
]
