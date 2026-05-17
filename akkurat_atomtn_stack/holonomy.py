#!/usr/bin/env python3
# holonomy.py
"""
Holonomy / adjoint-frame transport for AtomTN noncommutative flow edges.

Production role
---------------
This module converts matrix-valued NC edge flows X_{u->v} into stable SO(3)
adjoint-frame rotations used by the Hamiltonian builder for holonomy-coupled
edge terms.

Public API retained from earlier AtomTN scripts:

    decomp = GeneratorDecomposition(Lx, Ly, Lz, remove_trace=True)
    R = HolonomyBuilder(decomp=decomp).rotation_from_X(X_uv)

Production additions
--------------------
- Deterministic StepKey-aware caching via rotation_uv(...).
- Optional freeze-within-full-step semantics for RK stages.
- Cadence measured in full outer steps, not RK stages.
- Robust finite-value sanitation and graceful identity fallback in non-strict mode.
- Projection/decomposition helpers for su(2) coefficients, magnitudes, and
  reconstructed generators.
- Health/snapshot helpers for diagnostics and integration with Akkurat/AtomTN
  status surfaces.

Dependencies
------------
- numpy
- math_utils.py
- projection.py StepKey/make_step_key when available
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from math_utils import (
    _assert,
    antihermitianize,
    build_energy_observable_from_kappa,
    expm,
    fro_norm,
    hermitianize,
    hs_inner,
    real_orthonormalize_3x3,
    remove_trace as remove_trace_part,
)

try:  # keep exact step-key semantics aligned with projection.py
    from projection import StepKey, make_step_key  # type: ignore
except Exception:  # pragma: no cover - fallback for standalone use
    @dataclass(frozen=True)
    class StepKey:  # type: ignore[no-redef]
        full_step: int
        stage: int = 0
        bucket: int = 0

    def make_step_key(  # type: ignore[no-redef]
        *,
        full_step: int,
        stage: int = 0,
        cache_bucket_full_steps: int = 1,
    ) -> StepKey:
        bucket_every = max(int(cache_bucket_full_steps), 1)
        return StepKey(full_step=int(full_step), stage=int(stage), bucket=int(full_step) // bucket_every)


_EPS = 1e-12


# =============================================================================
# Helpers
# =============================================================================


def _finite_complex_matrix(x: Any, *, name: str = "matrix") -> np.ndarray:
    """Coerce to a finite complex128 2D matrix."""
    try:
        A = np.asarray(x, dtype=np.complex128)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {A.shape}")
    if A.size:
        real = np.nan_to_num(A.real, nan=0.0, posinf=0.0, neginf=0.0)
        imag = np.nan_to_num(A.imag, nan=0.0, posinf=0.0, neginf=0.0)
        A = (real + 1j * imag).astype(np.complex128, copy=False)
    return A


def _finite_real_matrix(x: Any, *, shape: Optional[Tuple[int, int]] = None, name: str = "matrix") -> np.ndarray:
    try:
        A = np.asarray(x, dtype=np.float64)
    except Exception as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if shape is not None and A.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {A.shape}")
    if A.size:
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    return A


def _is_antihermitian_like(M: np.ndarray, *, tol: float = 1e-8) -> bool:
    A = np.asarray(M, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
        return False
    n = max(fro_norm(A), _EPS)
    herm_err = fro_norm(A - A.conj().T) / n
    anti_err = fro_norm(A + A.conj().T) / n
    return bool(anti_err < herm_err and anti_err <= max(tol, 10.0 * np.finfo(float).eps))


def _observable_from_matrix(M: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Convert arbitrary matrix into a Hermitian observable for coefficient extraction."""
    m = str(mode or "auto").lower().strip()
    A = np.asarray(M, dtype=np.complex128)
    if m == "auto":
        m = "i_kappa" if _is_antihermitian_like(A) else "direct"
    if m in {"direct", "i_kappa", "kdagk"}:
        return build_energy_observable_from_kappa(A, mode=m)
    if m in {"hermitian", "herm"}:
        return hermitianize(A)
    if m in {"raw", "none"}:
        return A
    return hermitianize(A)


def _rotation_quality(R: np.ndarray) -> Dict[str, float]:
    A = _finite_real_matrix(R, shape=(3, 3), name="rotation")
    ortho_err = float(np.linalg.norm(A.T @ A - np.eye(3), ord="fro"))
    det = float(np.linalg.det(A))
    finite = float(np.all(np.isfinite(A)))
    return {"orthogonality_error": ortho_err, "determinant": det, "finite": finite}


def _identity3() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


# =============================================================================
# Generator decomposition
# =============================================================================

@dataclass
class GeneratorDecomposition:
    """
    Decompose matrices into the span of {Lx, Ly, Lz}.

    The generators are expected to be Hermitian k x k matrices.  Coefficients are
    computed with the Hilbert-Schmidt inner product normalized by generator norms:

        c_i = Re <G_i, A> / <G_i, G_i>

    where A is either the raw input or a Hermitian observable derived from it,
    depending on observable_mode.
    """

    Lx: np.ndarray
    Ly: np.ndarray
    Lz: np.ndarray
    remove_trace: bool = True
    observable_mode: str = "auto"       # "auto" | "direct" | "i_kappa" | "kdagk" | "raw"
    strict: bool = True

    def __post_init__(self) -> None:
        self.Lx = _finite_complex_matrix(self.Lx, name="Lx")
        self.Ly = _finite_complex_matrix(self.Ly, name="Ly")
        self.Lz = _finite_complex_matrix(self.Lz, name="Lz")
        if not (self.Lx.shape == self.Ly.shape == self.Lz.shape):
            raise ValueError("Lx, Ly, Lz must have matching square shapes")
        self.k = int(self.Lx.shape[0])
        self._basis = [hermitianize(self.Lx), hermitianize(self.Ly), hermitianize(self.Lz)]
        self._den = np.asarray(
            [max(float(np.real(hs_inner(g, g))), _EPS) for g in self._basis],
            dtype=np.float64,
        )
        self.observable_mode = str(self.observable_mode or "auto").lower().strip()

    @property
    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._basis[0], self._basis[1], self._basis[2])

    def _prepare(self, M: Any, *, observable_mode: Optional[str] = None) -> np.ndarray:
        A = _finite_complex_matrix(M, name="matrix to decompose")
        if A.shape != (self.k, self.k):
            raise ValueError(f"matrix shape {A.shape} does not match generator shape {(self.k, self.k)}")
        mode = self.observable_mode if observable_mode is None else str(observable_mode).lower().strip()
        A = _observable_from_matrix(A, mode=mode)
        if self.remove_trace:
            A = remove_trace_part(A)
        return A.astype(np.complex128, copy=False)

    def decompose_vec3(self, M: Any, *, observable_mode: Optional[str] = None) -> np.ndarray:
        """Return real coefficients [cx, cy, cz]."""
        try:
            A = self._prepare(M, observable_mode=observable_mode)
            c = np.zeros((3,), dtype=np.float64)
            for i, g in enumerate(self._basis):
                c[i] = float(np.real(hs_inner(g, A)) / self._den[i])
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
            return c
        except Exception:
            if self.strict:
                raise
            return np.zeros((3,), dtype=np.float64)

    def reconstruct(self, coeffs: Sequence[float]) -> np.ndarray:
        """Reconstruct Σ_i coeffs[i] G_i as a complex128 matrix."""
        c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if c.size != 3:
            raise ValueError(f"coeffs must have length 3, got {c.size}")
        out = c[0] * self._basis[0] + c[1] * self._basis[1] + c[2] * self._basis[2]
        return np.asarray(out, dtype=np.complex128)

    def project(self, M: Any, *, observable_mode: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (projected_matrix, coeffs)."""
        c = self.decompose_vec3(M, observable_mode=observable_mode)
        return self.reconstruct(c), c

    def magnitude(self, M: Any, *, observable_mode: Optional[str] = None) -> float:
        """Return Euclidean norm of the su(2) coefficient vector."""
        c = self.decompose_vec3(M, observable_mode=observable_mode)
        val = float(np.linalg.norm(c, ord=2))
        return val if np.isfinite(val) else 0.0

    def validate(self) -> None:
        _assert(self.k > 0, "generator dimension must be positive")
        for name, G in zip(("Lx", "Ly", "Lz"), self._basis):
            _assert(G.shape == (self.k, self.k), f"{name} shape mismatch")
            _assert(np.all(np.isfinite(G.real)) and np.all(np.isfinite(G.imag)), f"{name} contains non-finite values")
        _assert(np.all(self._den > 0), "generator denominators must be positive")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "GeneratorDecomposition",
            "k": int(self.k),
            "remove_trace": bool(self.remove_trace),
            "observable_mode": str(self.observable_mode),
            "basis_norms": [float(fro_norm(g)) for g in self._basis],
            "denominators": self._den.astype(float).tolist(),
            "strict": bool(self.strict),
        }


# =============================================================================
# Holonomy builder with direct cache
# =============================================================================

@dataclass
class HolonomyBuilder:
    """
    Build an adjoint-frame SO(3) rotation R_uv from matrix edge flow X_{u->v}.

    Pipeline:
      1. A = anti-Hermitian part of X.
      2. A is scaled by alpha, optionally normalized by ||A||_F.
      3. U = exp(A) acts on generators by conjugation.
      4. The adjoint action is decomposed into a real 3x3 matrix R.
      5. R is optionally projected to SO(3).

    The direct method rotation_from_X(X) preserves legacy behavior.  The cached
    method rotation_uv(...) adds deterministic cadence/freeze semantics.
    """

    decomp: GeneratorDecomposition
    alpha: float = 0.25
    orthonormalize: bool = True
    normalize_generator: bool = True
    generator_norm_floor: float = 1e-12
    max_alpha: float = 10.0
    strict: bool = False

    # Phase-4/5 cache/cadence knobs.  Atom may set these opportunistically.
    update_every_full_steps: int = 1
    freeze_within_step: bool = True
    cache_bucket_full_steps: int = 1
    cache_enabled: bool = True

    _R_cache: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _last_update_full_step: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _R_step: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)
    _last_error: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decomp, GeneratorDecomposition):
            # Allow duck-typed compatible objects but fail later if unusable.
            pass
        self.alpha = float(np.clip(float(self.alpha), -float(self.max_alpha), float(self.max_alpha)))
        self.generator_norm_floor = float(max(_EPS, self.generator_norm_floor))
        self.update_every_full_steps = int(max(1, self.update_every_full_steps))
        self.cache_bucket_full_steps = int(max(1, self.cache_bucket_full_steps))

    def clear_cache(self) -> None:
        self._R_cache.clear()
        self._last_update_full_step.clear()
        self._R_step.clear()
        self._hits = 0
        self._misses = 0

    def _scaled_generator(self, X: Any) -> np.ndarray:
        Xmat = _finite_complex_matrix(X, name="X_uv")
        if Xmat.shape != (int(self.decomp.k), int(self.decomp.k)):
            raise ValueError(f"X_uv shape {Xmat.shape} does not match decomp k={self.decomp.k}")

        A = antihermitianize(Xmat)
        nA = fro_norm(A)
        if not np.isfinite(nA) or nA <= self.generator_norm_floor:
            return np.zeros_like(A, dtype=np.complex128)

        if self.normalize_generator:
            A = A * (float(self.alpha) / max(nA, self.generator_norm_floor))
        else:
            A = A * float(self.alpha)
        return np.asarray(A, dtype=np.complex128)

    def rotation_from_X(self, X: Any) -> np.ndarray:
        """Return a 3x3 real adjoint-frame rotation for one edge matrix."""
        self._last_error = None
        try:
            A = self._scaled_generator(X)
            if fro_norm(A) <= self.generator_norm_floor:
                return _identity3()

            U = expm(A)
            if not np.all(np.isfinite(U.real)) or not np.all(np.isfinite(U.imag)):
                raise FloatingPointError("non-finite matrix exponential")

            G = list(self.decomp.basis)
            den = np.asarray([max(float(np.real(hs_inner(g, g))), _EPS) for g in G], dtype=np.float64)

            R = np.zeros((3, 3), dtype=np.float64)
            Udag = U.conj().T
            for j in range(3):
                transported = U @ G[j] @ Udag
                for i in range(3):
                    R[i, j] = float(np.real(hs_inner(G[i], transported)) / den[i])

            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
            if self.orthonormalize:
                R = real_orthonormalize_3x3(R)

            q = _rotation_quality(R)
            if not bool(q["finite"]) or not np.isfinite(q["determinant"]):
                raise FloatingPointError("invalid SO(3) rotation")
            if self.strict and (q["orthogonality_error"] > 1e-5 or abs(q["determinant"] - 1.0) > 1e-5):
                raise FloatingPointError(f"rotation quality failed: {q}")
            return R.astype(np.float64, copy=False)

        except Exception as exc:
            self._last_error = repr(exc)
            if self.strict:
                raise
            return _identity3()

    def _make_step(self, *, step_id: Optional[int], full_step: Optional[int], stage: int) -> StepKey:
        if full_step is None:
            full_step = int(step_id) if step_id is not None else 0
        return make_step_key(
            full_step=int(full_step),
            stage=int(stage),
            cache_bucket_full_steps=int(self.cache_bucket_full_steps),
        )

    def _should_refresh(self, uv: Tuple[int, int], step: StepKey) -> bool:
        if self.freeze_within_step and int(step.stage) != 0:
            return False
        last = self._last_update_full_step.get(uv)
        if last is None:
            return True
        return (int(step.full_step) - int(last)) >= int(max(1, self.update_every_full_steps))

    def rotation_uv(
        self,
        u: int,
        v: int,
        *,
        X_uv: Any,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
        force_refresh: bool = False,
    ) -> np.ndarray:
        """
        Cached rotation for edge (u, v) with deterministic step semantics.

        Backward compatible:
            rotation_uv(u, v, X_uv=M, step_id=sid)
        Preferred:
            rotation_uv(u, v, X_uv=M, full_step=k, stage=rk_stage)
        """
        u = int(u)
        v = int(v)
        step = self._make_step(step_id=step_id, full_step=full_step, stage=int(stage))
        uv = (u, v)
        key = (u, v, int(step.bucket))

        if not self.cache_enabled:
            self._misses += 1
            return self.rotation_from_X(X_uv)

        # Freeze semantics: if the edge already has a rotation for this full step,
        # all sub-stages reuse it exactly.
        if self.freeze_within_step and not force_refresh:
            prev_step = self._R_step.get(uv)
            if prev_step is not None and int(prev_step) == int(step.full_step):
                cached = self._R_cache.get(key)
                if cached is not None:
                    self._hits += 1
                    return cached.copy()

        if (not force_refresh) and key in self._R_cache and not self._should_refresh(uv, step):
            self._R_step[uv] = int(step.full_step)
            self._hits += 1
            return self._R_cache[key].copy()

        self._misses += 1
        R = self.rotation_from_X(X_uv)
        self._R_cache[key] = R.astype(np.float64, copy=True)
        self._last_update_full_step[uv] = int(step.full_step)
        self._R_step[uv] = int(step.full_step)
        return self._R_cache[key].copy()

    def rotation_from_edge_field(
        self,
        X: Any,
        u: int,
        v: int,
        *,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
        force_refresh: bool = False,
    ) -> np.ndarray:
        """Convenience wrapper for GraphVectorField-like objects."""
        edge_values = getattr(X, "edge_values", None)
        if not isinstance(edge_values, Mapping):
            if self.strict:
                raise TypeError("X must expose an edge_values mapping")
            return _identity3()
        X_uv = edge_values.get((int(u), int(v)))
        if X_uv is None:
            if self.strict:
                raise KeyError(f"edge ({u}, {v}) missing from flow field")
            return _identity3()
        return self.rotation_uv(
            int(u),
            int(v),
            X_uv=X_uv,
            step_id=step_id,
            full_step=full_step,
            stage=stage,
            force_refresh=force_refresh,
        )

    @staticmethod
    def transport_vec3(R: Any, coeffs: Sequence[float]) -> np.ndarray:
        """Apply a 3x3 holonomy rotation to a coefficient vector."""
        A = _finite_real_matrix(R, shape=(3, 3), name="rotation")
        c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if c.size != 3:
            raise ValueError(f"coeffs must have length 3, got {c.size}")
        out = A @ c
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    def transport_basis(self, R: Any, basis: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return transported basis matrices B_i = Σ_j R[i,j] G_j.

        This orientation matches the existing Hamiltonian builder convention:
            tilde_v[i] = Σ_j R[i,j] * L_v[j]
        """
        A = _finite_real_matrix(R, shape=(3, 3), name="rotation")
        G = list(basis) if basis is not None else list(self.decomp.basis)
        if len(G) != 3:
            raise ValueError("basis must contain exactly three matrices")
        out = []
        for i in range(3):
            M = A[i, 0] * np.asarray(G[0], dtype=np.complex128)
            M = M + A[i, 1] * np.asarray(G[1], dtype=np.complex128)
            M = M + A[i, 2] * np.asarray(G[2], dtype=np.complex128)
            out.append(hermitianize(M.astype(np.complex128)))
        return (out[0], out[1], out[2])

    def health_metrics(self) -> Dict[str, Any]:
        return {
            "kind": "HolonomyBuilder",
            "cache_entries": int(len(self._R_cache)),
            "cache_hits": int(self._hits),
            "cache_misses": int(self._misses),
            "hit_rate": float(self._hits / max(1, self._hits + self._misses)),
            "last_error": self._last_error,
            "is_stable": bool(self._last_error is None),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "HolonomyBuilder",
            "alpha": float(self.alpha),
            "orthonormalize": bool(self.orthonormalize),
            "normalize_generator": bool(self.normalize_generator),
            "generator_norm_floor": float(self.generator_norm_floor),
            "strict": bool(self.strict),
            "update_every_full_steps": int(self.update_every_full_steps),
            "freeze_within_step": bool(self.freeze_within_step),
            "cache_bucket_full_steps": int(self.cache_bucket_full_steps),
            "cache_enabled": bool(self.cache_enabled),
            "decomp": self.decomp.snapshot() if hasattr(self.decomp, "snapshot") else str(self.decomp),
            "health": self.health_metrics(),
        }


# =============================================================================
# Explicit cache wrapper retained for callers that prefer composition
# =============================================================================

@dataclass
class HolonomyCache:
    """
    Deterministic wrapper cache for a HolonomyBuilder.

    This class is kept for compatibility with earlier plans.  HolonomyBuilder now
    has an equivalent direct cache, but HolonomyCache remains useful when several
    independent cadence policies must share the same builder.
    """

    builder: HolonomyBuilder
    update_every_full_steps: int = 1
    freeze_within_step: bool = True
    cache_bucket_full_steps: int = 1

    _R_cache: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _last_update_full_step: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _R_step: Dict[Tuple[int, int], int] = field(default_factory=dict, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def clear(self) -> None:
        self._R_cache.clear()
        self._last_update_full_step.clear()
        self._R_step.clear()
        self._hits = 0
        self._misses = 0

    def _step(self, *, step_id: Optional[int], full_step: Optional[int], stage: int) -> StepKey:
        if full_step is None:
            full_step = int(step_id) if step_id is not None else 0
        return make_step_key(
            full_step=int(full_step),
            stage=int(stage),
            cache_bucket_full_steps=int(max(1, self.cache_bucket_full_steps)),
        )

    def _should_refresh(self, uv: Tuple[int, int], step: StepKey) -> bool:
        if self.freeze_within_step and int(step.stage) != 0:
            return False
        last = self._last_update_full_step.get(uv)
        if last is None:
            return True
        return (int(step.full_step) - int(last)) >= int(max(1, self.update_every_full_steps))

    def rotation_uv(
        self,
        u: int,
        v: int,
        *,
        X_uv: Any,
        step_id: Optional[int] = None,
        full_step: Optional[int] = None,
        stage: int = 0,
        force_refresh: bool = False,
    ) -> np.ndarray:
        u = int(u)
        v = int(v)
        step = self._step(step_id=step_id, full_step=full_step, stage=stage)
        uv = (u, v)
        key = (u, v, int(step.bucket))

        if self.freeze_within_step and not force_refresh:
            prev_step = self._R_step.get(uv)
            if prev_step is not None and int(prev_step) == int(step.full_step):
                cached = self._R_cache.get(key)
                if cached is not None:
                    self._hits += 1
                    return cached.copy()

        if (not force_refresh) and key in self._R_cache and not self._should_refresh(uv, step):
            self._R_step[uv] = int(step.full_step)
            self._hits += 1
            return self._R_cache[key].copy()

        self._misses += 1
        R = self.builder.rotation_from_X(X_uv)
        self._R_cache[key] = R.astype(np.float64, copy=True)
        self._last_update_full_step[uv] = int(step.full_step)
        self._R_step[uv] = int(step.full_step)
        return self._R_cache[key].copy()

    def health_metrics(self) -> Dict[str, Any]:
        return {
            "kind": "HolonomyCache",
            "cache_entries": int(len(self._R_cache)),
            "cache_hits": int(self._hits),
            "cache_misses": int(self._misses),
            "hit_rate": float(self._hits / max(1, self._hits + self._misses)),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "HolonomyCache",
            "update_every_full_steps": int(self.update_every_full_steps),
            "freeze_within_step": bool(self.freeze_within_step),
            "cache_bucket_full_steps": int(self.cache_bucket_full_steps),
            "health": self.health_metrics(),
        }


# =============================================================================
# Minimal smoke test
# =============================================================================


def _self_test() -> None:
    # Pauli-style spin-1/2 generators.
    Lx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Ly = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Lz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    dec = GeneratorDecomposition(Lx, Ly, Lz, remove_trace=True)
    builder = HolonomyBuilder(dec, alpha=0.2, strict=True)
    X = np.array([[0, 1j], [1j, 0]], dtype=np.complex128)
    R = builder.rotation_from_X(X)
    q = _rotation_quality(R)
    assert R.shape == (3, 3)
    assert q["orthogonality_error"] < 1e-8
    assert abs(q["determinant"] - 1.0) < 1e-8
    R0 = builder.rotation_uv(0, 1, X_uv=X, full_step=0, stage=0)
    R1 = builder.rotation_uv(0, 1, X_uv=2 * X, full_step=0, stage=1)
    assert np.allclose(R0, R1)
    print("holonomy.py self-test passed")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "GeneratorDecomposition",
    "HolonomyBuilder",
    "HolonomyCache",
    "StepKey",
    "make_step_key",
]
