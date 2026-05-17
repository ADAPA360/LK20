#!/usr/bin/env python3
# math_utils.py
"""
AtomTN shared numerical utilities.

This module is deliberately dependency-light and CPU/NumPy-first.  It provides
stable primitives used by the AtomTN runtime family:

- finite-value guards and dtype coercion
- Hermitian / anti-Hermitian matrix operations
- Hilbert-Schmidt inner products and Frobenius diagnostics
- small-matrix exponential and Hermitian eigensolver wrappers
- SO(3) projection for holonomy rotations
- SVD truncation with discarded-weight accounting
- QR helpers used by TTN canonicalization
- norm compensation / renormalization utilities

Compatibility target
--------------------
Existing AtomTN modules import these public symbols:
    _assert, clip_int, fro_norm, rel_fro_err, as_c128, as_f64,
    hermitianize, antihermitianize, remove_trace, hs_inner, expm,
    expm_via_eig, build_energy_observable_from_kappa,
    real_orthonormalize_3x3, TruncationStats, svd_truncate_by_rank,
    svd_truncate_by_tol, svd_truncate, qr_orthonormalize_cols,
    qr_orthonormalize_rows, norm2_from_singular_values,
    discarded_weight_from_singular_values, renormalize_tensor_in_place,
    eigh_hermitian_stable.

No PyTorch/autograd dependency.  No required SciPy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Constants
# =============================================================================

EPS: float = 1e-12
TINY: float = 1e-30


# =============================================================================
# Basic guards / coercion
# =============================================================================

def _assert(cond: bool, msg: str) -> None:
    """Raise ValueError with a consistent AtomTN error style."""
    if not bool(cond):
        raise ValueError(str(msg))


def is_finite_array(x: Any) -> bool:
    """Return True iff x can be viewed as an array and contains only finite values."""
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return True
        return bool(np.isfinite(arr).all())
    except Exception:
        return False


def require_finite(x: Any, name: str = "array") -> None:
    """Validate that x contains only finite values."""
    _assert(is_finite_array(x), f"{name} contains non-finite values")


def as_c128(A: Any, *, copy: bool = False, check_finite: bool = False, name: str = "array") -> np.ndarray:
    """Coerce to complex128 ndarray."""
    out = np.array(A, dtype=np.complex128, copy=copy)
    if check_finite:
        require_finite(out, name=name)
    return out


def as_f64(A: Any, *, copy: bool = False, check_finite: bool = False, name: str = "array") -> np.ndarray:
    """Coerce to float64 ndarray."""
    out = np.array(A, dtype=np.float64, copy=copy)
    if check_finite:
        require_finite(out, name=name)
    return out


def sanitize_real_array(A: Any, *, fill: float = 0.0, dtype: np.dtype = np.float64) -> np.ndarray:
    """Coerce to a real ndarray and replace NaN/inf by fill."""
    out = np.asarray(A, dtype=np.dtype(dtype)).copy()
    if out.size:
        mask = ~np.isfinite(out)
        if np.any(mask):
            out[mask] = np.dtype(dtype).type(fill)
    return out.astype(np.dtype(dtype), copy=False)


def sanitize_complex_array(A: Any, *, fill: complex = 0.0 + 0.0j) -> np.ndarray:
    """Coerce to complex128 and replace non-finite real/imag parts by fill."""
    out = np.asarray(A, dtype=np.complex128).copy()
    if out.size:
        mask = ~np.isfinite(out.real) | ~np.isfinite(out.imag)
        if np.any(mask):
            out[mask] = np.complex128(fill)
    return out


def clip_int(x: int, lo: int, hi: int) -> int:
    """Clip integer x into [lo, hi], handling accidentally reversed bounds."""
    lo_i = int(lo)
    hi_i = int(hi)
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    return int(np.clip(int(x), lo_i, hi_i))


def safe_norm(x: Any, ord: Optional[int | float | str] = None) -> float:
    """Finite-safe vector/matrix norm.  Returns 0.0 on invalid input."""
    try:
        arr = np.asarray(x)
        if arr.size == 0 or not np.isfinite(arr).all():
            return 0.0
        val = float(np.linalg.norm(arr, ord=ord))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def fro_norm(A: Any) -> float:
    """Frobenius norm via vector 2-norm."""
    return safe_norm(np.asarray(A).reshape(-1), ord=2)


def rel_fro_err(A: Any, B: Any, eps: float = EPS) -> float:
    """Relative Frobenius error ||A-B||_F / max(||A||_F, eps)."""
    Aa = np.asarray(A)
    Bb = np.asarray(B)
    _assert(Aa.shape == Bb.shape, f"rel_fro_err: shape mismatch {Aa.shape} vs {Bb.shape}")
    return float(fro_norm(Aa - Bb) / max(fro_norm(Aa), float(eps)))


# =============================================================================
# Hermitian / anti-Hermitian matrix helpers
# =============================================================================

def _require_square_matrix(A: np.ndarray, name: str) -> None:
    _assert(A.ndim == 2 and A.shape[0] == A.shape[1], f"{name}: square matrix required")


def hermitianize(A: Any) -> np.ndarray:
    """Return (A + A†)/2 as complex128."""
    M = as_c128(A)
    _require_square_matrix(M, "hermitianize")
    return ((M + M.conj().T) * 0.5).astype(np.complex128, copy=False)


def antihermitianize(A: Any) -> np.ndarray:
    """Return (A - A†)/2 as complex128."""
    M = as_c128(A)
    _require_square_matrix(M, "antihermitianize")
    return ((M - M.conj().T) * 0.5).astype(np.complex128, copy=False)


def remove_trace(A: Any) -> np.ndarray:
    """Return A - (Tr(A)/k) I."""
    M = as_c128(A)
    _require_square_matrix(M, "remove_trace")
    k = int(M.shape[0])
    return (M - (np.trace(M) / max(k, 1)) * np.eye(k, dtype=np.complex128)).astype(np.complex128)


def commutator(A: Any, B: Any) -> np.ndarray:
    """Return [A, B] = AB - BA."""
    Aa = as_c128(A)
    Bb = as_c128(B)
    _assert(Aa.shape == Bb.shape and Aa.ndim == 2 and Aa.shape[0] == Aa.shape[1], "commutator: shape mismatch")
    return (Aa @ Bb - Bb @ Aa).astype(np.complex128)


def anticommutator(A: Any, B: Any) -> np.ndarray:
    """Return {A, B} = AB + BA."""
    Aa = as_c128(A)
    Bb = as_c128(B)
    _assert(Aa.shape == Bb.shape and Aa.ndim == 2 and Aa.shape[0] == Aa.shape[1], "anticommutator: shape mismatch")
    return (Aa @ Bb + Bb @ Aa).astype(np.complex128)


# =============================================================================
# Inner products / observables
# =============================================================================

def hs_inner(A: Any, B: Any) -> complex:
    """
    Hilbert-Schmidt inner product normalized by matrix dimension:
        <A,B> = Tr(A† B) / k
    """
    Aa = as_c128(A)
    Bb = as_c128(B)
    _assert(Aa.shape == Bb.shape and Aa.ndim == 2 and Aa.shape[0] == Aa.shape[1], "hs_inner: shape mismatch")
    k = int(Aa.shape[0])
    return complex(np.trace(Aa.conj().T @ Bb) / max(k, 1))


def build_energy_observable_from_kappa(kappa_raw: Any, mode: str = "i_kappa") -> np.ndarray:
    """
    Convert κ_raw into a Hermitian observable K_H.

    Supported modes:
      - direct:  K_H = Herm(κ_raw)
      - i_kappa: K_H = Herm(i κ_raw), recommended when κ_raw is anti-Hermitian
      - kdagk:   K_H = Herm(κ_raw† κ_raw), positive semidefinite and robust
    """
    M = as_c128(kappa_raw, check_finite=True, name="kappa_raw")
    _require_square_matrix(M, "build_energy_observable_from_kappa")
    key = str(mode or "i_kappa").lower().strip()

    if key == "direct":
        KH = hermitianize(M)
    elif key == "i_kappa":
        KH = hermitianize(1j * M)
    elif key == "kdagk":
        KH = hermitianize(M.conj().T @ M)
    else:
        raise ValueError(f"build_energy_observable_from_kappa: unknown mode '{mode}'")

    require_finite(KH, name="energy observable")
    return KH.astype(np.complex128, copy=False)


# =============================================================================
# Matrix exponential / eigensolvers
# =============================================================================

def _expm_taylor_scaling_squaring(A: np.ndarray, order: int = 32) -> np.ndarray:
    """
    Conservative fallback matrix exponential using scaling and squaring with a
    truncated Taylor series.  Intended for small matrices when eigendecomposition
    is ill-conditioned or fails.
    """
    M = as_c128(A, check_finite=True, name="expm input")
    _require_square_matrix(M, "expm")

    nrm = fro_norm(M)
    s = int(max(0, np.ceil(np.log2(max(nrm, 1.0)))))
    B = M / float(2 ** s)

    k = int(M.shape[0])
    E = np.eye(k, dtype=np.complex128)
    term = np.eye(k, dtype=np.complex128)
    for j in range(1, int(max(4, order)) + 1):
        term = (term @ B) / float(j)
        E = E + term
        if fro_norm(term) <= EPS * max(1.0, fro_norm(E)):
            break

    for _ in range(s):
        E = E @ E
    return E.astype(np.complex128)


def expm(A: Any) -> np.ndarray:
    """
    Small dense matrix exponential.

    Primary path uses eigendecomposition, which is efficient for AtomTN's small
    fuzzy matrices.  If the eigenvector matrix is ill-conditioned or a numerical
    issue is detected, a scaling-and-squaring Taylor fallback is used.
    """
    M = as_c128(A, check_finite=True, name="expm input")
    _require_square_matrix(M, "expm")

    if M.size == 0:
        return M.copy()

    try:
        w, V = np.linalg.eig(M)
        cond = float(np.linalg.cond(V))
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError(f"ill-conditioned eigenbasis cond={cond}")
        Vinv = np.linalg.inv(V)
        E = V @ (np.exp(w)[:, None] * Vinv)
        if not np.all(np.isfinite(E)):
            raise FloatingPointError("non-finite exponential")
        return E.astype(np.complex128)
    except Exception:
        return _expm_taylor_scaling_squaring(M)


def expm_via_eig(A: Any) -> np.ndarray:
    """Backward-compatible alias."""
    return expm(A)


def eigh_hermitian_stable(
    H: Any,
    sort: bool = True,
    check_herm: bool = False,
    symmetrize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stable wrapper around np.linalg.eigh for Hermitian matrices.

    Returns:
      eigenvalues as float64, eigenvectors as complex128 columns.
    """
    A = as_c128(H, check_finite=True, name="Hermitian eigensolver input")
    _require_square_matrix(A, "eigh_hermitian_stable")

    if symmetrize:
        A = hermitianize(A)

    if check_herm:
        err = fro_norm(A - A.conj().T)
        _assert(err <= 1e-8 * max(fro_norm(A), EPS), f"H not Hermitian enough: ||H-H†||={err:.3e}")

    try:
        w, V = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        # Tiny diagonal jitter fallback for nearly defective numerical input.
        jitter = EPS * max(1.0, fro_norm(A))
        A2 = A + jitter * np.eye(A.shape[0], dtype=np.complex128)
        w, V = np.linalg.eigh(hermitianize(A2))

    if sort:
        idx = np.argsort(w)
        w = w[idx]
        V = V[:, idx]

    require_finite(w, name="Hermitian eigenvalues")
    require_finite(V, name="Hermitian eigenvectors")
    return w.astype(np.float64, copy=False), V.astype(np.complex128, copy=False)


# =============================================================================
# SO(3) projection helper
# =============================================================================

def real_orthonormalize_3x3(R: Any) -> np.ndarray:
    """Project a real 3x3 matrix to the closest proper rotation in SO(3)."""
    M = as_f64(R, check_finite=True, name="SO(3) matrix")
    _assert(M.shape == (3, 3), "real_orthonormalize_3x3: expected shape (3,3)")

    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Rout = U @ Vt
    if np.linalg.det(Rout) < 0.0:
        U[:, -1] *= -1.0
        Rout = U @ Vt
    require_finite(Rout, name="SO(3) projection")
    return Rout.astype(np.float64, copy=False)


# =============================================================================
# SVD truncation and accounting
# =============================================================================

@dataclass(frozen=True)
class TruncationStats:
    """
    Truncation quality metrics.

    kept_rank:
        Number of retained singular values.
    discarded_weight:
        Relative discarded Frobenius^2, i.e. sum(S_discarded^2)/sum(S_all^2).
    discarded_frob_sq:
        Absolute discarded Frobenius^2.
    total_frob_sq:
        Total Frobenius^2 in the untruncated spectrum.
    kept_frob_sq:
        Retained Frobenius^2.
    spectral_error:
        Largest discarded singular value, or 0 if no singular value was discarded.
    """

    kept_rank: int
    discarded_weight: float
    discarded_frob_sq: float
    total_frob_sq: float = 0.0
    kept_frob_sq: float = 0.0
    spectral_error: float = 0.0


def _singular_values_1d(S: Any) -> np.ndarray:
    s = np.asarray(S, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return s
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    # SVD returns sorted descending; preserve order but remove tiny negative noise.
    return np.maximum(s, 0.0)


def _stats_for_rank(S: np.ndarray, k: int) -> TruncationStats:
    s = _singular_values_1d(S)
    if s.size == 0:
        return TruncationStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    kk = int(max(0, min(int(k), s.size)))
    kept = s[:kk]
    disc = s[kk:]
    kept_f2 = float(np.sum(kept * kept))
    disc_f2 = float(np.sum(disc * disc))
    total_f2 = kept_f2 + disc_f2
    disc_w = float(disc_f2 / max(total_f2, TINY))
    spec_err = float(disc[0]) if disc.size else 0.0
    return TruncationStats(
        kept_rank=kk,
        discarded_weight=disc_w,
        discarded_frob_sq=disc_f2,
        total_frob_sq=total_f2,
        kept_frob_sq=kept_f2,
        spectral_error=spec_err,
    )


def svd_truncate_by_rank(S: Any, rank: int) -> Tuple[int, TruncationStats]:
    """Choose kept rank by fixed cap."""
    s = _singular_values_1d(S)
    if s.size == 0:
        return 0, _stats_for_rank(s, 0)
    k = int(max(0, min(int(rank), s.size)))
    return k, _stats_for_rank(s, k)


def svd_truncate_by_tol(S: Any, tol: float, keep_at_least_one: bool = True) -> Tuple[int, TruncationStats]:
    """
    Choose kept rank using relative singular-value tolerance:
        keep S_i >= tol * S_0
    """
    s = _singular_values_1d(S)
    if s.size == 0:
        return 0, _stats_for_rank(s, 0)

    s0 = float(s[0])
    if not np.isfinite(s0) or s0 <= 0.0:
        k = 1 if keep_at_least_one else 0
        return k, _stats_for_rank(s, k)

    threshold = max(0.0, float(tol)) * s0
    k = int(np.count_nonzero(s >= threshold))
    if keep_at_least_one:
        k = max(k, 1)
    k = min(k, s.size)
    return k, _stats_for_rank(s, k)


def _choose_svd_rank(
    S: np.ndarray,
    *,
    rank: Optional[int],
    tol: Optional[float],
    keep_at_least_one: bool,
) -> Tuple[int, TruncationStats]:
    """
    Combine tolerance and fixed-rank caps.

    If tol is supplied, it selects a rank.  If rank is also supplied, the kept
    rank is capped by rank.  This is safer for memory-sensitive TTN operations.
    """
    s = _singular_values_1d(S)
    if s.size == 0:
        return 0, _stats_for_rank(s, 0)

    if tol is not None:
        k_tol, _ = svd_truncate_by_tol(s, float(tol), keep_at_least_one=keep_at_least_one)
    else:
        k_tol = int(s.size)

    if rank is not None:
        k = min(k_tol, int(max(0, rank)))
    else:
        k = k_tol

    if keep_at_least_one:
        k = max(k, 1)
    k = min(k, int(s.size))
    return k, _stats_for_rank(s, k)


def svd_truncate(
    M: Any,
    rank: Optional[int] = None,
    tol: Optional[float] = None,
    full_matrices: bool = False,
    keep_at_least_one: bool = True,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TruncationStats]:
    """
    Compute a truncated SVD with discarded-weight accounting.

    Rules:
      - If tol is provided, keep S_i >= tol*S_0.
      - If rank is provided, cap the kept rank by rank.
      - If neither is provided, keep the full SVD.

    Returns:
      U_k, S_k, Vh_k, TruncationStats
    """
    A = np.asarray(M)
    _assert(A.ndim == 2, "svd_truncate: expected a 2D matrix")
    if check_finite:
        require_finite(A, name="svd_truncate input")

    if A.shape[0] == 0 or A.shape[1] == 0:
        dt = np.result_type(A.dtype, np.complex128 if np.iscomplexobj(A) else np.float64)
        U = np.zeros((A.shape[0], 0), dtype=dt)
        S = np.zeros((0,), dtype=np.float64)
        Vh = np.zeros((0, A.shape[1]), dtype=dt)
        stats = TruncationStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return U, S, Vh, stats

    try:
        U, S, Vh = np.linalg.svd(A, full_matrices=bool(full_matrices))
    except np.linalg.LinAlgError:
        # Tiny deterministic perturbation for rare convergence failures.
        eps = EPS * max(1.0, fro_norm(A))
        A2 = A.copy()
        diag_n = min(A2.shape)
        A2[np.arange(diag_n), np.arange(diag_n)] += eps
        U, S, Vh = np.linalg.svd(A2, full_matrices=bool(full_matrices))

    k, stats = _choose_svd_rank(S, rank=rank, tol=tol, keep_at_least_one=keep_at_least_one)
    U_k = U[:, :k].astype(np.complex128 if np.iscomplexobj(U) else U.dtype, copy=False)
    S_k = S[:k].astype(np.float64, copy=False)
    Vh_k = Vh[:k, :].astype(np.complex128 if np.iscomplexobj(Vh) else Vh.dtype, copy=False)
    return U_k, S_k, Vh_k, stats


# =============================================================================
# QR helpers
# =============================================================================

def qr_orthonormalize_cols(M: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Column QR: M = Q R, with Q column-orthonormal."""
    A = as_c128(M, check_finite=True, name="qr_orthonormalize_cols input")
    _assert(A.ndim == 2, "qr_orthonormalize_cols: expected 2D matrix")
    Q, R = np.linalg.qr(A, mode="reduced")
    require_finite(Q, name="QR Q")
    require_finite(R, name="QR R")
    return Q.astype(np.complex128, copy=False), R.astype(np.complex128, copy=False)


def qr_orthonormalize_rows(M: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row QR via QR on transpose.

    Returns:
      M_rows, transform
    where M_rows has orthonormal rows and transform is the left factor such that
    approximately M = transform @ M_rows.
    """
    A = as_c128(M, check_finite=True, name="qr_orthonormalize_rows input")
    _assert(A.ndim == 2, "qr_orthonormalize_rows: expected 2D matrix")
    Q, R = np.linalg.qr(A.T, mode="reduced")
    M_rows = Q.T.astype(np.complex128, copy=False)
    transform = R.T.astype(np.complex128, copy=False)
    require_finite(M_rows, name="row-orthonormalized matrix")
    require_finite(transform, name="row QR transform")
    return M_rows, transform


# =============================================================================
# Norm accounting / renormalization
# =============================================================================

def norm2_from_singular_values(S: Any) -> float:
    """Frobenius norm squared represented by singular values."""
    s = _singular_values_1d(S)
    return float(np.sum(s * s))


def discarded_weight_from_singular_values(S_full: Any, S_kept: Any) -> float:
    """Relative discarded Frobenius^2 after keeping S_kept out of S_full."""
    full = norm2_from_singular_values(S_full)
    kept = norm2_from_singular_values(S_kept)
    disc = max(float(full - kept), 0.0)
    return float(disc / max(full, TINY))


def renormalize_tensor_in_place(T: np.ndarray, target_norm2: float, eps: float = TINY) -> None:
    """Scale tensor in-place so its Frobenius norm squared equals target_norm2."""
    arr = np.asarray(T)
    cur = float(np.sum(np.abs(arr) ** 2))
    if not np.isfinite(cur) or cur <= float(eps):
        return
    tgt = max(float(target_norm2), 0.0)
    scale = float(np.sqrt(tgt / cur))
    T *= scale


def normalize_vector(x: Any, eps: float = EPS) -> np.ndarray:
    """Return L2-normalized complex vector; zero vector if norm is too small."""
    v = as_c128(x).reshape(-1)
    n = fro_norm(v)
    if n <= float(eps):
        return np.zeros_like(v, dtype=np.complex128)
    return (v / n).astype(np.complex128, copy=False)


__all__ = [
    "EPS",
    "TINY",
    "_assert",
    "is_finite_array",
    "require_finite",
    "clip_int",
    "safe_norm",
    "fro_norm",
    "rel_fro_err",
    "as_c128",
    "as_f64",
    "sanitize_real_array",
    "sanitize_complex_array",
    "hermitianize",
    "antihermitianize",
    "remove_trace",
    "commutator",
    "anticommutator",
    "hs_inner",
    "expm",
    "expm_via_eig",
    "build_energy_observable_from_kappa",
    "real_orthonormalize_3x3",
    "TruncationStats",
    "svd_truncate_by_rank",
    "svd_truncate_by_tol",
    "svd_truncate",
    "qr_orthonormalize_cols",
    "qr_orthonormalize_rows",
    "norm2_from_singular_values",
    "discarded_weight_from_singular_values",
    "renormalize_tensor_in_place",
    "normalize_vector",
    "eigh_hermitian_stable",
]
