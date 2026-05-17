#!/usr/bin/env python3
# constraints.py
"""
Operator-basis constraints / hooks for AtomTN.

This module provides the local operator basis used by LocalFiberBuilder and the
TTN Hamiltonian builders. It is intentionally NumPy-only and import-safe.

Current production role
-----------------------
- Provides deterministic local qudit operator bases for arbitrary physical
  dimension d >= 1.
- Embeds Pauli X/Y/Z in the top-left 2x2 block for d >= 2.
- Provides a seeded Hermitian auxiliary operator ``G`` for geometry / Adinkra
  placeholder coupling.
- Caches per-dimension bases because these matrices are requested repeatedly in
  tight tensor-network loops.

Forward-compatible role
-----------------------
The ``mode`` parameter prepares this hook for later true Adinkra / SUSY graph
operator generation while preserving the current public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np

try:  # local AtomTN import path
    from math_utils import hermitianize, fro_norm
except Exception:  # pragma: no cover - fallback for isolated import checks
    def hermitianize(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.complex128)
        return (A + A.conj().T) / 2.0

    def fro_norm(A: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(A).reshape(-1), ord=2))


_EPS = 1e-12
_SUPPORTED_MODES = {"random_hermitian", "susy_bipartite", "bipartite_susy"}


def _pauli2() -> Dict[str, np.ndarray]:
    """Return the canonical 2x2 Pauli basis as complex128 arrays."""
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def _copy_ops(ops: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a defensive deep copy of an operator dictionary."""
    return {str(k): np.asarray(v, dtype=np.complex128).copy() for k, v in ops.items()}


def _stable_seed(seed: int, d: int, mode: str) -> int:
    """
    Deterministic 32-bit seed mixer.

    Avoids Python's randomized hash() so that operator generation is stable
    across processes and Python invocations.
    """
    h = 2166136261
    payload = f"AdinkraConstraint::{int(seed)}::{int(d)}::{mode}".encode("utf-8", errors="ignore")
    for b in payload:
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _sanitize_complex_matrix(A: Any, *, name: str) -> np.ndarray:
    """
    Convert to complex128 and replace non-finite entries.

    The caller should check the original for non-finite values before calling
    this when strict failure semantics are required. This routine guarantees the
    returned array is finite unless the input cannot be coerced.
    """
    arr = np.asarray(A, dtype=np.complex128)
    if arr.size == 0:
        return arr.astype(np.complex128, copy=False)
    real = np.nan_to_num(arr.real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = np.nan_to_num(arr.imag, nan=0.0, posinf=0.0, neginf=0.0)
    out = (real + 1j * imag).astype(np.complex128, copy=False)
    if not np.all(np.isfinite(out.real)) or not np.all(np.isfinite(out.imag)):
        raise FloatingPointError(f"{name} contains non-finite values after sanitization")
    return out


def _normalize_operator(A: np.ndarray, *, name: str, eps: float = _EPS) -> np.ndarray:
    """Hermitianize and Frobenius-normalize an operator safely."""
    H = _sanitize_complex_matrix(hermitianize(A), name=f"{name}.hermitian")
    n = float(fro_norm(H))
    if not np.isfinite(n) or n <= float(eps):
        return np.zeros_like(H, dtype=np.complex128)
    return (H / n).astype(np.complex128, copy=False)


@dataclass
class AdinkraConstraint:
    """
    Deterministic operator-basis generator for local qudit fibers.

    Parameters
    ----------
    seed:
        Base seed for deterministic generation.
    mode:
        Operator generation mode for the auxiliary ``G`` operator.

        - ``"random_hermitian"``: seeded random Hermitian matrix normalized to
          Frobenius norm 1. This preserves the historical behavior.
        - ``"susy_bipartite"`` / ``"bipartite_susy"``: deterministic bipartite
          off-diagonal placeholder useful as a scaffold for later Adinkra / SUSY
          graph generators.
    cache_max_entries:
        Soft health threshold for cached dimensions. It does not evict entries;
        it only affects ``health_metrics()``.

    Public contract
    ---------------
    ``operator_basis(d)`` returns a fresh dictionary of complex128 arrays. Callers
    can mutate the returned matrices without corrupting this instance's cache.
    """

    seed: int = 0
    mode: str = "random_hermitian"
    cache_max_entries: int = 128

    _cache: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _nonfinite_seen: int = field(default=0, init=False, repr=False)
    _last_error: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.seed = int(self.seed)
        self.mode = str(self.mode or "random_hermitian").lower().strip()
        self.cache_max_entries = int(max(1, self.cache_max_entries))
        if self.mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported AdinkraConstraint mode: {self.mode!r}. Supported: {sorted(_SUPPORTED_MODES)}")

    # ------------------------------------------------------------------
    # Basis generation
    # ------------------------------------------------------------------

    @staticmethod
    def _pauli_embedding(d: int) -> Dict[str, np.ndarray]:
        """
        Build I/X/Y/Z for a d-dimensional qudit space.

        For d >= 2, X/Y/Z occupy the top-left 2x2 Pauli block and the remaining
        subspace is identity for X/Z and Y's inherited identity scaffold per the
        historical embedding convention requested for AtomTN compatibility.

        For d == 1, the explicit fallback is:
          X = [[1]], Y = [[0]], Z = [[1]].
        """
        d = int(d)
        if d <= 0:
            raise ValueError("operator dimension d must be positive")

        ops: Dict[str, np.ndarray] = {"I": np.eye(d, dtype=np.complex128)}

        if d >= 2:
            P = _pauli2()
            for name in ("X", "Y", "Z"):
                A = np.eye(d, dtype=np.complex128)
                A[:2, :2] = P[name]
                ops[name] = A.astype(np.complex128, copy=False)
        else:
            ops["X"] = np.eye(d, dtype=np.complex128)
            ops["Y"] = np.zeros((d, d), dtype=np.complex128)
            ops["Z"] = np.eye(d, dtype=np.complex128)

        return ops

    def _random_hermitian_G(self, d: int) -> np.ndarray:
        rng = np.random.default_rng(_stable_seed(self.seed, d, self.mode))
        A = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))).astype(np.complex128)
        return _normalize_operator(A, name="G.random_hermitian")

    def _susy_bipartite_G(self, d: int) -> np.ndarray:
        """
        Deterministic bipartite off-diagonal scaffold for future Adinkra modes.

        This is not a full Adinkra construction. It creates a Hermitian adjacency
        matrix connecting even-indexed basis states to odd-indexed basis states
        with deterministic signs. It gives the runtime a structured alternative
        to random ``G`` while keeping the current API stable.
        """
        G = np.zeros((d, d), dtype=np.complex128)
        if d <= 1:
            G[0, 0] = 1.0
            return G

        for i in range(d):
            for j in range(i + 1, d):
                if (i % 2) != (j % 2):
                    # Deterministic signed connectivity with seed dependence.
                    h = _stable_seed(self.seed + 17 * i + 31 * j, d, self.mode)
                    sign = -1.0 if (h & 1) else 1.0
                    G[i, j] = sign
                    G[j, i] = sign

        return _normalize_operator(G, name="G.susy_bipartite")

    def _build_G(self, d: int) -> np.ndarray:
        if self.mode == "random_hermitian":
            return self._random_hermitian_G(d)
        if self.mode in {"susy_bipartite", "bipartite_susy"}:
            return self._susy_bipartite_G(d)
        raise ValueError(f"Unsupported AdinkraConstraint mode: {self.mode!r}")

    def _validate_and_sanitize_ops(self, ops: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        bad: Dict[str, str] = {}

        for name, mat in ops.items():
            arr = np.asarray(mat, dtype=np.complex128)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                bad[str(name)] = f"non-square shape={arr.shape}"
                continue

            finite_before = bool(np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag)))
            arr = _sanitize_complex_matrix(arr, name=str(name))
            if not finite_before:
                bad[str(name)] = "non-finite values detected"
            out[str(name)] = arr

        if bad:
            self._nonfinite_seen += int(sum(1 for reason in bad.values() if "non-finite" in reason))
            self._last_error = f"operator_basis validation failed: {bad}"
            raise FloatingPointError(self._last_error)

        expected = {"I", "X", "Y", "Z", "G"}
        missing = sorted(expected.difference(out.keys()))
        if missing:
            self._last_error = f"operator_basis missing required operators: {missing}"
            raise KeyError(self._last_error)

        return out

    def operator_basis(self, d: int) -> Dict[str, np.ndarray]:
        """
        Return a deterministic local operator basis for dimension ``d``.

        The returned dictionary contains fresh copies of:
          ``I, X, Y, Z, G``.
        """
        d = int(d)
        if d <= 0:
            self._last_error = f"invalid operator dimension d={d}"
            raise ValueError(self._last_error)

        cached = self._cache.get(d)
        if cached is not None:
            return _copy_ops(cached)

        ops = self._pauli_embedding(d)
        ops["G"] = self._build_G(d)
        ops = self._validate_and_sanitize_ops(ops)

        self._cache[d] = _copy_ops(ops)
        self._last_error = ""
        return _copy_ops(ops)

    # ------------------------------------------------------------------
    # Diagnostics and cache control
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear cached operator bases while preserving diagnostics counters."""
        self._cache.clear()

    def cache_size(self) -> int:
        return int(len(self._cache))

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe runtime snapshot."""
        return {
            "kind": "AdinkraConstraint",
            "seed": int(self.seed),
            "mode": str(self.mode),
            "cache_size": int(len(self._cache)),
            "cached_dimensions": sorted(int(k) for k in self._cache.keys()),
            "cache_max_entries": int(self.cache_max_entries),
            "nonfinite_seen": int(self._nonfinite_seen),
            "last_error": self._last_error or None,
            "supported_modes": sorted(_SUPPORTED_MODES),
        }

    def health_metrics(self) -> Dict[str, Any]:
        """Return lightweight health diagnostics for Akkurat/AtomTN monitoring."""
        cache_size = int(len(self._cache))
        cache_too_large = bool(cache_size > int(self.cache_max_entries))
        has_nonfinite = bool(self._nonfinite_seen > 0)
        ok = bool((not cache_too_large) and (not has_nonfinite) and not self._last_error)
        return {
            "kind": "AdinkraConstraint",
            "is_stable": ok,
            "cache_size": cache_size,
            "cache_too_large": cache_too_large,
            "cache_max_entries": int(self.cache_max_entries),
            "has_nonfinite": has_nonfinite,
            "nonfinite_seen": int(self._nonfinite_seen),
            "last_error": self._last_error or None,
            "mode": str(self.mode),
        }


__all__ = ["AdinkraConstraint", "_pauli2"]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    c = AdinkraConstraint(seed=0)
    basis = c.operator_basis(4)
    print({k: tuple(v.shape) for k, v in basis.items()})
    print(c.health_metrics())
