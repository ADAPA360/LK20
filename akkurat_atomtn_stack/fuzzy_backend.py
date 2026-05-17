#!/usr/bin/env python3
# fuzzy_backend.py
"""
Fuzzy / noncommutative backend for AtomTN.

Production role
---------------
This module provides the matrix-valued graph calculus used by AtomTN's
noncommutative runtime:

- FuzzySU2: spin-l su(2) representation with k = 2l + 1.
- NCTwist: deterministic inner twist/reality map.
- NCFuzzyBackend: matrix-valued edge calculus, divergence, curvature proxies,
  twisted-reality projection, su(2) projection, and optional richer Hermitian
  basis projections.

The public API intentionally preserves compatibility with the existing AtomTN
scripts:

    backend = NCFuzzyBackend(adjacency, fuzzy_l=2, seed=0)
    edges = backend.oriented_edges()
    div = backend.div_edge_field(X_edge, N=64)
    kappa = backend.kappa_from_edge(X_edge, N=64)
    X_proj = backend.project_twisted_reality_edge(X_edge)
    M_su2, coeffs = backend.project_to_su2(M, remove_trace=True)

Design invariants
-----------------
- Fuzzy generators Lx, Ly, Lz are Hermitian complex128 matrices.
- Edge fields are dictionaries keyed by directed edge tuples (u, v).
- Matrix-valued flows use Mat_k(C) at each directed edge.
- Divergence is node-local outgoing sum: div(X)(u) = sum_v X_{u->v}.
- κ_raw(u) = 1/2 div(X)(u).
- Projection routines sanitize non-finite values instead of leaking NaNs.
- Deterministic construction: seed controls the twist unitary and optional random
  basis generation.

Projection note
---------------
κ_raw from the NC flow is often anti-Hermitian. For a Hermitian observable-style
projection into span{Lx,Ly,Lz}, the backend defaults to projection_mode="auto":
anti-Hermitian inputs are converted with Herm(iκ), while general inputs use
Herm(κ). This is more useful for energy-gauge workflows than blindly taking
Herm(κ), which can collapse anti-Hermitian κ to near zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from math_utils import (
    _assert,
    antihermitianize,
    build_energy_observable_from_kappa,
    fro_norm,
    hermitianize,
    hs_inner,
    remove_trace as remove_trace_part,
)


_EPS = 1e-12


# =============================================================================
# Small helpers
# =============================================================================


def _as_c128_matrix(x: Any, *, shape: Optional[Tuple[int, int]] = None, fill: complex = 0.0) -> np.ndarray:
    """Best-effort complex128 matrix coercion with finite sanitation."""
    try:
        A = np.asarray(x, dtype=np.complex128)
    except Exception:
        if shape is None:
            shape = (0, 0)
        return np.full(shape, fill, dtype=np.complex128)

    if shape is not None and A.shape != shape:
        out = np.full(shape, fill, dtype=np.complex128)
        if A.ndim == 2:
            r = min(shape[0], A.shape[0])
            c = min(shape[1], A.shape[1])
            out[:r, :c] = A[:r, :c]
        A = out

    A = np.asarray(A, dtype=np.complex128)
    if A.size:
        real = np.nan_to_num(A.real, nan=0.0, posinf=0.0, neginf=0.0)
        imag = np.nan_to_num(A.imag, nan=0.0, posinf=0.0, neginf=0.0)
        A = (real + 1j * imag).astype(np.complex128, copy=False)
    return A


def _matrix_is_antihermitian_like(M: np.ndarray, *, tol: float = 1e-8) -> bool:
    """Return True when M is much closer to anti-Hermitian than Hermitian."""
    A = np.asarray(M, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
        return False
    n = max(fro_norm(A), _EPS)
    herm_err = fro_norm(A - A.conj().T) / n
    anti_err = fro_norm(A + A.conj().T) / n
    return bool(anti_err < herm_err and anti_err <= max(tol, 10.0 * np.finfo(float).eps))


def _matrix_to_projection_observable(M: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    Convert an arbitrary matrix into a Hermitian observable before basis projection.

    mode:
      - direct:  Herm(M)
      - i_kappa: Herm(iM), useful when M is anti-Hermitian
      - kdagk:   Herm(M†M), positive semidefinite
      - auto:    use i_kappa when M is anti-Hermitian-like, else direct
    """
    m = str(mode or "auto").lower().strip()
    A = np.asarray(M, dtype=np.complex128)
    if m == "auto":
        m = "i_kappa" if _matrix_is_antihermitian_like(A) else "direct"
    if m in {"direct", "i_kappa", "kdagk"}:
        return build_energy_observable_from_kappa(A, mode=m)
    # Safe fallback for unknown modes.
    return hermitianize(A)


def _normalize_matrix(A: np.ndarray, *, target_norm: float = 1.0, eps: float = _EPS) -> np.ndarray:
    n = fro_norm(A)
    if not np.isfinite(n) or n <= eps:
        return np.zeros_like(A, dtype=np.complex128)
    return (float(target_norm) * A / n).astype(np.complex128, copy=False)


def _stable_sorted_adjacency(adjacency: Mapping[int, Iterable[int]]) -> Dict[int, List[int]]:
    """Return deterministic int adjacency with duplicate neighbors removed."""
    out: Dict[int, List[int]] = {}
    for u, nbrs in dict(adjacency).items():
        uu = int(u)
        vals = sorted({int(v) for v in nbrs if int(v) != uu})
        out[uu] = vals
    return out


# =============================================================================
# Fuzzy SU(2)
# =============================================================================


@dataclass
class FuzzySU2:
    """
    Spin-l representation of su(2), dimension k=2l+1.

    The returned generators are Hermitian and satisfy the angular momentum
    commutation relations up to numerical precision:
        [Li, Lj] = i eps_ijk Lk.
    """

    l: int

    def __post_init__(self) -> None:
        self.l = int(self.l)
        _assert(self.l >= 0, "FuzzySU2: l must be >= 0")
        self.k = 2 * self.l + 1
        self.Lx, self.Ly, self.Lz = self._build_generators(self.l)

        denom = float(np.sqrt(max(self.l * (self.l + 1), _EPS)))
        self.x = [
            (self.Lx / denom).astype(np.complex128),
            (self.Ly / denom).astype(np.complex128),
            (self.Lz / denom).astype(np.complex128),
        ]

    @staticmethod
    def _build_generators(l: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = 2 * int(l) + 1
        if k <= 0:
            raise ValueError("invalid spin dimension")

        # Basis ordering: |m>, m = l, l-1, ..., -l.
        m = np.arange(l, -l - 1, -1, dtype=np.float64)
        Lz = np.diag(m).astype(np.complex128)
        Lp = np.zeros((k, k), dtype=np.complex128)
        Lm = np.zeros((k, k), dtype=np.complex128)

        for i in range(k - 1):
            mi = float(m[i])
            coeff = float(np.sqrt(max(l * (l + 1) - mi * (mi - 1), 0.0)))
            Lm[i + 1, i] = coeff
            Lp[i, i + 1] = coeff

        Lx = (Lp + Lm) / 2.0
        Ly = (Lp - Lm) / (2.0j)
        return hermitianize(Lx), hermitianize(Ly), hermitianize(Lz)

    def generators(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.Lx, self.Ly, self.Lz

    def casimir(self) -> np.ndarray:
        return (self.Lx @ self.Lx + self.Ly @ self.Ly + self.Lz @ self.Lz).astype(np.complex128)

    def diagnostics(self) -> Dict[str, Any]:
        comm_xy = self.Lx @ self.Ly - self.Ly @ self.Lx
        rel = fro_norm(comm_xy - 1j * self.Lz) / max(fro_norm(self.Lz), _EPS)
        C = self.casimir()
        target = self.l * (self.l + 1) * np.eye(self.k, dtype=np.complex128)
        cas_err = fro_norm(C - target) / max(fro_norm(target), _EPS)
        return {
            "l": int(self.l),
            "k": int(self.k),
            "commutator_rel_error_xy": float(rel),
            "casimir_rel_error": float(cas_err),
        }


# =============================================================================
# Twisted reality helper
# =============================================================================


def random_unitary(k: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a deterministic Haar-ish unitary from a complex Gaussian QR."""
    k = int(k)
    _assert(k >= 1, "random_unitary: k must be positive")
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    A = (rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))).astype(np.complex128)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    phase = np.ones_like(d, dtype=np.complex128)
    nz = np.abs(d) > _EPS
    phase[nz] = d[nz] / np.abs(d[nz])
    Q = Q @ np.diag(np.conj(phase))
    return Q.astype(np.complex128)


@dataclass
class NCTwist:
    """Inner twist ς(A) = J A J†."""

    J: np.ndarray

    def __post_init__(self) -> None:
        J = _as_c128_matrix(self.J)
        _assert(J.ndim == 2 and J.shape[0] == J.shape[1], "NCTwist: J must be square")
        self.J = J

    def apply(self, A: np.ndarray) -> np.ndarray:
        X = _as_c128_matrix(A, shape=self.J.shape)
        return (self.J @ X @ self.J.conj().T).astype(np.complex128)

    def inverse_apply(self, A: np.ndarray) -> np.ndarray:
        X = _as_c128_matrix(A, shape=self.J.shape)
        return (self.J.conj().T @ X @ self.J).astype(np.complex128)


# =============================================================================
# Backend config
# =============================================================================


@dataclass
class NCFuzzyBackendConfig:
    fuzzy_l: int = 2
    seed: int = 0
    projection_mode: str = "auto"  # auto | direct | i_kappa | kdagk
    enforce_antisymmetric_pairs: bool = True
    twisted_reality_mix: float = 0.5
    finite_checks: bool = True


# =============================================================================
# Noncommutative calculus backend
# =============================================================================


@dataclass
class NCFuzzyBackend:
    """
    Matrix-valued calculus on a graph.

    Parameters
    ----------
    adjacency:
        Node -> list of neighbors. Values are normalized to deterministic int
        lists. Edges are assumed undirected at the graph level; directed fields
        may contain both (u,v) and (v,u).
    fuzzy_l:
        Spin representation parameter. Matrix dimension is k=2*fuzzy_l+1.
    seed:
        Controls the twist unitary.
    projection_mode:
        Observable conversion used by project_to_su2/project_to_basis. The
        default "auto" maps anti-Hermitian κ to Herm(iκ).
    """

    adjacency: Dict[int, List[int]]
    fuzzy_l: int = 2
    seed: int = 0
    projection_mode: str = "auto"
    enforce_antisymmetric_pairs: bool = True
    twisted_reality_mix: float = 0.5
    finite_checks: bool = True

    def __post_init__(self) -> None:
        self.adjacency = _stable_sorted_adjacency(self.adjacency)
        self.fuzzy_l = int(self.fuzzy_l)
        self.seed = int(self.seed)
        self.projection_mode = str(self.projection_mode or "auto").lower().strip()
        self.enforce_antisymmetric_pairs = bool(self.enforce_antisymmetric_pairs)
        self.twisted_reality_mix = float(np.clip(self.twisted_reality_mix, 0.0, 1.0))
        self.finite_checks = bool(self.finite_checks)

        self.fuzzy = FuzzySU2(self.fuzzy_l)
        self.k = int(self.fuzzy.k)

        rng = np.random.default_rng(self.seed)
        self.twist = NCTwist(J=random_unitary(self.k, rng))

        # Cached generator basis and denominators for stable projection.
        self._G: List[np.ndarray] = [self.fuzzy.Lx, self.fuzzy.Ly, self.fuzzy.Lz]
        self._den = np.array(
            [max(float(np.real(hs_inner(g, g))), _EPS) for g in self._G],
            dtype=np.float64,
        )
        self._higher_basis_cache: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def oriented_edges(self) -> List[Tuple[int, int]]:
        """Return undirected graph edges as deterministic (u,v) with u < v."""
        edges: List[Tuple[int, int]] = []
        for u in sorted(self.adjacency.keys()):
            for v in self.adjacency[u]:
                if int(u) < int(v):
                    edges.append((int(u), int(v)))
        return edges

    def directed_edges(self) -> List[Tuple[int, int]]:
        """Return all directed edges listed by adjacency."""
        out: List[Tuple[int, int]] = []
        for u in sorted(self.adjacency.keys()):
            for v in self.adjacency[u]:
                out.append((int(u), int(v)))
        return out

    @property
    def num_nodes(self) -> int:
        if not self.adjacency:
            return 0
        return int(max(max(self.adjacency.keys()), max((max(vs) if vs else -1) for vs in self.adjacency.values())) + 1)

    # ------------------------------------------------------------------
    # Edge field coercion and validation
    # ------------------------------------------------------------------

    def zero_matrix(self) -> np.ndarray:
        return np.zeros((self.k, self.k), dtype=np.complex128)

    def coerce_edge_field(self, X_edge: Mapping[Tuple[int, int], Any]) -> Dict[Tuple[int, int], np.ndarray]:
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for key, val in dict(X_edge).items():
            try:
                u, v = key
                e = (int(u), int(v))
            except Exception:
                continue
            out[e] = _as_c128_matrix(val, shape=(self.k, self.k))
        return out

    def validate_edge_field(self, X_edge: Mapping[Tuple[int, int], Any], *, require_pairs: bool = False) -> None:
        X = self.coerce_edge_field(X_edge)
        for (u, v), M in X.items():
            _assert(M.shape == (self.k, self.k), f"edge {(u, v)} has shape {M.shape}, expected {(self.k, self.k)}")
            if self.finite_checks:
                _assert(np.all(np.isfinite(M.real)) and np.all(np.isfinite(M.imag)), f"edge {(u, v)} contains non-finite values")
            if require_pairs:
                _assert((v, u) in X, f"missing reverse edge {(v, u)}")

    # ------------------------------------------------------------------
    # Divergence / curvature
    # ------------------------------------------------------------------

    def div_edge_field(self, X_edge: Mapping[Tuple[int, int], Any], N: Optional[int] = 64) -> Dict[int, np.ndarray]:
        """Compute div(X)(u) = sum_v X_{u->v}."""
        X = self.coerce_edge_field(X_edge)
        if N is None:
            N = max(self.num_nodes, 1 + max([u for e in X for u in e], default=-1))
        NN = int(max(0, N))
        div: Dict[int, np.ndarray] = {i: self.zero_matrix() for i in range(NN)}
        for (u, _v), M in X.items():
            if 0 <= int(u) < NN:
                div[int(u)] = div[int(u)] + M
        return div

    def kappa_from_edge(self, X_edge: Mapping[Tuple[int, int], Any], N: Optional[int] = 64) -> Dict[int, np.ndarray]:
        """κ_raw(u) = 1/2 div(X)(u)."""
        div = self.div_edge_field(X_edge, N=N)
        return {u: (0.5 * div[u]).astype(np.complex128) for u in div}

    def norm_node_field(self, A_node: Mapping[int, Any], *, N: int = 64) -> np.ndarray:
        """Return per-node Frobenius norms as shape (N,)."""
        out = np.zeros((int(N),), dtype=np.float64)
        for u, M in dict(A_node).items():
            uu = int(u)
            if 0 <= uu < int(N):
                out[uu] = fro_norm(_as_c128_matrix(M, shape=(self.k, self.k)))
        return out

    def edge_energy(self, X_edge: Mapping[Tuple[int, int], Any]) -> float:
        """Mean squared Frobenius norm over unique oriented graph edges."""
        X = self.coerce_edge_field(X_edge)
        vals: List[float] = []
        for u, v in self.oriented_edges():
            M = X.get((u, v), None)
            if M is None:
                M = -X.get((v, u), self.zero_matrix())
            n = fro_norm(M)
            vals.append(n * n)
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------
    # Reality / antisymmetry projections
    # ------------------------------------------------------------------

    def project_twisted_reality_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Twisted reality projection of one matrix:
            P(X) = (1-mix) X + mix * ς(X†)
        with mix=0.5 matching the original scaffold.
        """
        A = _as_c128_matrix(X, shape=(self.k, self.k))
        return ((1.0 - self.twisted_reality_mix) * A + self.twisted_reality_mix * self.twist.apply(A.conj().T)).astype(np.complex128)

    def project_antisymmetric_edge(self, X_edge: Mapping[Tuple[int, int], Any]) -> Dict[Tuple[int, int], np.ndarray]:
        """Enforce X_{v,u} = -X_{u,v} using canonical u<v edge pairs."""
        X = self.coerce_edge_field(X_edge)
        out: Dict[Tuple[int, int], np.ndarray] = {}
        all_pairs = {tuple(sorted((int(u), int(v)))) for (u, v) in X.keys() if int(u) != int(v)}
        # Include graph edges even if absent from X so sparse fields remain complete where possible.
        all_pairs.update((int(u), int(v)) for u, v in self.oriented_edges())

        for u, v in sorted(all_pairs):
            Auv = X.get((u, v), None)
            Avu = X.get((v, u), None)
            if Auv is None and Avu is None:
                continue
            if Auv is None:
                A = -Avu  # type: ignore[operator]
            elif Avu is None:
                A = Auv
            else:
                A = 0.5 * (Auv - Avu)
            A = _as_c128_matrix(A, shape=(self.k, self.k))
            out[(u, v)] = A
            out[(v, u)] = -A
        return out

    def project_twisted_reality_edge(self, X_edge: Mapping[Tuple[int, int], Any]) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Apply twisted-reality projection to an edge field.

        When enforce_antisymmetric_pairs=True, projection is performed pairwise on
        canonical u<v edges and the reverse edge is set to -forward. This keeps
        the edge-field contract expected by flow and Hamiltonian code.
        """
        X = self.coerce_edge_field(X_edge)

        if not self.enforce_antisymmetric_pairs:
            return {e: self.project_twisted_reality_matrix(M) for e, M in X.items()}

        Xanti = self.project_antisymmetric_edge(X)
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for u, v in self.oriented_edges():
            A = Xanti.get((u, v), None)
            if A is None:
                continue
            P = self.project_twisted_reality_matrix(A)
            out[(u, v)] = P
            out[(v, u)] = -P

        # Preserve any non-graph pairs that arrived in X_edge.
        for e, M in Xanti.items():
            if e not in out and (e[1], e[0]) not in out:
                u, v = e
                if u < v:
                    P = self.project_twisted_reality_matrix(M)
                    out[(u, v)] = P
                    out[(v, u)] = -P
        return out

    # ------------------------------------------------------------------
    # Basis projection
    # ------------------------------------------------------------------

    def _projection_observable(self, M: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
        return _matrix_to_projection_observable(M, mode=mode or self.projection_mode)

    def project_to_su2(
        self,
        M: np.ndarray,
        remove_trace: bool = True,
        *,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project a matrix into span{Lx,Ly,Lz}.

        Returns
        -------
        M_su2:
            Hermitian projected matrix.
        coeffs:
            Real coefficients [cx, cy, cz].
        """
        A0 = _as_c128_matrix(M, shape=(self.k, self.k))
        A = self._projection_observable(A0, mode=mode)
        if remove_trace:
            A = remove_trace_part(A)

        c = np.zeros((3,), dtype=np.float64)
        for i, g in enumerate(self._G):
            c[i] = float(np.real(hs_inner(g, A)) / self._den[i])

        Msu2 = (c[0] * self.fuzzy.Lx + c[1] * self.fuzzy.Ly + c[2] * self.fuzzy.Lz).astype(np.complex128)
        Msu2 = hermitianize(Msu2)
        if remove_trace:
            Msu2 = remove_trace_part(Msu2)
        return Msu2.astype(np.complex128), c

    def project_to_basis(
        self,
        M: np.ndarray,
        basis: Sequence[np.ndarray],
        remove_trace: bool = True,
        *,
        mode: Optional[str] = None,
        orthonormalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic Hermitian projection into a provided matrix basis.

        If orthonormalize=False, coefficients are computed by diagonal HS
        denominators. If basis elements are not approximately orthogonal, set
        orthonormalize=True to solve the Gram system.
        """
        if len(basis) == 0:
            return np.zeros((self.k, self.k), dtype=np.complex128), np.zeros((0,), dtype=np.float64)

        A = self._projection_observable(_as_c128_matrix(M, shape=(self.k, self.k)), mode=mode)
        if remove_trace:
            A = remove_trace_part(A)

        B = [hermitianize(_as_c128_matrix(x, shape=(self.k, self.k))) for x in basis]
        if remove_trace:
            B = [remove_trace_part(x) for x in B]

        if orthonormalize:
            G = np.array([[np.real(hs_inner(B[i], B[j])) for j in range(len(B))] for i in range(len(B))], dtype=np.float64)
            y = np.array([np.real(hs_inner(B[i], A)) for i in range(len(B))], dtype=np.float64)
            G = G + np.eye(len(B), dtype=np.float64) * _EPS
            try:
                c = np.linalg.solve(G, y)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(G, y, rcond=None)[0]
        else:
            den = np.array([max(float(np.real(hs_inner(Bi, Bi))), _EPS) for Bi in B], dtype=np.float64)
            c = np.array([float(np.real(hs_inner(Bi, A)) / den[i]) for i, Bi in enumerate(B)], dtype=np.float64)

        P = np.zeros((self.k, self.k), dtype=np.complex128)
        for ci, Bi in zip(c, B):
            P = P + float(ci) * Bi
        return hermitianize(P).astype(np.complex128), c.astype(np.float64)

    # ------------------------------------------------------------------
    # Optional higher-harmonic basis support
    # ------------------------------------------------------------------

    def su2_basis(self) -> List[np.ndarray]:
        return [g.copy() for g in self._G]

    def quadratic_basis(self, *, include_linear: bool = True) -> List[np.ndarray]:
        """
        Return a deterministic Hermitian traceless basis containing optional
        linear su(2) generators plus symmetrized quadratic harmonics.
        """
        key = f"quad:{int(include_linear)}:{self.k}"
        if key in self._higher_basis_cache:
            return [x.copy() for x in self._higher_basis_cache[key]]

        L = [self.fuzzy.Lx, self.fuzzy.Ly, self.fuzzy.Lz]
        basis: List[np.ndarray] = []
        if include_linear:
            basis.extend(remove_trace_part(g) for g in L)
        for i in range(3):
            for j in range(i, 3):
                Q = 0.5 * (L[i] @ L[j] + L[j] @ L[i])
                basis.append(remove_trace_part(hermitianize(Q)))

        # Drop near-zero basis elements.
        clean = []
        for B in basis:
            if fro_norm(B) > 1e-10:
                clean.append(_normalize_matrix(B))
        self._higher_basis_cache[key] = [x.copy() for x in clean]
        return clean

    def project_to_quadratic_basis(
        self,
        M: np.ndarray,
        *,
        include_linear: bool = True,
        remove_trace: bool = True,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.project_to_basis(
            M,
            self.quadratic_basis(include_linear=include_linear),
            remove_trace=remove_trace,
            mode=mode,
            orthonormalize=True,
        )

    # ------------------------------------------------------------------
    # Diagnostics / serialization
    # ------------------------------------------------------------------

    def health_metrics(self) -> Dict[str, Any]:
        J = self.twist.J
        unitary_err = fro_norm(J.conj().T @ J - np.eye(self.k, dtype=np.complex128))
        gen = self.fuzzy.diagnostics()
        return {
            "kind": "NCFuzzyBackend",
            "fuzzy_l": int(self.fuzzy_l),
            "k": int(self.k),
            "num_nodes": int(self.num_nodes),
            "num_edges": int(len(self.oriented_edges())),
            "projection_mode": str(self.projection_mode),
            "twisted_reality_mix": float(self.twisted_reality_mix),
            "enforce_antisymmetric_pairs": bool(self.enforce_antisymmetric_pairs),
            "twist_unitarity_error": float(unitary_err),
            "fuzzy": gen,
            "is_stable": bool(np.isfinite(unitary_err) and unitary_err < 1e-8 and gen.get("commutator_rel_error_xy", 1.0) < 1e-8),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "NCFuzzyBackend",
            "config": {
                "fuzzy_l": int(self.fuzzy_l),
                "seed": int(self.seed),
                "projection_mode": str(self.projection_mode),
                "enforce_antisymmetric_pairs": bool(self.enforce_antisymmetric_pairs),
                "twisted_reality_mix": float(self.twisted_reality_mix),
                "finite_checks": bool(self.finite_checks),
            },
            "num_nodes": int(self.num_nodes),
            "num_edges": int(len(self.oriented_edges())),
            "k": int(self.k),
            "health": self.health_metrics(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjacency": {str(k): [int(x) for x in v] for k, v in self.adjacency.items()},
            "fuzzy_l": int(self.fuzzy_l),
            "seed": int(self.seed),
            "projection_mode": str(self.projection_mode),
            "enforce_antisymmetric_pairs": bool(self.enforce_antisymmetric_pairs),
            "twisted_reality_mix": float(self.twisted_reality_mix),
            "finite_checks": bool(self.finite_checks),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NCFuzzyBackend":
        adj_raw = dict(payload.get("adjacency", {}))
        adjacency = {int(k): [int(x) for x in v] for k, v in adj_raw.items()}
        return cls(
            adjacency=adjacency,
            fuzzy_l=int(payload.get("fuzzy_l", 2)),
            seed=int(payload.get("seed", 0)),
            projection_mode=str(payload.get("projection_mode", "auto")),
            enforce_antisymmetric_pairs=bool(payload.get("enforce_antisymmetric_pairs", True)),
            twisted_reality_mix=float(payload.get("twisted_reality_mix", 0.5)),
            finite_checks=bool(payload.get("finite_checks", True)),
        )


__all__ = [
    "FuzzySU2",
    "NCTwist",
    "NCFuzzyBackendConfig",
    "NCFuzzyBackend",
    "random_unitary",
]
