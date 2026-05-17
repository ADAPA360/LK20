#!/usr/bin/env python3
# flow.py
"""
Flow solvers and diagnostics for AtomTN.

This module is the transport layer for the AtomTN runtime family.  It provides a
small, deterministic graph-flow substrate that supports both scalar
(commutative) edge fields and matrix-valued noncommutative / fuzzy edge fields.

Public API preserved
--------------------
- GraphVectorField
- FlowDiagnostics
- GraphCalculus
- GeodesicFlowConfig
- GeodesicFlowSolver
- NCGeodesicFlowSolver
- FlowMonitor

Design guarantees
-----------------
- NumPy-only core with finite-value sanitation at module boundaries.
- Deterministic integration; no hidden RNG use in solvers.
- Antisymmetric oriented-edge repair helpers for reservoir use cases.
- Conservative damping, diffusion, and cubic regularization guards.
- Diagnostics expose scalar divergence, finite-difference drift, flow energy,
  alarm scores, and noncommutative κ projections when a fuzzy backend exists.
- Compatible with AtomTN runtime scripts that pass either geometry.GraphCalculus
  or flow.GraphCalculus into solvers/monitors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from math_utils import _assert, antihermitianize, fro_norm, hermitianize
from fuzzy_backend import NCFuzzyBackend


_EPS = 1e-12
Edge = Tuple[int, int]


# =============================================================================
# Small helpers
# =============================================================================


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {
                "real": np.nan_to_num(arr.real, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist(),
                "imag": np.nan_to_num(arr.imag, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist(),
            }
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _sanitize_real_array(x: Any, *, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        arr = np.zeros(shape or (0,), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    if shape is not None and arr.shape != shape:
        out = np.zeros(shape, dtype=np.float64)
        flat = arr.reshape(-1)
        n = min(flat.size, out.size)
        if n:
            out.reshape(-1)[:n] = flat[:n]
        arr = out
    return arr


def _sanitize_complex_matrix(x: Any, k: int, *, antihermitian: bool = False, hermitian: bool = False) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.complex128)
    except Exception:
        arr = np.zeros((int(k), int(k)), dtype=np.complex128)
    if arr.shape != (int(k), int(k)):
        out = np.zeros((int(k), int(k)), dtype=np.complex128)
        m0 = min(out.shape[0], arr.shape[0] if arr.ndim >= 1 else 0)
        m1 = min(out.shape[1], arr.shape[1] if arr.ndim >= 2 else 0)
        if m0 and m1:
            out[:m0, :m1] = arr[:m0, :m1]
        arr = out
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex128, copy=False)
    if antihermitian:
        arr = antihermitianize(arr).astype(np.complex128, copy=False)
    if hermitian:
        arr = hermitianize(arr).astype(np.complex128, copy=False)
    return arr


def _edge_key(edge: Any) -> Edge:
    try:
        u, v = edge
        return int(u), int(v)
    except Exception as exc:
        raise ValueError(f"Invalid edge key: {edge!r}") from exc


def _extract_scalar(value: Any) -> float:
    """Scalar proxy used for mixed diagnostics and flow-energy summaries."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _finite_float(value)
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        if np.iscomplexobj(arr):
            v = float(np.linalg.norm(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)))
        else:
            v = float(np.linalg.norm(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)))
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _as_adjacency(adjacency: Mapping[int, Iterable[int]], num_nodes: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {i: [] for i in range(int(num_nodes))}
    for u, nbrs in dict(adjacency).items():
        uu = int(u)
        if uu not in out:
            out[uu] = []
        vals: List[int] = []
        for v in nbrs:
            vv = int(v)
            if vv != uu and vv not in vals:
                vals.append(vv)
        out[uu] = vals
    return out


# =============================================================================
# Core containers
# =============================================================================

@dataclass
class GraphVectorField:
    """
    Edge field on a directed graph.

    edge_values[(u, v)] is either a scalar float or a complex matrix.  Most
    AtomTN code stores both orientations explicitly, with F[v,u] = -F[u,v].
    The solvers tolerate missing reverse orientations but diagnostics are most
    meaningful when both are present.
    """

    edge_values: Dict[Edge, Any]
    matrix_valued: bool = False

    def __post_init__(self) -> None:
        self.edge_values = {_edge_key(k): v for k, v in dict(self.edge_values or {}).items()}
        self.matrix_valued = bool(self.matrix_valued)

    def copy(self) -> "GraphVectorField":
        vals: Dict[Edge, Any] = {}
        for e, v in self.edge_values.items():
            vals[e] = np.asarray(v).copy() if self.matrix_valued else float(_extract_scalar(v))
        return GraphVectorField(vals, matrix_valued=self.matrix_valued)

    def oriented_edges(self, *, unique: bool = True) -> List[Edge]:
        if not unique:
            return sorted(self.edge_values.keys())
        out = []
        seen = set()
        for u, v in sorted(self.edge_values.keys()):
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b))
        return out

    def scalar_values(self, *, unique: bool = True) -> np.ndarray:
        edges = self.oriented_edges(unique=unique)
        vals = []
        for e in edges:
            vals.append(_extract_scalar(self.edge_values.get(e, self.edge_values.get((e[1], e[0]), 0.0))))
        return np.asarray(vals, dtype=np.float64)

    def energy(self, *, unique: bool = True) -> float:
        vals = self.scalar_values(unique=unique)
        if vals.size == 0:
            return 0.0
        e = float(np.mean(vals * vals))
        return e if np.isfinite(e) else 0.0

    def max_abs(self) -> float:
        if not self.edge_values:
            return 0.0
        return float(max(_extract_scalar(v) for v in self.edge_values.values()))

    def sanitized(self, *, matrix_dim: Optional[int] = None, antisymmetric: bool = False) -> "GraphVectorField":
        vals: Dict[Edge, Any] = {}
        if self.matrix_valued:
            k = int(matrix_dim or 1)
            if k <= 0:
                k = 1
            for e, v in self.edge_values.items():
                vals[e] = _sanitize_complex_matrix(v, k)
        else:
            for e, v in self.edge_values.items():
                vals[e] = _finite_float(v)
        out = GraphVectorField(vals, matrix_valued=self.matrix_valued)
        return out.antisymmetrized(matrix_dim=matrix_dim) if antisymmetric else out

    def antisymmetrized(self, *, matrix_dim: Optional[int] = None) -> "GraphVectorField":
        """Return a field with explicit reverse edges F[v,u] = -F[u,v]."""
        vals: Dict[Edge, Any] = {}
        seen = set()
        k = int(matrix_dim or 1)
        for u, v in sorted(self.edge_values.keys()):
            if (u, v) in seen or (v, u) in seen:
                continue
            a = self.edge_values.get((u, v), None)
            b = self.edge_values.get((v, u), None)
            if self.matrix_valued:
                A = _sanitize_complex_matrix(a if a is not None else 0.0, k)
                B = _sanitize_complex_matrix(b if b is not None else 0.0, k)
                M = 0.5 * (A - B)
                vals[(u, v)] = M
                vals[(v, u)] = -M
            else:
                av = _finite_float(a, 0.0) if a is not None else 0.0
                bv = _finite_float(b, 0.0) if b is not None else 0.0
                x = 0.5 * (av - bv)
                vals[(u, v)] = float(x)
                vals[(v, u)] = float(-x)
            seen.add((u, v))
            seen.add((v, u))
        return GraphVectorField(vals, matrix_valued=self.matrix_valued)

    def to_dict(self, *, max_edges: Optional[int] = None) -> Dict[str, Any]:
        items = []
        for i, ((u, v), val) in enumerate(sorted(self.edge_values.items())):
            if max_edges is not None and i >= int(max_edges):
                break
            items.append({"u": int(u), "v": int(v), "value": _json_safe(val)})
        return {
            "matrix_valued": bool(self.matrix_valued),
            "edge_count": int(len(self.edge_values)),
            "unique_edge_count": int(len(self.oriented_edges(unique=True))),
            "energy": float(self.energy()),
            "max_abs": float(self.max_abs()),
            "edges": items,
        }


@dataclass
class FlowDiagnostics:
    """
    Monitoring values computed for a flow step.

    Commutative fields:
      - divX_scalar: |div(X)| per node.
      - DdivDt_scalar: finite-difference derivative of |div(X)|.
      - alarm_score: norm-style scalar alarm.

    Noncommutative fields:
      - kappa_node_mats_raw: κ_raw(u) = 1/2 div(X)(u).
      - kappa_node_mats_su2: backend Π_su2(κ_raw(u)).
      - kappa_su2_coeffs: coefficients in Lx,Ly,Lz basis, shape (N,3).
    """

    divX_scalar: np.ndarray
    DdivDt_scalar: Optional[np.ndarray] = None
    alarm_score: Optional[float] = None
    flow_energy: float = 0.0
    max_edge_norm: float = 0.0
    is_matrix_valued: bool = False
    stable: bool = True
    notes: List[str] = field(default_factory=list)

    kappa_node_mats_raw: Optional[Dict[int, np.ndarray]] = None
    kappa_node_mats_su2: Optional[Dict[int, np.ndarray]] = None
    kappa_su2_coeffs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.divX_scalar = _sanitize_real_array(self.divX_scalar).reshape(-1)
        if self.DdivDt_scalar is not None:
            self.DdivDt_scalar = _sanitize_real_array(self.DdivDt_scalar).reshape(-1)
        self.alarm_score = _finite_float(self.alarm_score, float(np.linalg.norm(self.divX_scalar)))
        self.flow_energy = _finite_float(self.flow_energy)
        self.max_edge_norm = _finite_float(self.max_edge_norm)
        self.stable = bool(self.stable and np.all(np.isfinite(self.divX_scalar)) and np.isfinite(self.alarm_score))
        if self.kappa_su2_coeffs is not None:
            self.kappa_su2_coeffs = _sanitize_real_array(self.kappa_su2_coeffs)

    def to_dict(self, *, include_matrices: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "divX_scalar": self.divX_scalar.astype(float).tolist(),
            "DdivDt_scalar": None if self.DdivDt_scalar is None else self.DdivDt_scalar.astype(float).tolist(),
            "alarm_score": float(self.alarm_score or 0.0),
            "flow_energy": float(self.flow_energy),
            "max_edge_norm": float(self.max_edge_norm),
            "is_matrix_valued": bool(self.is_matrix_valued),
            "stable": bool(self.stable),
            "notes": list(self.notes),
            "kappa_su2_coeffs": None if self.kappa_su2_coeffs is None else self.kappa_su2_coeffs.astype(float).tolist(),
        }
        if include_matrices:
            out["kappa_node_mats_raw"] = _json_safe(self.kappa_node_mats_raw)
            out["kappa_node_mats_su2"] = _json_safe(self.kappa_node_mats_su2)
        return out


# =============================================================================
# Commutative graph calculus
# =============================================================================

@dataclass
class GraphCalculus:
    adjacency: Dict[int, List[int]]
    num_nodes: int = 64

    def __post_init__(self) -> None:
        self.num_nodes = int(max(1, self.num_nodes))
        self.adjacency = _as_adjacency(self.adjacency, self.num_nodes)

    def oriented_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        seen = set()
        for u, nbrs in self.adjacency.items():
            for v in nbrs:
                uu, vv = int(u), int(v)
                a, b = (uu, vv) if uu < vv else (vv, uu)
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                edges.append((a, b))
        edges.sort()
        return edges

    def directed_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for u, nbrs in self.adjacency.items():
            for v in nbrs:
                if int(u) != int(v):
                    edges.append((int(u), int(v)))
        return sorted(edges)

    def div(self, F: Mapping[Edge, Any]) -> np.ndarray:
        """
        Divergence div(X)(u)=Σ_v X_{u->v}.  Missing edges contribute zero.
        """
        div = np.zeros((int(self.num_nodes),), dtype=np.float64)
        for (u, _v), val in dict(F).items():
            uu = int(u)
            if 0 <= uu < self.num_nodes:
                div[uu] += _finite_float(val)
        return np.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

    def laplacian(self, a: Any) -> np.ndarray:
        arr = _sanitize_real_array(a, shape=(int(self.num_nodes),))
        out = np.zeros_like(arr, dtype=np.float64)
        for u, nbrs in self.adjacency.items():
            uu = int(u)
            if 0 <= uu < self.num_nodes:
                out[uu] = sum(float(arr[int(v)] - arr[uu]) for v in nbrs if 0 <= int(v) < self.num_nodes)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient_from_node_scalar(self, phi: Any) -> Dict[Edge, float]:
        p = _sanitize_real_array(phi, shape=(int(self.num_nodes),))
        out: Dict[Edge, float] = {}
        for u, v in self.oriented_edges():
            val = float(p[v] - p[u])
            out[(u, v)] = val
            out[(v, u)] = -val
        return out

    def validate_field_edges(self, X: GraphVectorField) -> bool:
        if X.matrix_valued:
            return True
        known = set(self.directed_edges())
        return all(e in known for e in X.edge_values.keys())


# =============================================================================
# Flow configuration
# =============================================================================

@dataclass
class GeodesicFlowConfig:
    dt: float = 2e-2
    steps: int = 150
    damping: float = 0.03
    diffusion: float = 0.02
    twisted_reality: bool = True  # NC only

    # Production safety knobs.  Defaults preserve old behavior while bounding
    # pathological runtime states.
    beta: float = 0.05
    max_edge_norm: float = 1.0e6
    max_divergence_norm: float = 1.0e8
    symmetrize_edges: bool = True
    check_finite: bool = True

    def normalized(self) -> "GeodesicFlowConfig":
        dt = _finite_float(self.dt, 2e-2)
        steps = int(max(0, int(self.steps)))
        damping = max(0.0, _finite_float(self.damping, 0.03))
        diffusion = max(0.0, _finite_float(self.diffusion, 0.02))
        beta = max(0.0, _finite_float(self.beta, 0.05))
        max_edge_norm = max(1.0, _finite_float(self.max_edge_norm, 1.0e6))
        max_divergence_norm = max(1.0, _finite_float(self.max_divergence_norm, 1.0e8))
        return GeodesicFlowConfig(
            dt=dt,
            steps=steps,
            damping=damping,
            diffusion=diffusion,
            twisted_reality=bool(self.twisted_reality),
            beta=beta,
            max_edge_norm=max_edge_norm,
            max_divergence_norm=max_divergence_norm,
            symmetrize_edges=bool(self.symmetrize_edges),
            check_finite=bool(self.check_finite),
        )


# =============================================================================
# Scalar flow solver
# =============================================================================

class GeodesicFlowSolver:
    """
    Commutative scalar flow solver.

    The solver is a conservative hydrodynamic proxy:
      κ = 1/2 div(X)
      κ <- κ + diffusion * Δκ
      xdot = -(κ(v)-κ(u) + beta*x*|x|) - damping*x

    It is not a geometric proof object; it is a stable graph-dynamics substrate
    used by higher-level AtomTN components.
    """

    def __init__(self, calc: GraphCalculus):
        self.calc = calc

    def step(self, X: GraphVectorField, cfg: GeodesicFlowConfig) -> GraphVectorField:
        cfg = cfg.normalized()
        _assert(not X.matrix_valued, "GeodesicFlowSolver expects matrix_valued=False")

        X0 = X.sanitized(antisymmetric=cfg.symmetrize_edges)
        kappa = 0.5 * self.calc.div(X0.edge_values)
        if cfg.diffusion > 0:
            kappa = kappa + float(cfg.diffusion) * self.calc.laplacian(kappa)
        kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

        if cfg.check_finite:
            div_norm = float(np.linalg.norm(kappa))
            if not np.isfinite(div_norm) or div_norm > cfg.max_divergence_norm:
                raise FloatingPointError(f"Scalar flow divergence out of bounds: {div_norm}")

        new_edge: Dict[Edge, float] = {}
        for (u, v), raw in X0.edge_values.items():
            val = _finite_float(raw)
            uu, vv = int(u), int(v)
            if not (0 <= uu < self.calc.num_nodes and 0 <= vv < self.calc.num_nodes):
                continue
            transport = float((kappa[vv] - kappa[uu]) + cfg.beta * val * abs(val))
            xdot = -transport - float(cfg.damping) * val
            y = val + float(cfg.dt) * xdot
            if abs(y) > cfg.max_edge_norm:
                y = float(np.sign(y) * cfg.max_edge_norm)
            new_edge[(uu, vv)] = _finite_float(y)

        out = GraphVectorField(new_edge, matrix_valued=False)
        return out.antisymmetrized() if cfg.symmetrize_edges else out

    def integrate(self, X0: GraphVectorField, cfg: GeodesicFlowConfig) -> List[GraphVectorField]:
        cfg = cfg.normalized()
        X = X0.sanitized(antisymmetric=cfg.symmetrize_edges)
        out = [X]
        for _ in range(int(cfg.steps)):
            X = self.step(X, cfg)
            out.append(X)
        return out


# =============================================================================
# Noncommutative flow solver
# =============================================================================

class NCGeodesicFlowSolver:
    """
    Matrix-valued noncommutative flow solver.

    - κ_raw(u) = 1/2 div(X)(u)
    - xdot = -(κ(v)-κ(u)) - damping*X - beta*X X† X
    - optional twisted-reality projection through NCFuzzyBackend.
    """

    def __init__(self, backend: NCFuzzyBackend, num_nodes: int = 64):
        self.backend = backend
        self.num_nodes = int(max(1, num_nodes))

    @property
    def k(self) -> int:
        return int(getattr(self.backend, "k", getattr(getattr(self.backend, "fuzzy", None), "k", 1)))

    def _sanitize_nc_field(self, X: GraphVectorField, cfg: GeodesicFlowConfig) -> GraphVectorField:
        vals: Dict[Edge, np.ndarray] = {}
        k = self.k
        for e, M in X.edge_values.items():
            vals[e] = _sanitize_complex_matrix(M, k)
        out = GraphVectorField(vals, matrix_valued=True)
        return out.antisymmetrized(matrix_dim=k) if cfg.symmetrize_edges else out

    def step(self, X: GraphVectorField, cfg: GeodesicFlowConfig) -> GraphVectorField:
        cfg = cfg.normalized()
        _assert(X.matrix_valued, "NCGeodesicFlowSolver expects matrix_valued=True")

        X0 = self._sanitize_nc_field(X, cfg)
        kappa = self.backend.kappa_from_edge(X0.edge_values, N=self.num_nodes)  # type: ignore[arg-type]
        k = self.k

        new_edge: Dict[Edge, np.ndarray] = {}
        for (u, v), M_raw in X0.edge_values.items():
            uu, vv = int(u), int(v)
            M = _sanitize_complex_matrix(M_raw, k)
            Ku = _sanitize_complex_matrix(kappa.get(uu, np.zeros((k, k))), k)
            Kv = _sanitize_complex_matrix(kappa.get(vv, np.zeros((k, k))), k)

            cubic = M @ M.conj().T @ M
            xdot = -(Kv - Ku) - float(cfg.damping) * M - float(cfg.beta) * cubic
            Y = M + float(cfg.dt) * xdot
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex128, copy=False)

            yn = fro_norm(Y)
            if yn > cfg.max_edge_norm and yn > _EPS:
                Y = (cfg.max_edge_norm / yn) * Y
            new_edge[(uu, vv)] = Y.astype(np.complex128, copy=False)

        if cfg.symmetrize_edges:
            new_edge = GraphVectorField(new_edge, matrix_valued=True).antisymmetrized(matrix_dim=k).edge_values  # type: ignore[assignment]

        if cfg.twisted_reality:
            new_edge = self.backend.project_twisted_reality_edge(new_edge)  # type: ignore[arg-type]
            if cfg.symmetrize_edges:
                new_edge = GraphVectorField(new_edge, matrix_valued=True).antisymmetrized(matrix_dim=k).edge_values  # type: ignore[assignment]

        if cfg.check_finite:
            max_norm = max((fro_norm(v) for v in new_edge.values()), default=0.0)
            if not np.isfinite(max_norm) or max_norm > cfg.max_edge_norm * 1.01:
                raise FloatingPointError(f"NC flow edge norm out of bounds: {max_norm}")

        return GraphVectorField(new_edge, matrix_valued=True)

    def integrate(self, X0: GraphVectorField, cfg: GeodesicFlowConfig) -> List[GraphVectorField]:
        cfg = cfg.normalized()
        X = self._sanitize_nc_field(X0, cfg)
        out = [X]
        for _ in range(int(cfg.steps)):
            X = self.step(X, cfg)
            out.append(X)
        return out


# =============================================================================
# Flow monitor
# =============================================================================

class FlowMonitor:
    """
    Produces FlowDiagnostics for scalar or noncommutative flows.

    For commutative flow, pass calc=GraphCalculus.  For noncommutative flow,
    pass backend=NCFuzzyBackend.  If both are provided, the field's
    matrix_valued flag selects the branch.
    """

    def __init__(
        self,
        calc: Optional[Any] = None,
        backend: Optional[NCFuzzyBackend] = None,
        num_nodes: int = 64,
        *,
        alarm_scale: float = 1.0,
    ):
        self.calc = calc
        self.backend = backend
        self.num_nodes = int(max(1, num_nodes))
        self.alarm_scale = max(_EPS, _finite_float(alarm_scale, 1.0))

    def _comm_div(self, X: GraphVectorField) -> np.ndarray:
        _assert(self.calc is not None, "FlowMonitor: commutative diagnostics requires GraphCalculus")
        if hasattr(self.calc, "div"):
            return _sanitize_real_array(self.calc.div(X.edge_values), shape=(self.num_nodes,))
        raise ValueError("FlowMonitor calc object must provide div(...)")

    def diagnostics(self, X_prev: Optional[GraphVectorField], X: GraphVectorField, dt: float) -> FlowDiagnostics:
        dtv = max(_EPS, abs(_finite_float(dt, 1.0)))
        notes: List[str] = []

        if not X.matrix_valued:
            Xs = X.sanitized(antisymmetric=False)
            divX = self._comm_div(Xs)
            div_abs = np.abs(divX)

            Ddiv = None
            if X_prev is not None:
                try:
                    div_prev = self._comm_div(X_prev.sanitized(antisymmetric=False))
                    Ddiv = (np.abs(divX) - np.abs(div_prev)) / dtv
                except Exception as exc:
                    notes.append(f"Ddiv_failed:{exc!r}")
                    Ddiv = None

            flow_energy = Xs.energy()
            max_edge = Xs.max_abs()
            alarm = float(np.linalg.norm(div_abs, ord=2) / self.alarm_scale)
            stable = bool(np.isfinite(alarm) and np.isfinite(flow_energy) and np.isfinite(max_edge))
            return FlowDiagnostics(
                divX_scalar=div_abs,
                DdivDt_scalar=Ddiv,
                alarm_score=alarm,
                flow_energy=flow_energy,
                max_edge_norm=max_edge,
                is_matrix_valued=False,
                stable=stable,
                notes=notes,
            )

        # Noncommutative branch.
        _assert(self.backend is not None, "FlowMonitor: NC diagnostics requires NCFuzzyBackend")
        k = int(getattr(self.backend, "k", 1))
        Xs = X.sanitized(matrix_dim=k, antisymmetric=False)

        div_node = self.backend.div_edge_field(Xs.edge_values, N=self.num_nodes)  # type: ignore[arg-type]
        kappa_raw = {int(u): 0.5 * _sanitize_complex_matrix(M, k) for u, M in div_node.items()}

        kappa_su2: Dict[int, np.ndarray] = {}
        coeffs = np.zeros((self.num_nodes, 3), dtype=np.float64)
        for u in range(self.num_nodes):
            try:
                Msu2, c = self.backend.project_to_su2(kappa_raw[u], remove_trace=True)
                kappa_su2[u] = np.asarray(Msu2, dtype=np.complex128)
                coeffs[u, :] = _sanitize_real_array(c, shape=(3,))
            except Exception as exc:
                notes.append(f"su2_projection_failed_node_{u}:{exc!r}")
                kappa_su2[u] = np.zeros((k, k), dtype=np.complex128)

        div_norm = np.zeros((self.num_nodes,), dtype=np.float64)
        for u in range(self.num_nodes):
            div_norm[u] = fro_norm(_sanitize_complex_matrix(div_node.get(u, np.zeros((k, k))), k))

        Ddiv = None
        if X_prev is not None:
            try:
                Xp = X_prev.sanitized(matrix_dim=k, antisymmetric=False)
                div_prev = self.backend.div_edge_field(Xp.edge_values, N=self.num_nodes)  # type: ignore[arg-type]
                prev_norm = np.zeros((self.num_nodes,), dtype=np.float64)
                for u in range(self.num_nodes):
                    prev_norm[u] = fro_norm(_sanitize_complex_matrix(div_prev.get(u, np.zeros((k, k))), k))
                Ddiv = (div_norm - prev_norm) / dtv
            except Exception as exc:
                notes.append(f"Ddiv_nc_failed:{exc!r}")
                Ddiv = None

        vals = [fro_norm(_sanitize_complex_matrix(v, k)) for v in Xs.edge_values.values()]
        flow_energy = float(np.mean(np.asarray(vals, dtype=np.float64) ** 2)) if vals else 0.0
        max_edge = float(np.max(vals)) if vals else 0.0
        alarm = float(np.linalg.norm(div_norm, ord=2) / self.alarm_scale)
        stable = bool(np.isfinite(alarm) and np.isfinite(flow_energy) and np.isfinite(max_edge))

        return FlowDiagnostics(
            divX_scalar=div_norm,
            DdivDt_scalar=Ddiv,
            alarm_score=alarm,
            flow_energy=flow_energy,
            max_edge_norm=max_edge,
            is_matrix_valued=True,
            stable=stable,
            notes=notes,
            kappa_node_mats_raw=kappa_raw,
            kappa_node_mats_su2=kappa_su2,
            kappa_su2_coeffs=coeffs,
        )

    def diagnostics_sequence(self, traj: Sequence[GraphVectorField], dt: float) -> List[FlowDiagnostics]:
        out: List[FlowDiagnostics] = []
        prev: Optional[GraphVectorField] = None
        for X in traj:
            out.append(self.diagnostics(prev, X, dt))
            prev = X
        return out


__all__ = [
    "GraphVectorField",
    "FlowDiagnostics",
    "GraphCalculus",
    "GeodesicFlowConfig",
    "GeodesicFlowSolver",
    "NCGeodesicFlowSolver",
    "FlowMonitor",
]
