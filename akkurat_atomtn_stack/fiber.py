#!/usr/bin/env python3
# fiber.py
"""
Local fiber scheduling and operator-basis construction for AtomTN.

Production role
---------------
This module owns the local Hilbert-space dimension policy used by AtomTN TTN
states.  It is deliberately independent of flow integration and Hamiltonian
application.  Its responsibilities are:

- compute stable per-leaf physical dimensions from curvature scores and optional
  vibration context;
- enforce noncommutative projection constraints, especially d <= fuzzy.k;
- build deterministic local operator bases for commutative and projected fuzzy
  leaves;
- provide snapshot/health surfaces for Akkurat runtime diagnostics.

Public API retained from previous AtomTN scripts:

    cfg = LocalFiberConfig(d_uniform=16, d_min=8, d_max=32)
    fiber = LocalFiberBuilder(cfg, projection=atom.projection)
    d_leaf = fiber.leaf_dims(scores_leaf, vib)
    phys = fiber.make_phys_dim_map(tree, d_leaf)
    ops = fiber.base_operator_basis(d)

The module also re-exports AdinkraConstraint for older imports:

    from fiber import LocalFiberConfig, LocalFiberBuilder, AdinkraConstraint

Dependencies
------------
- numpy
- math_utils.py
- schedules.py
- constraints.py, projection.py, vibration.py, geometry.py when available
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import math
import numpy as np

try:
    from math_utils import _assert, clip_int, fro_norm, hermitianize, sanitize_real_array, safe_norm
except Exception:  # pragma: no cover - fallback for isolated linting
    def _assert(cond: bool, msg: str) -> None:
        if not bool(cond):
            raise ValueError(str(msg))

    def clip_int(x: int, lo: int, hi: int) -> int:
        lo_i, hi_i = int(lo), int(hi)
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        return int(np.clip(int(x), lo_i, hi_i))

    def fro_norm(A: Any) -> float:
        try:
            return float(np.linalg.norm(np.asarray(A).reshape(-1)))
        except Exception:
            return 0.0

    def hermitianize(A: Any) -> np.ndarray:
        X = np.asarray(A, dtype=np.complex128)
        return (X + X.conj().T) / 2.0

    def sanitize_real_array(A: Any, *, fill: float = 0.0, dtype: np.dtype = np.float64) -> np.ndarray:
        out = np.asarray(A, dtype=np.dtype(dtype)).copy()
        if out.size:
            out[~np.isfinite(out)] = np.dtype(dtype).type(fill)
        return out

    def safe_norm(x: Any, ord: Optional[int | float | str] = None) -> float:
        try:
            arr = np.asarray(x)
            if arr.size == 0 or not np.isfinite(arr).all():
                return 0.0
            val = float(np.linalg.norm(arr, ord=ord))
            return val if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

try:
    from schedules import ScoreNormConfig, adaptive_fiber_dims, normalize_scores_with_config
except Exception:  # pragma: no cover
    @dataclass(frozen=True)
    class ScoreNormConfig:  # type: ignore[no-redef]
        method: str = "minmax"
        eps: float = 1e-12
        clip_percentile: Optional[float] = None

    def normalize_scores_with_config(scores: Any, cfg: Optional[ScoreNormConfig] = None) -> np.ndarray:  # type: ignore[no-redef]
        arr = sanitize_real_array(scores, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return arr.astype(np.float64)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo <= 1e-12:
            return np.zeros_like(arr, dtype=np.float64)
        return ((arr - lo) / (hi - lo)).astype(np.float64)

    def adaptive_fiber_dims(  # type: ignore[no-redef]
        scores_leaf: Any,
        d_uniform: int,
        d_min: int,
        d_max: int,
        strength: float = 1.0,
        **_: Any,
    ) -> np.ndarray:
        ns = normalize_scores_with_config(scores_leaf)
        vals = np.rint(float(d_uniform) * (1.0 + float(strength) * ns)).astype(int)
        return np.clip(vals, int(d_min), int(d_max)).astype(int)

try:
    from constraints import AdinkraConstraint
except Exception:  # pragma: no cover
    @dataclass
    class AdinkraConstraint:  # type: ignore[no-redef]
        """Fallback seeded operator-basis generator."""

        seed: int = 0

        def operator_basis(self, d: int) -> Dict[str, np.ndarray]:
            d = int(max(1, d))
            rng = np.random.default_rng(int(self.seed) + 31 * d)
            ops: Dict[str, np.ndarray] = {"I": np.eye(d, dtype=np.complex128)}
            if d >= 2:
                X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
                Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
                Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
                for name, P in (("X", X2), ("Y", Y2), ("Z", Z2)):
                    A = np.eye(d, dtype=np.complex128)
                    A[:2, :2] = P
                    ops[name] = A
            else:
                ops.update({"X": np.eye(d, dtype=np.complex128), "Y": np.zeros((d, d), dtype=np.complex128), "Z": np.eye(d, dtype=np.complex128)})
            G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            G = hermitianize(G)
            ops["G"] = (G / max(fro_norm(G), 1e-12)).astype(np.complex128)
            return ops

try:  # imported only for type names / isinstance-safe annotations
    from projection import ProjectionLayer  # noqa: F401
except Exception:  # pragma: no cover
    ProjectionLayer = Any  # type: ignore

try:
    from vibration import VibrationModel  # noqa: F401
except Exception:  # pragma: no cover
    VibrationModel = Any  # type: ignore

try:
    from geometry import Tree  # noqa: F401
except Exception:  # pragma: no cover
    Tree = Any  # type: ignore


_EPS = 1e-12
_DEFAULT_LEAF_COUNT = 64


# =============================================================================
# Serialization / finite helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {"real": arr.real.astype(float).tolist(), "imag": arr.imag.astype(float).tolist()}
        return sanitize_real_array(arr, dtype=np.float64).astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _as_dim_vector(x: Any, *, leaf_count: int, fill: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64).reshape(-1) if x is not None else np.zeros((0,), dtype=np.int64)
    if arr.size == leaf_count:
        out = arr.copy()
    elif arr.size == 0:
        out = np.full((leaf_count,), int(fill), dtype=np.int64)
    elif arr.size < leaf_count:
        out = np.full((leaf_count,), int(fill), dtype=np.int64)
        out[: arr.size] = arr
    else:
        out = arr[:leaf_count].copy()
    return out.astype(np.int64, copy=False)


def _as_score_vector(scores: Any, *, leaf_count: int, strict: bool = True) -> np.ndarray:
    arr = sanitize_real_array(scores, dtype=np.float64).reshape(-1)
    if arr.size == leaf_count:
        return arr.astype(np.float64, copy=False)
    if strict:
        raise ValueError(f"scores_leaf must have shape ({leaf_count},), got ({arr.size},)")
    out = np.zeros((leaf_count,), dtype=np.float64)
    n = min(arr.size, leaf_count)
    if n:
        out[:n] = arr[:n]
    return out


def _projection_k(projection: Any) -> Optional[int]:
    try:
        fuzzy = getattr(projection, "fuzzy", None)
        if fuzzy is not None and hasattr(fuzzy, "k"):
            k = int(getattr(fuzzy, "k"))
            return k if k > 0 else None
    except Exception:
        pass
    return None


def _coerce_dim_bounds(d_min: int, d_uniform: int, d_max: int, *, cap: Optional[int] = None) -> Tuple[int, int, int]:
    cap_i = int(cap) if cap is not None else None
    mn = max(1, int(d_min))
    mx = max(1, int(d_max))
    if cap_i is not None:
        cap_i = max(1, cap_i)
        mn = min(mn, cap_i)
        mx = min(mx, cap_i)
    if mx < mn:
        mx = mn
    du = clip_int(int(d_uniform), mn, mx)
    return int(mn), int(du), int(mx)


def _round_to_multiple(vals: np.ndarray, multiple: int) -> np.ndarray:
    q = int(max(1, multiple))
    if q == 1:
        return np.rint(vals)
    return np.rint(vals / float(q)) * float(q)


# =============================================================================
# Operator basis helpers
# =============================================================================

def _pauli_embedded(d: int) -> Dict[str, np.ndarray]:
    d = int(max(1, d))
    I = np.eye(d, dtype=np.complex128)
    X = np.zeros((d, d), dtype=np.complex128)
    Y = np.zeros((d, d), dtype=np.complex128)
    Z = np.eye(d, dtype=np.complex128)

    if d == 1:
        X[0, 0] = 1.0
        Y[0, 0] = 0.0
        Z[0, 0] = 1.0
        return {"I": I, "X": X, "Y": Y, "Z": Z}

    X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X[:2, :2] = X2
    Y[:2, :2] = Y2
    Z[:2, :2] = Z2
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def _spin_generators(d: int) -> Dict[str, np.ndarray]:
    """Return Hermitian spin generators Lx/Ly/Lz for local dimension d."""
    d = int(max(1, d))
    if d == 1:
        z = np.zeros((1, 1), dtype=np.complex128)
        return {"Lx": z.copy(), "Ly": z.copy(), "Lz": z.copy()}

    # Spin j=(d-1)/2, m=j,j-1,...,-j.
    j = (d - 1) / 2.0
    m = np.arange(j, -j - 1.0, -1.0, dtype=np.float64)
    Lz = np.diag(m).astype(np.complex128)
    Lp = np.zeros((d, d), dtype=np.complex128)
    Lm = np.zeros((d, d), dtype=np.complex128)
    for i in range(d - 1):
        mi = m[i]
        coeff = math.sqrt(max(j * (j + 1.0) - mi * (mi - 1.0), 0.0))
        Lm[i + 1, i] = coeff
        Lp[i, i + 1] = coeff
    Lx = (Lp + Lm) / 2.0
    Ly = (Lp - Lm) / (2.0j)
    return {"Lx": hermitianize(Lx).astype(np.complex128), "Ly": hermitianize(Ly).astype(np.complex128), "Lz": hermitianize(Lz).astype(np.complex128)}


def _normalize_operator(A: np.ndarray, *, target_norm: Optional[float] = None, preserve_identity: bool = True) -> np.ndarray:
    X = np.asarray(A, dtype=np.complex128)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"operator must be square, got {X.shape}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex128, copy=False)
    if target_norm is None or target_norm <= 0:
        return X
    if preserve_identity and np.allclose(X, np.eye(X.shape[0], dtype=np.complex128), atol=1e-12, rtol=1e-12):
        return X
    n = fro_norm(X)
    if n <= _EPS:
        return X
    return (X * (float(target_norm) / n)).astype(np.complex128, copy=False)


def _operator_stats(ops: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    norms = {str(k): float(fro_norm(v)) for k, v in ops.items()}
    finite = all(bool(np.isfinite(np.asarray(v)).all()) for v in ops.values())
    herm_err = {}
    for k, v in ops.items():
        A = np.asarray(v)
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            herm_err[str(k)] = float(fro_norm(A - A.conj().T))
    return {"count": len(ops), "finite": bool(finite), "norms": norms, "hermitian_error": herm_err}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LocalFiberConfig:
    """Configuration for per-leaf local physical dimensions.

    Original fields are preserved.  Added fields are conservative and disabled
    unless explicitly set by callers.
    """

    d_uniform: int = 16
    d_min: int = 8
    d_max: int = 32
    adaptive_strength: float = 1.0
    vib_influence: float = 0.25
    seed: int = 0

    # Production additions.
    leaf_count: int = _DEFAULT_LEAF_COUNT
    strict_shapes: bool = True
    score_norm_method: str = "minmax"       # minmax | zscore | robust | sigmoid | none
    score_clip_percentile: Optional[float] = 99.0
    quantize_to: int = 1
    max_delta_per_update: Optional[int] = None
    smoothing_beta: float = 0.0             # 0 = disabled; else EMA dims before rounding
    min_operator_norm: float = 1e-12
    normalize_nonidentity_ops: bool = False
    operator_norm_target: float = 1.0
    include_spin_generators: bool = True
    include_pauli_fallback: bool = True
    cache_operator_basis: bool = True

    def normalized(self, *, projection: Any = None) -> "LocalFiberConfig":
        cap = _projection_k(projection)
        mn, du, mx = _coerce_dim_bounds(self.d_min, self.d_uniform, self.d_max, cap=cap)
        return LocalFiberConfig(
            d_uniform=du,
            d_min=mn,
            d_max=mx,
            adaptive_strength=float(max(0.0, self.adaptive_strength)),
            vib_influence=float(max(0.0, self.vib_influence)),
            seed=int(self.seed),
            leaf_count=int(max(1, self.leaf_count)),
            strict_shapes=bool(self.strict_shapes),
            score_norm_method=str(self.score_norm_method or "minmax"),
            score_clip_percentile=(None if self.score_clip_percentile is None else float(self.score_clip_percentile)),
            quantize_to=int(max(1, self.quantize_to)),
            max_delta_per_update=(None if self.max_delta_per_update is None else int(max(1, self.max_delta_per_update))),
            smoothing_beta=float(np.clip(self.smoothing_beta, 0.0, 0.999)),
            min_operator_norm=float(max(0.0, self.min_operator_norm)),
            normalize_nonidentity_ops=bool(self.normalize_nonidentity_ops),
            operator_norm_target=float(max(0.0, self.operator_norm_target)),
            include_spin_generators=bool(self.include_spin_generators),
            include_pauli_fallback=bool(self.include_pauli_fallback),
            cache_operator_basis=bool(self.cache_operator_basis),
        )


# =============================================================================
# Local fiber builder
# =============================================================================

class LocalFiberBuilder:
    """Compute local dimensions and local operator bases for AtomTN leaves."""

    def __init__(
        self,
        cfg: LocalFiberConfig,
        adinkra: Optional[AdinkraConstraint] = None,
        projection: Optional[Any] = None,
        include_pauli_fallback: Optional[bool] = None,
    ):
        if not isinstance(cfg, LocalFiberConfig):
            # Allows dict-like config payloads in checkpoints.
            if isinstance(cfg, Mapping):
                cfg = LocalFiberConfig(**dict(cfg))
            else:
                raise TypeError("cfg must be LocalFiberConfig or mapping")

        self.projection = projection
        self.cfg = cfg.normalized(projection=projection)
        if include_pauli_fallback is not None:
            self.cfg.include_pauli_fallback = bool(include_pauli_fallback)

        self.adinkra = adinkra if adinkra is not None else AdinkraConstraint(seed=int(self.cfg.seed))
        self.include_pauli_fallback = bool(self.cfg.include_pauli_fallback)

        self._basis_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._last_dims: Optional[np.ndarray] = None
        self._ema_dims: Optional[np.ndarray] = None
        self._last_scores: Optional[np.ndarray] = None
        self._last_vibration_factor: float = 0.0

        self._validate_projection_constraint()

    # ------------------------------------------------------------------
    # Validation / configuration
    # ------------------------------------------------------------------
    @property
    def projection_dim_cap(self) -> Optional[int]:
        return _projection_k(self.projection)

    def _validate_projection_constraint(self) -> None:
        cap = self.projection_dim_cap
        if cap is None:
            return
        if self.cfg.d_max > cap or self.cfg.d_uniform > cap or self.cfg.d_min > cap:
            raise ValueError(f"LocalFiberConfig violates projection constraint d<=k: bounds={self.cfg}, k={cap}")

    def _score_norm_cfg(self) -> ScoreNormConfig:
        try:
            return ScoreNormConfig(method=str(self.cfg.score_norm_method), clip_percentile=self.cfg.score_clip_percentile)
        except TypeError:
            return ScoreNormConfig(method=str(self.cfg.score_norm_method))

    def _apply_dimension_rate_limits(self, dims: np.ndarray) -> np.ndarray:
        out = np.asarray(dims, dtype=np.float64).reshape(-1)

        if self.cfg.smoothing_beta > 0.0:
            beta = float(self.cfg.smoothing_beta)
            if self._ema_dims is None or self._ema_dims.shape != out.shape:
                self._ema_dims = out.copy()
            else:
                self._ema_dims = (1.0 - beta) * self._ema_dims + beta * out
            out = self._ema_dims.copy()

        out = _round_to_multiple(out, self.cfg.quantize_to).astype(np.int64, copy=False)
        out = np.clip(out, self.cfg.d_min, self.cfg.d_max).astype(np.int64, copy=False)

        cap = self.projection_dim_cap
        if cap is not None:
            out = np.minimum(out, int(cap)).astype(np.int64, copy=False)

        if self.cfg.max_delta_per_update is not None and self._last_dims is not None and self._last_dims.shape == out.shape:
            md = int(self.cfg.max_delta_per_update)
            lo = self._last_dims.astype(np.int64) - md
            hi = self._last_dims.astype(np.int64) + md
            out = np.minimum(np.maximum(out, lo), hi).astype(np.int64, copy=False)
            out = np.clip(out, self.cfg.d_min, self.cfg.d_max).astype(np.int64, copy=False)
            if cap is not None:
                out = np.minimum(out, int(cap)).astype(np.int64, copy=False)

        return out.astype(np.int64, copy=False)

    @staticmethod
    def _vibration_factor(vib: Optional[Any]) -> float:
        if vib is None:
            return 0.0
        try:
            c = np.asarray(getattr(vib, "couplings", []), dtype=np.float64).reshape(-1)
            if c.size == 0:
                return 0.0
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            # Use a bounded combination of mean and RMS coupling.  The value is
            # clipped to [0,1] so vib_influence remains interpretable.
            mean_abs = float(np.mean(np.abs(c)))
            rms = float(np.sqrt(np.mean(c * c)))
            return float(np.clip(0.5 * (mean_abs + rms), 0.0, 1.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    def leaf_dims(self, scores_leaf: Optional[Any], vib: Optional[Any]) -> np.ndarray:
        """Compute a `(leaf_count,)` integer physical-dimension vector.

        `scores_leaf` may be `None`, in which case the uniform dimension is
        used.  If a ProjectionLayer is attached, all returned dimensions satisfy
        `d <= projection.fuzzy.k`.
        """
        leaf_count = int(self.cfg.leaf_count)
        cap = self.projection_dim_cap
        min_d, base_d, max_d = _coerce_dim_bounds(self.cfg.d_min, self.cfg.d_uniform, self.cfg.d_max, cap=cap)

        if scores_leaf is None:
            dims = np.full((leaf_count,), int(base_d), dtype=np.int64)
            scores = np.zeros((leaf_count,), dtype=np.float64)
        else:
            scores = _as_score_vector(scores_leaf, leaf_count=leaf_count, strict=bool(self.cfg.strict_shapes))
            if scores.size != leaf_count:
                scores = _as_score_vector(scores, leaf_count=leaf_count, strict=False)

            dims = adaptive_fiber_dims(
                scores,
                d_uniform=int(base_d),
                d_min=int(min_d),
                d_max=int(max_d),
                strength=float(self.cfg.adaptive_strength),
                norm=self._score_norm_cfg(),
                quantize_to=int(max(1, self.cfg.quantize_to)),
                cap=cap,
            ).astype(np.int64, copy=False)

        vib_factor = self._vibration_factor(vib)
        self._last_vibration_factor = float(vib_factor)
        if vib_factor > 0.0 and float(self.cfg.vib_influence) > 0.0:
            gain = 1.0 + float(self.cfg.vib_influence) * vib_factor
            dims = _round_to_multiple(dims.astype(np.float64) * gain, self.cfg.quantize_to).astype(np.int64, copy=False)
            dims = np.clip(dims, min_d, max_d).astype(np.int64, copy=False)
            if cap is not None:
                dims = np.minimum(dims, int(cap)).astype(np.int64, copy=False)

        dims = self._apply_dimension_rate_limits(dims)
        dims = np.maximum(dims, 1).astype(np.int64, copy=False)

        self._last_scores = scores.astype(np.float64, copy=True)
        self._last_dims = dims.copy()
        return dims.astype(int, copy=False)

    def make_phys_dim_map(self, tree: Any, d_leaf: Any) -> Dict[int, int]:
        """Map a leaf dimension vector to `{leaf_node_id: d}` using `tree.leaves`."""
        _assert(hasattr(tree, "leaves"), "tree must expose leaves")
        leaves = [int(x) for x in list(getattr(tree, "leaves"))]
        leaf_count = len(leaves)
        expected = int(self.cfg.leaf_count)
        if self.cfg.strict_shapes:
            _assert(leaf_count == expected, f"expected {expected} leaves, got {leaf_count}")

        fill = int(self.cfg.d_uniform)
        dims = _as_dim_vector(d_leaf, leaf_count=leaf_count, fill=fill)
        cap = self.projection_dim_cap
        if cap is not None:
            dims = np.minimum(dims, int(cap)).astype(np.int64, copy=False)
        dims = np.clip(dims, self.cfg.d_min, self.cfg.d_max).astype(np.int64, copy=False)
        return {int(leaves[i]): int(max(1, dims[i])) for i in range(leaf_count)}

    def uniform_leaf_dims(self) -> np.ndarray:
        """Return the projection-safe uniform dimension vector."""
        return self.leaf_dims(None, None)

    def last_leaf_dims(self) -> Optional[np.ndarray]:
        return None if self._last_dims is None else self._last_dims.copy()

    # ------------------------------------------------------------------
    # Operator bases
    # ------------------------------------------------------------------
    def clear_operator_cache(self) -> None:
        self._basis_cache.clear()

    def _adinkra_basis(self, d: int) -> Dict[str, np.ndarray]:
        try:
            ops = self.adinkra.operator_basis(int(d))
        except Exception:
            ops = AdinkraConstraint(seed=int(self.cfg.seed)).operator_basis(int(d))
        if not isinstance(ops, Mapping):
            raise TypeError("AdinkraConstraint.operator_basis must return a mapping")
        return {str(k): np.asarray(v, dtype=np.complex128) for k, v in ops.items()}

    def base_operator_basis(self, d: int) -> Dict[str, np.ndarray]:
        """Return deterministic local operators for a physical dimension `d`.

        The dictionary always contains at least `I`, `X`, `Y`, `Z`, and `G`.
        With `include_spin_generators=True`, it also contains `Lx`, `Ly`, and
        `Lz`.  Projection-aware `L*` operators are added later by
        `projection.projected_ops(...)` in the Hamiltonian builder, overriding
        these generic spin generators where appropriate.
        """
        d = int(max(1, d))
        cap = self.projection_dim_cap
        if cap is not None and d > int(cap):
            raise ValueError(f"requested local basis dimension d={d} exceeds projection fuzzy k={cap}")

        if self.cfg.cache_operator_basis and d in self._basis_cache:
            return {k: v.copy() for k, v in self._basis_cache[d].items()}

        ops = self._adinkra_basis(d)

        if self.include_pauli_fallback:
            for name, A in _pauli_embedded(d).items():
                if name not in ops or np.asarray(ops[name]).shape != (d, d):
                    ops[name] = A

        # Guarantee identity is exact.
        ops["I"] = np.eye(d, dtype=np.complex128)

        if self.cfg.include_spin_generators:
            for name, A in _spin_generators(d).items():
                if name not in ops or np.asarray(ops[name]).shape != (d, d):
                    ops[name] = A

        if "G" not in ops or np.asarray(ops["G"]).shape != (d, d):
            rng = np.random.default_rng(int(self.cfg.seed) + 7919 * d)
            G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            G = hermitianize(G)
            ops["G"] = G / max(fro_norm(G), _EPS)

        cleaned: Dict[str, np.ndarray] = {}
        for name, A in ops.items():
            arr = np.asarray(A, dtype=np.complex128)
            if arr.shape != (d, d):
                # Discard incompatible custom operators rather than letting a
                # later Hamiltonian apply fail with an opaque shape error.
                continue
            if name != "I":
                # Most local terms assume Hermitian observables. Preserve exact
                # I, symmetrize other basis entries defensively.
                arr = hermitianize(arr)
            target = self.cfg.operator_norm_target if self.cfg.normalize_nonidentity_ops else None
            arr = _normalize_operator(arr, target_norm=target, preserve_identity=True)
            if name != "I" and fro_norm(arr) < float(self.cfg.min_operator_norm):
                # Avoid silently exporting a dead operator.
                if name in {"X", "Y", "Z", "Lx", "Ly", "Lz", "G"}:
                    fallback = _spin_generators(d).get(name)
                    if fallback is None:
                        fallback = _pauli_embedded(d).get(name, np.eye(d, dtype=np.complex128))
                    arr = np.asarray(fallback, dtype=np.complex128)
            cleaned[str(name)] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex128, copy=False)

        # Ensure required keys survived shape cleanup.
        for name, A in _pauli_embedded(d).items():
            cleaned.setdefault(name, A)
        if self.cfg.include_spin_generators:
            for name, A in _spin_generators(d).items():
                cleaned.setdefault(name, A)
        cleaned.setdefault("G", hermitianize(cleaned.get("Z", np.eye(d, dtype=np.complex128))).astype(np.complex128))
        cleaned["I"] = np.eye(d, dtype=np.complex128)

        if self.cfg.cache_operator_basis:
            self._basis_cache[d] = {k: v.copy() for k, v in cleaned.items()}
        return {k: v.copy() for k, v in cleaned.items()}

    def operator_basis_for_tree(self, tree: Any, phys_dims: Optional[Mapping[int, int]] = None) -> Dict[int, Dict[str, np.ndarray]]:
        """Return local operator bases for every leaf in a tree."""
        leaves = [int(x) for x in list(getattr(tree, "leaves", []))]
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for lid in leaves:
            d = int(phys_dims[lid]) if phys_dims is not None and lid in phys_dims else int(self.cfg.d_uniform)
            out[lid] = self.base_operator_basis(d)
        return out

    # ------------------------------------------------------------------
    # Diagnostics / state surfaces
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        cap = self.projection_dim_cap
        return {
            "kind": "LocalFiberBuilder",
            "config": _json_safe(self.cfg),
            "projection_dim_cap": None if cap is None else int(cap),
            "include_pauli_fallback": bool(self.include_pauli_fallback),
            "basis_cache_dims": sorted(int(k) for k in self._basis_cache.keys()),
            "last_dims": None if self._last_dims is None else self._last_dims.astype(int).tolist(),
            "last_scores_norm": None if self._last_scores is None else float(safe_norm(self._last_scores)),
            "last_vibration_factor": float(self._last_vibration_factor),
        }

    def health_metrics(self) -> Dict[str, Any]:
        cap = self.projection_dim_cap
        dims = self._last_dims
        ok_projection = True
        if cap is not None and dims is not None:
            ok_projection = bool(np.all(dims <= int(cap)))
        basis_ok = True
        basis_stats: Dict[str, Any] = {}
        try:
            ops = self.base_operator_basis(int(self.cfg.d_uniform))
            basis_stats = _operator_stats(ops)
            basis_ok = bool(basis_stats.get("finite", False))
        except Exception as exc:
            basis_ok = False
            basis_stats = {"error": repr(exc)}

        return {
            "kind": "LocalFiberBuilder",
            "is_stable": bool(ok_projection and basis_ok),
            "projection_constraint_ok": bool(ok_projection),
            "basis_ok": bool(basis_ok),
            "last_dim_min": None if dims is None else int(np.min(dims)),
            "last_dim_mean": None if dims is None else float(np.mean(dims)),
            "last_dim_max": None if dims is None else int(np.max(dims)),
            "basis_stats": basis_stats,
        }

    def __repr__(self) -> str:
        cap = self.projection_dim_cap
        cap_s = "none" if cap is None else str(cap)
        return (
            "LocalFiberBuilder("
            f"d_uniform={self.cfg.d_uniform}, d_min={self.cfg.d_min}, d_max={self.cfg.d_max}, "
            f"projection_cap={cap_s}, leaf_count={self.cfg.leaf_count})"
        )


__all__ = [
    "LocalFiberConfig",
    "LocalFiberBuilder",
    "AdinkraConstraint",
]
