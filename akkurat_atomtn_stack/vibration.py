#!/usr/bin/env python3
# vibration.py
"""
Vibration / phonon-bath model for AtomTN.

This module stays independent of geometry, TTN state, flow, and Hamiltonian
application code. It provides deterministic frequency grids, spectral-density
matching, direction sampling, serialization, and stable diagnostics for the
AtomTN runtime family.

Public compatibility target
---------------------------
Existing scripts call:

    VibrationModel.build("linear", 0.1, 10.0, n=32)
    VibrationModel.build(
        grid_kind="fractal", w_min=..., w_max=..., spectral_kind="ohmic", ...
    )

and then access:

    model.frequencies
    model.couplings
    model.directions
    model.meta
    model.validate()

Those semantics are preserved.

Design notes
------------
- NumPy-only; no SciPy dependency.
- All arrays are finite-normalized on construction.
- Frequencies must be strictly positive.
- Couplings are finite and non-negative by default after spectral matching.
- Direction vectors, when present, are shaped (N, 3) and normalized.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from math_utils import _assert, safe_norm
except Exception:  # pragma: no cover - standalone fallback
    def _assert(cond: bool, msg: str) -> None:
        if not bool(cond):
            raise ValueError(str(msg))

    def safe_norm(x: Any) -> float:
        try:
            arr = np.asarray(x)
            if arr.size == 0 or not np.isfinite(arr).all():
                return 0.0
            out = float(np.linalg.norm(arr.reshape(-1)))
            return out if np.isfinite(out) else 0.0
        except Exception:
            return 0.0


_EPS = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================

def _as_float_array(x: Any, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    """Coerce to finite float64 ndarray."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like") from exc
    if ndim is not None and int(arr.ndim) != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if arr.size:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float64, copy=False)


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return np.nan_to_num(obj.astype(float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _normalize_rows(X: np.ndarray, *, eps: float = _EPS) -> np.ndarray:
    X = _as_float_array(X, name="directions", ndim=2)
    if X.shape[1] != 3:
        raise ValueError(f"directions must have shape (N, 3), got {X.shape}")
    if X.size == 0:
        return X.reshape(0, 3).astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    good = np.isfinite(n) & (n > float(eps))
    out = np.zeros_like(X, dtype=np.float64)
    out[good[:, 0]] = X[good[:, 0]] / n[good[:, 0]]
    return out.astype(np.float64, copy=False)


def _stable_seed(seed: int, salt: int = 0) -> int:
    return int((int(seed) * 1664525 + 1013904223 + int(salt)) & 0xFFFFFFFF)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class VibrationBuildConfig:
    """Declarative configuration for VibrationModel.build_from_config(...)."""

    grid_kind: str = "linear"
    w_min: float = 0.1
    w_max: float = 10.0
    n: int = 64
    spectral_kind: str = "ohmic"
    alpha: float = 1.0
    omega_c: float = 10.0
    coupling_normalization: str = "max"  # "max" | "l2" | "none"
    coupling_scale: float = 1.0
    directions_n: int = 0
    seed: int = 0
    fractal_levels: int = 3
    fractal_branching: int = 4
    fractal_exponent: float = 2.0
    jitter: float = 0.0
    sort_frequencies: bool = True


# =============================================================================
# Frequency grids and directions
# =============================================================================

def fibonacci_sphere(n: int, seed: int = 0) -> np.ndarray:
    """
    Deterministically sample approximately uniform directions on S^2.

    Returns an array of shape (n, 3). For n=0, returns (0, 3).
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    rng = np.random.default_rng(_stable_seed(seed, salt=17))
    offset = float(rng.random() * 2.0 * np.pi)

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * np.pi * (1.0 - 1.0 / phi)

    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = offset + golden_angle * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return _normalize_rows(np.stack([x, y, z], axis=1))


def fibonacci_frequency_grid(w_min: float, w_max: float, n: int) -> np.ndarray:
    """Warped log-like grid using a Fibonacci-ratio interpolation."""
    w_min = float(w_min)
    w_max = float(w_max)
    n = int(n)
    _assert(w_min > 0.0 and w_max > w_min and n >= 2, "bad fibonacci grid params")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t_warp = (phi ** t - 1.0) / (phi - 1.0)
    return w_min * (w_max / w_min) ** t_warp


def fractal_frequency_grid(
    w_min: float,
    w_max: float,
    levels: int,
    branching_ratio: int,
    exponent: float,
) -> np.ndarray:
    """
    Build a deterministic fractal band grid.

    The number of grid points is branching_ratio ** levels. Each recursive band
    contributes one representative frequency chosen by a power-law tilt.
    """
    w_min = float(w_min)
    w_max = float(w_max)
    levels = int(levels)
    branching_ratio = int(branching_ratio)
    exponent = float(exponent)

    _assert(w_min > 0.0 and w_max > w_min, "bad fractal frequency range")
    _assert(levels >= 1 and branching_ratio >= 2, "bad fractal levels/branching")
    _assert(exponent > 0.0, "fractal_exponent must be > 0")

    bands = [(w_min, w_max)]
    for _ in range(levels):
        next_bands = []
        for a, b in bands:
            # Linear edge split keeps the prototype's historical behavior while
            # representative frequencies are multiplicative inside each band.
            edges = np.linspace(a, b, branching_ratio + 1, dtype=np.float64)
            for i in range(branching_ratio):
                lo = float(max(edges[i], _EPS))
                hi = float(max(edges[i + 1], lo * (1.0 + 1e-9)))
                next_bands.append((lo, hi))
        bands = next_bands

    tilt = 1.0 / (1.0 + exponent)
    vals = np.array([a * (b / a) ** tilt for a, b in bands], dtype=np.float64)
    vals.sort()
    return vals


def chebyshev_frequency_grid(w_min: float, w_max: float, n: int) -> np.ndarray:
    """Chebyshev-node grid remapped to [w_min, w_max], denser near endpoints."""
    w_min = float(w_min)
    w_max = float(w_max)
    n = int(n)
    _assert(w_min > 0.0 and w_max > w_min and n >= 1, "bad chebyshev grid params")
    j = np.arange(n, dtype=np.float64)
    x = np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))
    t = 0.5 * (1.0 - x)
    w = w_min + (w_max - w_min) * t
    w.sort()
    return w.astype(np.float64)


def build_frequency_grid(
    grid_kind: str,
    w_min: float,
    w_max: float,
    *,
    n: int = 64,
    seed: int = 0,
    fractal_levels: int = 3,
    fractal_branching: int = 4,
    fractal_exponent: float = 2.0,
    jitter: float = 0.0,
    sort_frequencies: bool = True,
) -> np.ndarray:
    """Build a deterministic positive frequency grid."""
    gk = str(grid_kind).lower().strip()
    w_min = float(w_min)
    w_max = float(w_max)
    n = int(n)
    _assert(w_min > 0.0 and w_max > w_min, "bad frequency bounds")

    if gk == "linear":
        _assert(n >= 1, "linear grid requires n>=1")
        w = np.linspace(w_min, w_max, n, dtype=np.float64)
    elif gk in {"log", "logarithmic", "geom", "geometric"}:
        _assert(n >= 1, "log grid requires n>=1")
        w = np.geomspace(w_min, w_max, n).astype(np.float64)
    elif gk == "fibonacci":
        w = fibonacci_frequency_grid(w_min, w_max, n)
    elif gk == "fractal":
        w = fractal_frequency_grid(
            w_min,
            w_max,
            levels=int(fractal_levels),
            branching_ratio=int(fractal_branching),
            exponent=float(fractal_exponent),
        )
    elif gk == "chebyshev":
        w = chebyshev_frequency_grid(w_min, w_max, n)
    else:
        raise ValueError(f"unknown grid_kind: {grid_kind}")

    w = _as_float_array(w, name="frequencies", ndim=1)
    if float(jitter) > 0.0 and w.size > 1:
        rng = np.random.default_rng(_stable_seed(seed, salt=91))
        # Multiplicative jitter preserves positivity. Keep it small by design.
        amp = min(float(jitter), 0.25)
        w = w * np.exp(rng.normal(0.0, amp, size=w.shape))
        w = np.clip(w, w_min, w_max)

    w = np.maximum(w, _EPS)
    if sort_frequencies:
        w.sort()
    return w.astype(np.float64, copy=False)


# =============================================================================
# Spectral densities and coupling matching
# =============================================================================

def spectral_density(
    omega: Any,
    kind: str = "ohmic",
    alpha: float = 1.0,
    omega_c: float = 10.0,
) -> np.ndarray:
    """
    Evaluate spectral density templates.

    Supported kinds:
      - ohmic:      J(w) = w^alpha exp(-w/omega_c)
      - powerlaw:   J(w) = w^alpha
      - debye:      J(w) = w^2 for w <= omega_c else 0
      - flat:       J(w) = 1
      - lorentzian: centered at omega_c with width alpha (or 1 if alpha<=0)
      - subohmic:   shorthand for alpha=0.5 ohmic
      - superohmic: shorthand for alpha=3.0 ohmic
    """
    w = np.maximum(_as_float_array(omega, name="omega"), _EPS)
    k = str(kind).lower().strip()
    a = float(alpha)
    wc = max(float(omega_c), _EPS)

    if k == "subohmic":
        a = 0.5
        k = "ohmic"
    elif k == "superohmic":
        a = 3.0
        k = "ohmic"

    if k == "ohmic":
        J = (w ** a) * np.exp(-w / wc)
    elif k == "powerlaw":
        J = w ** a
    elif k == "debye":
        J = np.where(w <= wc, w * w, 0.0)
    elif k == "flat":
        J = np.ones_like(w, dtype=np.float64)
    elif k == "lorentzian":
        gamma = max(abs(a), 1.0)
        J = (gamma * gamma) / ((w - wc) * (w - wc) + gamma * gamma)
    else:
        raise ValueError(f"unknown spectral density: {kind}")

    return np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def _frequency_widths(frequencies: np.ndarray) -> np.ndarray:
    w = _as_float_array(frequencies, name="frequencies", ndim=1)
    if w.size <= 1:
        return np.ones_like(w, dtype=np.float64)

    idx = np.argsort(w)
    ws = w[idx]
    dw = np.zeros_like(ws, dtype=np.float64)
    dw[1:-1] = 0.5 * (ws[2:] - ws[:-2])
    dw[0] = ws[1] - ws[0]
    dw[-1] = ws[-1] - ws[-2]
    dw = np.maximum(dw, _EPS)

    out = np.empty_like(dw)
    out[idx] = dw
    return out


def normalize_couplings(couplings: Any, *, mode: str = "max", scale: float = 1.0) -> np.ndarray:
    """Normalize finite couplings with stable zero handling."""
    g = _as_float_array(couplings, name="couplings", ndim=1)
    g = np.maximum(g, 0.0)
    mode = str(mode).lower().strip()

    if g.size == 0:
        return g.astype(np.float64)

    if mode == "max":
        denom = float(np.max(np.abs(g)))
    elif mode in {"l2", "norm"}:
        denom = float(np.linalg.norm(g))
    elif mode in {"none", "raw"}:
        denom = 1.0
    else:
        raise ValueError(f"unknown coupling normalization mode: {mode}")

    if denom > _EPS and np.isfinite(denom):
        g = g / denom
    else:
        g = np.zeros_like(g, dtype=np.float64)

    g = g * float(scale)
    return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def match_discrete_couplings(
    frequencies: Any,
    J_of_w: Any,
    *,
    normalization: str = "max",
    scale: float = 1.0,
) -> np.ndarray:
    """
    Convert a continuous spectral density J(w) into discrete couplings g_i.

    Approximation:
        g_i^2 ≈ max(J(w_i), 0) * Δw_i

    Couplings are then normalized according to `normalization`.
    """
    w = _as_float_array(frequencies, name="frequencies", ndim=1)
    J = _as_float_array(J_of_w, name="J_of_w", ndim=1)
    _assert(w.shape == J.shape, "frequency / spectral-density shape mismatch")

    if w.size == 0:
        return np.zeros(0, dtype=np.float64)

    dw = _frequency_widths(w)
    g2 = np.maximum(J, 0.0) * dw
    g = np.sqrt(np.maximum(g2, 0.0))
    return normalize_couplings(g, mode=normalization, scale=scale)


# =============================================================================
# Vibration model
# =============================================================================

@dataclass
class VibrationModel:
    """Discrete vibration bath: frequencies, couplings, optional directions."""

    frequencies: np.ndarray
    couplings: np.ndarray
    directions: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequencies = _as_float_array(self.frequencies, name="frequencies", ndim=1)
        self.couplings = _as_float_array(self.couplings, name="couplings", ndim=1)
        if self.directions is not None:
            self.directions = _normalize_rows(self.directions)
        self.meta = dict(self.meta or {})
        self.validate()

    # ------------------------------------------------------------------
    # Validation / diagnostics
    # ------------------------------------------------------------------
    def validate(self) -> None:
        _assert(self.frequencies.ndim == 1, "frequencies must be 1D")
        _assert(self.couplings.ndim == 1, "couplings must be 1D")
        _assert(self.frequencies.shape == self.couplings.shape, "frequency/coupling shape mismatch")
        _assert(np.all(np.isfinite(self.frequencies)), "frequencies contain non-finite values")
        _assert(np.all(np.isfinite(self.couplings)), "couplings contain non-finite values")
        _assert(np.all(self.frequencies > 0.0), "frequencies must be positive")
        if self.directions is not None:
            _assert(self.directions.ndim == 2 and self.directions.shape[1] == 3, "directions must be (N,3)")
            _assert(np.all(np.isfinite(self.directions)), "directions contain non-finite values")
        return None

    @property
    def size(self) -> int:
        return int(self.frequencies.size)

    @property
    def n_modes(self) -> int:
        return int(self.frequencies.size)

    def copy(self) -> "VibrationModel":
        return VibrationModel(
            frequencies=self.frequencies.copy(),
            couplings=self.couplings.copy(),
            directions=None if self.directions is None else self.directions.copy(),
            meta=dict(self.meta),
        )

    def stats(self) -> Dict[str, Any]:
        w = self.frequencies
        g = self.couplings
        return {
            "n": int(w.size),
            "frequency_min": float(np.min(w)) if w.size else 0.0,
            "frequency_max": float(np.max(w)) if w.size else 0.0,
            "frequency_mean": float(np.mean(w)) if w.size else 0.0,
            "frequency_std": float(np.std(w)) if w.size else 0.0,
            "coupling_min": float(np.min(g)) if g.size else 0.0,
            "coupling_max": float(np.max(g)) if g.size else 0.0,
            "coupling_mean": float(np.mean(g)) if g.size else 0.0,
            "coupling_l2": safe_norm(g),
            "has_directions": self.directions is not None,
            "directions_n": 0 if self.directions is None else int(self.directions.shape[0]),
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            "[VibrationModel]\n"
            f"- modes: {s['n']}\n"
            f"- frequency range: {s['frequency_min']:.6g} .. {s['frequency_max']:.6g}\n"
            f"- coupling max: {s['coupling_max']:.6g}\n"
            f"- coupling l2: {s['coupling_l2']:.6g}\n"
            f"- directions: {s['directions_n']}\n"
        )

    def effective_frequency(self, *, weighted: bool = True) -> float:
        """Return mean frequency, optionally weighted by coupling energy."""
        if self.frequencies.size == 0:
            return 0.0
        if not weighted:
            return float(np.mean(self.frequencies))
        weights = self.couplings * self.couplings
        s = float(np.sum(weights))
        if not np.isfinite(s) or s <= _EPS:
            return float(np.mean(self.frequencies))
        return float(np.sum(weights * self.frequencies) / s)

    def thermal_occupancy(self, temperature: float, *, kB: float = 1.0) -> np.ndarray:
        """
        Bose-Einstein occupancy n(w)=1/(exp(w/(kB*T))-1), with stable clipping.
        """
        T = max(float(temperature), _EPS)
        beta_w = np.clip(self.frequencies / max(float(kB) * T, _EPS), 1e-12, 80.0)
        out = 1.0 / np.expm1(beta_w)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def coupling_envelope(self, temperature: Optional[float] = None) -> np.ndarray:
        """Return couplings, optionally thermally amplified by sqrt(2n+1)."""
        g = self.couplings.astype(np.float64, copy=True)
        if temperature is not None:
            nbar = self.thermal_occupancy(float(temperature))
            g = g * np.sqrt(2.0 * nbar + 1.0)
        return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def with_scaled_couplings(self, scale: float) -> "VibrationModel":
        out = self.copy()
        out.couplings = (out.couplings * float(scale)).astype(np.float64)
        out.meta["coupling_scale_applied"] = float(scale)
        out.validate()
        return out

    def rescale_frequency_range(self, w_min: float, w_max: float) -> "VibrationModel":
        """Affine-rescale current frequencies into a new positive range."""
        w_min = float(w_min)
        w_max = float(w_max)
        _assert(w_min > 0.0 and w_max > w_min, "bad target frequency range")
        out = self.copy()
        old = out.frequencies
        if old.size == 0 or float(np.max(old) - np.min(old)) <= _EPS:
            out.frequencies = np.full_like(old, 0.5 * (w_min + w_max), dtype=np.float64)
        else:
            t = (old - float(np.min(old))) / max(float(np.max(old) - np.min(old)), _EPS)
            out.frequencies = w_min + t * (w_max - w_min)
        out.meta["rescaled_frequency_range"] = [w_min, w_max]
        out.validate()
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "VibrationModel",
            "frequencies": self.frequencies.astype(float).tolist(),
            "couplings": self.couplings.astype(float).tolist(),
            "directions": None if self.directions is None else self.directions.astype(float).tolist(),
            "meta": _json_safe(self.meta),
            "stats": self.stats(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VibrationModel":
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        return cls(
            frequencies=np.asarray(payload.get("frequencies", []), dtype=np.float64),
            couplings=np.asarray(payload.get("couplings", []), dtype=np.float64),
            directions=(None if payload.get("directions", None) is None else np.asarray(payload.get("directions"), dtype=np.float64)),
            meta=dict(payload.get("meta", {})),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(self.to_dict()), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "VibrationModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Static builders retained for compatibility
    # ------------------------------------------------------------------
    @staticmethod
    def spectral_density(
        omega: Any,
        kind: str = "ohmic",
        alpha: float = 1.0,
        omega_c: float = 10.0,
    ) -> np.ndarray:
        return spectral_density(omega, kind=kind, alpha=alpha, omega_c=omega_c)

    @staticmethod
    def match_discrete_couplings(frequencies: Any, J_of_w: Any) -> np.ndarray:
        # Historical behavior normalized by max(g).
        return match_discrete_couplings(frequencies, J_of_w, normalization="max", scale=1.0)

    @staticmethod
    def build(
        grid_kind: str,
        w_min: float,
        w_max: float,
        n: int = 64,
        spectral_kind: str = "ohmic",
        alpha: float = 1.0,
        omega_c: float = 10.0,
        directions_n: int = 0,
        seed: int = 0,
        fractal_levels: int = 3,
        fractal_branching: int = 4,
        fractal_exponent: float = 2.0,
        coupling_normalization: str = "max",
        coupling_scale: float = 1.0,
        jitter: float = 0.0,
        sort_frequencies: bool = True,
    ) -> "VibrationModel":
        """
        Build a VibrationModel from a frequency grid and spectral density.

        Parameters are intentionally backward-compatible with the prior AtomTN
        runtime. Additional optional parameters are keyword-only-compatible in
        practice for existing callers because they appear after the old fields.
        """
        cfg = VibrationBuildConfig(
            grid_kind=str(grid_kind),
            w_min=float(w_min),
            w_max=float(w_max),
            n=int(n),
            spectral_kind=str(spectral_kind),
            alpha=float(alpha),
            omega_c=float(omega_c),
            coupling_normalization=str(coupling_normalization),
            coupling_scale=float(coupling_scale),
            directions_n=int(directions_n),
            seed=int(seed),
            fractal_levels=int(fractal_levels),
            fractal_branching=int(fractal_branching),
            fractal_exponent=float(fractal_exponent),
            jitter=float(jitter),
            sort_frequencies=bool(sort_frequencies),
        )
        return VibrationModel.build_from_config(cfg)

    @staticmethod
    def build_from_config(cfg: VibrationBuildConfig) -> "VibrationModel":
        w = build_frequency_grid(
            cfg.grid_kind,
            cfg.w_min,
            cfg.w_max,
            n=cfg.n,
            seed=cfg.seed,
            fractal_levels=cfg.fractal_levels,
            fractal_branching=cfg.fractal_branching,
            fractal_exponent=cfg.fractal_exponent,
            jitter=cfg.jitter,
            sort_frequencies=cfg.sort_frequencies,
        )
        J = spectral_density(w, kind=cfg.spectral_kind, alpha=cfg.alpha, omega_c=cfg.omega_c)
        g = match_discrete_couplings(
            w,
            J,
            normalization=cfg.coupling_normalization,
            scale=cfg.coupling_scale,
        )

        dirs = None
        if int(cfg.directions_n) > 0:
            dirs = fibonacci_sphere(int(cfg.directions_n), seed=int(cfg.seed))

        meta = {
            "grid": str(cfg.grid_kind).lower().strip(),
            "n": int(w.size),
            "requested_n": int(cfg.n),
            "spectral": str(cfg.spectral_kind).lower().strip(),
            "alpha": float(cfg.alpha),
            "omega_c": float(cfg.omega_c),
            "w_min": float(cfg.w_min),
            "w_max": float(cfg.w_max),
            "coupling_normalization": str(cfg.coupling_normalization),
            "coupling_scale": float(cfg.coupling_scale),
            "directions_n": int(cfg.directions_n),
            "seed": int(cfg.seed),
            "fractal_levels": int(cfg.fractal_levels),
            "fractal_branching": int(cfg.fractal_branching),
            "fractal_exponent": float(cfg.fractal_exponent),
            "jitter": float(cfg.jitter),
        }
        return VibrationModel(frequencies=w, couplings=g, directions=dirs, meta=meta)


# =============================================================================
# Convenience aliases
# =============================================================================

def build_vibration_model(*args: Any, **kwargs: Any) -> VibrationModel:
    """Alias for VibrationModel.build(...)."""
    return VibrationModel.build(*args, **kwargs)


__all__ = [
    "VibrationBuildConfig",
    "VibrationModel",
    "fibonacci_sphere",
    "fibonacci_frequency_grid",
    "fractal_frequency_grid",
    "chebyshev_frequency_grid",
    "build_frequency_grid",
    "spectral_density",
    "normalize_couplings",
    "match_discrete_couplings",
    "build_vibration_model",
]


if __name__ == "__main__":  # lightweight smoke test
    vm = VibrationModel.build("linear", 0.1, 10.0, n=32, directions_n=8, seed=42)
    vm.validate()
    print(vm.summary())
