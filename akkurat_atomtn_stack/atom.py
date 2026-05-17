#!/usr/bin/env python3
# atom.py
"""
AtomTN atom wrapper and demo runner.

This module wires together the core AtomTN substrate:
- geometry.TetraMesh64 -> Tree + GraphCalculus;
- optional noncommutative/fuzzy backend, projection, generator decomposition,
  and holonomy transport;
- vibration attachment;
- TTNState initialization through LocalFiberBuilder;
- flow simulation through scalar/NC geodesic solvers and FlowMonitor;
- Hamiltonian building and TTN time evolution.

Public compatibility
--------------------
The API used by the current AtomTN runtime family is preserved:

    Atom(...)
    Atom.setup(...)
    Atom.switch_backend(...)
    Atom.attach_vibration(...)
    Atom.init_state(...)
    Atom.build_initial_flow(...)
    Atom.simulate_flow(...)
    Atom.evolve(...)
    Atom.report(...)

Production additions are deliberately additive: import-safe path setup,
structured snapshots, health metrics, stable NC layer rebuilding, cache/cadence
synchronization, and finite-state guards.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Local import setup
# =============================================================================


def _add_local_paths() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, here.parent.parent):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return here.parent


_ATOMTN_DIR = _add_local_paths()

from math_utils import _assert, antihermitianize, fro_norm  # noqa: E402
from geometry import TetraMesh64, Tree  # noqa: E402
from vibration import VibrationModel  # noqa: E402
from fiber import LocalFiberBuilder, LocalFiberConfig, AdinkraConstraint  # noqa: E402
from ttn_state import TTNState  # noqa: E402
from flow import (  # noqa: E402
    GraphVectorField,
    FlowDiagnostics,
    GeodesicFlowConfig,
    GeodesicFlowSolver,
    NCGeodesicFlowSolver,
    FlowMonitor,
)
from fuzzy_backend import NCFuzzyBackend  # noqa: E402
from projection import ProjectionLayer  # noqa: E402
from holonomy import HolonomyBuilder, GeneratorDecomposition  # noqa: E402
from hamiltonian import HamiltonianBuildConfig, TreeMPOBuilder  # noqa: E402
from evolve import ApplyConfig, TTNEvolveConfig, TTNTimeEvolver  # noqa: E402


# =============================================================================
# Generic helpers
# =============================================================================

_EPS = 1e-12


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _maybe_setattr(obj: Any, name: str, value: Any) -> None:
    """Set optional tuning attributes only if the object supports them."""
    try:
        if obj is not None and hasattr(obj, name):
            setattr(obj, name, value)
    except Exception:
        pass


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {"real": np.real(arr).astype(float).tolist(), "imag": np.imag(arr).astype(float).tolist()}
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "snapshot") and callable(getattr(obj, "snapshot")):
        try:
            return _json_safe(obj.snapshot())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _safe_norm(x: Any) -> float:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return 0.0
        val = float(np.linalg.norm(arr.reshape(-1)))
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _state_norm_squared(state: Optional[TTNState]) -> float:
    if state is None:
        return 0.0
    try:
        val = float(state.amplitude_norm_squared())
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _count_nonfinite_state(state: Optional[TTNState]) -> int:
    if state is None:
        return 0
    total = 0
    for T in state.tensors.values():
        arr = np.asarray(T)
        total += int(arr.size - np.count_nonzero(np.isfinite(arr)))
    return int(total)


def _as_adjacency(obj: Any) -> Dict[int, List[int]]:
    raw = dict(obj)
    return {int(k): sorted({int(v) for v in vals if int(v) != int(k)}) for k, vals in raw.items()}


def _field_max_norm(X: Optional[GraphVectorField]) -> float:
    if X is None:
        return 0.0
    try:
        vals = []
        for v in X.edge_values.values():
            vals.append(_safe_norm(v))
        return float(max(vals, default=0.0))
    except Exception:
        return 0.0


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return dict(asdict(cfg))
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {"repr": repr(cfg)}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AtomCacheConfig:
    """Cadence/caching knobs shared by projection, holonomy, and builder caches."""

    update_every_full_steps: int = 1
    cache_bucket_full_steps: int = 1
    freeze_within_step: bool = True

    def normalized(self) -> "AtomCacheConfig":
        return AtomCacheConfig(
            update_every_full_steps=int(max(1, self.update_every_full_steps)),
            cache_bucket_full_steps=int(max(1, self.cache_bucket_full_steps)),
            freeze_within_step=bool(self.freeze_within_step),
        )


@dataclass
class AtomRuntimeConfig:
    """Optional production knobs for Atom setup and runtime maintenance."""

    projection_strategy: str = "energy_based"
    projection_energy_mode: str = "i_kappa"
    projection_overlap_track: bool = True
    holonomy_alpha: float = 0.25
    holonomy_orthonormalize: bool = True
    decomp_observable_mode: str = "auto"
    cache: AtomCacheConfig = field(default_factory=AtomCacheConfig)
    strict_finite_checks: bool = True
    init_normalize: bool = True

    def normalized(self) -> "AtomRuntimeConfig":
        return AtomRuntimeConfig(
            projection_strategy=str(self.projection_strategy or "energy_based"),
            projection_energy_mode=str(self.projection_energy_mode or "i_kappa"),
            projection_overlap_track=bool(self.projection_overlap_track),
            holonomy_alpha=float(self.holonomy_alpha),
            holonomy_orthonormalize=bool(self.holonomy_orthonormalize),
            decomp_observable_mode=str(self.decomp_observable_mode or "auto"),
            cache=self.cache.normalized() if isinstance(self.cache, AtomCacheConfig) else AtomCacheConfig(**dict(self.cache)).normalized(),
            strict_finite_checks=bool(self.strict_finite_checks),
            init_normalize=bool(self.init_normalize),
        )


# =============================================================================
# Atom wrapper
# =============================================================================

@dataclass
class Atom:
    element: str
    Z: int
    isotope_A: Optional[int] = None

    noncommutative: bool = False
    fuzzy_l: int = 2
    seed: int = 0

    geometry: TetraMesh64 = field(default_factory=TetraMesh64)
    vibration: Optional[VibrationModel] = None
    runtime: AtomRuntimeConfig = field(default_factory=AtomRuntimeConfig)

    # Compiled/runtime objects.
    tree: Optional[Tree] = None
    state: Optional[TTNState] = None

    backend: Optional[NCFuzzyBackend] = None
    calc: Optional[Any] = None

    projection: Optional[ProjectionLayer] = None
    decomp: Optional[GeneratorDecomposition] = None
    holonomy: Optional[HolonomyBuilder] = None

    X_traj: Optional[List[GraphVectorField]] = None
    flow_diags: Optional[List[FlowDiagnostics]] = None
    ttn_traj: Optional[List[TTNState]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.element = str(self.element)
        self.Z = int(self.Z)
        self.isotope_A = None if self.isotope_A is None else int(self.isotope_A)
        self.noncommutative = bool(self.noncommutative)
        self.fuzzy_l = int(max(0, self.fuzzy_l))
        self.seed = int(self.seed)
        self.runtime = self.runtime.normalized() if isinstance(self.runtime, AtomRuntimeConfig) else AtomRuntimeConfig(**dict(self.runtime)).normalized()
        self.metadata = dict(self.metadata or {})

    # ------------------------------------------------------------------
    # Setup / backend wiring
    # ------------------------------------------------------------------

    def setup(self, tree_mode: str = "balanced", arity: int = 4) -> None:
        """Compile geometry tree/calculus and initialize optional NC layers."""
        self.tree = self.geometry.compile_tree(mode=str(tree_mode), arity=int(arity), seed=int(self.seed))
        self.calc = self.geometry.compile_calculus(num_nodes=64)
        self.metadata["setup"] = {
            "tree_mode": str(tree_mode),
            "arity": int(arity),
            "ts": _now_iso(),
            "atomtn_dir": str(_ATOMTN_DIR),
        }

        if self.noncommutative:
            self._init_noncommutative_layers(rebuild_backend=True)
        else:
            self._clear_noncommutative_layers(clear_backend=True)

        self._invalidate_trajectories()

    def _invalidate_trajectories(self) -> None:
        self.X_traj = None
        self.flow_diags = None
        self.ttn_traj = None

    def _ensure_setup(self) -> None:
        if self.tree is None or self.calc is None:
            self.setup()

    def _clear_noncommutative_layers(self, *, clear_backend: bool = True) -> None:
        if clear_backend:
            self.backend = None
        self.projection = None
        self.decomp = None
        self.holonomy = None

    def _sync_cache_knobs(self, bucket: Optional[int] = None) -> None:
        cfg = self.runtime.cache.normalized()
        if bucket is not None:
            cfg.cache_bucket_full_steps = int(max(1, bucket))

        for obj in (self.projection, self.holonomy):
            _maybe_setattr(obj, "update_every_full_steps", int(cfg.update_every_full_steps))
            _maybe_setattr(obj, "cache_bucket_full_steps", int(cfg.cache_bucket_full_steps))
            _maybe_setattr(obj, "freeze_within_step", bool(cfg.freeze_within_step))
            # Legacy aliases used by some earlier classes.
            _maybe_setattr(obj, "update_every", int(cfg.update_every_full_steps))
            _maybe_setattr(obj, "cache_bucket", int(cfg.cache_bucket_full_steps))

    def _init_noncommutative_layers(self, *, rebuild_backend: bool = False) -> None:
        self._ensure_setup_without_nc_recursion()
        _assert(self.calc is not None, "Atom setup incomplete: calc is missing")

        if rebuild_backend or self.backend is None:
            self.backend = NCFuzzyBackend(
                _as_adjacency(self.geometry.adjacency),
                fuzzy_l=int(self.fuzzy_l),
                seed=int(self.seed),
            )

        self.projection = ProjectionLayer(
            fuzzy=self.backend.fuzzy,
            strategy=str(self.runtime.projection_strategy),
            overlap_track=bool(self.runtime.projection_overlap_track),
            update_every_full_steps=int(self.runtime.cache.update_every_full_steps),
            cache_bucket_full_steps=int(self.runtime.cache.cache_bucket_full_steps),
            freeze_within_step=bool(self.runtime.cache.freeze_within_step),
            energy_mode=str(self.runtime.projection_energy_mode),
        )

        self.decomp = GeneratorDecomposition(
            Lx=self.backend.fuzzy.Lx,
            Ly=self.backend.fuzzy.Ly,
            Lz=self.backend.fuzzy.Lz,
            remove_trace=True,
            observable_mode=str(self.runtime.decomp_observable_mode),
        )

        self.holonomy = HolonomyBuilder(
            decomp=self.decomp,
            alpha=float(self.runtime.holonomy_alpha),
            orthonormalize=bool(self.runtime.holonomy_orthonormalize),
            update_every_full_steps=int(self.runtime.cache.update_every_full_steps),
            cache_bucket_full_steps=int(self.runtime.cache.cache_bucket_full_steps),
            freeze_within_step=bool(self.runtime.cache.freeze_within_step),
        )
        self._sync_cache_knobs()

    def _ensure_setup_without_nc_recursion(self) -> None:
        """Ensure tree/calculus exist without recursively creating NC layers."""
        if self.tree is not None and self.calc is not None:
            return
        prev_nc = bool(self.noncommutative)
        self.noncommutative = False
        try:
            self.setup()
        finally:
            self.noncommutative = prev_nc

    def switch_backend(self, noncommutative: bool) -> None:
        """Dynamically switch between commutative and noncommutative physics."""
        target = bool(noncommutative)
        self._ensure_setup_without_nc_recursion()
        if self.noncommutative == target:
            if target and (self.backend is None or self.projection is None or self.decomp is None or self.holonomy is None):
                self._init_noncommutative_layers(rebuild_backend=self.backend is None)
            return

        self.noncommutative = target
        if target:
            self._init_noncommutative_layers(rebuild_backend=(self.backend is None))
        else:
            self._clear_noncommutative_layers(clear_backend=True)
        self._invalidate_trajectories()

    # ------------------------------------------------------------------
    # UI/math events and vibration
    # ------------------------------------------------------------------

    def get_math_events(self) -> List[Dict[str, Any]]:
        """Return compact event descriptors for visualization/HUD overlays."""
        events: List[Dict[str, Any]] = [
            {"type": "geometry", "text": "M = TetraMesh64"},
            {"type": "spectral", "text": "ω ∼ spectral grid"},
        ]
        if self.state is not None:
            events.append({"type": "state", "text": "Ψ = TTNState(tree, tensors)"})
        if self.vibration is not None:
            grid = self.vibration.meta.get("grid", "unknown") if hasattr(self.vibration, "meta") else "unknown"
            events.append({"type": "vibration", "text": f"Bath: {grid}"})
        if self.noncommutative:
            events.append({"type": "commutator", "text": "[X_μ, X_ν] ≠ 0", "highlight": True})
            if self.projection is not None:
                events.append({"type": "projection", "text": "Pκ: Mat_k → ℂ^d"})
            if self.holonomy is not None:
                events.append({"type": "holonomy", "text": "Holonomy: Uγ acts by adjoint transport"})
        return events

    def attach_vibration(self, vib: VibrationModel) -> None:
        if hasattr(vib, "validate") and callable(getattr(vib, "validate")):
            vib.validate()
        self.vibration = vib
        self.metadata["vibration_attached_at"] = _now_iso()

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def init_state(self, *, fiber: LocalFiberBuilder, bond_dim: Union[int, Mapping[int, int]] = 4) -> None:
        """Initialize a normalized TTNState using the supplied fiber builder."""
        self._ensure_setup()
        _assert(self.tree is not None, "call setup() first")

        leaf_count = len(self.tree.leaves)
        d0 = int(getattr(getattr(fiber, "cfg", object()), "d_uniform", 4))
        d_leaf = np.full((leaf_count,), max(1, d0), dtype=int)

        # Hard NC invariant: if a fuzzy projection exists, local physical dims
        # must not exceed k.  This also protects callers that built LocalFiber
        # without passing projection=atom.projection.
        k_cap = None
        if self.projection is not None and hasattr(self.projection, "fuzzy"):
            k_cap = int(getattr(self.projection.fuzzy, "k", 0) or 0)
        elif getattr(fiber, "projection", None) is not None:
            try:
                k_cap = int(fiber.projection.fuzzy.k)
            except Exception:
                k_cap = None
        if k_cap is not None and k_cap > 0:
            d_leaf = np.minimum(d_leaf, k_cap).astype(int)

        if hasattr(fiber, "make_phys_dim_map"):
            phys = fiber.make_phys_dim_map(self.tree, d_leaf)
        else:
            phys = {int(self.tree.leaves[i]): int(d_leaf[i]) for i in range(leaf_count)}

        self.state = TTNState.random(
            self.tree,
            phys_dims_leaf=phys,
            bond_dim=bond_dim,
            seed=int(self.seed),
            normalize=bool(self.runtime.init_normalize),
            metadata={"atom": self.element, "Z": int(self.Z), "created_at": _now_iso()},
        )
        if bool(self.runtime.init_normalize):
            self.state.normalize_in_place()
        if bool(self.runtime.strict_finite_checks):
            self.state.validate()
        self.ttn_traj = None

    # ------------------------------------------------------------------
    # Flow initialization and simulation
    # ------------------------------------------------------------------

    def build_initial_flow(self, scale: float = 0.1) -> GraphVectorField:
        """
        Return an antisymmetric oriented edge field.

        Commutative mode stores scalar floats.  NC mode stores k×k complex
        anti-Hermitian matrices projected through twisted reality.
        """
        self._ensure_setup()
        _assert(self.calc is not None, "calc missing; call setup()")
        rng = np.random.default_rng(int(self.seed) + 999)
        sc = _finite_float(scale, 0.1)
        edges = list(self.calc.oriented_edges())

        if not self.noncommutative:
            vals: Dict[Tuple[int, int], float] = {}
            for u, v in edges:
                val = float(sc * rng.normal())
                vals[(int(u), int(v))] = val
                vals[(int(v), int(u))] = -val
            return GraphVectorField(edge_values=vals, matrix_valued=False)

        if self.backend is None:
            self._init_noncommutative_layers(rebuild_backend=True)
        _assert(self.backend is not None, "backend missing in NC mode")
        k = int(self.backend.k)
        vals_m: Dict[Tuple[int, int], np.ndarray] = {}
        for u, v in edges:
            A = (rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))).astype(np.complex128)
            A = antihermitianize(A)
            nA = max(fro_norm(A), _EPS)
            A = (sc / nA) * A
            vals_m[(int(u), int(v))] = A.astype(np.complex128)
            vals_m[(int(v), int(u))] = (-A).astype(np.complex128)

        vals_m = self.backend.project_twisted_reality_edge(vals_m)
        return GraphVectorField(edge_values=vals_m, matrix_valued=True)

    def _make_flow_solver_and_monitor(self) -> Tuple[Any, FlowMonitor]:
        self._ensure_setup()
        _assert(self.calc is not None, "calc missing; call setup()")
        if not self.noncommutative:
            return GeodesicFlowSolver(self.calc), FlowMonitor(calc=self.calc, backend=None, num_nodes=64)
        if self.backend is None:
            self._init_noncommutative_layers(rebuild_backend=True)
        _assert(self.backend is not None, "backend missing in NC mode")
        return NCGeodesicFlowSolver(self.backend, num_nodes=64), FlowMonitor(calc=None, backend=self.backend, num_nodes=64)

    def simulate_flow(self, cfg: GeodesicFlowConfig) -> None:
        """Generate a flow trajectory and matching diagnostics."""
        cfg = cfg.normalized() if hasattr(cfg, "normalized") else cfg
        solver, monitor = self._make_flow_solver_and_monitor()
        X0 = self.build_initial_flow(scale=0.1)
        traj = solver.integrate(X0, cfg)

        diags: List[FlowDiagnostics] = []
        prev: Optional[GraphVectorField] = None
        dt = float(getattr(cfg, "dt", 1.0))
        for X in traj:
            diags.append(monitor.diagnostics(prev, X, dt))
            prev = X

        self.X_traj = traj
        self.flow_diags = diags
        self.ttn_traj = None

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def _normalize_hamiltonian_for_backend(self, H_cfg: HamiltonianBuildConfig) -> HamiltonianBuildConfig:
        if hasattr(H_cfg, "normalized"):
            H_cfg = H_cfg.normalized()
        if not self.noncommutative:
            return H_cfg

        # If caller used commutative defaults in NC mode, promote them to the NC
        # Hamiltonian modes expected by projection/holonomy.
        if str(H_cfg.onsite_mode).lower().strip() == "zfield":
            H_cfg.onsite_mode = "holographic_su2"
        if str(H_cfg.edge_mode).lower().strip() == "zz":
            H_cfg.edge_mode = "holonomy_su2"
        return H_cfg

    def _ensure_flow_for_evolution(self) -> None:
        if self.X_traj is not None and self.flow_diags is not None:
            return
        self.simulate_flow(GeodesicFlowConfig())

    def evolve(
        self,
        *,
        fiber: LocalFiberBuilder,
        evo_cfg: TTNEvolveConfig,
        H_cfg: Optional[HamiltonianBuildConfig] = None,
    ) -> None:
        """Evolve the current TTN state along the simulated flow trajectory."""
        _assert(self.state is not None, "init_state() first")
        self._ensure_flow_for_evolution()
        _assert(self.X_traj is not None and self.flow_diags is not None, "simulate_flow() first")
        _assert(self.calc is not None, "calc missing")

        if H_cfg is None:
            H_cfg = getattr(evo_cfg, "H_cfg", None)
        _assert(H_cfg is not None, "Atom.evolve: missing H_cfg")
        H_cfg = self._normalize_hamiltonian_for_backend(H_cfg)

        bucket = max(int(getattr(evo_cfg, "step_bucket_every", 1)), 1)
        self._sync_cache_knobs(bucket=bucket)

        builder = TreeMPOBuilder(
            self.calc,
            cfg=H_cfg,
            decomp=self.decomp,
            projection=self.projection,
            holonomy=self.holonomy,
            cache_bucket=bucket,
        )
        evolver = TTNTimeEvolver(self.calc)
        traj = evolver.integrate_with_flow(
            self.state,
            fiber=fiber,
            builder=builder,
            X_traj=self.X_traj,
            vib=self.vibration,
            flow_diags=self.flow_diags,
            cfg=evo_cfg,
            seed=int(self.seed),
        )
        self.ttn_traj = traj
        if traj:
            self.state = traj[-1].clone()

    # ------------------------------------------------------------------
    # Measurements / diagnostics
    # ------------------------------------------------------------------

    def latest_flow(self) -> Optional[GraphVectorField]:
        if self.X_traj:
            return self.X_traj[-1]
        return None

    def latest_flow_diag(self) -> Optional[FlowDiagnostics]:
        if self.flow_diags:
            return self.flow_diags[-1]
        return None

    def health_metrics(self) -> Dict[str, Any]:
        qn = _state_norm_squared(self.state)
        nonfinite = _count_nonfinite_state(self.state)
        diag = self.latest_flow_diag()
        flow_alarm = _finite_float(getattr(diag, "alarm_score", 0.0), 0.0) if diag is not None else 0.0
        flow_energy = _finite_float(getattr(diag, "flow_energy", 0.0), 0.0) if diag is not None else 0.0
        max_edge = _field_max_norm(self.latest_flow())
        stable = bool(
            self.state is not None
            and nonfinite == 0
            and math.isfinite(qn)
            and 0.01 <= qn <= 100.0
            and math.isfinite(flow_alarm)
            and math.isfinite(flow_energy)
            and math.isfinite(max_edge)
        )
        return {
            "kind": "Atom",
            "element": self.element,
            "Z": int(self.Z),
            "noncommutative": bool(self.noncommutative),
            "is_stable": stable,
            "has_state": self.state is not None,
            "nonfinite_tensor_count": int(nonfinite),
            "quantum_norm_squared": float(qn),
            "flow_alarm_score": float(flow_alarm),
            "flow_energy": float(flow_energy),
            "max_edge_norm": float(max_edge),
            "flow_steps": 0 if self.X_traj is None else max(0, len(self.X_traj) - 1),
            "evolution_snapshots": 0 if self.ttn_traj is None else len(self.ttn_traj),
            "backend_k": None if self.backend is None else int(self.backend.k),
        }

    def snapshot(self, *, include_state_summary: bool = True, include_trajectories: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": "Atom",
            "element": self.element,
            "Z": int(self.Z),
            "isotope_A": self.isotope_A,
            "noncommutative": bool(self.noncommutative),
            "fuzzy_l": int(self.fuzzy_l),
            "seed": int(self.seed),
            "runtime": _json_safe(self.runtime),
            "geometry": _json_safe(self.geometry),
            "has_tree": self.tree is not None,
            "has_state": self.state is not None,
            "has_vibration": self.vibration is not None,
            "backend": None if self.backend is None else {"fuzzy_l": int(self.backend.fuzzy_l), "k": int(self.backend.k)},
            "health": self.health_metrics(),
            "metadata": _json_safe(self.metadata),
            "ts": _now_iso(),
        }
        if include_state_summary and self.state is not None:
            try:
                ds = [int(self.state.phys_dims[l]) for l in self.state.tree.leaves]
                bonds = [int(v) for v in self.state.parent_bond_dims.values()]
                out["state_summary"] = {
                    "leaf_d_min": int(min(ds)) if ds else 0,
                    "leaf_d_max": int(max(ds)) if ds else 0,
                    "leaf_d_mean": float(np.mean(ds)) if ds else 0.0,
                    "bond_min": int(min(bonds)) if bonds else 0,
                    "bond_max": int(max(bonds)) if bonds else 0,
                    "norm_squared": float(_state_norm_squared(self.state)),
                }
            except Exception as exc:
                out["state_summary_error"] = repr(exc)
        if self.vibration is not None:
            out["vibration"] = _json_safe(self.vibration)
        if include_trajectories:
            out["flow_traj_len"] = 0 if self.X_traj is None else len(self.X_traj)
            out["flow_diags"] = _json_safe(self.flow_diags or [])
            out["ttn_traj_len"] = 0 if self.ttn_traj is None else len(self.ttn_traj)
        return out

    def report(self) -> str:
        """Human-readable state report for demos and smoke tests."""
        lines: List[str] = []
        lines.append(f"Atom(element={self.element}, Z={self.Z}, isotope_A={self.isotope_A})")
        lines.append("Backend: " + ("noncommutative/fuzzy (energy-gauge + holonomy)" if self.noncommutative else "commutative"))

        if self.noncommutative and self.backend is not None:
            lines.append(f"  fuzzy_l={self.backend.fuzzy_l}  k={self.backend.k}")
            lines.append(f"  projection={type(self.projection).__name__ if self.projection is not None else 'None'}")
            lines.append(f"  holonomy={type(self.holonomy).__name__ if self.holonomy is not None else 'None'}")

        if self.vibration is not None:
            lines.append("VibrationModel: attached")
            try:
                lines.append(f"  grid={self.vibration.meta.get('grid')}, n={int(self.vibration.frequencies.size)}")
                lines.append(f"  coupling_norm={_safe_norm(self.vibration.couplings):.6f}")
            except Exception:
                pass

        if self.state is not None:
            self.state.validate()
            ds = [int(self.state.phys_dims[l]) for l in self.state.tree.leaves]
            bd = [int(v) for v in self.state.parent_bond_dims.values()]
            lines.append("State: TTNState")
            lines.append(f"  leaf d: min={min(ds)} max={max(ds)} mean={float(np.mean(ds)):.2f}")
            lines.append(f"  bond: min={min(bd)} max={max(bd)} mean={float(np.mean(bd)):.2f}")
            lines.append(f"  norm^2={self.state.amplitude_norm_squared():.6f}")

        if self.X_traj is not None and self.flow_diags is not None and len(self.flow_diags) > 0:
            last = self.flow_diags[-1]
            lines.append(f"Flow: steps={len(self.X_traj) - 1} alarm={_finite_float(getattr(last, 'alarm_score', 0.0)):.6f}")
            try:
                lines.append(f"  last ||divX||2={float(np.linalg.norm(last.divX_scalar)):.6f}")
                lines.append(f"  flow_energy={_finite_float(getattr(last, 'flow_energy', 0.0)):.6e}")
            except Exception:
                pass
            if getattr(last, "kappa_su2_coeffs", None) is not None:
                mag = np.linalg.norm(np.asarray(last.kappa_su2_coeffs), axis=1)
                lines.append(f"  κ_su2 coeff mean|c|={float(np.mean(mag)):.6e}  max|c|={float(np.max(mag)):.6e}")
                lines.append(f"  κ_su2(0) coeffs [cx,cy,cz]={np.asarray(last.kappa_su2_coeffs)[0]}")

        if self.ttn_traj is not None and len(self.ttn_traj) > 0:
            last_state = self.ttn_traj[-1]
            ds = [int(last_state.phys_dims[l]) for l in last_state.tree.leaves]
            lines.append(f"Evolution snapshots={len(self.ttn_traj)}")
            lines.append(f"  last leaf d: min={min(ds)} max={max(ds)} mean={float(np.mean(ds)):.2f}")
            lines.append(f"  last norm^2={last_state.amplitude_norm_squared():.6f}")

        hm = self.health_metrics()
        lines.append(f"Health: stable={hm['is_stable']} nonfinite={hm['nonfinite_tensor_count']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence metadata
    # ------------------------------------------------------------------

    def save_metadata_json(self, path: Union[str, Path], *, include_trajectories: bool = False) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_safe(self.snapshot(include_trajectories=include_trajectories)), indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Demo
# =============================================================================


def _demo(noncommutative: bool = False) -> None:
    atom = Atom(element="H", Z=1, isotope_A=1, noncommutative=noncommutative, fuzzy_l=4, seed=7)
    atom.setup(tree_mode="balanced", arity=4)

    vib = VibrationModel.build(
        grid_kind="fractal",
        w_min=0.1,
        w_max=50.0,
        spectral_kind="ohmic",
        alpha=1.0,
        omega_c=20.0,
        directions_n=32,
        seed=42,
        fractal_levels=3,
        fractal_branching=4,
        fractal_exponent=2.2,
    )
    atom.attach_vibration(vib)

    d_uniform = 16
    if noncommutative and atom.backend is not None:
        d_uniform = min(d_uniform, int(atom.backend.k))

    fiber_cfg = LocalFiberConfig(
        d_uniform=d_uniform,
        d_min=min(8, d_uniform),
        d_max=max(8, min(32, d_uniform * 2)),
        adaptive_strength=1.0,
        vib_influence=0.25,
        seed=7,
    )
    fiber = LocalFiberBuilder(
        fiber_cfg,
        adinkra=AdinkraConstraint(seed=7),
        projection=atom.projection,
        include_pauli_fallback=True,
    )

    atom.init_state(fiber=fiber, bond_dim=4)

    print("==== Initial ====")
    print(atom.report())
    print()

    flow_cfg = GeodesicFlowConfig(dt=2e-2, steps=40, damping=0.03, diffusion=0.02, twisted_reality=True)
    atom.simulate_flow(flow_cfg)

    print("==== After flow ====")
    print(atom.report())
    print()

    H_cfg = HamiltonianBuildConfig(
        onsite_scale=0.5,
        onsite_mode=("holographic_su2" if noncommutative else "zfield"),
        vib_scale=0.02,
        vib_op="I",
        edge_scale=0.25,
        edge_mode=("holonomy_su2" if noncommutative else "zz"),
        hop_scale=1.0,
        extra_scale=0.0,
        extra_op="G",
    )

    evo_cfg = TTNEvolveConfig(
        dt=5e-3,
        steps=10,
        method="rk4_end_truncate",
        renormalize_every=1,
        step_bucket_every=1,
        apply_config=ApplyConfig(
            apply_truncate_rank=(12 if not noncommutative else 10),
            apply_truncate_tol=None,
            canonicalize_every=1,
            apply_grouping="lca_routed",
        ),
        post_step_truncate_rank=(12 if not noncommutative else 10),
        post_step_truncate_tol=None,
    )

    atom.evolve(fiber=fiber, evo_cfg=evo_cfg, H_cfg=H_cfg)

    print("==== After evolution ====")
    print(atom.report())
    print()

    if atom.ttn_traj:
        stride = max(1, len(atom.ttn_traj) // 5)
        norms = [st.amplitude_norm_squared() for st in atom.ttn_traj[::stride]]
        print("Norm^2 samples:", ["{:.6f}".format(x) for x in norms])


if __name__ == "__main__":
    print("\n######## COMMUTATIVE DEMO ########\n")
    _demo(noncommutative=False)

    print("\n######## NONCOMMUTATIVE / FUZZY DEMO ########\n")
    _demo(noncommutative=True)


__all__ = [
    "Atom",
    "AtomRuntimeConfig",
    "AtomCacheConfig",
]
