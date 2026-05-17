# "C:\\Users\\ali_z\\ANU AI\\Akkurat\\cognitive_model_3\\ncp.py"
# ============================================================================
# Project Chimera / Akkurat - Production Neural Substrate v3.0
#
# ROLE
# ----
# Shared bounded recurrent substrate for cognitive lobes.
#
# DESIGN PRINCIPLES
# -----------------
# - Neural substrate only: no orchestration, no tool routing, no symbolic planning.
# - Supports TensorTrain recurrent weights via tn.py, with strict or graceful dense
#   fallback depending on configuration.
# - Stable continuous-time hidden dynamics with adaptive time constants.
# - Explicit state serialization, full checkpoint save/load, health inspection,
#   and maintenance hooks.
# - Backward compatible with existing lobe wrappers that call:
#       TensorizedNCP(...).step(feature_vector, return_state=True)
# ============================================================================

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# TensorTrain import
# -----------------------------------------------------------------------------
TENSORTRAIN_AVAILABLE = False
TT_IMPORT_ERROR: Optional[Exception] = None

try:
    try:
        from .tn import TensorTrain, TTConfig, get_factors, tensor_train_svd  # type: ignore
        TENSORTRAIN_AVAILABLE = True
    except Exception:
        try:
            from tn import TensorTrain, TTConfig, get_factors, tensor_train_svd  # type: ignore
            TENSORTRAIN_AVAILABLE = True
        except Exception:
            # Common project layout: Akkurat/tn.py and Akkurat/cognitive_model_3/ncp.py
            root = Path(__file__).resolve().parents[1]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from tn import TensorTrain, TTConfig, get_factors, tensor_train_svd  # type: ignore
            TENSORTRAIN_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only when tn.py unavailable
    TT_IMPORT_ERROR = e

    def get_factors(n: int) -> Tuple[int, int]:
        n = int(n)
        if n <= 0:
            return (1, 1)
        a = int(np.floor(np.sqrt(n)))
        while a > 1 and n % a != 0:
            a -= 1
        return (max(1, a), max(1, n // max(1, a)))

    class TensorTrain:  # type: ignore[no-redef]
        pass

    @dataclass(frozen=True)
    class TTConfig:  # type: ignore[no-redef]
        dtype: np.dtype = np.float32
        device: str = "cpu"
        check_finite: bool = True

    def tensor_train_svd(*args: Any, **kwargs: Any) -> Optional[Any]:  # type: ignore[no-redef]
        return None


def _np_from_any_array(x: Any) -> np.ndarray:
    """Return a NumPy array from NumPy/CuPy-like arrays."""
    if hasattr(x, "__cuda_array_interface__"):
        try:
            import cupy as cp  # type: ignore
            return cp.asnumpy(x)  # type: ignore[no-any-return]
        except Exception:
            pass
    return np.asarray(x)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _coerce_float64_vector(x: Any, expected_dim: Optional[int] = None, name: str = "input") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape, got shape {arr.shape}")
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    if expected_dim is not None and arr.size != int(expected_dim):
        raise ValueError(f"{name} size mismatch: expected {expected_dim}, got {arr.size}")
    return arr


def _safe_clip_inplace(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    np.clip(arr, float(lo), float(hi), out=arr)
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _sanitize_vector(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    mask = ~np.isfinite(out)
    if np.any(mask):
        out[mask] = float(fill_value)
    return out


def _count_nonfinite(arr: np.ndarray) -> int:
    a = np.asarray(arr)
    return int(np.size(a) - np.count_nonzero(np.isfinite(a)))


def _saturation_fraction(arr: np.ndarray, threshold: float = 0.98) -> float:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a) >= float(threshold)))


def _safe_spectral_norm(mat: np.ndarray) -> Optional[float]:
    try:
        s = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
        if s.size == 0:
            return 0.0
        return float(s[0])
    except Exception:
        return None


def _safe_norm(arr: np.ndarray) -> float:
    try:
        n = float(np.linalg.norm(np.asarray(arr, dtype=np.float64)))
        return n if np.isfinite(n) else 0.0
    except Exception:
        return 0.0


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# -----------------------------------------------------------------------------
# Legacy neuron object
# -----------------------------------------------------------------------------
class LTCNeuron:
    """
    Lightweight compatibility mirror for a neuron-like state container.

    TensorizedNCP._state is authoritative. These objects are retained for legacy
    inspection and isolated experiments.
    """

    def __init__(self, neuron_id: int, membrane_capacitance: float = 1.0, bias: float = 0.0):
        self.neuron_id = int(neuron_id)
        self.state = 0.0
        self.membrane_capacitance = float(membrane_capacitance)
        self.bias = float(bias)

    def __repr__(self) -> str:
        return (
            f"LTCNeuron(id={self.neuron_id}, state={self.state:.4f}, "
            f"C={self.membrane_capacitance:.4f}, bias={self.bias:.4f})"
        )


# -----------------------------------------------------------------------------
# Optional legacy sparse NCP
# -----------------------------------------------------------------------------
class NeuralCircuitPolicy:
    """
    Legacy sparse NCP retained for backward compatibility and comparison tests.

    Not used by the Akkurat production lobe stack.
    """

    def __init__(
        self,
        policy_id: str,
        num_neurons: int,
        num_inputs: int,
        num_outputs: int,
        sparsity: float = 0.8,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
    ):
        self.policy_id = str(policy_id)
        self.num_neurons = int(num_neurons)
        self.num_inputs = int(num_inputs)
        self.num_outputs = int(num_outputs)
        self.verbose = bool(verbose)
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

        if self.num_inputs <= 0 or self.num_outputs <= 0 or self.num_neurons <= 0:
            raise ValueError("num_neurons, num_inputs, and num_outputs must be positive")
        if self.num_inputs + self.num_outputs > self.num_neurons:
            raise ValueError("num_inputs + num_outputs cannot exceed num_neurons")

        self.neurons = [LTCNeuron(neuron_id=i, bias=0.0) for i in range(self.num_neurons)]
        self.input_neuron_indices = list(range(self.num_inputs))
        self.output_neuron_indices = list(range(self.num_neurons - self.num_outputs, self.num_neurons))
        self.synapses = self._create_sparse_wiring(float(sparsity))

    def _create_sparse_wiring(self, sparsity: float) -> List[Tuple[int, int, float]]:
        sparsity = float(np.clip(sparsity, 0.0, 0.999))
        synapses: List[Tuple[int, int, float]] = []
        fan_in_est = max(1, int(round((1.0 - sparsity) * (self.num_neurons - 1))))
        scale = 1.0 / np.sqrt(fan_in_est)

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    continue
                if self.rng.random() < sparsity:
                    continue
                weight = float(self.rng.normal(0.0, 1.0) * scale)
                synapses.append((i, j, weight))

        if self.verbose:
            print(
                f"[LegacySparseNCP:{self.policy_id}] neurons={self.num_neurons} "
                f"synapses={len(synapses)}"
            )
        return synapses

    def reset_state(self, value: float = 0.0) -> None:
        v = float(value)
        for n in self.neurons:
            n.state = v

    def get_state(self) -> np.ndarray:
        return np.array([n.state for n in self.neurons], dtype=np.float64)

    def set_state(self, new_state: Union[np.ndarray, Sequence[float]]) -> None:
        arr = _coerce_float64_vector(new_state, expected_dim=self.num_neurons, name="new_state")
        for i, n in enumerate(self.neurons):
            n.state = float(arr[i])

    def step(self, inputs: Any, dt: float = 1.0) -> np.ndarray:
        x = _coerce_float64_vector(inputs, expected_dim=self.num_inputs, name="inputs")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        input_currents = np.zeros((self.num_neurons,), dtype=np.float64)
        for i, val in enumerate(x):
            input_currents[self.input_neuron_indices[i]] += float(val)

        src_states = np.array([n.state for n in self.neurons], dtype=np.float64)
        src_acts = _sigmoid(src_states)

        for s_idx, t_idx, w in self.synapses:
            input_currents[t_idx] += float(w) * src_acts[s_idx]

        _safe_clip_inplace(input_currents, -10.0, 10.0)

        for i, n in enumerate(self.neurons):
            C = max(1e-8, float(n.membrane_capacitance))
            tau = 10.0
            leakage = C / tau
            d_state = (-leakage * n.state + n.bias + input_currents[i]) / C
            n.state = float(np.tanh(n.state + dt * d_state))

        return np.array([self.neurons[i].state for i in self.output_neuron_indices], dtype=np.float64)

    def snapshot(self) -> Dict[str, Any]:
        state = self.get_state()
        return {
            "policy_id": self.policy_id,
            "kind": "legacy_sparse_ncp",
            "num_neurons": self.num_neurons,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "synapse_count": len(self.synapses),
            "hidden_state_norm": float(np.linalg.norm(state)),
            "hidden_state_mean": float(np.mean(state)) if state.size else 0.0,
            "hidden_state_std": float(np.std(state)) if state.size else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralCircuitPolicy(id='{self.policy_id}', neurons={self.num_neurons}, "
            f"inputs={self.num_inputs}, outputs={self.num_outputs}, synapses={len(self.synapses)})"
        )


# -----------------------------------------------------------------------------
# Production TensorizedNCP config
# -----------------------------------------------------------------------------
@dataclass
class TensorizedNCPConfig:
    policy_id: str
    num_inputs: int
    num_hidden: int
    num_outputs: int

    bond_dim: Union[int, Sequence[int]] = 4
    tau_min: float = 1.0
    tau_max: float = 50.0
    tt_modes: Optional[Sequence[int]] = None
    output_activation: str = "linear"  # linear | tanh | sigmoid
    output_scale: float = 1.0
    tau_jitter_frac: float = 0.0
    safety_clamp: Optional[float] = None
    tt_strict: bool = True

    leaky_lambda: float = 0.1
    num_freq_gates: int = 0
    num_mvm_outputs: int = 0

    verbose: bool = False
    rng_seed: Optional[int] = None
    strict_runtime_checks: bool = True
    force_dense: bool = False
    check_finite: bool = True
    mirror_neurons_each_step: bool = False

    # TT initialization: dense stable operator -> TT-SVD gives a meaningful recurrent backbone.
    tt_init_from_dense: bool = True
    tt_energy_tol: Optional[float] = 0.999
    tt_device: str = "cpu"

    # Runtime guardrails.
    input_clip: float = 10.0
    current_clip: float = 5.0
    output_clip: Optional[float] = None
    max_dt: float = 100.0


# -----------------------------------------------------------------------------
# Production TensorizedNCP
# -----------------------------------------------------------------------------
class TensorizedNCP:
    """
    Production-grade Tensorized Neural Circuit Policy.

    Intended usage:
        ncp = TensorizedNCP(...)
        y, aux = ncp.step(feature_vector, return_state=True)

    Core guarantees:
    - Numeric vector input only.
    - Stable bounded hidden-state dynamics.
    - TT recurrent operator when available/configured.
    - Dense fallback when configured and allowed.
    - State serialization, full checkpointing, health metrics, and maintenance hooks.
    """

    CHECKPOINT_VERSION = 3

    def __init__(
        self,
        policy_id: str,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        bond_dim: Union[int, Iterable[int]] = 4,
        tau_min: float = 1.0,
        tau_max: float = 50.0,
        tt_modes: Optional[Iterable[int]] = None,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = True,
        output_activation: str = "linear",
        output_scale: float = 1.0,
        tau_jitter_frac: float = 0.0,
        safety_clamp: Optional[float] = None,
        tt_strict: bool = True,
        leaky_lambda: float = 0.1,
        num_freq_gates: int = 0,
        num_mvm_outputs: int = 0,
        strict_runtime_checks: bool = True,
        force_dense: Optional[bool] = None,
        check_finite: bool = True,
        mirror_neurons_each_step: bool = False,
        tt_init_from_dense: bool = True,
        tt_energy_tol: Optional[float] = 0.999,
        tt_device: str = "cpu",
        input_clip: float = 10.0,
        current_clip: float = 5.0,
        output_clip: Optional[float] = None,
        max_dt: float = 100.0,
    ):
        cfg_bond: Union[int, List[int]]
        if isinstance(bond_dim, Iterable) and not isinstance(bond_dim, (str, bytes)):
            cfg_bond = [int(max(1, x)) for x in list(bond_dim)]
        else:
            cfg_bond = int(max(1, int(bond_dim)))

        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()

        self.config = TensorizedNCPConfig(
            policy_id=str(policy_id),
            num_inputs=int(num_inputs),
            num_hidden=int(num_hidden),
            num_outputs=int(num_outputs),
            bond_dim=cfg_bond,
            tau_min=float(tau_min),
            tau_max=float(tau_max),
            tt_modes=list(tt_modes) if tt_modes is not None else None,
            output_activation=str(output_activation).lower().strip(),
            output_scale=float(output_scale),
            tau_jitter_frac=float(tau_jitter_frac),
            safety_clamp=None if safety_clamp is None else float(safety_clamp),
            tt_strict=bool(tt_strict),
            leaky_lambda=float(leaky_lambda),
            num_freq_gates=int(num_freq_gates),
            num_mvm_outputs=int(num_mvm_outputs),
            verbose=bool(verbose),
            rng_seed=None,
            strict_runtime_checks=bool(strict_runtime_checks),
            force_dense=bool(force_dense) if force_dense is not None else False,
            check_finite=bool(check_finite),
            mirror_neurons_each_step=bool(mirror_neurons_each_step),
            tt_init_from_dense=bool(tt_init_from_dense),
            tt_energy_tol=tt_energy_tol,
            tt_device=str(tt_device).lower().strip(),
            input_clip=float(input_clip),
            current_clip=float(current_clip),
            output_clip=None if output_clip is None else float(output_clip),
            max_dt=float(max_dt),
        )

        self.policy_id = self.config.policy_id
        self.num_inputs = self.config.num_inputs
        self.num_hidden = self.config.num_hidden
        self.num_outputs = self.config.num_outputs
        self.verbose = self.config.verbose
        self.output_activation = self.config.output_activation
        if self.output_activation not in {"linear", "tanh", "sigmoid"}:
            self.output_activation = "linear"
        self.output_scale = float(self.config.output_scale)
        self.safety_clamp = self.config.safety_clamp
        self.tt_strict = bool(self.config.tt_strict)
        self.strict_runtime_checks = bool(self.config.strict_runtime_checks)
        self.leaky_lambda = float(np.clip(self.config.leaky_lambda, 0.0, 1.0))
        self.num_freq_gates = int(max(0, self.config.num_freq_gates))
        self.num_mvm_outputs = int(max(0, self.config.num_mvm_outputs))
        self.check_finite = bool(self.config.check_finite)
        self.mirror_neurons_each_step = bool(self.config.mirror_neurons_each_step)

        self._step_count = 0
        self._created_at = _utc_ts()
        self._last_maintenance_at: Optional[str] = None

        if self.num_inputs <= 0 or self.num_hidden <= 0 or self.num_outputs <= 0:
            raise ValueError("num_inputs, num_hidden, and num_outputs must all be > 0")
        if self.config.tau_min <= 0 or self.config.tau_max <= 0:
            raise ValueError("tau_min and tau_max must be > 0")
        if self.config.tau_min >= self.config.tau_max:
            raise ValueError("tau_min must be < tau_max")
        if self.config.input_clip <= 0 or self.config.current_clip <= 0:
            raise ValueError("input_clip and current_clip must be > 0")

        # Bond rank normalization.
        if isinstance(self.config.bond_dim, list):
            self.bond_dims_list = [int(max(1, x)) for x in self.config.bond_dim]
            self.bond_dim = int(self.bond_dims_list[0]) if self.bond_dims_list else 4
        else:
            self.bond_dim = int(max(1, self.config.bond_dim))
            self.bond_dims_list = [self.bond_dim]

        # Canonical hidden state and vectorized LTC parameters.
        self._state = np.zeros((self.num_hidden,), dtype=np.float64)
        self._membrane_capacitance = np.ones((self.num_hidden,), dtype=np.float64)
        self._bias = self.rng.normal(0.0, 0.01, size=(self.num_hidden,)).astype(np.float64)

        # Hidden neurons retained for compatibility/debug only.
        self.hidden_neurons = [
            LTCNeuron(neuron_id=i, membrane_capacitance=float(self._membrane_capacitance[i]), bias=float(self._bias[i]))
            for i in range(self.num_hidden)
        ]

        # Dense input/output and auxiliary heads.
        self.input_weights = self.rng.normal(0.0, 0.1, size=(self.num_hidden, self.num_inputs)).astype(np.float64)
        self.output_weights = self.rng.normal(0.0, 0.1, size=(self.num_outputs, self.num_hidden)).astype(np.float64)

        self.freq_gate_weights: Optional[np.ndarray] = None
        if self.num_freq_gates > 0:
            self.freq_gate_weights = self.rng.normal(0.0, 0.1, size=(self.num_freq_gates, self.num_hidden)).astype(np.float64)

        self.mvm_weights: Optional[np.ndarray] = None
        if self.num_mvm_outputs > 0:
            self.mvm_weights = self.rng.normal(0.0, 0.1, size=(self.num_mvm_outputs, self.num_hidden)).astype(np.float64)

        # Adaptive tau parameters.
        self.tc_W_h = self.rng.normal(0.0, 0.05, size=(self.num_hidden, self.num_hidden)).astype(np.float64)
        self.tc_W_x = self.rng.normal(0.0, 0.05, size=(self.num_hidden, self.num_inputs)).astype(np.float64)
        self.tc_b = self.rng.normal(0.0, 0.2, size=(self.num_hidden,)).astype(np.float64)

        base_tau_min = float(self.config.tau_min)
        base_tau_max = float(self.config.tau_max)
        jitter = float(max(0.0, self.config.tau_jitter_frac))
        if jitter > 0.0:
            eps = self.rng.uniform(-jitter, jitter, size=(self.num_hidden,)).astype(np.float64)
            self._tau_min_arr = base_tau_min * (1.0 + eps)
            self._tau_max_arr = base_tau_max * (1.0 + eps)
            bad = self._tau_min_arr >= self._tau_max_arr
            self._tau_min_arr[bad] = base_tau_min
            self._tau_max_arr[bad] = base_tau_max
            self._tau_min_arr = np.maximum(self._tau_min_arr, 1e-6)
            self._tau_max_arr = np.maximum(self._tau_max_arr, self._tau_min_arr + 1e-6)
        else:
            self._tau_min_arr = np.full((self.num_hidden,), base_tau_min, dtype=np.float64)
            self._tau_max_arr = np.full((self.num_hidden,), base_tau_max, dtype=np.float64)

        # TT shapes.
        self._tt_in_shape: Tuple[int, ...]
        self._tt_out_shape: Tuple[int, ...]
        self._infer_tt_shapes()

        # Recurrent operator.
        env_force_dense = str(os.getenv("NCP_FORCE_DENSE", "")).strip().lower() in ("1", "true", "yes", "on")
        self._force_dense_effective = bool(self.config.force_dense) or bool(env_force_dense)
        self._use_tt = False
        self.recurrent_weights: Optional[TensorTrain] = None
        self.recurrent_dense: Optional[np.ndarray] = None
        self._recurrent_param_count: Optional[int] = None
        self._tt_init_error: Optional[str] = None
        self._tt_bond_dims_actual: List[int] = []

        self._init_recurrent_operator()

        # Runtime caches for diagnostics.
        self._last_input = np.zeros((self.num_inputs,), dtype=np.float64)
        self._last_input_signal = np.zeros((self.num_hidden,), dtype=np.float64)
        self._last_recurrent_signal = np.zeros((self.num_hidden,), dtype=np.float64)
        self._last_total_currents = np.zeros((self.num_hidden,), dtype=np.float64)
        self._last_dynamic_tau = np.full((self.num_hidden,), self.config.tau_min, dtype=np.float64)
        self._last_output = np.zeros((self.num_outputs,), dtype=np.float64)
        self._last_aux_outputs: Dict[str, np.ndarray] = {}
        self._last_error: Optional[str] = None

        if self.verbose:
            mode = "TT" if self._use_tt else "Dense"
            print(
                f"[TensorizedNCP:{self.policy_id}] mode={mode} "
                f"hidden={self.num_hidden} inputs={self.num_inputs} outputs={self.num_outputs} "
                f"recurrent_params={self.parameter_count()} total_params={self.total_parameter_count()}"
            )

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Union[TensorizedNCPConfig, Dict[str, Any]], *, rng: Optional[np.random.Generator] = None) -> "TensorizedNCP":
        if isinstance(cfg, dict):
            cfg = TensorizedNCPConfig(**cfg)
        return cls(
            policy_id=cfg.policy_id,
            num_inputs=cfg.num_inputs,
            num_hidden=cfg.num_hidden,
            num_outputs=cfg.num_outputs,
            bond_dim=cfg.bond_dim,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            tt_modes=cfg.tt_modes,
            rng=rng,
            verbose=cfg.verbose,
            output_activation=cfg.output_activation,
            output_scale=cfg.output_scale,
            tau_jitter_frac=cfg.tau_jitter_frac,
            safety_clamp=cfg.safety_clamp,
            tt_strict=cfg.tt_strict,
            leaky_lambda=cfg.leaky_lambda,
            num_freq_gates=cfg.num_freq_gates,
            num_mvm_outputs=cfg.num_mvm_outputs,
            strict_runtime_checks=cfg.strict_runtime_checks,
            force_dense=cfg.force_dense,
            check_finite=cfg.check_finite,
            mirror_neurons_each_step=cfg.mirror_neurons_each_step,
            tt_init_from_dense=cfg.tt_init_from_dense,
            tt_energy_tol=cfg.tt_energy_tol,
            tt_device=cfg.tt_device,
            input_clip=cfg.input_clip,
            current_clip=cfg.current_clip,
            output_clip=cfg.output_clip,
            max_dt=cfg.max_dt,
        )

    def config_dict(self) -> Dict[str, Any]:
        d = asdict(self.config)
        # Normalize sequences for JSON stability.
        if isinstance(d.get("bond_dim"), tuple):
            d["bond_dim"] = list(d["bond_dim"])
        if isinstance(d.get("tt_modes"), tuple):
            d["tt_modes"] = list(d["tt_modes"])
        return d

    def _infer_tt_shapes(self) -> None:
        if self.config.tt_modes is not None:
            modes = tuple(int(m) for m in self.config.tt_modes if int(m) > 0)
            prod = int(np.prod(modes)) if modes else 0
            if prod != self.num_hidden:
                if self.tt_strict:
                    raise ValueError(
                        f"[{self.policy_id}] tt_modes product ({prod}) must equal num_hidden ({self.num_hidden})"
                    )
                if self.verbose:
                    print(
                        f"[{self.policy_id}] WARNING: tt_modes product ({prod}) != num_hidden ({self.num_hidden}); "
                        "falling back to inferred factorization."
                    )
                f1, f2 = get_factors(self.num_hidden)
                self._tt_in_shape = (int(f1), int(f2))
                self._tt_out_shape = (int(f1), int(f2))
            else:
                self._tt_in_shape = modes
                self._tt_out_shape = modes
        else:
            f1, f2 = get_factors(self.num_hidden)
            self._tt_in_shape = (int(f1), int(f2))
            self._tt_out_shape = (int(f1), int(f2))

    def _resolve_tt_bond_dims(self, num_cores: int) -> List[int]:
        if num_cores <= 1:
            return []
        needed = num_cores - 1
        if isinstance(self.config.bond_dim, list):
            if len(self.bond_dims_list) != needed:
                msg = f"[{self.policy_id}] bond_dims length mismatch: expected {needed}, got {len(self.bond_dims_list)}"
                if self.tt_strict:
                    raise ValueError(msg)
                if self.verbose:
                    print(f"{msg}. Using repeated first bond rank instead.")
                return [self.bond_dim] * needed
            return [int(max(1, x)) for x in self.bond_dims_list]
        return [self.bond_dim] * needed

    @staticmethod
    def _stable_dense_init(h: int, rng: np.random.Generator, *, gamma: float = 0.95) -> np.ndarray:
        """Orthogonal-ish recurrent initialization with spectral control."""
        h = int(h)
        if h <= 0:
            return np.zeros((0, 0), dtype=np.float64)
        A = rng.normal(0.0, 1.0, size=(h, h))
        q, r = np.linalg.qr(A)
        d = np.sign(np.diag(r))
        d[d == 0] = 1.0
        q = q * d
        W = (q * 0.9).astype(np.float64)
        smax = _safe_spectral_norm(W)
        if smax is not None and smax > gamma and smax > 0:
            W *= gamma / smax
        return W

    def _init_recurrent_operator(self) -> None:
        if self._force_dense_effective:
            self._use_tt = False
            self.recurrent_dense = self._stable_dense_init(self.num_hidden, self.rng)
            self.recurrent_weights = None
            self._recurrent_param_count = int(self.recurrent_dense.size)
            self._tt_bond_dims_actual = []
            return

        if not TENSORTRAIN_AVAILABLE:
            msg = f"[{self.policy_id}] TensorTrain unavailable"
            if TT_IMPORT_ERROR is not None:
                msg += f": {TT_IMPORT_ERROR}"
            self._tt_init_error = msg
            if self.tt_strict:
                raise RuntimeError(msg)
            self._use_tt = False
            self.recurrent_dense = self._stable_dense_init(self.num_hidden, self.rng)
            self.recurrent_weights = None
            self._recurrent_param_count = int(self.recurrent_dense.size)
            self._tt_bond_dims_actual = []
            return

        num_cores = len(self._tt_in_shape)
        bond_dims = self._resolve_tt_bond_dims(num_cores)
        max_rank = max(bond_dims) if bond_dims else self.bond_dim

        try:
            if bool(self.config.tt_init_from_dense):
                dense_init = self._stable_dense_init(self.num_hidden, self.rng)
                tt = tensor_train_svd(
                    dense_init,
                    output_dims=list(self._tt_out_shape),
                    input_dims=list(self._tt_in_shape),
                    max_bond_dim=int(max_rank),
                    dtype=np.float32,
                    energy_tol=self.config.tt_energy_tol,
                    device=self.config.tt_device,
                    check_finite=self.check_finite,
                )
                if tt is None:
                    raise RuntimeError("tensor_train_svd returned None")
                self.recurrent_weights = tt
            else:
                self.recurrent_weights = TensorTrain(
                    output_dims=list(self._tt_out_shape),
                    input_dims=list(self._tt_in_shape),
                    bond_dims=bond_dims,
                    rng=self.rng,
                    config=TTConfig(dtype=np.float32, device=self.config.tt_device, check_finite=self.check_finite),
                    init_scale=1e-2,
                )

            self._use_tt = True
            self.recurrent_dense = None
            self._tt_bond_dims_actual = list(self.recurrent_weights.bond_ranks()) if hasattr(self.recurrent_weights, "bond_ranks") else list(bond_dims)
            self._recurrent_param_count = int(self.recurrent_weights.parameter_count()) if hasattr(self.recurrent_weights, "parameter_count") else None

        except Exception as e:
            msg = f"[{self.policy_id}] TensorTrain recurrent init failed: {e}"
            self._tt_init_error = msg
            if self.tt_strict:
                raise RuntimeError(msg) from e
            if self.verbose:
                print(f"{msg}. Falling back to dense recurrent operator.")
            self._use_tt = False
            self.recurrent_weights = None
            self.recurrent_dense = self._stable_dense_init(self.num_hidden, self.rng)
            self._recurrent_param_count = int(self.recurrent_dense.size)
            self._tt_bond_dims_actual = []

    # -------------------------------------------------------------------------
    # Core math
    # -------------------------------------------------------------------------
    def _recurrent_matvec(self, h_prev: np.ndarray) -> np.ndarray:
        h_prev = _coerce_float64_vector(h_prev, expected_dim=self.num_hidden, name="h_prev")

        if not self._use_tt:
            if self.recurrent_dense is None:
                raise RuntimeError(f"[{self.policy_id}] recurrent_dense is not initialized")
            out = self.recurrent_dense @ h_prev
            return _sanitize_vector(out)

        if self.recurrent_weights is None:
            raise RuntimeError(f"[{self.policy_id}] recurrent_weights is not initialized")

        try:
            if hasattr(self.recurrent_weights, "apply"):
                out = self.recurrent_weights.apply(h_prev.astype(np.float32, copy=False))
            else:
                out = self.recurrent_weights.contract_with_vector(h_prev.astype(np.float32, copy=False))
        except Exception as e:
            raise RuntimeError(f"[{self.policy_id}] TensorTrain contraction failed: {e}") from e

        out = np.asarray(_np_from_any_array(out), dtype=np.float64).reshape(-1)
        if out.size != self.num_hidden:
            raise ValueError(f"[{self.policy_id}] TT matvec shape mismatch: expected {self.num_hidden}, got {out.size}")
        return _sanitize_vector(out)

    def _dynamic_tau(self, x: np.ndarray) -> np.ndarray:
        gate_pre = self.tc_W_h @ self._state + self.tc_W_x @ x + self.tc_b
        gate = _sigmoid(gate_pre)
        tau = self._tau_min_arr + gate * (self._tau_max_arr - self._tau_min_arr)
        return _sanitize_vector(np.maximum(tau, 1e-6))

    def _vectorized_ltc_update(self, total_currents: np.ndarray, dynamic_tau: np.ndarray, dt: float) -> None:
        total_currents = _coerce_float64_vector(total_currents, expected_dim=self.num_hidden, name="total_currents")
        dynamic_tau = _coerce_float64_vector(dynamic_tau, expected_dim=self.num_hidden, name="dynamic_tau")

        C = np.maximum(_sanitize_vector(self._membrane_capacitance, 1.0), 1e-8)
        bias = _sanitize_vector(self._bias, 0.0)

        tau = np.clip(dynamic_tau, 1e-6, np.inf)
        leakage = C / tau
        d_state = (-leakage * self._state + bias + total_currents) / C

        if self.safety_clamp is not None and self.safety_clamp > 0:
            _safe_clip_inplace(d_state, -float(self.safety_clamp), float(self.safety_clamp))

        potential_new_state = np.tanh(self._state + float(dt) * d_state)
        self._state = (1.0 - self.leaky_lambda) * self._state + self.leaky_lambda * potential_new_state
        self._state = _sanitize_vector(self._state)

        if self.mirror_neurons_each_step:
            self.sync_neuron_mirror_from_vectors()

    def _activate_output(self, out: np.ndarray) -> np.ndarray:
        y = np.asarray(out, dtype=np.float64)
        if self.output_activation == "tanh":
            y = np.tanh(y)
        elif self.output_activation == "sigmoid":
            y = _sigmoid(y)
        if self.output_scale != 1.0:
            y = y * self.output_scale
        if self.config.output_clip is not None and self.config.output_clip > 0:
            _safe_clip_inplace(y, -float(self.config.output_clip), float(self.config.output_clip))
        return _sanitize_vector(y)

    # -------------------------------------------------------------------------
    # Public state API
    # -------------------------------------------------------------------------
    def sync_neuron_mirror_from_vectors(self) -> None:
        for i, n in enumerate(self.hidden_neurons):
            n.state = float(self._state[i])
            n.membrane_capacitance = float(self._membrane_capacitance[i])
            n.bias = float(self._bias[i])

    def reset_state(self, value: float = 0.0) -> None:
        v = float(value)
        self._state.fill(v)
        if self.mirror_neurons_each_step:
            self.sync_neuron_mirror_from_vectors()
        else:
            for n in self.hidden_neurons:
                n.state = v

        self._step_count = 0
        self._last_input.fill(0.0)
        self._last_input_signal.fill(0.0)
        self._last_recurrent_signal.fill(0.0)
        self._last_total_currents.fill(0.0)
        self._last_dynamic_tau.fill(self.config.tau_min)
        self._last_output.fill(0.0)
        self._last_aux_outputs = {}
        self._last_error = None

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, new_state: Union[np.ndarray, Sequence[float]]) -> None:
        arr = _coerce_float64_vector(new_state, expected_dim=self.num_hidden, name="new_state")
        self._state = np.clip(arr, -1.0, 1.0).astype(np.float64, copy=True)
        if self.mirror_neurons_each_step:
            self.sync_neuron_mirror_from_vectors()
        else:
            for i, n in enumerate(self.hidden_neurons):
                n.state = float(self._state[i])

    def serialize_state(self) -> Dict[str, Any]:
        """Serialize dynamic runtime state only, not model weights."""
        return {
            "schema": "TensorizedNCP.runtime_state.v1",
            "policy_id": self.policy_id,
            "num_inputs": self.num_inputs,
            "num_hidden": self.num_hidden,
            "num_outputs": self.num_outputs,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "hidden_state": self._state.astype(np.float64).tolist(),
            "step_count": int(self._step_count),
            "tau_min": float(self.config.tau_min),
            "tau_max": float(self.config.tau_max),
            "leaky_lambda": float(self.leaky_lambda),
            "output_activation": str(self.output_activation),
            "output_scale": float(self.output_scale),
            "num_freq_gates": int(self.num_freq_gates),
            "num_mvm_outputs": int(self.num_mvm_outputs),
            "tt_in_shape": list(self._tt_in_shape),
            "tt_out_shape": list(self._tt_out_shape),
            "tt_bond_dims": list(self._tt_bond_dims_actual),
            "created_at": self._created_at,
            "serialized_at": _utc_ts(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        nh = int(payload.get("num_hidden", self.num_hidden))
        if nh != self.num_hidden:
            raise ValueError(f"[{self.policy_id}] load_state hidden dim mismatch: expected {self.num_hidden}, got {nh}")
        vec = payload.get("hidden_state", None)
        if vec is None:
            raise ValueError(f"[{self.policy_id}] load_state missing 'hidden_state'")
        self.set_state(vec)
        self._step_count = int(payload.get("step_count", self._step_count))

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def parameter_count(self) -> int:
        return int(self._recurrent_param_count) if self._recurrent_param_count is not None else -1

    def total_parameter_count(self) -> int:
        total = 0
        total += int(self._recurrent_param_count) if self._recurrent_param_count is not None and self._recurrent_param_count >= 0 else self.num_hidden * self.num_hidden
        total += int(self.input_weights.size)
        total += int(self.output_weights.size)
        total += int(self.tc_W_h.size + self.tc_W_x.size + self.tc_b.size)
        total += int(self._membrane_capacitance.size + self._bias.size)
        if self.freq_gate_weights is not None:
            total += int(self.freq_gate_weights.size)
        if self.mvm_weights is not None:
            total += int(self.mvm_weights.size)
        return int(total)

    def auxiliary_head_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {}
        if self.freq_gate_weights is not None:
            manifest["freq_gates"] = {"size": int(self.num_freq_gates), "activation": "sigmoid"}
        if self.mvm_weights is not None:
            manifest["mvm_predictions"] = {"size": int(self.num_mvm_outputs), "activation": "linear"}
        return manifest

    def snapshot(self) -> Dict[str, Any]:
        state = self._state
        tau = self._last_dynamic_tau
        return {
            "policy_id": self.policy_id,
            "kind": "tensorized_ncp",
            "num_inputs": self.num_inputs,
            "num_hidden": self.num_hidden,
            "num_outputs": self.num_outputs,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "tt_available": bool(TENSORTRAIN_AVAILABLE),
            "tt_in_shape": list(self._tt_in_shape),
            "tt_out_shape": list(self._tt_out_shape),
            "tt_bond_dims": list(self._tt_bond_dims_actual),
            "tt_init_error": self._tt_init_error,
            "output_activation": self.output_activation,
            "output_scale": float(self.output_scale),
            "hidden_state_norm": float(np.linalg.norm(state)),
            "hidden_state_mean": float(np.mean(state)) if state.size else 0.0,
            "hidden_state_std": float(np.std(state)) if state.size else 0.0,
            "tau_min": float(np.min(tau)) if tau.size else 0.0,
            "tau_mean": float(np.mean(tau)) if tau.size else 0.0,
            "tau_max": float(np.max(tau)) if tau.size else 0.0,
            "parameter_count": self.parameter_count(),
            "total_parameter_count": self.total_parameter_count(),
            "auxiliary_heads": self.auxiliary_head_manifest(),
            "step_count": int(self._step_count),
            "created_at": self._created_at,
            "last_maintenance_at": self._last_maintenance_at,
            "last_error": self._last_error,
        }

    def estimate_recurrent_gain_power(self, iters: int = 12, seed: Optional[int] = None) -> float:
        rng = np.random.default_rng(seed if seed is not None else 12345)
        v = rng.normal(size=(self.num_hidden,)).astype(np.float64)
        n = _safe_norm(v)
        if n <= 1e-12:
            return 0.0
        v /= n
        last_n = 0.0
        for _ in range(int(max(1, iters))):
            w = self._recurrent_matvec(v)
            last_n = _safe_norm(w)
            if last_n <= 1e-12:
                return 0.0
            v = w / last_n
        return float(last_n)

    def health_metrics(self, include_expensive: bool = False) -> Dict[str, Any]:
        state = self._state
        tau = self._last_dynamic_tau
        inp = self._last_input_signal
        rec = self._last_recurrent_signal
        cur = self._last_total_currents
        out = self._last_output

        spectral_estimate = None
        if include_expensive:
            if self.recurrent_dense is not None:
                spectral_estimate = _safe_spectral_norm(self.recurrent_dense)
            elif self.recurrent_weights is not None:
                try:
                    if hasattr(self.recurrent_weights, "estimate_operator_norm"):
                        spectral_estimate = float(self.recurrent_weights.estimate_operator_norm(num_iters=12))
                    else:
                        spectral_estimate = self.estimate_recurrent_gain_power(iters=12)
                except Exception:
                    spectral_estimate = None

        metrics = {
            "policy_id": self.policy_id,
            "recurrent_mode": "tt" if self._use_tt else "dense",
            "hidden_norm": float(np.linalg.norm(state)),
            "hidden_mean": float(np.mean(state)) if state.size else 0.0,
            "hidden_std": float(np.std(state)) if state.size else 0.0,
            "hidden_saturation_fraction": _saturation_fraction(state),
            "tau_min": float(np.min(tau)) if tau.size else 0.0,
            "tau_mean": float(np.mean(tau)) if tau.size else 0.0,
            "tau_max": float(np.max(tau)) if tau.size else 0.0,
            "tau_span": float(np.max(tau) - np.min(tau)) if tau.size else 0.0,
            "input_signal_norm": float(np.linalg.norm(inp)),
            "recurrent_signal_norm": float(np.linalg.norm(rec)),
            "total_current_norm": float(np.linalg.norm(cur)),
            "output_norm": float(np.linalg.norm(out)),
            "nonfinite_hidden_count": _count_nonfinite(state),
            "nonfinite_tau_count": _count_nonfinite(tau),
            "nonfinite_output_count": _count_nonfinite(out),
            "recurrent_gain_estimate": spectral_estimate,
            "is_saturated": bool(_saturation_fraction(state) >= 0.25),
            "has_nan": bool(_count_nonfinite(state) > 0 or _count_nonfinite(tau) > 0 or _count_nonfinite(out) > 0),
            "degraded_mode": bool(not self._use_tt and self._tt_init_error is not None),
            "step_count": int(self._step_count),
        }
        metrics["is_stable"] = bool(
            not metrics["has_nan"]
            and metrics["hidden_norm"] < (self.num_hidden * 1.25)
            and metrics["tau_min"] > 0.0
            and metrics["output_norm"] < max(1.0, self.num_outputs * 10.0)
        )
        return metrics

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------
    def apply_regularization(self, target_norm: float = 1.0, *, use_power_estimate: bool = True) -> None:
        """Stability-enforcing recurrent regularization."""
        target_norm = float(target_norm)
        if target_norm <= 0:
            raise ValueError("target_norm must be > 0")

        if not self._use_tt:
            if self.recurrent_dense is None:
                return
            current_norm = _safe_spectral_norm(self.recurrent_dense)
            if current_norm is not None and current_norm > target_norm and current_norm > 0:
                self.recurrent_dense *= target_norm / current_norm
                self._last_maintenance_at = _utc_ts()
                if self.verbose:
                    print(f"[{self.policy_id}] Dense recurrent spectral clamp: {current_norm:.4f} -> {target_norm:.4f}")
            return

        if self.recurrent_weights is None:
            return

        try:
            current_norm = self.estimate_recurrent_gain_power(iters=12) if use_power_estimate else None
            if current_norm is not None and current_norm > target_norm and current_norm > 0:
                self.recurrent_weights = self.recurrent_weights.scale(float(target_norm / current_norm))
                if hasattr(self.recurrent_weights, "round"):
                    self.recurrent_weights = self.recurrent_weights.round(
                        max_rank=max(self._tt_bond_dims_actual) if self._tt_bond_dims_actual else self.bond_dim,
                        energy_tol=0.999,
                    )
                self._recurrent_param_count = int(self.recurrent_weights.parameter_count()) if hasattr(self.recurrent_weights, "parameter_count") else self._recurrent_param_count
                self._last_maintenance_at = _utc_ts()
                if self.verbose:
                    print(f"[{self.policy_id}] TT recurrent regularized: {current_norm:.4f} -> {target_norm:.4f}")
        except Exception as e:
            self._last_error = f"regularization_failed:{e}"
            if self.verbose:
                print(f"[{self.policy_id}] WARNING: TT regularization failed: {e}")

    def canonicalize_recurrent(self, direction: str = "left") -> None:
        if not self._use_tt or self.recurrent_weights is None:
            return
        direction = str(direction).lower().strip()
        if direction == "left":
            self.recurrent_weights = self.recurrent_weights.left_orthonormalize()
        elif direction == "right":
            self.recurrent_weights = self.recurrent_weights.right_orthonormalize()
        else:
            raise ValueError("direction must be 'left' or 'right'")
        self._recurrent_param_count = int(self.recurrent_weights.parameter_count())
        self._last_maintenance_at = _utc_ts()

    def round_recurrent(self, max_rank: Optional[int] = None, energy_tol: Optional[float] = 0.999) -> None:
        if not self._use_tt or self.recurrent_weights is None:
            return
        self.recurrent_weights = self.recurrent_weights.round(
            max_rank=max_rank if max_rank is not None else (max(self._tt_bond_dims_actual) if self._tt_bond_dims_actual else self.bond_dim),
            energy_tol=energy_tol,
        )
        self._recurrent_param_count = int(self.recurrent_weights.parameter_count())
        self._tt_bond_dims_actual = list(self.recurrent_weights.bond_ranks()) if hasattr(self.recurrent_weights, "bond_ranks") else self._tt_bond_dims_actual
        self._last_maintenance_at = _utc_ts()

    # -------------------------------------------------------------------------
    # Forward step
    # -------------------------------------------------------------------------
    def step(self, inputs: Any, dt: float = 1.0, return_state: bool = False):
        """
        Forward step for a single numeric feature vector.

        Parameters
        ----------
        inputs:
            1D numeric vector or array-like coercible to shape (num_inputs,).
        dt:
            Positive integration step. Non-finite or non-positive dt is coerced to 1.0.
        return_state:
            When True, returns (primary_output, aux_dict). Existing lobe wrappers rely on this.
        """
        if isinstance(inputs, dict):
            raise TypeError(
                f"[{self.policy_id}] TensorizedNCP.step() accepts numeric vectors only. "
                "Structured dict inputs must be encoded by the calling lobe."
            )

        x = _coerce_float64_vector(inputs, expected_dim=self.num_inputs, name="inputs")
        _safe_clip_inplace(x, -float(self.config.input_clip), float(self.config.input_clip))

        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        dt = float(np.clip(dt, 1e-9, max(1e-9, self.config.max_dt)))

        self._step_count += 1
        self._last_input = x.copy()

        try:
            input_signal = self.input_weights @ x
            recurrent_signal = self._recurrent_matvec(self._state)
            total_currents = input_signal + recurrent_signal
            _safe_clip_inplace(total_currents, -float(self.config.current_clip), float(self.config.current_clip))

            dynamic_tau = self._dynamic_tau(x)
            self._vectorized_ltc_update(total_currents, dynamic_tau, dt)

            out = self.output_weights @ self._state
            out = self._activate_output(out)

            aux_outputs: Dict[str, Any] = {}
            if self.freq_gate_weights is not None:
                freq_gates = _sigmoid(self.freq_gate_weights @ self._state)
                aux_outputs["freq_gates"] = freq_gates.astype(np.float64, copy=False)
            if self.mvm_weights is not None:
                mvm_predictions = self.mvm_weights @ self._state
                aux_outputs["mvm_predictions"] = _sanitize_vector(mvm_predictions)

            self._last_input_signal = _sanitize_vector(input_signal)
            self._last_recurrent_signal = _sanitize_vector(recurrent_signal)
            self._last_total_currents = _sanitize_vector(total_currents)
            self._last_dynamic_tau = _sanitize_vector(dynamic_tau)
            self._last_output = out.copy()
            self._last_aux_outputs = {k: np.asarray(v, dtype=np.float64).copy() for k, v in aux_outputs.items() if isinstance(v, np.ndarray)}
            self._last_error = None

            if self.strict_runtime_checks:
                if _count_nonfinite(self._state) > 0 or _count_nonfinite(self._last_dynamic_tau) > 0 or _count_nonfinite(self._last_output) > 0:
                    raise FloatingPointError(f"[{self.policy_id}] non-finite runtime values detected")

        except Exception as e:
            self._last_error = str(e)
            raise

        if return_state:
            aux = {
                "hidden_state": self._state.copy(),
                "dynamic_tau": self._last_dynamic_tau.copy(),
                "input_signal": self._last_input_signal.copy(),
                "recurrent_signal": self._last_recurrent_signal.copy(),
                "total_currents": self._last_total_currents.copy(),
                "output_vector": self._last_output.copy(),
                "runtime_flags": {
                    "recurrent_mode": "tt" if self._use_tt else "dense",
                    "strict_runtime_checks": bool(self.strict_runtime_checks),
                    "tt_degraded_to_dense": bool(not self._use_tt and self._tt_init_error is not None),
                    "step_count": int(self._step_count),
                },
                "health": self.health_metrics(include_expensive=False),
            }
            aux.update(aux_outputs)
            return out.copy(), aux

        return out.copy()

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    def save_checkpoint(self, path: Union[str, os.PathLike[str]]) -> None:
        """Save full model weights plus runtime state to a compressed .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, Any] = {
            "checkpoint_version": np.asarray(self.CHECKPOINT_VERSION, dtype=np.int64),
            "config_json": np.asarray(_json_dumps_stable(self.config_dict())),
            "snapshot_json": np.asarray(_json_dumps_stable(self.snapshot())),
            "state": self._state.astype(np.float64),
            "membrane_capacitance": self._membrane_capacitance.astype(np.float64),
            "bias": self._bias.astype(np.float64),
            "input_weights": self.input_weights.astype(np.float64),
            "output_weights": self.output_weights.astype(np.float64),
            "tc_W_h": self.tc_W_h.astype(np.float64),
            "tc_W_x": self.tc_W_x.astype(np.float64),
            "tc_b": self.tc_b.astype(np.float64),
            "tau_min_arr": self._tau_min_arr.astype(np.float64),
            "tau_max_arr": self._tau_max_arr.astype(np.float64),
            "step_count": np.asarray(self._step_count, dtype=np.int64),
            "recurrent_mode": np.asarray("tt" if self._use_tt else "dense"),
            "tt_in_shape": np.asarray(self._tt_in_shape, dtype=np.int64),
            "tt_out_shape": np.asarray(self._tt_out_shape, dtype=np.int64),
            "tt_bond_dims_actual": np.asarray(self._tt_bond_dims_actual, dtype=np.int64),
        }
        if self.freq_gate_weights is not None:
            arrays["freq_gate_weights"] = self.freq_gate_weights.astype(np.float64)
        if self.mvm_weights is not None:
            arrays["mvm_weights"] = self.mvm_weights.astype(np.float64)
        if self.recurrent_dense is not None:
            arrays["recurrent_dense"] = self.recurrent_dense.astype(np.float64)
        if self.recurrent_weights is not None and hasattr(self.recurrent_weights, "cores_data"):
            cores = getattr(self.recurrent_weights, "cores_data")
            arrays["tt_num_cores"] = np.asarray(len(cores), dtype=np.int64)
            for i, core in enumerate(cores):
                arrays[f"tt_core_{i}"] = _np_from_any_array(core).astype(np.float32)

        np.savez_compressed(str(p), **arrays)

    @classmethod
    def load_checkpoint(cls, path: Union[str, os.PathLike[str]], *, rng: Optional[np.random.Generator] = None) -> "TensorizedNCP":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        data = np.load(str(p), allow_pickle=False)
        try:
            cfg_json = str(data["config_json"].item())
            cfg_dict = json.loads(cfg_json)
            cfg = TensorizedNCPConfig(**cfg_dict)
            obj = cls.from_config(cfg, rng=rng)

            obj._state = np.asarray(data["state"], dtype=np.float64).reshape(obj.num_hidden)
            obj._membrane_capacitance = np.asarray(data["membrane_capacitance"], dtype=np.float64).reshape(obj.num_hidden)
            obj._bias = np.asarray(data["bias"], dtype=np.float64).reshape(obj.num_hidden)
            obj.input_weights = np.asarray(data["input_weights"], dtype=np.float64).reshape(obj.num_hidden, obj.num_inputs)
            obj.output_weights = np.asarray(data["output_weights"], dtype=np.float64).reshape(obj.num_outputs, obj.num_hidden)
            obj.tc_W_h = np.asarray(data["tc_W_h"], dtype=np.float64).reshape(obj.num_hidden, obj.num_hidden)
            obj.tc_W_x = np.asarray(data["tc_W_x"], dtype=np.float64).reshape(obj.num_hidden, obj.num_inputs)
            obj.tc_b = np.asarray(data["tc_b"], dtype=np.float64).reshape(obj.num_hidden)
            obj._tau_min_arr = np.asarray(data["tau_min_arr"], dtype=np.float64).reshape(obj.num_hidden)
            obj._tau_max_arr = np.asarray(data["tau_max_arr"], dtype=np.float64).reshape(obj.num_hidden)
            obj._step_count = int(np.asarray(data["step_count"]).item()) if "step_count" in data.files else 0

            if "freq_gate_weights" in data.files:
                obj.freq_gate_weights = np.asarray(data["freq_gate_weights"], dtype=np.float64).reshape(obj.num_freq_gates, obj.num_hidden)
            if "mvm_weights" in data.files:
                obj.mvm_weights = np.asarray(data["mvm_weights"], dtype=np.float64).reshape(obj.num_mvm_outputs, obj.num_hidden)

            mode = str(data["recurrent_mode"].item()) if "recurrent_mode" in data.files else ("tt" if obj._use_tt else "dense")
            obj._tt_in_shape = tuple(int(x) for x in np.asarray(data["tt_in_shape"], dtype=np.int64).tolist()) if "tt_in_shape" in data.files else obj._tt_in_shape
            obj._tt_out_shape = tuple(int(x) for x in np.asarray(data["tt_out_shape"], dtype=np.int64).tolist()) if "tt_out_shape" in data.files else obj._tt_out_shape
            obj._tt_bond_dims_actual = [int(x) for x in np.asarray(data["tt_bond_dims_actual"], dtype=np.int64).tolist()] if "tt_bond_dims_actual" in data.files else obj._tt_bond_dims_actual

            if mode == "dense" and "recurrent_dense" in data.files:
                obj._use_tt = False
                obj.recurrent_dense = np.asarray(data["recurrent_dense"], dtype=np.float64).reshape(obj.num_hidden, obj.num_hidden)
                obj.recurrent_weights = None
                obj._recurrent_param_count = int(obj.recurrent_dense.size)
            elif mode == "tt" and "tt_num_cores" in data.files:
                if not TENSORTRAIN_AVAILABLE:
                    raise RuntimeError("Cannot load TT checkpoint because TensorTrain is unavailable")
                ncores = int(np.asarray(data["tt_num_cores"]).item())
                cores = [np.asarray(data[f"tt_core_{i}"], dtype=np.float32) for i in range(ncores)]
                obj.recurrent_weights = TensorTrain(
                    output_dims=list(obj._tt_out_shape),
                    input_dims=list(obj._tt_in_shape),
                    cores_data=cores,
                    config=TTConfig(dtype=np.float32, device=obj.config.tt_device, check_finite=obj.check_finite),
                )
                obj.recurrent_dense = None
                obj._use_tt = True
                obj._recurrent_param_count = int(obj.recurrent_weights.parameter_count()) if hasattr(obj.recurrent_weights, "parameter_count") else None
            elif "recurrent_dense" in data.files:
                obj._use_tt = False
                obj.recurrent_dense = np.asarray(data["recurrent_dense"], dtype=np.float64).reshape(obj.num_hidden, obj.num_hidden)
                obj.recurrent_weights = None
                obj._recurrent_param_count = int(obj.recurrent_dense.size)

            obj.sync_neuron_mirror_from_vectors()
            return obj
        finally:
            data.close()

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        mode = "tt" if self._use_tt else "dense"
        rp = self.parameter_count()
        return (
            f"TensorizedNCP(id='{self.policy_id}', mode='{mode}', hidden={self.num_hidden}, "
            f"recurrent_params={rp:,}, total_params={self.total_parameter_count():,})"
        )


# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
def _self_test() -> None:
    print("=== TensorizedNCP production substrate self-test ===")
    rng = np.random.default_rng(42)

    ncp = TensorizedNCP(
        policy_id="selftest_ncp",
        num_inputs=8,
        num_hidden=64,
        num_outputs=12,
        bond_dim=4,
        rng=rng,
        verbose=True,
        output_activation="tanh",
        leaky_lambda=0.5,
        tau_min=2.0,
        tau_max=18.0,
        num_freq_gates=2,
        num_mvm_outputs=3,
        tt_strict=False,
        mirror_neurons_each_step=False,
    )

    x = rng.normal(size=(8,))
    y, aux = ncp.step(x, return_state=True)
    assert y.shape == (12,)
    assert aux["hidden_state"].shape == (64,)
    assert np.isfinite(y).all()
    assert aux["health"]["is_stable"]

    print("Output shape:", y.shape)
    print("Hidden state shape:", aux["hidden_state"].shape)
    print("Tau mean:", round(float(np.mean(aux["dynamic_tau"])), 4))
    print("Snapshot:", ncp.snapshot())
    print("Health:", ncp.health_metrics(include_expensive=False))

    payload = ncp.serialize_state()
    old_state = ncp.get_state().copy()
    ncp.reset_state()
    ncp.load_state(payload)
    assert np.allclose(old_state, ncp.get_state())

    tmp = Path(os.getenv("NCP_SELFTEST_CHECKPOINT", "./_ncp_selftest_checkpoint.npz"))
    ncp.save_checkpoint(tmp)
    loaded = TensorizedNCP.load_checkpoint(tmp, rng=np.random.default_rng(123))
    assert loaded.num_hidden == ncp.num_hidden
    assert np.allclose(loaded.get_state(), ncp.get_state())
    try:
        tmp.unlink()
    except Exception:
        pass

    print("State reloaded. Hidden norm:", round(float(np.linalg.norm(ncp.get_state())), 6))
    print("Checkpoint load OK. Loaded mode:", loaded.snapshot()["recurrent_mode"])
    print("=== Done ===")


if __name__ == "__main__":
    _self_test()
