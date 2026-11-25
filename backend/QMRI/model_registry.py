"""
model_registry.py
=================
Single source of truth for all supported QMRI models (IVIM, DCE).

This file replaces the former ``model_config.py``. That module was an older
design (one lazy builder per model wiring the forward_model_* / sample_parameters_*
/ lsq_* modules); its only remaining users were the VFA model and the H3/H4/H7-H10
scripts, all retired, plus the ``model_config=None`` fallback in
backend/NN/train.py and prediction.py (now pointed here).

Each model entry specifies everything needed to run the hypothesis experiments:
  - forward model (parameters -> signal)
  - parameter bounds and names
  - default b-values / time-points
  - default SNR
  - LSQ fitting function
  - parameter sampling function
  - signal simulation function

Usage
-----
    from backend.QMRI.model_registry import get_model_config
    cfg = get_model_config('IVIM')   # or 'DCE' (case-insensitive)

    # cfg.param_names         list[str]
    # cfg.bounds              list[tuple(float,float)]
    # cfg.param_distribution  dict  {name: (lo, hi)}
    # cfg.xvalues             np.ndarray   (b-values / timepoints)
    # cfg.default_snr         float
    # cfg.n_inputs            int   (= len(xvalues), NN input width)
    # cfg.n_params            int   (= len(param_names))
    # cfg.fit_lsq_1d          Callable(xvalues, signal, x0) -> np.ndarray (N_params,)
    # cfg.sample_parameters   Callable(N, distribution) -> (params, coeffs)
    # cfg.simulate_signals    Callable(params, xvalues, SNR) -> list[np.ndarray]
    # cfg.parameters_to_signal Callable(*params_scalar, xvalues) -> torch.Tensor
    # cfg.coefficients_to_parameters  Callable(coefficients_tensor) -> tuple of tensors
    # cfg.parameters_to_coefficients  Callable(params_np) -> np.ndarray coefficients
"""

from __future__ import annotations
import dataclasses
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelConfig:
    name: str
    param_names: list
    bounds: list            # list of (lo, hi) per parameter
    param_distribution: dict
    xvalues: np.ndarray     # b-values or time-points
    default_snr: float

    # Callables -- populated by the registration functions in the *_model.py files
    fit_lsq_1d: Callable = None
    sample_parameters: Callable = None
    simulate_signals: Callable = None
    parameters_to_signal: Callable = None
    coefficients_to_parameters: Callable = None
    parameters_to_coefficients: Callable = None

    # Optional: AIF for DCE
    extra: dict = dataclasses.field(default_factory=dict)

    @property
    def n_params(self):
        return len(self.param_names)

    @property
    def n_inputs(self):
        return len(self.xvalues)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelConfig] = {}
_MODULES_LOADED = False


def register_model(cfg: ModelConfig):
    _REGISTRY[cfg.name.upper()] = cfg


# ---------------------------------------------------------------------------
# Lazy registration: import model modules which call register_model()
# ---------------------------------------------------------------------------

def _ensure_registered():
    """Import all model modules so they self-register (once).

    Uses a flag rather than ``if not _REGISTRY``: otherwise importing one
    model module directly (e.g. ``ivim_model``) first would make the registry
    non-empty and silently skip the others.
    """
    global _MODULES_LOADED
    if _MODULES_LOADED:
        return
    _MODULES_LOADED = True          # set first: guards against re-entrancy
    from backend.QMRI import ivim_model   # noqa: F401
    from backend.QMRI import dce_model    # noqa: F401


def get_model_config(model_name: str) -> ModelConfig:
    """Return the ModelConfig for *model_name* (case-insensitive)."""
    _ensure_registered()
    key = model_name.upper()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_models() -> list:
    _ensure_registered()
    return list(_REGISTRY.keys())