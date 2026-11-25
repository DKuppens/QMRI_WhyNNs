"""
backend/experiment_utils.py
===========================
Shared utilities for the hypothesis experiments H1-H10 (single module; replaces
the former pair backend/experiment_utils.py + experiments/experiment_utils.py).

All functions are model-agnostic. Model-specific information comes from the
``ModelConfig`` returned by ``backend.QMRI.model_registry.get_model_config``.

Import-cycle note: this module deliberately does NOT import model_registry (or
the NN package) at module level. ``ModelConfig`` methods such as
``cfg.normalize`` import *this* module lazily, so a top-level import of the
registry here would create a cycle. The NN functions are imported inside
``train_and_predict``.

Contents
--------
  normalization     normalize_params, denormalize_params, rmse_normalized
  Jacobian / CRLB   compute_jacobian_normalized, condition_number,
                    jacobian_column_norms, jacobian_scaling, compute_crlb
  generic helpers   get_prior_mean, signals_to_array
  experiment flow   simulate_data, train_and_predict, run_lsq_array   (from the
                    former experiments/experiment_utils.py)
  output            fig_path, and the figure_data re-exports

``compute_jacobian_normalized`` keeps both historical call styles:
    compute_jacobian_normalized(params, xvalues, bounds, parameters_to_signal_fn, eps=1e-7)
    compute_jacobian_normalized(params, cfg, eps=1e-7)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                  # type hints only
    from backend.QMRI.model_registry import ModelConfig


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_params(params: np.ndarray, bounds: list) -> np.ndarray:
    """Map raw parameters to [0, 1] using known physical bounds."""
    bounds_arr = np.array(bounds)
    return (params - bounds_arr[:, 0]) / (bounds_arr[:, 1] - bounds_arr[:, 0])


def denormalize_params(params_norm: np.ndarray, bounds: list) -> np.ndarray:
    """Map [0, 1] parameters back to raw units."""
    bounds_arr = np.array(bounds)
    return params_norm * (bounds_arr[:, 1] - bounds_arr[:, 0]) + bounds_arr[:, 0]


def rmse_normalized(
    params_pred: np.ndarray,
    params_true_norm: np.ndarray,
    bounds: list,
) -> np.ndarray:
    """
    Per-parameter RMSE in normalized [0, 1] space.

    Parameters
    ----------
    params_pred      : (N, n_params) raw predictions
    params_true_norm : (N, n_params) already-normalized ground truth
    bounds           : list of (lo, hi)

    Returns
    -------
    (n_params,) RMSE vector
    """
    pred_norm = normalize_params(params_pred, bounds)
    return np.sqrt(np.mean((pred_norm - params_true_norm) ** 2, axis=0))


# ---------------------------------------------------------------------------
# Jacobian and condition number
# ---------------------------------------------------------------------------

def compute_jacobian_normalized(
    params: np.ndarray,
    xvalues,
    bounds=None,
    parameters_to_signal_fn=None,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Numerical Jacobian, with columns normalized by physical parameter ranges.

    Two call styles (see module docstring): explicit
    ``(params, xvalues, bounds, parameters_to_signal_fn, eps)`` or
    ``(params, cfg, eps)``. Returns (N_x, n_params).

    Parameters are passed to ``parameters_to_signal_fn`` as float64 NumPy
    scalars on purpose: wrapping them in ``torch.tensor(float(p))`` would make
    them float32, and a 1e-7 finite-difference step is then below float32
    resolution for O(1) parameters (S0, f, M0), giving errors of ~20%.
    """
    import torch

    if hasattr(xvalues, 'xvalues') and hasattr(xvalues, 'bounds'):   # cfg style
        cfg = xvalues
        if isinstance(bounds, (int, float)):         # compute_jacobian_normalized(p, cfg, 1e-6)
            eps = float(bounds)
        xvalues = cfg.xvalues
        bounds = cfg.bounds
        parameters_to_signal_fn = cfg.parameters_to_signal

    params = np.asarray(params, dtype=np.float64)
    x_t = torch.tensor(np.asarray(xvalues), dtype=torch.float64)
    J = np.zeros((len(xvalues), len(params)))
    for i in range(len(params)):
        p_plus  = params.copy(); p_plus[i]  += eps
        p_minus = params.copy(); p_minus[i] -= eps

        s_plus  = parameters_to_signal_fn(*p_plus,  x_t).numpy()
        s_minus = parameters_to_signal_fn(*p_minus, x_t).numpy()

        J[:, i] = (s_plus - s_minus) / (2 * eps)
        param_range = bounds[i][1] - bounds[i][0]
        J[:, i] *= param_range

    return J


def condition_number(J: np.ndarray) -> float:
    sv = np.linalg.svd(J, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else np.inf


def jacobian_column_norms(J: np.ndarray) -> np.ndarray:
    return np.linalg.norm(J, axis=0)


def jacobian_scaling(J: np.ndarray) -> np.ndarray:
    norms = jacobian_column_norms(J)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return 1.0 / norms


# ---------------------------------------------------------------------------
# CRLB
# ---------------------------------------------------------------------------

def compute_crlb(
    params: np.ndarray,
    xvalues: np.ndarray,
    sigma_noise: float,
    bounds: list,
    parameters_to_signal_fn,
) -> np.ndarray:
    """
    Cramér-Rao Lower Bound in normalized [0,1] parameter space.

    Returns (n_params,) variance vector.
    """
    J = compute_jacobian_normalized(params, xvalues, bounds,
                                    parameters_to_signal_fn)
    FIM = (J.T @ J) / sigma_noise ** 2
    try:
        crlb = np.diag(np.linalg.inv(FIM))
    except np.linalg.LinAlgError:
        crlb = np.full(len(params), np.inf)
    return np.maximum(crlb, 0.0)


# ---------------------------------------------------------------------------
# Generic experiment helpers
# ---------------------------------------------------------------------------

def get_prior_mean(cfg) -> np.ndarray:
    """Return midpoint of each parameter range as initial LSQ guess."""
    return np.array([(lo + hi) / 2 for lo, hi in cfg.bounds])


def signals_to_array(signals) -> np.ndarray:
    """Convert a list of (N_x,) arrays to (N, N_x) np.ndarray."""
    if isinstance(signals, np.ndarray):
        return signals
    return np.stack(signals, axis=0)


# ---------------------------------------------------------------------------
# Data simulation  (from the former experiments/experiment_utils.py)
# ---------------------------------------------------------------------------

def simulate_data(cfg: "ModelConfig", n_train: int, n_test: int, snr: float,
                  seed_train: int = None, seed_test: int = None):
    """
    Sample parameters and simulate signals for train and test sets, through the
    model's own registered ``sample_parameters`` / ``simulate_signals`` (the
    same ones H1/H2/H5/H6 call directly). The old version dispatched on
    ``cfg.name == 'ivim'`` etc. and called the ``sample_parameters_*`` modules
    itself, which silently broke for registry configs (their names are
    upper-case) and bypassed the registered simulator.

    Returns
    -------
    params_train, coeffs_train, signals_train,
    params_test,  signals_test,  params_test_norm
    """
    def sample(n, seed):
        if seed is None:
            return cfg.sample_parameters(n, cfg.param_distribution)
        return cfg.sample_parameters(n, cfg.param_distribution, seed_nr=seed)

    params_train, coeffs_train = sample(n_train, seed_train)
    params_test, _             = sample(n_test,  seed_test)

    signals_train    = cfg.simulate_signals(params_train, cfg.xvalues, snr)
    signals_test     = cfg.simulate_signals(params_test,  cfg.xvalues, snr)
    params_test_norm = normalize_params(params_test, cfg.bounds)

    return (params_train, coeffs_train, signals_train,
            params_test,  signals_test,  params_test_norm)


# ---------------------------------------------------------------------------
# NN helpers
# ---------------------------------------------------------------------------

def train_and_predict(cfg: "ModelConfig", signals_train, coeffs_train,
                      signals_test,
                      batch_size=64, nn_depth=6, nn_hidden=1000):
    """Train the NN and return (model, params_nn)."""
    from backend.NN.train import train_NN            # lazy: avoids import cycles
    from backend.NN.prediction import predict_NN

    model     = train_NN(signals_train, coeffs_train, cfg.xvalues,
                         model_config=cfg, batch_size=batch_size,
                         nn_depth=nn_depth, nn_hidden=nn_hidden)
    params_nn = predict_NN(model, signals_test, cfg.xvalues,
                           model_config=cfg)
    return model, params_nn


# ---------------------------------------------------------------------------
# LSQ helpers
# ---------------------------------------------------------------------------

def run_lsq_array(cfg: "ModelConfig", signals_test):
    """
    Run single-start LSQ (from the prior mean) on every test sample.

    Returns (N_test, N_params) array. The registered ``fit_lsq_1d`` applies the
    model's own bounds (the old ``bounds`` argument was ignored by it anyway).
    """
    x0 = get_prior_mean(cfg)
    return np.array([
        cfg.fit_lsq_1d(cfg.xvalues, s, x0)
        for s in signals_test
    ])


# ---------------------------------------------------------------------------
# Figure save path helper
# ---------------------------------------------------------------------------

def fig_path(figure_name: str, model_name: str, base_dir: str = 'figures') -> str:
    """
    Return a save path that includes the model name, e.g.
    'figures/ivim/H1_local_minima_rmse.png'.
    (The hypothesis scripts themselves use ``plot_style.outdir``.)
    """
    import os
    out = os.path.join(base_dir, model_name)
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, figure_name)


# ---------------------------------------------------------------------------
# Figure-data persistence (re-draw figures without recomputing)
# ---------------------------------------------------------------------------
# The implementation lives in backend/figure_data.py; re-exported here so the
# experiment scripts can reach it from the shared utilities module as well.

from backend.figure_data import (   # noqa: E402,F401
    data_dir_for, save_table, save_arrays, save_meta,
    load_table, load_arrays, load_meta,
)