"""
ivim_model.py
=============
IVIM (Intra-Voxel Incoherent Motion) model registration.

Registers the IVIM model with the central model registry.
All IVIM-specific logic lives here; hypothesis scripts remain model-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import curve_fit

from backend.QMRI.model_registry import ModelConfig, register_model, _REGISTRY


# ---------------------------------------------------------------------------
# IVIM forward model
# ---------------------------------------------------------------------------

def ivim_parameters_to_signal(D, Dp, f, S0, bvalues):
    """
    IVIM bi-exponential signal model.
    S(b) = S0 * [ f * exp(-b*Dp) + (1-f) * exp(-b*D) ]

    Accepts scalar np.float64 / float or torch.Tensor for each parameter.
    bvalues may be np.ndarray or torch.Tensor.
    Returns torch.Tensor of shape (N_b,) or (N, N_b).
    """
    if isinstance(D,  np.ndarray): D  = torch.from_numpy(D)
    if isinstance(Dp, np.ndarray): Dp = torch.from_numpy(Dp)
    if isinstance(f,  np.ndarray): f  = torch.from_numpy(f)
    if isinstance(S0, np.ndarray): S0 = torch.from_numpy(S0)
    if isinstance(bvalues, np.ndarray): bvalues = torch.from_numpy(bvalues)

    relative_signal = f * torch.exp(-bvalues * Dp) + (1 - f) * torch.exp(-bvalues * D)
    return S0 * relative_signal


def ivim_coefficients_to_parameters(coefficients):
    """
    Map unconstrained NN output coefficients to physical IVIM parameters
    via sigmoid squashing.

    Args:
        coefficients: (N, 4) tensor — raw NN output

    Returns:
        D, Dp, f, S0 — each (N, 1) tensor in physical units
    """
    ranges = {'D': [0.0003, 0.0035], 'Dp': [0.005, 0.1], 'f': [0, 0.75], 'S0': [0.5, 1.5]}
    Dmin,  Dmax  = ranges['D']
    Dpmin, Dpmax = ranges['Dp']
    fmin,  fmax  = ranges['f']
    S0min, S0max = ranges['S0']

    D  = Dmin  + torch.sigmoid(coefficients[:, 0].unsqueeze(1)) * (Dmax  - Dmin)
    Dp = Dpmin + torch.sigmoid(coefficients[:, 1].unsqueeze(1)) * (Dpmax - Dpmin)
    f  = fmin  + torch.sigmoid(coefficients[:, 2].unsqueeze(1)) * (fmax  - fmin)
    S0 = S0min + torch.sigmoid(coefficients[:, 3].unsqueeze(1)) * (S0max - S0min)
    return D, Dp, f, S0


def ivim_parameters_to_coefficients(params: np.ndarray) -> np.ndarray:
    """
    Convert physical IVIM parameters (N, 4) to normalized [0,1] coefficients.
    """
    bounds = np.array([[0.0003, 0.0035], [0.005, 0.1], [0.0, 0.75], [0.5, 1.5]])
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (params - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def sample_ivim_parameters(N: int, parameter_ranges: dict, seed_nr=None):
    """
    Sample N sets of IVIM parameters uniformly within parameter_ranges.

    Returns
    -------
    parameters   : (N, 4) np.ndarray  [D, Dp, f, S0]
    coefficients : (N, 4) np.ndarray  in [0, 1]
    """
    rg = np.random.RandomState(seed_nr)

    Dmin,  Dmax  = parameter_ranges['D']
    Dpmin, Dpmax = parameter_ranges['Dp']
    fmin,  fmax  = parameter_ranges['f']
    S0min, S0max = parameter_ranges['S0']

    D_coeff  = rg.uniform(0, 1, (N, 1))
    Dp_coeff = rg.uniform(0, 1, (N, 1))
    f_coeff  = rg.uniform(0, 1, (N, 1))
    S0_coeff = rg.uniform(0, 1, (N, 1))

    coefficients = np.concatenate([D_coeff, Dp_coeff, f_coeff, S0_coeff], axis=1)
    D  = Dmin  + D_coeff  * (Dmax  - Dmin)
    Dp = Dpmin + Dp_coeff * (Dpmax - Dpmin)
    f  = fmin  + f_coeff  * (fmax  - fmin)
    S0 = S0min + S0_coeff * (S0max - S0min)

    parameters = np.concatenate([D, Dp, f, S0], axis=1)
    return parameters, coefficients


# ---------------------------------------------------------------------------
# Signal simulation
# ---------------------------------------------------------------------------

def _simulate_ivim_signal_single(D, Dp, f, S0, bvalues, SNR, rg):
    bvalues = np.asarray(bvalues)
    bvalues.sort()
    b0_bool = bvalues == 0

    signal = ivim_parameters_to_signal(
        torch.tensor(float(D)), torch.tensor(float(Dp)),
        torch.tensor(float(f)), torch.tensor(float(S0)),
        torch.tensor(bvalues, dtype=torch.float64),
    ).cpu().detach().numpy()

    noise_r = rg.normal(0, 1.0 / SNR, len(bvalues))
    noise_i = rg.normal(0, 1.0 / SNR, len(bvalues))
    signal  = np.sqrt((signal + noise_r) ** 2 + noise_i ** 2)

    return signal


def simulate_ivim_signals(parameters: np.ndarray, bvalues: np.ndarray,
                          SNR: float, seed_nr=None) -> list:
    """Simulate a list of noisy IVIM signals."""
    rg = np.random.RandomState(seed_nr)
    return [
        _simulate_ivim_signal_single(*parameters[i], bvalues, SNR, rg)
        for i in range(parameters.shape[0])
    ]


# ---------------------------------------------------------------------------
# LSQ fitting
# ---------------------------------------------------------------------------

IVIM_BOUNDS_DEFAULT = ([0.0003, 0.005, 0.0, 0.5], [0.0035, 0.1, 0.75, 1.5])


def fit_ivim_lsq_1d(bvalues, signal, x0, bounds=IVIM_BOUNDS_DEFAULT):
    """
    Fit a single IVIM curve using Levenberg-Marquardt / trust-region.

    Parameters
    ----------
    bvalues : array-like (N_b,)
    signal  : array-like (N_b,)
    x0      : array-like (4,)  initial guess [D, Dp, f, S0]
    bounds  : 2-tuple of lists

    Returns
    -------
    np.ndarray (4,)  [D, Dp, f, S0]
    """
    def ivim_fn(b, D, Dp, f, S0):
        return S0 * (f * np.exp(-b * Dp) + (1 - f) * np.exp(-b * D))

    try:
        params, _ = curve_fit(ivim_fn, bvalues, signal, p0=x0, bounds=bounds,
                              maxfev=5000)
    except RuntimeError:
        params = np.array(x0)
    return np.array(params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

B_VALUES_IVIM = np.array([0, 10, 20, 40, 80, 150, 300, 500, 800], dtype=float)

PARAM_DIST_IVIM = {
    'D':  (0.0003, 0.0035),
    'Dp': (0.005,  0.1),
    'f':  (0.0,    0.75),
    'S0': (0.5,    1.5),
}

BOUNDS_IVIM = [
    (0.0003, 0.0035),
    (0.005,  0.1),
    (0.0,    0.75),
    (0.5,    1.5),
]


def _fit_ivim_lsq_dispatch(xvalues, signal, x0):
    return fit_ivim_lsq_1d(xvalues, signal, x0)


if 'IVIM' not in _REGISTRY:
    register_model(ModelConfig(
        name='IVIM',
        param_names=['D', 'Dp', 'f', 'S0'],
        bounds=BOUNDS_IVIM,
        param_distribution=PARAM_DIST_IVIM,
        xvalues=B_VALUES_IVIM,
        default_snr=20.0,
        fit_lsq_1d=_fit_ivim_lsq_dispatch,
        sample_parameters=sample_ivim_parameters,
        simulate_signals=simulate_ivim_signals,
        parameters_to_signal=ivim_parameters_to_signal,
        coefficients_to_parameters=ivim_coefficients_to_parameters,
        parameters_to_coefficients=ivim_parameters_to_coefficients,
    ))
