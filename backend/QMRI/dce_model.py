"""
dce_model.py
============
DCE (Dynamic Contrast-Enhanced) MRI model registration.

Uses the Extended Tofts (Kety) model with a population-average Cosine4 AIF
(Orton et al. 2008, Phys. Med. Biol. 53:1225; as used in DCE-NET, Apache 2.0).
The forward model lives in this module (pure torch, analytic, batched and
differentiable: gradients flow from the predicted signal back to
(ke, ve, vp, dt), which makes self-supervised training through the forward
model possible) and is wrapped for the model registry, the signal simulator and
the LSQ fitter, so that the NN forward pass, the simulated data and the LSQ fit
all use exactly the same signal model. (It used to live in forward_model_dce.py.)

Parameters:
    ke   — elimination rate constant  [0.01, 3.0 min⁻¹]
    ve   — extravascular extracellular volume fraction [0.01, 0.8]
    vp   — plasma volume fraction [0.001, 0.2]
    dt   — time delay (bolus arrival) [0.0, 0.5 min]

The xvalues for DCE are time-points (minutes).

Model
-----
Cosine4 AIF (4 parameters: ab, mb, ae, me), with t' = t - t0 - dt:

    Cp(t) = ab * CosineBolus(t'; mb)  +  ab * ae * [CosineBolus(.; mb) (x) exp(-me t)](t')

    CosineBolus(t; m) = 1 - cos(m t)  for 0 <= t < 2*pi/m, else 0

(first-pass bolus plus an exponentially equilibrating washout phase, ``(x)`` =
convolution).  Extended Tofts tissue curve:

    Ct(t) = vp * Cp(t) + ve * ke * [Cp (x) exp(-ke t)](t)

All convolutions are evaluated analytically (no numerical integration on the
sampling grid), in float64 internally; the result is cast back to the dtype of
the incoming parameters.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import curve_fit

from backend.QMRI.model_registry import ModelConfig, register_model, _REGISTRY


# ---------------------------------------------------------------------------
# DCE parameter names, ranges and default time axis
# ---------------------------------------------------------------------------

DCE_PARAM_NAMES = ['ke', 've', 'vp', 'dt']
DCE_BOUNDS = [
    (0.01,  3.0),     # ke  [min^-1]
    (0.01,  0.8),     # ve  [-]
    (0.001, 0.2),     # vp  [-]
    (0.0,   0.5),     # dt  [min]
]
DCE_PARAM_RANGES = dict(zip(DCE_PARAM_NAMES, DCE_BOUNDS))
DCE_TIMEPOINTS = np.linspace(0, 5, 40)      # 40 time points over 5 minutes

_TWO_PI = 2.0 * math.pi
_DK_MIN = 1e-5      # |ke - me| below this is displaced to avoid a 0/0 (see below)


# ---------------------------------------------------------------------------
# Analytic building blocks (torch, float64, differentiable)
# ---------------------------------------------------------------------------

def _cosine_bolus(t, m):
    """1 - cos(m t) on [0, 2*pi/m), 0 elsewhere."""
    z = m * t
    inside = (z >= 0) & (z < _TWO_PI)
    return torch.where(inside, 1.0 - torch.cos(z), torch.zeros_like(z))


def _special_cosine_exp(x, y):
    """
    S(x, y) = int_0^1 (1 - cos(y s)) exp(-x (1 - s)) ds   (closed form).

    (1 - exp(-x)) / x is evaluated with expm1 and a series for tiny |x|, so
    neither the value nor the gradient produces NaN at x = 0.
    """
    small   = x.abs() < 1e-6
    x_safe  = torch.where(small, torch.ones_like(x), x)
    exp_term = torch.where(small,
                           1.0 - x / 2.0 + x * x / 6.0,
                           -torch.expm1(-x) / x_safe)
    trig_term = x * (1.0 - torch.cos(y)) - y * torch.sin(y)
    return (trig_term + y * y * exp_term) / (x * x + y * y + 1e-24)


def _conv_bolus_exp(t, m, k):
    """
    CosineBolus(.; m) (x) exp(-k t), evaluated at t.

        0 < t < tB :  t  * S(k t,  m t)
        t >= tB    :  tB * S(k tB, m tB) * exp(-k (t - tB))        tB = 2*pi/m
        t <= 0     :  0

    Written with clamps instead of masks, so every branch stays finite and the
    gradients w.r.t. both t (-> dt) and k (-> ke) are well defined.
    """
    tB   = _TWO_PI / m
    t1   = t.clamp(0.0, tB)
    y_in = t1 * _special_cosine_exp(k * t1, m * t1)
    tail = torch.exp(-k * (t - tB).clamp(min=0.0))
    return y_in * tail


def _conv_bolus_exp_exp(t, m, k1, k2):
    """
    CosineBolus(.; m) (x) exp(-k1 t) (x) exp(-k2 t)
      = [ConvBolusExp(t; m, k1) - ConvBolusExp(t; m, k2)] / (k2 - k1).

    k1 : python float (fixed AIF parameter me), k2 : tensor (ke).
    The divided difference is 0/0 at k2 == k1.  For |k2 - k1| < _DK_MIN the
    value is evaluated at k1 +/- _DK_MIN (relative value error ~1e-5, in a
    sliver of ke of width 2e-5) while the gradient still flows to k2
    (straight-through), so training never sees NaN/inf.
    """
    dk      = k2 - k1
    near    = dk.abs() < _DK_MIN
    sgn     = torch.where(dk >= 0, torch.ones_like(dk), -torch.ones_like(dk))
    k2_safe = torch.where(near, k1 + sgn * _DK_MIN, k2)
    k2_eff  = k2 + (k2_safe - k2).detach()
    y1 = _conv_bolus_exp(t, m, k1)
    y2 = _conv_bolus_exp(t, m, k2_eff)
    return (y1 - y2) / (k2_eff - k1)


# ---------------------------------------------------------------------------
# Cosine4 AIF
# ---------------------------------------------------------------------------

def get_default_aif(Hct=0.4):
    """
    Population-based Cosine4 AIF for H&N patients.

    Returns a dict with keys ab, mb, ae, me, t0 (see module docstring).
    ``ab`` is the bolus amplitude, ``mb`` the bolus rate (2*pi/mb = bolus
    duration), ``ae``/``me`` amplitude and rate of the equilibration phase.
    """
    ab_raw = 0.8959 / (1.0 - Hct)
    ae_raw = 1.3354 / (1.0 - Hct)
    mb     = 25.5671
    me     = 0.2130

    tb = 2.0 * math.pi / mb
    ab = ab_raw / tb
    # value of (CosineBolus (x) exp(-me t)) at the end of the bolus:
    f64 = dict(dtype=torch.float64)
    sc  = tb * float(_special_cosine_exp(torch.tensor(tb * me, **f64),
                                         torch.tensor(tb * mb, **f64)))
    ae = ae_raw / ab / sc

    return {'ab': ab, 'mb': mb, 'ae': ae, 'me': me, 't0': 0.0}


_DEFAULT_AIF = get_default_aif()


def cosine4_aif(t, aif):
    """Cp(t) for a (shifted) time tensor t (any shape); zero for t <= 0."""
    return (aif['ab'] * _cosine_bolus(t, aif['mb'])
            + aif['ab'] * aif['ae'] * _conv_bolus_exp(t, aif['mb'], aif['me']))


def cosine4_aif_ext_kety_ce(t, aif, ke):
    """Extravascular term  ke * [Cp (x) exp(-ke t)](t)  of the Extended Tofts model."""
    return ke * aif['ab'] * (
        _conv_bolus_exp(t, aif['mb'], ke)
        + aif['ae'] * _conv_bolus_exp_exp(t, aif['mb'], aif['me'], ke)
    )


def cosine4_aif_numpy(t, aif=None, dt=0.0):
    """Cp(t - t0 - dt) as a float64 numpy array (for plotting / inspection)."""
    aif = _DEFAULT_AIF if aif is None else aif
    with torch.no_grad():
        ts = torch.as_tensor(np.asarray(t, dtype=np.float64)) - aif['t0'] - dt
        return cosine4_aif(ts, aif).numpy()


# ---------------------------------------------------------------------------
# Coefficients -> parameters (NN output head)
# ---------------------------------------------------------------------------

def dce_coefficients_to_parameters(coefficients):
    """
    Map unconstrained NN outputs (N, 4) to (ke, ve, vp, dt) via a sigmoid onto
    the ranges in ``DCE_PARAM_RANGES``.  Each output is an (N, 1) tensor.
    """
    out = []
    for i, name in enumerate(DCE_PARAM_NAMES):
        lo, hi = DCE_PARAM_RANGES[name]
        out.append(lo + torch.sigmoid(coefficients[:, i].unsqueeze(1)) * (hi - lo))
    return tuple(out)


# ---------------------------------------------------------------------------
# Parameters -> signal
# ---------------------------------------------------------------------------

def _column(x, device):
    """-> float64 (N, 1) tensor; tensors keep their autograd graph."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64).reshape(-1, 1)
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=device).reshape(-1, 1)


def dce_signal_torch(timepoints, ke, ve, vp, dt, aif=None):
    """
    Batched, differentiable Extended Tofts / Cosine4 tissue curve.

    Args:
        timepoints : (N_t,) array or tensor, minutes
        ke, ve, vp, dt : floats, or tensors of shape (N,), (N, 1) (batch)
        aif        : AIF dict from get_default_aif() (default if None)

    Returns:
        (N, N_t) tensor with the dtype of the incoming parameter tensors
        (float64 if the parameters are python/numpy numbers).  Gradients flow
        to every parameter that is a tensor with requires_grad.
    """
    aif = _DEFAULT_AIF if aif is None else aif

    tensors = [x for x in (ke, ve, vp, dt) if isinstance(x, torch.Tensor)]
    device  = tensors[0].device if tensors else torch.device('cpu')
    out_dtype = (tensors[0].dtype if tensors and tensors[0].is_floating_point()
                 else torch.float64)

    ke_, ve_, vp_, dt_ = (_column(x, device) for x in (ke, ve, vp, dt))

    if isinstance(timepoints, torch.Tensor):
        t = timepoints.detach().to(device=device, dtype=torch.float64)
    else:
        t = torch.as_tensor(np.asarray(timepoints, dtype=np.float64), device=device)
    t = t.reshape(1, -1)

    t_shift = t - aif['t0'] - dt_                       # (N, N_t)
    cp = cosine4_aif(t_shift, aif)
    ce = cosine4_aif_ext_kety_ce(t_shift, aif, ke_)
    ct = vp_ * cp + ve_ * ce
    return ct.to(out_dtype)


# ---------------------------------------------------------------------------
# Population-average Cosine4 AIF
# ---------------------------------------------------------------------------

AIF = _DEFAULT_AIF


def _population_aif(t: np.ndarray) -> np.ndarray:
    """
    Cosine4 AIF Cp(t) in mM (t in minutes, bolus arrival at t0 = 0, no dt shift).
    """
    return cosine4_aif_numpy(t, AIF)


# ---------------------------------------------------------------------------
# Extended Tofts signal model
# ---------------------------------------------------------------------------

def _dce_signal_extended_tofts(t, ke, ve, vp, dt, aif=None) -> np.ndarray:
    """
    Extended Tofts model with Cosine4 AIF for ONE parameter set (floats).
    Ct(t) = vp*Cp(t) + ve * ke * [Cp (x) exp(-ke t)](t)   (analytic convolution).

    Returns a float64 numpy array (N_t,).  Used by the simulator and the LSQ
    fitter; not differentiable (use ``dce_parameters_to_signal`` for that).
    """
    with torch.no_grad():
        return dce_signal_torch(t, ke, ve, vp, dt, aif)[0].cpu().numpy()


def dce_parameters_to_signal(ke, ve, vp, dt, timepoints, aif=None):
    """
    Generic DCE signal function matching the model registry API.

    Handles two calling conventions:

    1. Scalar / 1-D  — called from LSQ or single-sample evaluation:
         ke, ve, vp, dt are Python floats, 0-D tensors, or shape-(1,) arrays.
         Returns torch.Tensor of shape (N_t,).

    2. Batched — called from the training loop where coefficients_to_parameters
         returns (N, 1) tensors (N = batch size, N = 1 allowed):
         ke, ve, vp, dt are shape-(N, 1) tensors.
         Returns torch.Tensor of shape (N, N_t).

    The computation is differentiable: gradients flow from the returned signal
    to every parameter tensor that requires grad (self-supervised training).
    The output has the dtype of the parameter tensors (float64 for plain
    Python / numpy numbers).

    timepoints : np.ndarray or 1-D torch.Tensor of shape (N_t,), minutes.
    aif        : optional Cosine4 AIF dict (default: get_default_aif()).
    """
    def _is_batched(x):
        if isinstance(x, torch.Tensor):
            return x.dim() == 2 or x.numel() > 1
        if isinstance(x, np.ndarray):
            return x.ndim == 2 or x.size > 1
        return False

    signal = dce_signal_torch(timepoints, ke, ve, vp, dt, aif)    # (N, N_t)
    return signal if _is_batched(ke) else signal[0]


def dce_parameters_to_coefficients(params: np.ndarray) -> np.ndarray:
    """Convert physical DCE parameters (N, 4) to normalized [0,1] coefficients."""
    bounds = np.array(DCE_BOUNDS)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (params - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

PARAM_DIST_DCE = dict(DCE_PARAM_RANGES)

BOUNDS_DCE = list(DCE_BOUNDS)


def sample_dce_parameters(N: int, parameter_ranges: dict, seed_nr=None):
    """
    Sample N sets of DCE parameters uniformly.

    Returns
    -------
    parameters   : (N, 4) np.ndarray  [ke, ve, vp, dt]
    coefficients : (N, 4) np.ndarray  in [0, 1]
    """
    rg = np.random.RandomState(seed_nr)

    names = ['ke', 've', 'vp', 'dt']
    cols = []
    coeffs = []
    for name in names:
        lo, hi = parameter_ranges[name]
        c = rg.uniform(0, 1, (N, 1))
        cols.append(lo + c * (hi - lo))
        coeffs.append(c)

    parameters   = np.concatenate(cols,   axis=1)
    coefficients = np.concatenate(coeffs, axis=1)
    return parameters, coefficients


# ---------------------------------------------------------------------------
# Signal simulation
# ---------------------------------------------------------------------------

TIMEPOINTS_DCE = DCE_TIMEPOINTS   # 40 timepoints over 5 minutes


def _simulate_dce_signal_single(ke, ve, vp, dt, timepoints, SNR, rg):
    signal = _dce_signal_extended_tofts(timepoints, ke, ve, vp, dt)
    smax = np.max(signal)
    noise_r = rg.normal(0, smax / SNR, len(timepoints))
    noise_i = rg.normal(0, smax / SNR, len(timepoints))
    signal  = np.sqrt((signal + noise_r) ** 2 + noise_i ** 2)

    #mean_signal = np.mean(np.abs(signal))
    #if mean_signal > 0:
    #    signal /= mean_signal
    return signal


def simulate_dce_signals(parameters: np.ndarray, timepoints: np.ndarray,
                         SNR: float, seed_nr=None) -> list:
    rg = np.random.RandomState(seed_nr)
    return [
        _simulate_dce_signal_single(*parameters[i], timepoints, SNR, rg)
        for i in range(parameters.shape[0])
    ]


# ---------------------------------------------------------------------------
# LSQ fitting
# ---------------------------------------------------------------------------

_bounds_arr        = np.array(DCE_BOUNDS)
DCE_BOUNDS_DEFAULT = (list(_bounds_arr[:, 0]), list(_bounds_arr[:, 1]))


def fit_dce_lsq_1d(timepoints, signal, x0, bounds=DCE_BOUNDS_DEFAULT):
    """
    Fit a single DCE curve using curve_fit against the Extended Tofts model.

    Parameters
    ----------
    timepoints : array-like (N_t,)
    signal     : array-like (N_t,)
    x0         : array-like (4,)  [ke, ve, vp, dt]

    Returns
    -------
    np.ndarray (4,)
    """
    def dce_fn(t, ke, ve, vp, dt):
        return _dce_signal_extended_tofts(t, ke, ve, vp, dt)

    try:
        params, _ = curve_fit(dce_fn, timepoints, signal, p0=x0,
                              bounds=bounds, maxfev=10000)
    except RuntimeError:
        params = np.array(x0)
    return np.array(params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _dce_params_to_signal_api(ke, ve, vp, dt, timepoints):
    return dce_parameters_to_signal(ke, ve, vp, dt, timepoints)


if 'DCE' not in _REGISTRY:
    register_model(ModelConfig(
        name='DCE',
        param_names=list(DCE_PARAM_NAMES),
        bounds=BOUNDS_DCE,
        param_distribution=PARAM_DIST_DCE,
        xvalues=TIMEPOINTS_DCE,
        default_snr=20.0,
        fit_lsq_1d=fit_dce_lsq_1d,
        sample_parameters=sample_dce_parameters,
        simulate_signals=simulate_dce_signals,
        parameters_to_signal=_dce_params_to_signal_api,
        coefficients_to_parameters=dce_coefficients_to_parameters,
        parameters_to_coefficients=dce_parameters_to_coefficients,
        extra={'aif': _population_aif, 'aif_params': AIF,
               'timepoints': TIMEPOINTS_DCE},
    ))