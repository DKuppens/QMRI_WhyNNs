"""
backend/NN/prediction.py
========================
Model-agnostic NN prediction for QMRI parameter estimation.

Usage
-----
    from backend.QMRI.model_registry import get_model_config
    cfg    = get_model_config('IVIM')   # or 'DCE'
    params = predict_NN(model, signals, cfg.xvalues,
                        model_config=cfg)

Backward-compatible call (IVIM assumed when no config is given)
---------------------------------------------------------------
    params = predict_NN(model, signals, bvalues)
"""

import torch
import numpy as np


def predict_NN(model, signals, acq_values, model_config=None):
    """
    Run trained NN inference and return physical parameters.

    Args:
        model        : trained NN instance
        signals      : list or (N, N_acq) array
        acq_values   : (N_acq,) acquisition axis array
        model_config : ModelConfig; defaults to IVIM if None

    Returns:
        params : (N, N_params) numpy array in raw physical units
    """
    if model_config is None:
        from backend.QMRI.model_registry import get_model_config
        model_config = get_model_config('IVIM')

    coeffs_to_params = model_config.coefficients_to_parameters
    device           = model.device

    signals_arr = np.stack(signals, axis=0) if isinstance(signals, list) else signals
    signals_t   = torch.tensor(signals_arr, dtype=torch.float).to(device)

    model.eval()
    with torch.no_grad():
        model_coefficients = model(signals_t)

    params_tuple = coeffs_to_params(model_coefficients)
    params = np.concatenate(
        [p.cpu().detach().numpy() for p in params_tuple], axis=1
    )
    return params