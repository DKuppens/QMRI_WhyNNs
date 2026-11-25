"""
backend/nn_variants.py
======================
The two NN training regimes that every experiment compares against LSQ:

    sup   supervised       parameter_loss_fraction = 1  (loss on the parameters)
    self  self-supervised  parameter_loss_fraction = 0  (loss on the signal)

Both are trained on the same data with the same architecture; only
``parameter_loss_fraction`` (see ``train_NN`` in backend/NN/train.py) differs.
This module is the single place that defines their label, colour and marker,
so every figure draws them identically: pastel blue + square = supervised,
pastel green + triangle = self-supervised (LSQ keeps its pastel orange).

Usage
-----
    from backend.nn_variants import (NN_VARIANTS, train_variants,
                                     predict_variants)

    models = train_variants(signals_train, coeffs_train, cfg.xvalues, cfg)
    preds  = predict_variants(models, signals_test, cfg.xvalues, cfg)
    # preds['sup'], preds['self']  -> (N, n_params) arrays
"""

import numpy as np
import torch

from backend.NN.train import train_NN
from backend.NN.prediction import predict_NN
from backend.plot_style import COLORS

# Self-supervised colours (not in plot_style.COLORS): a pastel green and a
# lighter tint of it, used where a script also needs an "uncorrelated" shade
# (H5), mirroring COLORS['NN'] / COLORS['NN_UNCOR'] for the supervised NN.
SELF_COLOR       = '#8FD19E'
SELF_COLOR_UNCOR = '#C3E8CA'

NN_VARIANTS = (
    dict(key='sup',  label='NN sup.',      plf=1,
         color=COLORS['NN'],  color_uncor=COLORS['NN_UNCOR'],
         marker='s', ls='-'),
    dict(key='self', label='NN self-sup.', plf=0,
         color=SELF_COLOR,    color_uncor=SELF_COLOR_UNCOR,
         marker='^', ls='-'),
)
VARIANT_KEYS = tuple(v['key'] for v in NN_VARIANTS)


def variant(key):
    """Look up a variant dict by its key ('sup' or 'self')."""
    for v in NN_VARIANTS:
        if v['key'] == key:
            return v
    raise KeyError(f"unknown NN variant {key!r}; choose from {VARIANT_KEYS}")


def train_variants(signals_train, coeffs_train, xvalues, cfg, seed=None,
                   **train_kw):
    """
    Train one NN per variant on the same data and return {key: model}.

    If ``seed`` is given, torch's RNG is re-seeded with it before *each*
    training, so both variants start from the same initial weights and the
    only difference between them is the loss. Extra keyword arguments are
    forwarded to ``train_NN`` (but not ``parameter_loss_fraction``).
    """
    models = {}
    for v in NN_VARIANTS:
        print(f"  Training {v['label']} "
              f"(parameter_loss_fraction={v['plf']})...")
        if seed is not None:
            torch.manual_seed(seed)
        models[v['key']] = train_NN(
            signals_train, coeffs_train, xvalues, model_config=cfg,
            parameter_loss_fraction=v['plf'], **train_kw)
    return models


def predict_variants(models, signals, xvalues, cfg):
    """Predict with every trained variant; returns {key: (N, n_params) array}."""
    return {k: np.asarray(predict_NN(m, signals, xvalues, model_config=cfg))
            for k, m in models.items()}