"""
backend/NN/train.py
===================
Model-agnostic NN training loop for QMRI parameter estimation.

Usage
-----
Pass a ModelConfig to ``train_NN``:

    from backend.QMRI.model_registry import get_model_config
    cfg   = get_model_config('IVIM')   # or 'DCE'
    model = train_NN(signals_train, coeffs_train, cfg.xvalues,
                     model_config=cfg)

Backward-compatible call (IVIM assumed when no config is given)
---------------------------------------------------------------
    model = train_NN(signals_train, coeffs_train, bvalues)
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from backend.NN.NN import NN


def train_NN(signals_train, params_train, acq_values,
             model_config=None,
             batch_size=64, nn_depth=6, nn_hidden=1000,
             lr=3e-4, epochs=200, steps_per_epoch=100,
             parameter_loss_fraction=1):
    """
    Train the QMRI parameter estimation NN.

    Args:
        signals_train            : list of (N_acq,) arrays
        params_train             : list/array of (N_params,) coefficients [0,1]
        acq_values               : (N_acq,) acquisition axis array
        model_config             : ModelConfig; defaults to IVIM if None
        batch_size               : mini-batch size
        nn_depth                 : layers per parallel head
        nn_hidden                : hidden layer width
        lr                       : Adam learning rate
        epochs                   : training epochs
        steps_per_epoch          : gradient steps per epoch
        parameter_loss_fraction  : weight of parameter-space loss

    Returns:
        model : trained NN
    """
    if model_config is None:
        from backend.QMRI.model_registry import get_model_config
        model_config = get_model_config('IVIM')

    coeffs_to_params = model_config.coefficients_to_parameters
    params_to_signal = model_config.parameters_to_signal
    n_params         = model_config.n_params

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model      = NN(device,
                    input_channels=len(acq_values),
                    output_channels=n_params,
                    depth=nn_depth,
                    hidden=nn_hidden).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = torch.nn.MSELoss(reduction='mean').to(device)
    acq_tensor = torch.tensor(acq_values, dtype=torch.float).to(device)

    data = list(zip(signals_train, params_train))

    for epoch in range(1, epochs + 1):
        steps        = 0
        cum_sig      = 0.0
        cum_par      = 0.0
        cum_tot      = 0.0
        print(f'epoch {epoch} of {epochs}')

        model.train()
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        for batch in dataloader:
            if steps > steps_per_epoch:
                break

            optimizer.zero_grad()
            signals, parameter_coeffs = batch

            parameter_coeffs = torch.tensor(
                np.array(parameter_coeffs), dtype=torch.float).to(device)
            signals = torch.tensor(
                np.array(signals), dtype=torch.float).to(device)

            model_coefficients = model(signals)

            # Parameter-space loss
            sigmoid_coeffs = torch.sigmoid(model_coefficients)
            parameter_loss = criterion(sigmoid_coeffs, parameter_coeffs)

            # Signal-space loss
            params_tuple = coeffs_to_params(model_coefficients)
            signal_pred  = params_to_signal(*params_tuple, acq_tensor)
            signal_loss  = criterion(signal_pred.to(device), signals.to(device))

            total_loss = ((1 - parameter_loss_fraction) * signal_loss
                          + parameter_loss_fraction * parameter_loss)
            total_loss.backward()
            optimizer.step()

            cum_sig += signal_loss.cpu().detach().numpy()
            cum_par += parameter_loss.cpu().detach().numpy()
            cum_tot += total_loss.cpu().detach().numpy()
            steps   += 1

        print(f'  signal loss: {cum_sig/steps:.6f}  |  '
              f'parameter loss: {cum_par/steps:.6f}  |  '
              f'total loss: {cum_tot/steps:.6f}')

    print('training finished')
    return model