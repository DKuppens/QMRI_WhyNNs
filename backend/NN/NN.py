"""
backend/NN/NN.py
================
Model-agnostic neural network for QMRI parameter estimation.

The network architecture is a set of parallel fully-connected heads
(one per output parameter), each using BatchNorm + ReLU layers.
The only model-specific configuration is:
  - input_channels : number of acquisition points (b-values, flip angles, …)
  - output_channels: number of parameters to estimate
"""

import torch


def linear_act_norm_dropout(input_channels, output_channels, use_batchnorm, dropout_ratio):
    module_list = torch.nn.ModuleList()
    module_list.extend([torch.nn.Linear(input_channels, output_channels)])
    if use_batchnorm:
        module_list.extend([torch.nn.BatchNorm1d(output_channels)])
    module_list.extend([torch.nn.ReLU()])
    if dropout_ratio > 0:
        module_list.extend([torch.nn.Dropout(dropout_ratio)])
    return module_list


def build_module_list(depth, input_channels, hidden_channels, output_channels,
                      dropout, use_batchnorm):
    module_list = torch.nn.ModuleList()
    assert depth > 1, 'minimum depth is 2'
    if depth == 2:
        hidden_channels = output_channels
    module_list.extend(
        linear_act_norm_dropout(input_channels, hidden_channels, use_batchnorm, dropout)
    )
    if depth > 2:
        for _ in range(depth - 2):
            module_list.extend(
                linear_act_norm_dropout(hidden_channels, hidden_channels, use_batchnorm, dropout)
            )
    module_list.extend([torch.nn.Linear(hidden_channels, int(output_channels))])
    return module_list


class NN(torch.nn.Module):
    """
    Model-agnostic parallel-head NN for QMRI parameter estimation.

    Args:
        device          : 'cuda' or 'cpu'
        input_channels  : number of acquisition points  (model-specific)
        output_channels : number of parameters to estimate (model-specific)
        depth           : number of layers per head (default 6)
        hidden          : width of hidden layers (default 1000)
        dropout         : dropout probability (default 0)
        use_batchnorm   : use BatchNorm (default True)
    """

    def __init__(self, device, input_channels=9, output_channels=4,
                 depth=6, hidden=1000, dropout=0, use_batchnorm=True):
        super(NN, self).__init__()

        self.device          = device
        self.input_channels  = input_channels
        self.output_channels = output_channels
        self.depth           = depth
        self.hidden_channels = hidden
        self.dropout         = dropout
        self.use_batchnorm   = use_batchnorm

        if output_channels > 8:
            raise NotImplementedError(
                f'Parallel heads support up to 8 outputs; got {output_channels}.'
            )

        self._encoders = torch.nn.ModuleList()
        for _ in range(output_channels):
            ml = build_module_list(depth, input_channels, hidden, 1,
                                   dropout, use_batchnorm)
            self._encoders.append(torch.nn.Sequential(*ml).to(device))

    def forward(self, x):
        return torch.cat([enc(x) for enc in self._encoders], dim=1)
