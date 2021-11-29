# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Remark on the documentation:

Due to an issue in autoapi, the classes `NeuralNetworkStatefreeOutputReductor`,
`NeuralNetworkInstationaryReductor`, `NeuralNetworkInstationaryStatefreeOutputReductor`,
`EarlyStoppingScheduler` and `CustomDataset` do not appear in the documentation,
see https://github.com/pymor/pymor/issues/1343.
"""

from pymor.config import config


if config.HAVE_TORCH:
    from numbers import Number

    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils as utils


    class NeuralNetworkReductor:
        """Reduced Basis reductor relying on artificial neural networks.

        This is a reductor that constructs a reduced basis using proper
        orthogonal decomposition and trains a neural network that approximates
        the mapping from parameter space to coefficients of the full-order
        solution in the reduced basis.
        The approach is described in
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set

        def reduce(self, hidden_layers='[(N+P)*3, (N+P)*3]', activation_function=torch.tanh,
                   optimizer=optim.LBFGS, epochs=1000, batch_size=20, learning_rate=1.,
                   restarts=10, seed=0):
            """Reduce by training artificial neural networks.
            """

    class NeuralNetworkStatefreeOutputReductor:
        """Output reductor relying on artificial neural networks.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     validation_loss=None):
            assert 0 < validation_ratio < 1 or validation_set

        def _compute_layer_sizes(self, hidden_layers):
            """Compute the number of neurons in the layers of the neural network."""
            # determine the numbers of neurons in the hidden layers
            if isinstance(hidden_layers, str):
                hidden_layers = eval(hidden_layers, {'N': self.fom.dim_output, 'P': self.fom.parameters.dim})
            # input and output size of the neural network are prescribed by the
            # dimension of the parameter space and the output dimension
            assert isinstance(hidden_layers, list)
            return [self.fom.parameters.dim, ] + hidden_layers + [self.fom.dim_output, ]


    class NeuralNetworkInstationaryReductor(NeuralNetworkReductor):
        """Reduced Basis reductor for instationary problems relying on artificial neural networks.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set
