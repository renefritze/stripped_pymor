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

        Parameters
        ----------
        fom
            The full-order to reduce.
        training_set
            Set of use for POD and training of the
            neural network.
        validation_set
            Set of use for validation in the training
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        basis_size
            Desired size of the reduced basis. If `None`, rtol, atol or l2_err must
            be provided.
        rtol
            Relative tolerance the basis should guarantee on the training set.
        atol
            Absolute tolerance the basis should guarantee on the training set.
        l2_err
            L2-approximation error the basis should not exceed on the training
            set.
        pod_params
            Dict of additional parameters for the POD-method.
        ann_mse
            If `'like_basis'`, the mean squared error of the neural network on
            the training set should not exceed the error of projecting onto the basis.
            If `None`, the neural network with smallest validation error is
            used to build the ROM.
            If a tolerance is prescribed, the mean squared error of the neural
            network on the training set should not exceed this threshold.
            Training is interrupted if a neural network that undercuts the
            error tolerance is found.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set

        def reduce(self, hidden_layers='[(N+P)*3, (N+P)*3]', activation_function=torch.tanh,
                   optimizer=optim.LBFGS, epochs=1000, batch_size=20, learning_rate=1.,
                   restarts=10, seed=0):
            """Reduce by training artificial neural networks.

            Parameters
            ----------
            hidden_layers
                Number of neurons in the hidden layers. Can either be fixed or
                a Python expression string depending on the reduced basis size
                respectively output dimension `N` and the total dimension of
                the
            activation_function
                Activation function to use between the hidden layers.
            optimizer
                Algorithm to use as optimizer during training.
            epochs
                Maximum number of epochs for training.
            batch_size
                Batch size to use if optimizer allows mini-batching.
            learning_rate
                Step size to use in each optimization step.
            restarts
                Number of restarts of the training algorithm. Since the training
                results highly depend on the initial starting point, i.e. the
                initial weights and biases, it is advisable to train multiple
                neural networks by starting with different initial values and
                choose that one performing best on the validation set.
            seed
                Seed to use for various functions in PyTorch. Using a fixed seed,
                it is possible to reproduce former results.

            Returns
            -------
            rom
                Reduced-order
            """

    class NeuralNetworkStatefreeOutputReductor:
        """Output reductor relying on artificial neural networks.

        This is a reductor that trains a neural network that approximates
        the mapping from parameter space to output space.

        Parameters
        ----------
        fom
            The full-order to reduce.
        training_set
            Set of
            neural network.
        validation_set
            Set of
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        validation_loss
            The validation loss to reach during training. If `None`, the neural
            network with the smallest validation loss is returned.
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

        This is a reductor that constructs a reduced basis using proper
        orthogonal decomposition and trains a neural network that approximates
        the mapping from parameter and time space to coefficients of the
        full-order solution in the reduced basis.
        The approach is described in

        Parameters
        ----------
        fom
            of the neural network.
        validation_ratio
            Fraction of the training set to use for validation in the training
            of the neural network (only used if no validation set is provided).
        basis_size
            Desired size of the reduced basis. If `None`, rtol, atol or l2_err must
            be provided.
        rtol
            Relative tolerance the basis should guarantee on the training set.
        atol
            Absolute tolerance the basis should guarantee on the training set.
        l2_err
            L2-approximation error the basis should not exceed on the training
            set.
        pod_params
            Dict of additional parameters for the POD-method.
        ann_mse
            If `'like_basis'`, the mean squared error of the neural network on
            the training set should not exceed the error of projecting onto the basis.
            If `None`, the neural network with smallest validation error is
            used to build the ROM.
            If a tolerance is prescribed, the mean squared error of the neural
            network on the training set should not exceed this threshold.
            Training is interrupted if a neural network that undercuts the
            error tolerance is found.
        """

        def __init__(self, fom, training_set, validation_set=None, validation_ratio=0.1,
                     basis_size=None, rtol=0., atol=0., l2_err=0., pod_params=None,
                     ann_mse='like_basis'):
            assert 0 < validation_ratio < 1 or validation_set

