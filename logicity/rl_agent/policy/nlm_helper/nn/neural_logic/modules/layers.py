#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : layers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/27/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import torch.nn as nn

__all__ = [
    'LinearLayer', 'MLPLayer'
]

class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'identity':
        return Identity()
    elif act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))

class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

        self.in_features = in_features
        self.out_features = out_features

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True, last_activation=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation
        self.flatten = flatten
        self.last_activation = last_activation

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=self.batch_norm, dropout=self.dropout, activation=self.activation)
            modules.append(layer)
        if self.last_activation:
            layer = LinearLayer(dims[-2], dims[-1], batch_norm=self.batch_norm, dropout=self.dropout, activation=self.activation)
        else:
            layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)

class MLPModel(MLPLayer):
    pass