# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np


def compute_loss(y, tx, w, loss_function='mse'):
    """
    Computes and returns the loss using the given function in parameters.

    :param y: labels
    :param tx: features
    :param w: weights
    :param loss_function: Which loss function to use (mse or mae for now) 
    :return: Computed loss
    """
    e = y - tx @ w
    N = len(y)
    if loss_function == 'mse':
        return e @ e / (2 * N)
    else:
        return np.abs(e).mean()
