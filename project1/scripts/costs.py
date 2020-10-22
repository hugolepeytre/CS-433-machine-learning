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


def compute_log_likelihood(y, tx, w):
    """compute the loss: negative log likelihood."""
    N = len(y)
    prediction = tx @ w
    sum_logs = np.log(np.exp(prediction) + 1).sum()
    error = -y.T @ prediction
    return (sum_logs + error)/N
