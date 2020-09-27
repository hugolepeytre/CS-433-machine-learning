# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np


def compute_loss(y, tx, w, loss_function='mse'):
    """
    :param y:
    :param tx:
    :param w:
    :param loss_function:
    :return:
    """
    e = y - tx @ w
    N = len(y)
    if loss_function == 'mse':
        return e @ e / (2 * N)
    else:
        return np.abs(e).mean()
