# -*- coding: utf-8 -*-
""" Grid Search"""

from costs import *
import numpy as np


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """
    :param y: labels
    :param tx: features
    :param w0: list of values for the first parameter
    :param w1: list of values for the second parameter
    :return: array of losses where losses[i,j] is the loss computed using w0[i] and w1[j]
    """
    losses = np.zeros((len(w0), len(w1)))
    for i, w_0 in np.ndenumerate(w0):
        for j, w_1 in np.ndenumerate(w1):
            w = np.array([w_0, w_1])
            losses[i, j] = compute_loss(y, tx, w)
    return losses
