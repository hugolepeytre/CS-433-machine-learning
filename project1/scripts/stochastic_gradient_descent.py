# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from gradient_descent import *
from helpers import *
from costs import *


def compute_stoch_gradient(y, tx, w, loss_function='mse'):
    """

    :param y: Labels
    :param tx: Feature points
    :param w: Weights
    :param loss_function: Which loss function to use (mse or mae for now)
    :return: Gradient vector
    """
    return compute_gradient(y, tx, w, loss_function)


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """

    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weights
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :param loss_function: loss_function: Which loss function to use (mse or mae for now)
    :return: History of losses and weights through descent
    """
    batch_size = 10000
    ws = []
    losses = []
    w = initial_w
    # batch_iter can only send out each element once, so for a big amount of iterations or big batches, we have to
    # repeat the function call
    max_num_batches = y.shape[0]//batch_size
    if max_iters <= max_num_batches:
        n_batches = max_iters
        iterations = 1
    else:
        n_batches = max_num_batches
        iterations = max_iters//n_batches

    for i in range(iterations):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=n_batches):
            loss = compute_loss(y_batch, tx_batch, w, loss_function)
            grad = compute_stoch_gradient(y_batch, tx_batch, w, loss_function)
            w = w - gamma * grad
            ws.append(w)
            losses.append(loss)

    return ws[-1], losses[-1]
