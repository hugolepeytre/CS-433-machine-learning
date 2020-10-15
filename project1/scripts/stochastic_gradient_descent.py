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


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_function='mse'):
    """

    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weights
    :param batch_size: Size of random subsets of features at each step
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :param loss_function: loss_function: Which loss function to use (mse or mae for now)
    :return: History of losses and weights through descent
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = 0
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            loss = compute_loss(y_batch, tx_batch, w, loss_function)
            grad = compute_stoch_gradient(y_batch, tx_batch, w, loss_function)
            w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={lo}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, lo=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]
