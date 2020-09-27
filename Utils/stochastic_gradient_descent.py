# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from gradient_descent import *
from helpers import *
from costs import *


def compute_stoch_gradient(y, tx, w, loss_function='mse'):
    """

    :param y:
    :param tx:
    :param w:
    :param loss_function:
    :return:
    """
    return gradient_descent(y, tx, w, loss_function)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """

    :param y:
    :param tx:
    :param initial_w:
    :param batch_size:
    :param max_iters:
    :param gamma:
    :return:
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = 0
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            loss = compute_loss(y_batch, tx_batch, w)
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={lo}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, lo=loss, w0=w[0], w1=w[1]))

    return losses, ws
