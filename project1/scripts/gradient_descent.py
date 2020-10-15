# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *


def compute_gradient(y, tx, w, loss_function='mse'):
    """

    :param y: Labels
    :param tx: Feature points
    :param w: Weights
    :param loss_function: Which loss function to use (mse or mae for now)
    :return: Gradient vector
    """
    e = y - tx @ w
    N = len(y)
    if loss_function == 'mse':
        return -tx.T @ e / N
    else:
        return -tx.T @ np.sign(e) / N


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """

    :param y: Labels, dim: N
    :param tx: Feature points, dim: NxD
    :param initial_w: Initial weights, dim: D
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :param loss_function: loss_function: Which loss function to use (mse or mae for now)
    :return: History of losses and weights through descent
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_function)
        grad = compute_gradient(y, tx, w, loss_function)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={lo}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, lo=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]
