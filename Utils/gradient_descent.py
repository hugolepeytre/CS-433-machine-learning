# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *


def compute_gradient(y, tx, w, loss_function='mse'):
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
        return -tx.T @ e / N
    else:
        return -tx.T @ np.sign(e) / N


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :param loss_function:
    :return:
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

    return losses, ws
