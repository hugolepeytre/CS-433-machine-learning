"""Gradient Descent Model"""
from costs import *


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """
    Operates a gradient descent using either mean square error or mean absolute error
    :param y: Labels, dim: N
    :param tx: Feature points, dim: NxD
    :param initial_w: Initial weights, dim: D
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :param loss_function: loss_function: Which loss function to use (mse or mae for now)
    :return: History of losses and weights through descent
    """
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_function)
        grad = compute_gradient(y, tx, w, loss_function)
        w = w - gamma * grad

    return w, loss
