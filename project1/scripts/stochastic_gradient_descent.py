"""Stochastic Gradient Descent Model"""
from helpers import *
from costs import *


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """
    Performs a stochastic gradient descent using either mean square error or mean absolute error
    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weights
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :param loss_function: loss_function: Which loss function to use (mse or mae for now)
    :return: Last weight/loss pair
    """
    batch_size = 500
    loss = 0
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
            grad = compute_gradient(y_batch, tx_batch, w, loss_function)
            w = w - gamma * grad

    return w, loss
