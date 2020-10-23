import numpy as np
from helpers import *


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    :param y: Labels, dim: N
    :param tx: Feature points, dim: NxD
    :param initial_w: Initial weights, dim: D
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :return: History of losses and weights through descent
    """
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse(y, tx, w)
        grad = compute_gradient_mse(y, tx, w)
        w = w - gamma * grad

    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weights
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :return: Last weight/loss pair
    """
    batch_size = 1
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
            loss = compute_mse(y_batch, tx_batch, w)
            grad = compute_gradient_mse(y_batch, tx_batch, w)
            w = w - gamma * grad

    return w, loss


def least_squares(y, tx):
    """
    :param y: Labels
    :param tx: Feature vector
    :return: (w, loss), the optimal weight vector found, and the training loss
    """

    w, _, _, _ = np.linalg.lstsq(tx.T @ tx, tx.T @ y, rcond=None)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    :param y: Labels
    :param tx: Feature points
    :param lambda_: Regularization parameter
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    lambda_p = 2 * len(y) * lambda_
    w, _, r, _ = np.linalg.lstsq(tx.T @ tx + lambda_p, tx.T @ y, rcond=None)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning rate
    :return: The last weight/loss pair
    """
    batch_size = 1000
    w = initial_w
    loss = 0

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
            loss = compute_log_likelihood(y_batch, tx_batch, w)
            grad = calculate_gradient_neg_log_likelihood(y_batch, tx_batch, w)
            w = w - gamma*grad
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Trains a regularized logistic regression model
    :param y: Labels
    :param tx: Feature points
    :param lambda_: Regularization parameter
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning rate
    :return: The last weight/loss pair
    """
    batch_size = 1000
    w = initial_w
    loss = 0
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
            loss = compute_log_likelihood(y_batch, tx_batch, w) + np.linalg.norm(w) * lambda_ / 2
            grad = calculate_gradient_neg_log_likelihood(y_batch, tx_batch, w) + lambda_*w
            w = w - gamma*grad
    return w, loss


"""
Auxiliary Methods
"""


def compute_gradient_mse(y, tx, w):
    """
    Computes the gradient of the mean square error
    :param y: Labels
    :param tx: Feature points
    :param w: Weights
    :return: Gradient vector
    """
    e = y - tx @ w
    N = len(y)
    return -tx.T @ e / N


def compute_mse(y, tx, w):
    """
    Computes and returns the loss using the given function in parameters.

    :param y: labels
    :param tx: features
    :param w: weights
    :return: Computed loss
    """
    e = y - tx @ w
    N = len(y)
    return e @ e / (2 * N)


def sigmoid(t):
    """apply the sigmoid function on t."""
    inv_exp = np.exp(-t)
    return 1/(inv_exp+1)


def compute_log_likelihood(y, tx, w):
    """compute the negative log likelihood."""
    N = len(y)
    prediction = tx @ w
    sum_logs = np.log(np.exp(prediction) + 1).sum()
    error = -y.T @ prediction
    return (sum_logs + error)/N


def calculate_gradient_neg_log_likelihood(y, tx, w):
    """compute the gradient of the negative log-likelihood."""
    N = len(y)
    return (tx.T @ (sigmoid(tx @ w) - y))/N