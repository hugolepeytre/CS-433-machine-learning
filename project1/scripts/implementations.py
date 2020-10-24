import numpy as np
from helpers import batch_iter


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
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


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weights
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning Rate
    :return: Last weight/loss pair
    """
    batch_size = 1
    w = initial_w

    # batch_iter can only send out each element once, so for a big amount 
    # of iterations or big batches, we have to repeat the function call
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
    Least squares regression using normal equations
    :param y: Labels
    :param tx: Feature vector
    :return: (w, loss), the optimal weight vector found, and the training loss
    """

    w, _, _, _ = np.linalg.lstsq(tx.T @ tx, tx.T @ y, rcond=None)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    :param y: Labels
    :param tx: Feature points
    :param lambda_: Regularization parameter
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    lambda_p = 2 * len(y) * lambda_
    w = np.linalg.solve(tx.T @ tx + (lambda_p*np.eye(tx.shape[1])), tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD
    :param y: Labels
    :param tx: Feature points
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning rate
    :return: The last weight/loss pair
    """
    w = initial_w
    loss = 0

    for i in range(max_iters):
        loss = compute_log_likelihood(y, tx, w)
        grad = calculate_gradient_neg_log_likelihood(y, tx, w)
        w = w - gamma*grad
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD
    :param y: Labels
    :param tx: Feature points
    :param lambda_: Regularization parameter
    :param initial_w: Initial weight vector
    :param max_iters: Number of iterations of gradient descent
    :param gamma: Learning rate
    :return: The last weight/loss pair
    """
    w = initial_w
    loss = 0

    for i in range(max_iters):
        loss = compute_log_likelihood(y, tx, w) + np.linalg.norm(w) * lambda_ / 2
        grad = calculate_gradient_neg_log_likelihood(y, tx, w) + lambda_*w
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
    return 1 / (1 + np.exp(-t))


def compute_log_likelihood(y, tx, w):
    """compute the negative log likelihood."""
    eta = tx @ w
    sum_logs = np.log(np.exp(eta) + 1).sum()
    error = -y.T @ eta
    return sum_logs + error


def calculate_gradient_neg_log_likelihood(y, tx, w):
    """compute the gradient of the negative log-likelihood."""
    return tx.T @ (sigmoid(tx @ w) - y)