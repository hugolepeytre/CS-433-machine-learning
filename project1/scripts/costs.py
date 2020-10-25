"""Functions used to compute the losses and gradients of mse and negative log-likelihood."""
import numpy as np


def compute_loss(y, tx, w, loss_function='mse'):
    """
    Computes and returns the loss using the given function in parameters.
    :param y: Labels
    :param tx: Features
    :param w: Weights
    :param loss_function: Which loss function to use (mse, mae, negative log-likehood)
    :return: Computed loss
    """
    prediction = tx @ w
    e = y - prediction
    N = len(y)
    if loss_function == 'mse':
        return e @ e / (2 * N)
    elif loss_function == 'mae':
        return np.abs(e).mean()
    elif loss_function == 'log-likelihood':
        sum_logs = np.log(np.exp(prediction) + 1).sum()
        error = -y.T @ prediction
        return (sum_logs + error) / N
    else:
        raise ValueError('Loss function unavailable')


def compute_gradient(y, tx, w, loss_function='mse'):
    """
    Computes the gradient for mse or mae
    :param y: Labels
    :param tx: Feature points
    :param w: Weights
    :param loss_function: Which loss function to use (mse, mae, negative log-likelihood)
    :return: Gradient vector
    """
    e = y - tx @ w
    N = len(y)
    if loss_function == 'mse':
        return -tx.T @ e / N
    elif loss_function == 'mae':
        return -tx.T @ np.sign(e) / N
    elif loss_function == 'log-likelihood':
        return (tx.T @ (sigmoid(tx @ w) - y)) / N
    else:
        raise ValueError('Loss function unavailable')


def sigmoid(t):
    """Apply the sigmoid function on t."""
    inv_exp = np.exp(-t)
    return 1/(inv_exp+1)
