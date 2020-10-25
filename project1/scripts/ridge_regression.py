"""Ridge Regression Model"""
from costs import *


def ridge_regression(y, tx, lambda_):
    """
    Solves the ridge regression and returns the weight vector and loss
    :param y: Labels
    :param tx: Feature points
    :param lambda_: Regularization parameter
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    lambda_p = 2 * len(y) * lambda_
    w = np.linalg.solve(tx.T @ tx + (lambda_p*np.eye(tx.shape[1])), tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss
