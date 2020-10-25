"""Least Squares Regression Model"""
from costs import *


def least_squares(y, tx):
    """
    Solves the least squares regression and returns the weight vector and loss
    :param y: Labels
    :param tx: Feature vector
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    w, _, _, _ = np.linalg.lstsq(tx.T @ tx, tx.T @ y, rcond=None)
    loss = compute_loss(y, tx, w)
    return w, loss
