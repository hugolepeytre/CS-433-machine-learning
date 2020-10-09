from costs import *


def least_squares(y, tx):
    """

    :param y: Labels
    :param tx: Feature vector
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss(y, tx, w)
