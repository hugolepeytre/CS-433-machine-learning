from costs import *


def least_squares(y, tx):
    """

    :param y: Labels
    :param tx: Feature vector
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    
    w, _, r, _ = np.linalg.lstsq(tx.T @ tx, tx.T @ y, rcond=None)
    print("Matrix X^TX has rank ", r)
    return w, compute_loss(y, tx, w)
