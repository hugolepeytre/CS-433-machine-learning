from costs import *


def ridge_regression(y, tx, lambda_):
    """

    :param y: Labels
    :param tx: Feature points
    :param lambda_: Regularization parameter
    :return: (w, loss), the optimal weight vector found, and the training loss
    """
    lambda_p = 2 * len(y) * lambda_
    w, _, r, _ = np.linalg.lstsq(tx.T @ tx + lambda_p, tx.T @ y, rcond=None)
    print(tx.shape)
    print("Matrix X^TX + lambda has rank ", r)
    return w, compute_loss(y, tx, w)
