from helpers import *
from costs import *


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Performs a stochastic gradient descent on negative log-likelihood with regularization parameter
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
            loss = compute_loss(y_batch, tx_batch, w, 'log-likelihood') + np.linalg.norm(w) * lambda_ / 2
            grad = compute_gradient(y_batch, tx_batch, w, 'log-likelihood') + lambda_*w
            w = w - gamma * grad
    return w, loss
