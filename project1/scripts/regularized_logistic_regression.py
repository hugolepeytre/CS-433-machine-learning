from helpers import *
from costs import *
from logistic_regression import *


def gradient_descent_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using regularized logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_log_likelihood(y, tx, w) + np.linalg.norm(w) * lambda_ / 2
    grad = calculate_gradient(y, tx, w) + lambda_*w
    w = w - gamma * grad
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """

    :param y:
    :param tx:
    :param lambda_:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
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
            loss, w = gradient_descent_step(y_batch, tx_batch, w, gamma, lambda_)
    return w, loss
