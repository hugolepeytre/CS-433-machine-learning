from helpers import *
from costs import *


def sigmoid(t):
    """apply the sigmoid function on t."""
    inv_exp = np.exp(-t)
    return 1/(inv_exp+1)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    N = len(y)
    return (tx.T @ (sigmoid(tx @ w) - y))/N


def gradient_descent_step(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_log_likelihood(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    return loss, w


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
            loss, w = gradient_descent_step(y_batch, tx_batch, w, gamma)
    return w, loss
