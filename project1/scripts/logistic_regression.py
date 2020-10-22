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

    :param y:
    :param tx:
    :param initial_w:
    :param max_iters:
    :param gamma:
    :return:
    """
    batch_size = 1000
    w = initial_w
    loss = 0
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            loss, w = gradient_descent_step(y_batch, tx_batch, w, gamma)
        # if i % 10 == 0:
        #     print("Loss iteration {} : {}".format(i, loss))
    return w, loss
