from helpers import *


def sigmoid(t):
    """apply the sigmoid function on t."""
    inv_exp = np.exp(-t)
    return 1/(inv_exp+1)


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    N = len(y)
    prediction = tx @ w
    sum_logs = np.log(np.exp(prediction) + 1).sum()
    error = -y.T @ prediction
    return (sum_logs + error)/N


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    N = len(y)
    return (tx.T @ (sigmoid(tx @ w) - y))/N


def gradient_descent_step(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
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
        if i % 10 == 0:
            print("Loss iteration {} : {}".format(i, loss))
    return w, loss
