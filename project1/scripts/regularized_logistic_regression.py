from helpers import *
from costs import *
from logistic_regression import *

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = np.squeeze(sigmoid(tx @ w))
    S = np.diag((pred * (1 - pred)))
    return tx.T @ S @ tx

def gradient_descent_step(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using regularized logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_log_likelihood(y, tx, w) + np.linalg.norm(w) * lambda_ / 2
    grad = calculate_gradient(y, tx, w) + lambda_*w
    hess = calculate_hessian(y, tx, w) + lambda_
    w = w - gamma * np.linalg.inv(hess) @ grad
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
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            loss, w = gradient_descent_step(y_batch, tx_batch, w, gamma, lambda_)
        if i % 10 == 0:
            print("Loss iteration {} : {}".format(i, loss))
    return w, loss
