from gradient_descent import gradient_descent
from stochastic_gradient_descent import stochastic_gradient_descent
from least_squares import least_squares
from ridge_regression import ridge_regression
from logistic_regression import logistic_regression
from regularized_logistic_regression import reg_logistic_regression

from build_polynomial import build_poly_2D
from costs import *


def model_data(y, tx, model, initial_w=[], max_iters=1000, gamma=0.1, lambda_=0.1, poly_exp=1):
    """
    How to use : Specify the machine learning model to use, and give the necessary parameters as named arguments
    :param y: Labels
    :param tx: Feature points
    :param model: Which machine learning model to use, argument should be the name as a string
    (available : gradient_descent, stochastic_gradient_descent, least_squares, ridge_regression,
    logistic_regression, regularized_logistic_regression)
    :param initial_w: Initial weights. Defaults to zeroes
    :param max_iters: number of iterations
    :param gamma: learning rate
    :param lambda_: Regularization parameter
    :param poly_exp: If above 1, what degree to raise the features to
    :return: the last weight vector, and loss
    """
    if poly_exp > 1:
        tx = build_poly_2D(tx, poly_exp)

    if len(initial_w) == 0:
        initial_w = np.zeros(tx.shape[1])

    if model == 'gradient_descent':
        w, loss = gradient_descent(y, tx, initial_w, max_iters, gamma)
    elif model == 'stochastic_gradient_descent':
        w, loss = stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)
    elif model == 'least_squares':
        w, loss = least_squares(y, tx)
    elif model == 'ridge_regression':
        w, loss = ridge_regression(y, tx, lambda_)
    elif model == 'logistic_regression':
        y = y/2 + 0.5
        w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
    elif model == 'regularized_logistic_regression':
        y = y/2 + 0.5
        w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    else:
        raise ValueError('Invalid model')
    return w, loss


def get_loss(y, tx, w):
    """
    Returns the mse
    """
    return compute_loss(y, tx, w)


def get_log_likelihood(y, tx, w):
    """
    Returns the negative log-likelihood
    Takes care of the change in labels from {-1,-1} to {0,1}
    """
    y = y/2 + 0.5
    return compute_log_likelihood(y, tx, w)
