from gradient_descent import gradient_descent
from stochastic_gradient_descent import stochastic_gradient_descent
from least_squares import least_squares
from ridge_regression import ridge_regression
from logistic_regression import logistic_regression
from regularized_logistic_regression import reg_logistic_regression

from build_polynomial import build_poly_2D
from costs import *


def model_data(y, tx, model, initial_w=0, max_iters=1000, gamma=0.1, lambda_=0.1, poly_exp=1):
    """

    :param y:
    :param tx:
    :param model: Which machine learning model to use
    :param initial_w:
    :param max_iters:
    :param gamma:
    :param lambda_:
    :param poly_exp: If above one, expand the data
    :return:
    """
    if initial_w == 0:
        initial_w = np.zeros(tx.shape[1])
    w = 0
    loss = 0
    if poly_exp > 1:
        tx = build_poly_2D(tx, poly_exp)

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

    return w, loss


def get_loss(y, tx, w):
    return compute_loss(y, tx, w)


def get_log_likelihood(y, tx, w):
    return compute_log_likelihood(y, tx, w)
