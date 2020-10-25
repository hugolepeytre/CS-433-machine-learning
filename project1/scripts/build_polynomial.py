"""Implements polynomial expansion of the data"""
import numpy as np


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=1 up to j=degree.
    :param x: Vector of values to raise to the powers of 1 to degree
    :param degree: Maximum power to raise x to
    :return: Matrix of dimensions (len(x), degree), with column i being x^(i+1)
    """
    result = x
    current = x
    for i in range(degree - 1):
        current = current*x
        result = np.vstack([result, current])
    return result.T


def build_poly_2D(x, degree):
    """
    Same as above but x is a matrix. It is assumed that the first column is the biais,
    and thus not expanded
    :param x: Matrix of values to raise to the powers of 1 to degree
    :param degree: Maximum power to raise x to
    :return: Matrix of dimensions (x.shape[0], x.shape[1] * degree))
    """
    # We don't do the expansion on the bias column
    expanded_features = np.hstack([build_poly(column, degree) for column in x.T[1:]])
    return np.c_[x[:, 0], expanded_features]
