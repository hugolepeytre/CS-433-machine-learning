# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: Vector of values to raise to the powers of 0 to degree
    :param degree: Maximum power to raise x to
    :return: Matrix of dimensions (len(x), degree + 1), with column i being x^i
    """
    result = x
    current = x
    for i in range(degree - 1):
        current = current*x
        result = np.vstack([result, current])
    return result.T


def build_poly_2D(x, degree):
    """
    Same as above but x is a matrix
    :param x: Matrix of values to raise to the powers of 0 to degree
    :param degree: Maximum power to raise x to
    :return: Matrix of dimensions (x.shape[0], x.shape[1] * (degree + 1))
    """
    return np.hstack([build_poly(column, degree) for column in x.T])
