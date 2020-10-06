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
    poly = [[y**i for i in range(degree + 1)] for y in x]
    return np.array(poly)
