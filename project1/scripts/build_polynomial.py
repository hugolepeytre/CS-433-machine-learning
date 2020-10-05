# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.c_[np.ones(len(x)), x]
    for i in range(2, degree):
        poly = np.c_[poly, np.power(x,i)]
    return poly
