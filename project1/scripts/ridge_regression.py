import numpy as np

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_p = 2 * len(y) * lambda_
    return np.linalg.inv(tx.T @ tx + lambda_p * np.identity(tx.shape[1])) @ tx.T @ y