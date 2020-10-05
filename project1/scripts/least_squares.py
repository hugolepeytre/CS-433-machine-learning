import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w