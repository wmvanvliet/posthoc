import numpy as np

def unit_gain_normalizer(normalizer, X, y, pattern, coef):
    """Ensures that pattern @ coef.T == 1"""
    return 1 / coef.dot(pattern)

def unit_weight_norm_normalizer(normalizer, X, y, pattern, coef):
    """Ensures that |coef| == 1"""
    return coef / np.linalg.norm(coef, axis=1, keepdims=True)

def y_normalizer(normalizer, X, y, pattern, coef):
    """Normalizes with cov(y)^-1"""
    return np.linalg.pinv(y.T.dot(y))
