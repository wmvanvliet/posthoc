"""
Some ways of modifying the normalizer.
"""
from numpy.linalg import pinv, norm


def unit_gain(normalizer, X, y, pattern, coef):
    """Ensures that pattern @ coef.T == 1"""
    return pinv(coef.dot(pattern))


def true_labels(normalizer, X, y, pattern, coef):
    """Normalizes with the true labels: cov(y)^-1"""
    return pinv(y.T.dot(y))


def lstsq(normalizer, X, y, pattern, coef):
    """Normalizes to minimize the least-squares-error with the given y"""
    y_hat = X.dot(coef.T).ravel()
    return y_hat.dot(y) / y_hat.dot(y_hat)
