"""
Some more scoring functions to use with scikit-learn model selection functions.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.utils.extmath import log_logistic


def logistic_loss_score(model, X, y):
    """
    Logistic loss score.

    This measure serves as an indicator for the quality of a binary classifier.
    It is computed by applying the sigmoid function to the raw output of the
    model.
    """
    if hasattr(model, 'decision_function'):
        y_hat = model.decision_function(X)
    elif hasattr(model, 'coef_'):
        y_hat = X.dot(model.coef_.T)
    else:
        y_hat = model.predict(X)

    # Minimize logistic loss
    perf = np.sum(log_logistic(y * y_hat))
    return perf


def correlation_score(model, X, y):
    """
    Pearson correlation score.

    Pearson correlation is a scale-free indicator for the quality of a
    regression model. Use this when you don't care about the scaling of the
    output, just whether the output is a good predictor of the target variable.
    """
    return pearsonr(model.predict(X), y)[0]
