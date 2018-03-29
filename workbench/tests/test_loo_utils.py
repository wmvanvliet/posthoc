from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_true, assert_greater
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut

from workbench import Workbench
from workbench.loo_utils import loo_patterns_from_model


def _gen_data(noise_scale=2, zero_mean=False):
    """Generate some testing data.

    Parameters
    ----------
    noise_scale : float
        The amount of noise (in standard deviations) to add to the data.
    zero_mean : bool
        Whether X and y should be zero-mean (across samples) or not.
        Defaults to False.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The measured data.
    Y : ndarray, shape (n_samples, n_targets)
        The latent variables generating the data.
    A : ndarray, shape (n_features, n_targets)
        The forward model, mapping the latent variables (=Y) to the measured
        data (=X).
    """
    # Fix random seed for consistent tests
    random = np.random.RandomState(42)

    N = 100  # Number of samples
    M = 5  # Number of features

    # Y has 3 targets and the following covariance:
    cov_Y = np.array([
        [10, 1, 2],
        [1,  5, 1],
        [2,  1, 3],
    ]).astype(float)
    mean_Y = np.array([1, -3, 7])
    Y = random.multivariate_normal(mean_Y, cov_Y, size=N)
    Y -= Y.mean(axis=0)

    # The pattern (=forward model)
    A = np.array([
        [1, 10, -3],
        [4,  1,  8],
        [3, -2,  4],
        [1,  1,  1],
        [7,  6,  0],
    ]).astype(float)

    # The noise covariance matrix
    cov_noise = np.array([
        [1.25,  0.89,  1.06,  0.99,  1.27],
        [0.89,  1.10,  1.17,  1.08,  1.14],
        [1.06,  1.17,  1.32,  1.28,  1.36],
        [0.99,  1.08,  1.28,  1.37,  1.34],
        [1.27,  1.14,  1.36,  1.34,  1.60],
    ])
    mean_noise = np.zeros(M)
    noise = random.multivariate_normal(mean_noise, cov_noise, size=N)

    # Construct X
    X = Y.dot(A.T)
    X += noise_scale * noise

    if zero_mean:
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)

    return X, Y, A


def test_loo_patterns_from_model():
    X, y, A = _gen_data(noise_scale=0.1)

    # Normal path
    model = Ridge(fit_intercept=False)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = Ridge(fit_intercept=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = Ridge(normalize=True)
    p0 = Workbench(model).fit(X[1:], y[1:]).pattern_normalized_
    p1, _ = next(loo_patterns_from_model(model, X, y))
    assert_allclose(p0, p1)

    # Optimized path for LinearRegression
    model = LinearRegression(fit_intercept=False)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = LinearRegression(fit_intercept=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = LinearRegression(normalize=True)
    p0 = Workbench(model).fit(X[1:], y[1:]).pattern_normalized_
    p1, _ = next(loo_patterns_from_model(model, X, y))
    assert_allclose(p0, p1)
