from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut

from posthoc import Workbench
from posthoc.utils import gen_data
from posthoc.loo_utils import (loo, loo_mean_norm, loo_patterns_from_model,
                               loo_ols_regression, loo_kernel_regression)


def test_loo():
    """Test in-place leave-one-out generation."""
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    X_loo = loo(X)
    assert_array_equal(next(X_loo), np.array([[4, 5, 6], [7, 8, 9]]))
    assert_array_equal(next(X_loo), np.array([[1, 2, 3], [7, 8, 9]]))
    assert_array_equal(next(X_loo), np.array([[4, 5, 6], [1, 2, 3]]))

    X_loo = loo(X, axis=1)
    assert_array_equal(next(X_loo), np.array([[2, 3], [5, 6], [8, 9]]))
    assert_array_equal(next(X_loo), np.array([[1, 3], [4, 6], [7, 9]]))
    assert_array_equal(next(X_loo), np.array([[2, 1], [5, 4], [8, 7]]))


def test_loo_mean_norm():
    """Test efficient LOO computation of mean and norm."""
    X, y, _ = gen_data(N=5)
    means_norms = loo_mean_norm(X)
    for train, _ in LeaveOneOut().split(X, y):
        mean, norm = next(means_norms)
        assert_allclose(mean, X[train].mean(axis=0, keepdims=True))
        assert_allclose(norm, np.std(X[train], axis=0, keepdims=True, ddof=3))


def test_loo_patterns_from_model():
    """Test efficient generation of LOO patterns and normalizers."""
    X, y, A = gen_data(N=10, noise_scale=0)

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

    # Test with normalization
    model = Ridge(normalize=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_normalized_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    # Optimized path for LinearRegression
    model = LinearRegression(fit_intercept=False)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1, atol=1E-12)
        assert_allclose(n0, n1, atol=1E-12)

    model = LinearRegression(fit_intercept=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1, atol=1E-12)
        assert_allclose(n0, n1, atol=1E-12)

    # Test with normalization
    model = LinearRegression(normalize=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_normalized_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1, atol=1E-12)
        assert_allclose(n0, n1, atol=1E-12)


def test_loo_ols_regression():
    """Test generating LOO regression coefficients."""
    X, y, _ = gen_data(N=10)

    coef_gen = loo_ols_regression(X, y, fit_intercept=False, normalize=False)
    base = LinearRegression(fit_intercept=False, normalize=False)
    for X_, y_, coef_ in zip(loo(X), loo(y), coef_gen):
        assert_allclose(base.fit(X_, y_).coef_, coef_, rtol=1E-6)

    coef_gen = loo_ols_regression(X, y, fit_intercept=True, normalize=False)
    base = LinearRegression(fit_intercept=True, normalize=False)
    for X_, y_, coef_ in zip(loo(X), loo(y), coef_gen):
        assert_allclose(base.fit(X_, y_).coef_, coef_, rtol=1E-6)

    coef_gen = loo_ols_regression(X, y, fit_intercept=True, normalize=True)
    base = LinearRegression(fit_intercept=True, normalize=True)
    for X_, y_, coef_ in zip(loo(X), loo(y), coef_gen):
        assert_allclose(base.fit(X_, y_).coef_, coef_, rtol=1E-7)

def test_loo_kernel_regression():
    """Test generating LOO regression coefs using kernel formulation."""
    from sklearn import datasets
    X, y = datasets.make_regression(n_samples=10, n_features=1000,
                                    n_targets=1)

    coef_gen = loo_kernel_regression(X, y, fit_intercept=False, normalize=False)
    base = LinearRegression(fit_intercept=False, normalize=False)
    for X_, y_, coef_ in zip(loo(X), loo(y), coef_gen):
        assert_allclose(base.fit(X_, y_).coef_, coef_)

    coef_gen = loo_kernel_regression(X, y, fit_intercept=True, normalize=False)
    base = LinearRegression(fit_intercept=True, normalize=False)
    for X_, y_, coef_ in zip(loo(X), loo(y), coef_gen):
        print(base.fit(X_, y_).intercept_)
        assert_allclose(base.fit(X_, y_).coef_, coef_)
