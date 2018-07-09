from __future__ import print_function
from __future__ import division

from numpy.testing import assert_allclose
from nose.tools import assert_true, assert_greater
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import numpy as np

from workbench import Workbench, WorkbenchOptimizer, ShrinkageUpdater
from workbench.utils import gen_data


def _compare_models(wb, base, X, atol=0, rtol=1E-7):
    """Compare a Workbench model with a base model."""
    assert_allclose(wb.coef_, base.coef_, atol, rtol)
    assert_allclose(wb.intercept_, base.intercept_, atol, rtol)

    y_hat_wb = wb.predict(X)
    y_hat_base = base.predict(X)
    assert_allclose(y_hat_wb, y_hat_base, atol, rtol)


def test_pattern_computation():
    """Test computation of the pattern from a linear model."""
    X, y, A = gen_data(noise_scale=0)

    assert_true((X.std(axis=0) != 0).all())
    assert_true((X.mean(axis=0) != 0).all())

    assert_true((y.std(axis=0) != 0).all())
    assert_true((y.mean(axis=0) != 0).all())

    for normalize in [False, True]:
        # The base model is a simple OLS regressor
        ols = LinearRegression(normalize=normalize)

        # Don't specify any modifications, should trigger shortcut path
        wb = Workbench(ols).fit(X, y)

        # Test pattern computation
        assert_allclose(wb.pattern_, A, atol=1E-7)


def test_inverse_predict():
    X, y, _ = gen_data(noise_scale=0)
    w = Workbench(LinearRegression()).fit(X, y)
    m = LinearRegression().fit(y, X)
    assert_allclose(w.pattern_, m.coef_, atol=1E-15)
    assert_allclose(w.inverse_predict(y), m.predict(y))


def test_identity_transform():
    """Test disassembling and re-assembling a model as-is."""
    X, y, A = gen_data()
    train = np.arange(500)  # Samples used as training set
    test = train + 500  # Samples used as test set

    # The base model is a simple OLS regressor
    ols = LinearRegression().fit(X[train], y[train])

    # Specify identity modifications.
    def cov_modifier(cov, X, y):
        return cov

    def cov_updater(X, y):
        n = X.shape[1]  # Number of features
        return np.zeros((n, n))

    def pattern_modifier(pattern, X, y):
        return pattern

    def normalizer_modifier(normalizer, X_train, y_train, cov_X, coef):
        return normalizer

    # Using a modifier function
    wb = Workbench(ols, cov_modifier=cov_modifier,
                   pattern_modifier=pattern_modifier,
                   normalizer_modifier=normalizer_modifier,
                   method='traditional')
    wb.fit(X[train], y[train])
    _compare_models(wb, ols, X[test])

    # Using an updater function. We have to use the "traditional" solver.
    # The "kernel" solver cannot be used, as an update rule of all zero's does
    # not have an inverse.
    wb = Workbench(LinearRegression(), cov_updater=cov_updater,
                   pattern_modifier=pattern_modifier,
                   normalizer_modifier=normalizer_modifier,
                   method='traditional')
    wb.fit(X[train], y[train])
    _compare_models(wb, ols, X[test])


def test_ridge_regression():
    """Test post-hoc adaptation of OLS to be a ridge regressor."""
    X, y, A = gen_data()
    train = np.arange(500)  # Samples used as training set
    test = train + 500  # Samples used as test set

    # The shrinkage parameter
    alpha = 5

    # The ridge regressor we're going to imitate
    ridge = Ridge(alpha, normalize=False).fit(X[train], y[train])

    # The OLS regressor we're going to adapt to be a ridge regressor
    ols = LinearRegression(normalize=False).fit(X[train], y[train])

    # Perform the post-hoc adaptation using different code paths
    def cov_modifier(cov, X, y):
        n = X.shape[1]  # Number of features
        return cov + alpha * np.eye(n)

    def cov_updater(X, y):
        n = X.shape[1]  # Number of features
        return alpha * np.eye(n)

    wb = Workbench(ols, cov_modifier=cov_modifier, method='traditional')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test])

    wb = Workbench(ols, cov_updater=cov_updater, method='traditional')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test])

    # Try the "kernel" method
    wb = Workbench(ols, cov_updater=cov_updater, method='kernel')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test], rtol=1E-5)

    # Test using a CovUpdater object
    cov_updater = ShrinkageUpdater(alpha, scale_by_trace=False)
    wb = Workbench(ols, cov_updater=cov_updater, method='traditional')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test])

    wb = Workbench(ols, cov_updater=cov_updater, method='kernel')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test], rtol=1E-5)


def test_post_hoc_modification():
    """Test post-hoc modification of the model."""
    X, y, A = gen_data(noise_scale=10)

    train = np.arange(5)         # Samples used as training set
    test = np.arange(5, len(X))  # Samples used as test set

    # The OLS regressor we're going to try and optimize
    ols = LinearRegression().fit(X[train], y[train])
    ols_score = ols.score(X[test], y[test])

    # Use post-hoc adaptation to swap out the cov matrix estimated on the
    # training data only, with one that was estimated on all data.
    def cov_modifier(cov, X_train, y_train):
        X_ = X - X.mean(axis=0)
        return X_.T.dot(X_) / len(X_)

    # Use post-hoc adaptation to swap out the estimated pattern with the actual
    # pattern.
    def pattern_modifier(pattern, X_train, y_train):
        return A  # The ground truth pattern

    # Use post-hoc adaptation to swap out the normalizer estimated on the
    # training data only, with one that was estimated on all data.
    def normalizer_modifier(normalizer, X_train, y_train, cov_X, coef):
        y_ = y - y.mean(axis=0)
        return y_.T.dot(y_) / len(y_)  # The ground truth normalizer

    wb = Workbench(LinearRegression(),
                   cov_modifier=cov_modifier,
                   pattern_modifier=pattern_modifier,
                   normalizer_modifier=normalizer_modifier)
    wb.fit(X[train], y[train])

    # The new regressor should perform better
    wb_score = wb.score(X[test], y[test])
    assert_greater(wb_score, ols_score)
    assert_greater(wb_score, 0.9)  # The new regression should be awesome


def test_workbench_optimizer():
    """Test using an optimizer to fine-tune parameters."""
    X, y, A = gen_data(noise_scale=50, N=100)

    for method in ['auto', 'traditional', 'kernel']:
        wbo = WorkbenchOptimizer(LinearRegression(normalize=True),
                                 cov_updater=ShrinkageUpdater(), scoring='r2',
                                 method=method, cov_param_x0=[0.2],
                                 cov_param_bounds=[(0.1, None)],
                                 optimizer_options={'maxiter': 100})
        wbo.fit(X, y)

        # Weights of the WorkbenchOptimizer should equal the weights of a
        # Workbench initialized with the optimal parameters.
        wb = Workbench(LinearRegression(normalize=True),
                       cov_updater=ShrinkageUpdater(*wbo.cov_updater_params_),
                       method=method)
        wb.fit(X, y)
        assert_allclose(wbo.coef_, wb.coef_)
        assert_allclose(wbo.intercept_, wb.intercept_)

        # Weights of the WorkbenchOptimizer should equal the weights of a
        # Ridge initialized with the optimal parameters.
        rr = Ridge(alpha=wbo.cov_updater_params_[0], normalize=True)
        rr.fit(X, y)
        assert_allclose(wbo.coef_, rr.coef_)
        assert_allclose(wbo.intercept_, rr.intercept_)


def test_workbench_optimizer2():
    """Test using an optimizer to fine-tune parameters."""
    X, y, A = gen_data(noise_scale=10)

    train = np.arange(5)         # Samples used as training set
    test = np.arange(5, len(X))  # Samples used as test set

    # The OLS regressor we're going to try and optimize
    ols = LinearRegression().fit(X[train], y[train])
    ols_score = ols.score(X[test], y[test])

    X_ = X - X.mean(axis=0)
    cov_X = X_.T.dot(X_) / len(X_)

    def cov_updater(X, y, alpha=0.5):
        cov = X.T.dot(X)
        return (1 - alpha) * cov + alpha * cov_X * len(X)

    def pattern_modifier(pattern, X, y, beta=0.5):
        return (1 - beta) * pattern + beta * A

    wb = WorkbenchOptimizer(LinearRegression(), cov_updater=cov_updater,
                            pattern_modifier=pattern_modifier,
                            cov_param_x0=[0.5], cov_param_bounds=[(0, 1)],
                            pattern_param_x0=[0.5],
                            pattern_param_bounds=[(0, 1)],
                            method='traditional', scoring='r2')
    wb.fit(X[train], y[train])
    wb_score = wb.score(X[test], y[test])

    rr = RidgeCV().fit(X[train], y[train])
    rr_score = rr.score(X[test], y[test])

    print('OLS:', ols_score, 'RR:', rr_score, 'WB:', wb_score)
    assert_greater(wb_score, ols_score)
    assert_greater(wb_score, 0.5)  # The new regression should be awesome
