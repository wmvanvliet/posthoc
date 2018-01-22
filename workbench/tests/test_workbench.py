from __future__ import print_function
from __future__ import division

from numpy.testing import assert_allclose
from nose.tools import assert_true, assert_greater
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import numpy as np

from workbench import Workbench, WorkbenchOptimizer, ShrinkageUpdater


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

    N = 1000  # Number of samples
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

    # Y = Y[:, :1]
    # A = A[:1, :1]
    # noise = noise[:, :1]

    # Construct X
    X = Y.dot(A.T)
    X += noise_scale * noise

    if zero_mean:
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)

    return X, Y, A


def _compare_models(wb, base, X, atol=0, rtol=1E-7):
    """Compare a Workbench model with a base model."""
    assert_allclose(wb.coef_, base.coef_, atol, rtol)
    assert_allclose(wb.intercept_, base.intercept_, atol, rtol)

    y_hat_wb = wb.predict(X)
    y_hat_base = base.predict(X)
    assert_allclose(y_hat_wb, y_hat_base, atol, rtol)


def test_pattern_computation():
    """Test computation of the pattern from a linear model."""
    X, y, A = _gen_data(noise_scale=0)

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


def test_identity_transform():
    """Test disassembling and re-assembling a model as-is."""
    X, y, A = _gen_data()
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
    X, y, A = _gen_data()
    train = np.arange(500)  # Samples used as training set
    test = train + 500  # Samples used as test set

    # The shrinkage parameter
    alpha = 5

    # The ridge regressor we're going to imitate
    ridge = Ridge(alpha, normalize=True).fit(X[train], y[train])

    # The OLS regressor we're going to adapt to be a ridge regressor
    ols = LinearRegression(normalize=True).fit(X[train], y[train])

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
    _compare_models(wb, ridge, X[test])

    # Test using a CovUpdater object
    wb = Workbench(ols, cov_updater=ShrinkageUpdater(alpha),
                   method='traditional')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test])

    wb = Workbench(ols, cov_updater=ShrinkageUpdater(alpha), method='kernel')
    wb.fit(X[train], y[train])
    _compare_models(wb, ridge, X[test])


def test_post_hoc_modification():
    """Test post-hoc modification of the model."""
    X, y, A = _gen_data(noise_scale=10)

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
    X, y, A = _gen_data(noise_scale=10)

    train = np.arange(5)         # Samples used as training set
    test = np.arange(5, len(X))  # Samples used as test set

    # The OLS regressor we're going to try and optimize
    ols = LinearRegression().fit(X[train], y[train])
    ols_score = ols.score(X[test], y[test])

    wb = WorkbenchOptimizer(LinearRegression(), cov_updater=ShrinkageUpdater())
    wb.fit(X[train], y[train])
    wb_score = wb.score(X[test], y[test])

    # The optimized model should be better
    assert_greater(wb_score, ols_score)


def test_workbench_optimizer2():
    """Test using an optimizer to fine-tune parameters."""
    X, y, A = _gen_data(noise_scale=10)

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
                            pattern_param_bounds=[(0, 1)])
    wb.fit(X[train], y[train])
    wb_score = wb.score(X[test], y[test])

    rr = RidgeCV().fit(X[train], y[train])
    rr_score = rr.score(X[test], y[test])

    print('OLS:', ols_score, 'RR:', rr_score, 'WB:', wb_score)
    assert_greater(wb_score, ols_score)
    assert_greater(wb_score, 0.5)  # The new regression should be awesome
