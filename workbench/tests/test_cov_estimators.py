from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from numpy.linalg import pinv

from workbench import cov_estimators
from workbench.utils import gen_data
from workbench.loo_utils import loo


class TestEmpirical():
    """Test empirical estimation of covariance."""
    def test_basic_cov(self):
        """Test basic cov estimation."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        emp = cov_estimators.Empirical().fit(X)
        assert_allclose(emp.cov, X.T.dot(X))
        assert_allclose(emp.cov_inv, pinv(X.T.dot(X)))

    def test_inv_dot(self):
        """Test inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=100)
        emp = cov_estimators.Empirical().fit(X)
        assert_allclose(emp.inv_dot(X, P), pinv(X.T.dot(X)).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=10)
        emp = cov_estimators.Empirical().fit(X)
        for X_, XP in zip(loo(X), emp.loo_inv_dot(X, [P] * 10)):
            assert_allclose(XP, pinv(X_.T.dot(X_)).dot(P))


class TestL2():
    """Test estimation of covariance with l2 regularization."""
    def test_basic_cov(self):
        """Test basic cov estimation."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 1

        l2 = cov_estimators.L2(alpha, scale_by_var=False).fit(X)
        assert_allclose(l2.cov, X.T.dot(X))
        assert_allclose(l2.cov_reg, X.T.dot(X) + alpha * np.eye(m))
        assert_allclose(l2.cov_inv, pinv(X.T.dot(X) + alpha * np.eye(m)))

    def test_scale_by_var(self):
        """Test scaling by var."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 1

        mod_alpha = alpha * np.trace(X.T.dot(X)) / m
        l2 = cov_estimators.L2(alpha, scale_by_var=True).fit(X)
        assert_allclose(l2.cov, X.T.dot(X))
        assert_allclose(l2.cov_reg, X.T.dot(X) + mod_alpha * np.eye(m))
        assert_allclose(l2.cov_inv, pinv(X.T.dot(X) + mod_alpha * np.eye(m)))

    def test_update(self):
        """Test updating the alpha parameter."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 1
        l2 = cov_estimators.L2(alpha, scale_by_var=False).fit(X)
        new_alpha = 2
        l2 = l2.update(X, new_alpha)
        assert_allclose(l2.cov_reg, X.T.dot(X) + new_alpha * np.eye(m))

        # Test with scale_by_var
        mod_alpha = alpha * np.trace(X.T.dot(X)) / m
        l2 = cov_estimators.L2(mod_alpha, scale_by_var=True).fit(X)
        new_mod_alpha = new_alpha * np.trace(X.T.dot(X)) / m
        l2 = l2.update(X, new_alpha)
        assert_allclose(l2.cov_reg, X.T.dot(X) + new_mod_alpha * np.eye(m))

    def test_inv_dot(self):
        """Test inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 1
        l2 = cov_estimators.L2(alpha, scale_by_var=False).fit(X)
        assert_allclose(l2.inv_dot(X, P),
                        pinv(X.T.dot(X) + alpha * np.eye(m)).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=10)
        m = X.shape[1]
        alpha = 1
        l2 = cov_estimators.L2(alpha, scale_by_var=False).fit(X)
        for X_, XP in zip(loo(X), l2.loo_inv_dot(X, [P] * 10)):
            assert_allclose(XP, pinv(X_.T.dot(X_) + alpha * np.eye(m)).dot(P))

        # Test with scale_by_var
        l2 = cov_estimators.L2(alpha, scale_by_var=True).fit(X)
        for X_, XP in zip(loo(X), l2.loo_inv_dot(X, [P] * 10)):
            mod_alpha = alpha * np.trace(X_.T.dot(X_)) / m
            assert_allclose(XP,
                            pinv(X_.T.dot(X_) + mod_alpha * np.eye(m)).dot(P))


class TestL2Kernel():
    """Test kernel formulation of L2 regularization."""
    def test_inv_dot(self):
        """Test inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        m = X.shape[1]
        P = np.random.randn(m, 1)
        alpha = 1
        l2 = cov_estimators.L2Kernel(alpha, scale_by_var=False).fit(X)
        assert_allclose(l2.inv_dot(X, P),
                        pinv(X.T.dot(X) + alpha * np.eye(m)).dot(P))

        # Test with scale_by_var
        mod_alpha = alpha * np.trace(X.T.dot(X)) / m
        l2 = cov_estimators.L2Kernel(alpha, scale_by_var=True).fit(X)
        assert_allclose(l2.inv_dot(X, P),
                        pinv(X.T.dot(X) + mod_alpha * np.eye(m)).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        m = X.shape[1]
        P = np.random.randn(m, 1)
        alpha = 1
        l2 = cov_estimators.L2Kernel(alpha, scale_by_var=False).fit(X)
        for X_, XP in zip(loo(X), l2.loo_inv_dot(X, [P] * 10)):
            assert_allclose(XP, pinv(X_.T.dot(X_) + alpha * np.eye(m)).dot(P))

        # Test with scale_by_var
        l2 = cov_estimators.L2Kernel(alpha, scale_by_var=True).fit(X)
        for X_, XP in zip(loo(X), l2.loo_inv_dot(X, [P] * 10)):
            mod_alpha = alpha * np.trace(X_.T.dot(X_)) / m
            assert_allclose(XP,
                            pinv(X_.T.dot(X_) + mod_alpha * np.eye(m)).dot(P))

    def test_update(self):
        """Test updating the alpha parameter."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        m = X.shape[1]
        P = np.random.randn(m, 1)
        alpha = 1
        l2 = cov_estimators.L2Kernel(alpha, scale_by_var=False).fit(X)
        new_alpha = 2
        l2 = l2.update(X, new_alpha)
        assert_allclose(l2.inv_dot(X, P),
                        pinv(X.T.dot(X) + new_alpha * np.eye(m)).dot(P))

        # Test with scale_by_var
        mod_alpha = alpha * np.trace(X.T.dot(X)) / m
        l2 = cov_estimators.L2Kernel(mod_alpha, scale_by_var=True).fit(X)
        new_mod_alpha = new_alpha * np.trace(X.T.dot(X)) / m
        l2 = l2.update(X, new_alpha)
        assert_allclose(l2.inv_dot(X, P),
                        pinv(X.T.dot(X) + new_mod_alpha * np.eye(m)).dot(P))


class TestShrinkage():
    """Test estimation of covariance with shrinkage regularization."""
    def test_basic_cov(self):
        """Test basic cov estimation."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)

        shrink = cov_estimators.Shrinkage(alpha).fit(X)
        assert_allclose(shrink.cov, X.T.dot(X))
        assert_allclose(shrink.cov_reg,
                        (1 - alpha) * X.T.dot(X) + alpha * target)
        assert_allclose(shrink.cov_inv,
                        pinv((1 - alpha) * X.T.dot(X) + alpha * target))

    def test_update(self):
        """Test updating the alpha parameter."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)
        shrink = cov_estimators.Shrinkage(alpha).fit(X)
        assert_allclose(shrink.cov_reg,
                        (1 - alpha) * X.T.dot(X) + alpha * target)

        new_alpha = 0.75
        shrink = shrink.update(X, new_alpha)
        assert_allclose(shrink.cov_reg,
                        (1 - new_alpha) * X.T.dot(X) + new_alpha * target)

    def test_inv_dot(self):
        """Test inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=100)
        m = X.shape[1]
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)
        shrink = cov_estimators.Shrinkage(alpha).fit(X)
        assert_allclose(shrink.inv_dot(X, P),
                        pinv((1 - alpha) * X.T.dot(X) + alpha * target).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=10)
        m = X.shape[1]
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)
        shrink = cov_estimators.Shrinkage().fit(X)
        for X_, XP in zip(loo(X), shrink.loo_inv_dot(X, [P] * 10)):
            assert_allclose(
                XP, pinv((1 - alpha) * X_.T.dot(X_) + alpha * target).dot(P))


class TestShrinkageKernel():
    """Test kernel formulation of shrinkage regularization."""
    def test_inv_dot(self):
        """Test inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        m = X.shape[1]
        P = np.random.randn(m, 1)
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)
        shrink = cov_estimators.ShrinkageKernel(alpha).fit(X)
        assert_allclose(shrink.inv_dot(X, P),
                        pinv((1 - alpha) * X.T.dot(X) + alpha * target).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        m = X.shape[1]
        P = np.random.randn(m, 1)
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)
        shrink = cov_estimators.ShrinkageKernel().fit(X)
        for X_, XP in zip(loo(X), shrink.loo_inv_dot(X, [P] * 10)):
            assert_allclose(
                XP, pinv((1 - alpha) * X_.T.dot(X_) + alpha * target).dot(P))

    def test_update(self):
        """Test updating the alpha parameter."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        m = X.shape[1]
        P = np.random.randn(m, 1)
        alpha = 0.5
        target = (np.trace(X.T.dot(X)) / m) * np.eye(m)
        shrink = cov_estimators.ShrinkageKernel(alpha).fit(X)
        new_alpha = 0.75
        shrink = shrink.update(X, new_alpha)
        assert_allclose(
            shrink.inv_dot(X, P),
            pinv((1 - new_alpha) * X.T.dot(X) + new_alpha * target).dot(P))


def _func(X, y=None):
    return X.T.dot(X) / 10


class TestFunction():
    """Test estimating covariance with a custom function."""

    def test_basic_cov(self):
        """Test basic cov estimation."""
        X, _, _ = gen_data(zero_mean=True, N=100)
        emp = cov_estimators.Function(_func).fit(X)
        assert_allclose(emp.cov, _func(X))
        assert_allclose(emp.cov_inv, pinv(_func(X)))

    def test_inv_dot(self):
        """Test inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=100)
        emp = cov_estimators.Function(_func).fit(X)
        assert_allclose(emp.inv_dot(X, P), pinv(_func(X)).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        X, _, P = gen_data(zero_mean=True, N=10)
        emp = cov_estimators.Function(_func).fit(X)
        for X_, XP in zip(loo(X), emp.loo_inv_dot(X, [P] * 10)):
            assert_allclose(XP, pinv(_func(X_)).dot(P))
