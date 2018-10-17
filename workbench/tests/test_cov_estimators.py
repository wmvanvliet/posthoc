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


def _func(X, y=None):
    return X.T.dot(X) / 10


class TestFunction():
    """Test estimating covariance with a custom function."""
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
                            pinv(X_.T.dot(X_) + mod_alpha * np.eye(m)).dot(P),
                            atol=1e-4)

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
        shrink = cov_estimators.ShrinkageKernel().fit(X)
        for X_, XP in zip(loo(X), shrink.loo_inv_dot(X, [P] * 10)):
            target = (np.trace(X_.T.dot(X_)) / m) * np.eye(m)
            assert_allclose(
                XP, pinv((1 - alpha) * X_.T.dot(X_) + alpha * target).dot(P),
                atol=1e-4)

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


class TestKronecker():
    """Test estimation of covariance with Kronkecker regularization."""
    def test_basic_cov(self):
        """Test basic cov estimation."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=100, n_features=10,
                                        n_targets=1)
        cov = X.T.dot(X)
        m = X.shape[1]

        # Compute spatial covariance
        X_ = X.reshape(100, 2, 5).transpose(1, 0, 2).reshape(2, -1)
        gamma = np.trace(cov) / m
        spat_cov = X_.dot(X_.T)

        # Compute the shrinkage target
        alpha = 0.9
        beta = 0.3
        target = beta * np.kron(spat_cov, np.eye(5)) + (1 - beta) * cov
        target = alpha * gamma * np.eye(m) + (1 - alpha) * target

        # Test the Kronkecker object
        kron = cov_estimators.Kronecker(2, 5, alpha, beta).fit(X)
        assert_allclose(kron.cov, cov)
        assert_allclose(kron.diag_loading, gamma)
        assert_allclose(kron.outer_mat_, spat_cov)
        assert_allclose(kron.cov_reg, target)

    def test_inv_dot(self):
        """Test inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=100, n_features=10,
                                        n_targets=1)
        cov = X.T.dot(X)
        m = X.shape[1]

        # Compute spatial covariance
        X_ = X.reshape(100, 2, 5).transpose(1, 0, 2).reshape(2, -1)
        gamma = np.trace(cov) / m
        spat_cov = X_.dot(X_.T)

        # Compute the shrinkage target
        alpha = 0.9
        beta = 0.3
        target = beta * np.kron(spat_cov, np.eye(5)) + (1 - beta) * cov
        target = alpha * gamma * np.eye(m) + (1 - alpha) * target

        # Test the KronkeckerKernel object
        kron = cov_estimators.Kronecker(2, 5, alpha, beta).fit(X)
        assert_allclose(kron.outer_mat_, spat_cov)
        assert_allclose(kron.diag_loading, gamma)
        assert_allclose(kron.cov_reg, target)

        # Test the inv_dot method
        P = np.random.randn(m, 1)
        assert_allclose(kron.inv_dot(X, P), pinv(target).dot(P))

    def test_loo_inv_dot(self):
        """Test loo_inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=10,
                                        n_targets=1)
        m = X.shape[1]
        alpha = 0.9
        beta = 0.3

        # Test the inv_dot method
        P = np.random.randn(m, 1)
        kron = cov_estimators.Kronecker(2, 5, alpha, beta).fit(X)
        for X_, XP in zip(loo(X), kron.loo_inv_dot(X, [P] * 10)):
            cov = X_.T.dot(X_)

            # Compute spatial covariance
            X_ = X_.reshape(9, 2, 5).transpose(1, 0, 2).reshape(2, -1)
            gamma = np.trace(cov) / m
            spat_cov = X_.dot(X_.T)

            # Compute the shrinkage target
            target = beta * np.kron(spat_cov, np.eye(5)) + (1 - beta) * cov
            target = alpha * gamma * np.eye(m) + (1 - alpha) * target
            assert_allclose(XP, pinv(target).dot(P))
            print('good')

    def test_kronecker_dot(self):
        """Test efficient Kronecker dot function."""
        outer_mat = np.random.randn(5, 5)
        X = np.random.randn(100, 10)
        kron = cov_estimators.Kronecker(5, 2).fit(X)
        assert_allclose(kron._kronecker_dot(outer_mat, X.T),
                        np.kron(outer_mat, np.eye(2)).dot(X.T))

    def test_update(self):
        """Test updating the alpha and beta parameters."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=100, n_features=10,
                                        n_targets=1)
        cov = X.T.dot(X)
        m = X.shape[1]

        # Compute spatial covariance
        X_ = X.reshape(100, 2, 5).transpose(1, 0, 2).reshape(2, -1)
        gamma = np.trace(cov) / m
        spat_cov = X_.dot(X_.T)

        # Compute the shrinkage target
        alpha = 0.9
        beta = 0.3
        target = beta * np.kron(spat_cov, np.eye(5)) + (1 - beta) * cov
        target = alpha * gamma * np.eye(m) + (1 - alpha) * target

        # Initialize Kronecker object with different parameters
        kron = cov_estimators.Kronecker(2, 5, alpha=0.5, beta=0.7).fit(X)
        # Update to current parameters
        kron = kron.update(X, alpha, beta)
        assert_allclose(kron.cov, cov)
        assert_allclose(kron.diag_loading, gamma)
        assert_allclose(kron.outer_mat_, spat_cov)
        assert_allclose(kron.cov_reg, target)

        P = np.random.randn(m, 1)
        assert_allclose(kron.inv_dot(X, P), pinv(target).dot(P))


class TestKroneckerKernel():
    """Test kernel formulation of kronecker regularization."""
    def test_inv_dot(self):
        """Test inv_dot method."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=100, n_features=1000,
                                        n_targets=1)
        cov = X.T.dot(X)
        m = X.shape[1]

        # Compute spatial covariance
        X_ = X.reshape(100, 20, 50).transpose(1, 0, 2).reshape(20, -1)
        gamma = np.trace(cov) / m
        spat_cov = X_.dot(X_.T)

        # Compute the shrinkage target
        alpha = 0.9
        beta = 0.3
        target = beta * np.kron(spat_cov, np.eye(50)) + (1 - beta) * cov
        target = alpha * gamma * np.eye(m) + (1 - alpha) * target

        # Test the KronkeckerKernel object
        kron = cov_estimators.KroneckerKernel(20, 50, alpha, beta).fit(X)
        assert_allclose(kron.outer_mat_, spat_cov)
        assert_allclose(kron.diag_loading, gamma)

        # Test the inv_dot method
        P = np.random.randn(m, 1)
        assert_allclose(kron.inv_dot(X, P), pinv(target).dot(P))

    def test_update(self):
        """Test updating the alpha parameter."""
        from sklearn import datasets
        X, _ = datasets.make_regression(n_samples=10, n_features=1000,
                                        n_targets=1)
        cov = X.T.dot(X)
        m = X.shape[1]
        gamma = np.trace(cov) / m
        spat_cov = X.reshape(20, -1).dot(X.reshape(20, -1).T)
        P = np.random.randn(m, 1)
        alpha = 0.5
        beta = 0.3
        new_alpha = 0.7
        new_beta = 0.5
        new_target = new_alpha * gamma * np.eye(m)
        new_target += (1 - new_alpha)
        new_target *= new_beta * np.kron(spat_cov, np.eye(50)) + (1 - new_beta) * cov
        kron = cov_estimators.KroneckerKernel(20, 50, alpha, beta).fit(X)
        kron = kron.update(X, new_alpha, new_beta)
        assert_allclose(kron.inv_dot(X, P), pinv(new_target).dot(P))

    def test_kronecker_dot(self):
        """Test efficient Kronecker dot function."""
        outer_mat = np.random.randn(5, 5)
        X = np.random.randn(100, 10)
        kron = cov_estimators.KroneckerKernel(5, 2).fit(X)
        assert_allclose(kron._kronecker_dot(outer_mat, X.T),
                        np.kron(outer_mat, np.eye(2)).dot(X.T))
