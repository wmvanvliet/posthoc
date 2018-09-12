from copy import deepcopy
import numpy as np
from numpy.linalg import multi_dot, pinv

from .loo_utils import loo, loo_kern_inv


class CovEstimator(object):
    def fit(self, X, y=None):
        return self

    def update(self, X):
        return self.copy()

    def inv_dot(self):
        raise NotImplemented('This function must be implemented in a subclass')

    def loo_inv_dot(self):
        raise NotImplemented('This function must be implemented in a subclass')

    def get_x0(self):
        return []

    def get_bounds(self):
        return []

    def copy(self):
        return deepcopy(self)


class Empirical(CovEstimator):
    def fit(self, X, y=None):
        self.cov_inv = pinv(X.T @ X)
        return self

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        return self.cov_inv @ P

    def loo_inv_dot(self, X, Ps):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        for x, P in zip(X, Ps):
            # Update cov_inv using Sherman–Morrison formula
            tmp = self.cov_inv @ x[:, np.newaxis]
            cov_inv = self.cov_inv + (tmp @ tmp.T) / (1 + x @ tmp)
            yield cov_inv @ P


class L2(Empirical):
    def __init__(self, alpha=1, scale_by_var=True):
        super().__init__()
        self.alpha = alpha
        self.scale_by_var = scale_by_var

    def fit(self, X, y=None):
        cov = X.T @ X
        if self.scale_by_var:
            self.mean_var = np.trace(cov) / len(cov)
        # Add to the diagonal in-place
        cov_reg = cov.copy()
        cov_reg.flat[::len(cov_reg) + 1] += self.alpha * self.mean_var
        self.cov = cov
        self.cov_reg = cov_reg
        self.cov_inv = pinv(cov_reg)
        return self

    def update(self, X, alpha):
        l2 = L2(alpha, self.scale_by_var)
        l2.cov = self.cov
        l2.mean_var = self.mean_var
        l2.cov_reg = self.cov.copy()
        l2.cov_reg.flat[::len(l2.cov_reg) + 1] += alpha * l2.mean_var
        l2.cov_inv = pinv(l2.cov_reg)
        return l2

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0, None)]


class Shrinkage(Empirical):
    def __init__(self, alpha=0.5):
        if not (0 <= alpha <= 1):
            raise ValueError('alpha must be between 0 and 1')
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y=None):
        cov = X.T @ X
        self.mean_var = np.trace(cov) / len(cov)
        # Add to the diagonal in-place
        cov_reg = (1 - self.alpha) * cov.copy()
        cov_reg.flat[::len(cov_reg) + 1] += self.alpha * self.mean_var
        self.cov = cov
        self.cov_reg = cov_reg
        self.cov_inv = pinv(cov_reg)
        return self

    def update(self, X, alpha):
        if not (0 <= alpha <= 1):
            raise ValueError('alpha must be between 0 and 1')
        s = Shrinkage(alpha)
        s.cov = self.cov
        s.mean_var = self.mean_var
        s.cov_reg = (1 - alpha) * s.cov_reg.copy()
        s.cov_reg.flat[::len(s.cov_reg) + 1] += alpha * s.mean_var
        s.cov_inv = pinv(s.cov_reg)
        return s

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0, None)]


class _InversionLemma(CovEstimator):
    def compute_AB_parts(self, X):
        raise NotImplemented('This function must be implemented in a subclass')

    def fit(self, X, y=None):
        A_inv, B = self.compute_AB_parts(X)
        G = A_inv * B
        K = X @ G
        K.flat[::len(K) + 1] += 1

        self.A_inv = A_inv
        self.G = G
        self.K = K
        self.K_inv = pinv(K)

        return self

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        A_inv_P = self.A_inv * P
        return A_inv_P - multi_dot((self.G, self.K_inv, X, A_inv_P))

    def loo_inv_dot(self, X, Ps):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        Xs = loo(X)
        Gs = loo(self.G, axis=1)
        K_invs = loo_kern_inv(self.K)

        for G, K_inv, X, P in zip(Gs, K_invs, Xs, Ps):
            A_inv_P = self.A_inv * P
            yield A_inv_P - multi_dot((G, K_inv, X, A_inv_P))


class L2Kernel(_InversionLemma):
    def __init__(self, alpha=1, scale_by_var=True):
        if alpha < 1E-15:
            raise ValueError('alpha must be greater than zero')
        self.alpha = alpha
        self.scale_by_var = scale_by_var

    def compute_AB_parts(self, X):
        A_inv = 1 / self.alpha
        if self.scale_by_var:
            mean_var = np.sum(X ** 2) / X.shape[1]
            A_inv /= mean_var
        B = X.T
        return A_inv, B

    def update(self, X, alpha):
        return L2Kernel(alpha, self.scale_by_var).fit(X)

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0.01, None)]


class ShrinkageKernel(_InversionLemma):
    def __init__(self, alpha=0.5, scale_by_var=True):
        if alpha < 1E-15:
            raise ValueError('alpha must be greater than zero')
        self.alpha = alpha
        self.scale_by_var = scale_by_var

    def compute_AB_parts(self, X):
        A_inv = 1 / self.alpha
        if self.scale_by_var:
            mean_var = np.sum(X ** 2) / X.shape[1]
            A_inv /= mean_var
        B = (1 - self.alpha) * X.T
        return A_inv, B

    def update(self, X, alpha):
        return ShrinkageKernel(alpha, self.scale_by_var).fit(X)

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0.01, 1)]


class KroneckerKernel(_InversionLemma):
    """
    Attributes
    ----------
    outer_cov_ : ndarray, shape (outer_size, outer_size)
        The estimated outer matrix of the covariance. Becomes available after
        calling the `fit` method.
    """
    def __init__(self, outer_size=None, inner_size=None, alpha=0.5,
                 beta=0.5):
        if outer_size is None and inner_size is None:
            raise ValueError('Either the `outer_size` or `inner_size` '
                             'parameter must be specified.')
        self.outer_size = outer_size
        self.inner_size = inner_size
        self.alpha = alpha
        self.beta = beta

    def _check_X(self, X):
        """Check whether X is compatible with inner_size and inner_size."""
        n_samples, n_features = X.shape

        if n_features != self.outer_size * self.inner_size:
            raise ValueError(
                'Number of features of the given data ({n_features}) is '
                'incompatible with the outer ({outer_size}) and inner '
                '({inner_size}) sizes of the Kronecker structure. '
                '({n_features} * {outer_size} != {n_features}). '.format(
                    n_features=n_features,
                    outer_size=self.outer_size,
                    inner_size=self.inner_size,
                )
            )

    def fit(self, X, y=None):
        """Estimate the outer matrix of the covariance of a given dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data to estimate the covariance of.
        y : ndarray, shape (n_samples, n_target)
            The target labels. Unused.

        Returns
        -------
        self : instance of KroneckerUpdater
            A version of this object with the outer covariance matrix
            estimated and ready for the `update` function to be called.
        """
        n_samples, n_features = X.shape

        if self.inner_size is None:
            self.inner_size = n_features // self.outer_size
        if self.outer_size is None:
            self.outer_size = n_features // self.inner_size

        self._check_X(X)

        X_ = X.reshape(n_samples, self.outer_size, self.inner_size)
        X_ = X_.transpose(1, 0, 2).reshape(-1, n_samples * self.inner_size)
        self.Gamma = X_.dot(X_.T) / self.inner_size
        self.diag_loading = np.trace(self.Gamma) / self.outer_size

        self._compute_kernel(X)
        return self

    def _compute_kernel(self, X):
        A = (1 - self.alpha) * self.beta * self.Gamma
        A.flat[::self.outer_size + 1] += self.alpha * self.diag_loading
        A_inv = pinv(A)
        # B = (1 - self.alpha) * (1 - self.beta) * X.T
        G = self._kronecker_dot(A_inv, X.T)
        K = (X @ G) * (1 - self.alpha) * (1 - self.beta)
        K.flat[::len(K) + 1] += 1

        self.A_inv = A_inv
        self.G = G
        self.K = K
        self.K_inv = pinv(K)

    def update(self, X, alpha=None, beta=None):
        if self.Gamma is None:
            raise RuntimeError('First run `fit`.')

        if alpha is None:
            alpha = self.alpha  # use default value
        if beta is None:
            beta = self.beta  # use default value

        k = KroneckerKernel(self.outer_size, self.inner_size, alpha, beta)
        k.Gamma = self.Gamma
        k.diag_loading = self.diag_loading
        k._compute_kernel(X)
        return k

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        A_inv_P = self._kronecker_dot(self.A_inv, P)
        return A_inv_P - multi_dot((self.G, self.K_inv, X, A_inv_P))

    def loo_inv_dot(self, X, Ps):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        Xs = loo(X)
        Gs = loo(self.G, axis=1)
        K_invs = loo_kern_inv(self.K)

        for G, K_inv, X, P in zip(Gs, K_invs, Xs, Ps):
            A_inv_P = self._kronecker_dot(self.A_inv, P)
            yield A_inv_P - multi_dot((G, K_inv, X, A_inv_P))

    def _kronecker_dot(self, A, B):
        result = A.dot(B.reshape(self.outer_size, -1))
        return result.reshape(B.shape)

    def get_x0(self):
        return [self.alpha, self.beta]

    def get_bounds(self):
        return [(0.01, 1), (0, 1)]

    def __repr__(self):
        return 'KroneckerKernel(alpha={}, beta={})'.format(self.alpha,
                                                           self.beta)
