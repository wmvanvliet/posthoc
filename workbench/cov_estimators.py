# encoding: utf-8
"""
Different ways of estimating the covariance matrix.

These classes are highly optimized to be used for efficient leave-one-out
estimation of the inverse covariance matrix. This optimization is vital for the
WorkbenchOptimizer to not be impractically slow.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from copy import deepcopy
import numpy as np
from numpy.linalg import multi_dot, pinv

from .loo_utils import loo, loo_kern_inv


class CovEstimator(object):
    """Abstract base class for covariance estimation methods."""
    def fit(self, X):
        return self

    def update(self, X):
        return self.copy()

    def inv_dot(self, P):
        """Computes inv(cov(X)) @ P"""
        raise NotImplemented('This function must be implemented in a subclass')

    def loo_inv_dot(self, X, Ps, remove_mean=False):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        for X_, P in zip(loo(X), Ps):
            if remove_mean:
                X_ = X_ - X_.mean(axis=0)
            yield self.fit(X_).inv_dot(X_, P)

    def get_x0(self):
        return []

    def get_bounds(self):
        return []

    def copy(self):
        return deepcopy(self)


class Function(CovEstimator):
    """Estimate covariance using a user specified function."""
    def __init__(self, func):
        self.func = func

    def fit(self, X):
        return self

    def inv_dot(self, X, P):
        return pinv(self.func(X)).dot(P)

    def __repr__(self):
        return 'Function({})'.format(self.func)


class Empirical(CovEstimator):
    """Empirical estimation of the covariance matrix."""
    def fit(self, X):
        self.cov = X.T @ X
        self.cov_inv = pinv(X.T @ X)
        return self

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        return self.cov_inv @ P

    def loo_inv_dot(self, X, Ps, remove_mean=False):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        for X_, P in zip(X, Ps):
            # Update cov_inv using Sherman–Morrison formula
            tmp = self.cov_inv @ X_[:, np.newaxis]
            cov_inv = self.cov_inv + (tmp @ tmp.T) / (1 - X_ @ tmp)
            yield cov_inv @ P

    def __repr__(self):
        return 'Empirical()'


class L2(CovEstimator):
    """
    Estimation of the covariance matrix using L2 regularization.

    This approach is more efficient than :class:`L2Kernel`
    if #samples > #features.

    Parameters
    ----------
    alpha : float
        The amount of l2-regularization (0 to infinity) to apply to the
        empirical covariance matrix. Defaults to 1.
    scale_by_var : bool
        Whether to scale ``alpha`` by the mean variance. This makes ``alpha``
        independant of the scale of the data. Defaults to ``True``.
    """
    def __init__(self, alpha=1, scale_by_var=True):
        super().__init__()
        self.alpha = alpha
        self.scale_by_var = scale_by_var

    def fit(self, X):
        cov = X.T @ X
        # Add to the diagonal in-place
        cov_reg = cov.copy()
        if self.scale_by_var:
            self.mean_var = np.trace(cov) / len(cov)
            cov_reg.flat[::len(cov_reg) + 1] += self.alpha * self.mean_var
        else:
            cov_reg.flat[::len(cov_reg) + 1] += self.alpha
        self.cov = cov
        self.cov_reg = cov_reg
        self.cov_inv = pinv(cov_reg)
        return self

    def update(self, X, alpha):
        l2 = L2(alpha, self.scale_by_var)
        l2.cov = self.cov
        l2.cov_reg = self.cov.copy()
        if self.scale_by_var:
            l2.mean_var = self.mean_var
            l2.cov_reg.flat[::len(l2.cov_reg) + 1] += alpha * l2.mean_var
        else:
            l2.cov_reg.flat[::len(l2.cov_reg) + 1] += alpha
        l2.cov_inv = pinv(l2.cov_reg)
        return l2

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        return self.cov_inv @ P

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0, None)]

    def __repr__(self):
        return 'L2(alpha={}, scale_by_var={})'.format(self.alpha,
                                                      self.scale_by_var)


class Shrinkage(CovEstimator):
    """
    Estimation of the covariance matrix using Shrinkage regularization.

    This approach is more efficient than :class:`ShrinkageKernel`
    if #samples > #features.

    Parameters
    ----------
    alpha : float
        The amount of shrinkage (0 to 1) to apply to the empirical covariance
        matrix. Defaults to 0.5
    """
    def __init__(self, alpha=0.5):
        if not (0 <= alpha <= 1):
            raise ValueError('alpha must be between 0 and 1')
        super().__init__()
        self.alpha = alpha

    def fit(self, X):
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
        s.cov_reg = (1 - alpha) * self.cov.copy()
        s.cov_reg.flat[::len(s.cov_reg) + 1] += alpha * s.mean_var
        s.cov_inv = pinv(s.cov_reg)
        return s

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        return self.cov_inv @ P

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0, None)]

    def __repr__(self):
        return 'Shrinkage(alpha={})'.format(self.alpha)


class Kronecker(CovEstimator):
    """
    Estimation of the covariance matrix using Kronecker shrinkage.

    With Kronecker regularization, the covariance matrix is assumed to consist
    of the Kronecker product of two smaller sub-matrices [1]_. For example, for
    spatio-temporal signals, the covariance matrix can be approximated by the
    Kronecker product of the spatial and temporal covariance matrices.
    Kronecker shrinkage regularization shrinks the two sub-matrices
    independantly.

    This approach is more efficient than :class:`KroneckerKernel`
    if #samples > #features.

    Parameters
    ----------
    outer_size : int | None
        The size of the outer (=first) matrix of the Kronecker product.
        Either ``outer_size`` or ``inner_size`` must be specified.
    inner_size : int | None
        The size of the inner (=second) matrix of the Kronecker product.
        Either ``outer_size`` or ``inner_size`` must be specified.
    alpha : float
        The amount of shrinkage (0 to 1) to apply to the outer matrix.
        Defaults to 0.5
    beta : float
        The amount of shrinkage (0 to 1) to apply to the inner matrix.
        Defaults to 0.5

    Attributes
    ----------
    outer_mat_ : ndarray, shape (outer_size, outer_size)
        The estimated outer matrix of the covariance. Becomes available after
        calling the `fit` method.

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Kronecker_product
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

    def fit(self, X):
        n_samples, n_features = X.shape

        if self.inner_size is None:
            self.inner_size = n_features // self.outer_size
        if self.outer_size is None:
            self.outer_size = n_features // self.inner_size

        self._check_X(X)

        X_ = X.reshape(n_samples, self.outer_size, self.inner_size)
        X_ = X_.transpose(1, 0, 2).reshape(-1, n_samples * self.inner_size)
        self.outer_mat_ = X_.dot(X_.T)
        self.diag_loading = np.trace(self.outer_mat_) / (self.outer_size * self.inner_size)
        self.cov = X.T.dot(X)
        self._perform_shrinkage()
        self.cov_inv = pinv(self.cov_reg)
        return self

    def _perform_shrinkage(self):
        """Perform Kronecker shrinkage."""
        self.cov_reg = self.beta * np.kron(self.outer_mat_, np.eye(self.inner_size))
        self.cov_reg += (1 - self.beta) * self.cov
        self.cov_reg *= (1 - self.alpha)
        self.cov_reg.flat[::len(self.cov_reg) + 1] += self.alpha * self.diag_loading

    def update(self, X, alpha=None, beta=None):
        if self.outer_mat_ is None:
            raise RuntimeError('First run `fit`.')

        if alpha is None:
            alpha = self.alpha  # use default value
        if beta is None:
            beta = self.beta  # use default value

        k = Kronecker(self.outer_size, self.inner_size, alpha, beta)
        k.outer_mat_ = self.outer_mat_
        k.diag_loading = self.diag_loading
        k.cov = self.cov
        k._perform_shrinkage()
        k.cov_inv = pinv(k.cov_reg)
        return k

    def _kronecker_dot(self, A, B):
        result = A.dot(B.reshape(self.outer_size, -1))
        return result.reshape(B.shape)

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        return self.cov_inv @ P

    def get_x0(self):
        return [self.alpha, self.beta]

    def get_bounds(self):
        return [(0.01, 1), (0, 1)]

    def __repr__(self):
        return 'Kronecker(alpha={}, beta={})'.format(self.alpha, self.beta)


class _InversionLemma(CovEstimator):
    """Abstract base class for CovEstimator's that use the inversion lemma.

    When #samples < #features, the computation can be sped up by using the
    Woodbury matrix inversion lemma [1]_:

    (A − B D⁻¹ C)⁻¹ = A⁻¹ + A⁻¹ B (D − C A⁻¹ B)⁻¹ C A⁻¹

    And a reformulation of this lemma [2]_:

    A⁻¹ B (D − C A⁻¹ B)⁻¹ = (A − B D⁻¹ C)⁻¹ B D

    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    .. [2]: The second form described in:
            https://www.stats.ox.ac.uk/~lienart/blog_linalg_invlemmas.html
    """
    def compute_AB_parts(self, X):
        raise NotImplemented('This function must be implemented in a subclass')

    def fit(self, X):
        A_inv, B = self.compute_AB_parts(X)
        G = A_inv * B
        print(G)
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

    def loo_inv_dot(self, X, Ps, remove_mean=False):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        Xs = loo(X)
        Gs = loo(self.G, axis=1)
        K_invs = loo_kern_inv(self.K)

        for G, K_inv, X_, x, P in zip(Gs, K_invs, Xs, X, Ps):
            if remove_mean:
                X_ = X_ - X_.mean(axis=0)
            A_inv_loo = self.A_inv
            if self.scale_by_var:
                A_inv_loo /= 1 - x.dot(x) / (X.shape[1] * self.mean_var)
            print(G)
            A_inv_P = A_inv_loo * P
            yield A_inv_P - multi_dot((G, K_inv, X_, A_inv_P))


class L2Kernel(_InversionLemma):
    """
    L2 estimation of the covariance using the kernel formulation.

    This approach is more efficient than ``L2`` if #features > #samples.

    Parameters
    ----------
    alpha : float
        The amount of l2-regularization (>0 to infinity) to apply to the
        empirical covariance matrix. Must be greater than zero. Defaults to 1.
    scale_by_var : bool
        Whether to scale ``alpha`` by the mean variance. This makes ``alpha``
        independant of the scale of the data. Defaults to ``True``.
    """
    def __init__(self, alpha=1, scale_by_var=True):
        if alpha < 1E-15:
            raise ValueError('alpha must be greater than zero')
        self.alpha = alpha
        self.scale_by_var = scale_by_var

    def compute_AB_parts(self, X):
        A_inv = 1 / self.alpha
        if self.scale_by_var:
            self.mean_var = np.sum(X ** 2) / X.shape[1]
            A_inv /= self.mean_var
        B = X.T
        return A_inv, B

    def update(self, X, alpha):
        return L2Kernel(alpha, self.scale_by_var).fit(X)

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0.01, None)]

    def __repr__(self):
        return 'L2Kernel(alpha={}, scale_by_var={})'.format(self.alpha,
                                                            self.scale_by_var)


class ShrinkageKernel(_InversionLemma):
    """
    Shrinkage estimation of the covariance using the kernel formulation.

    This approach is more efficient than ``Shrinkage`` if #features > #samples.

    Parameters
    ----------
    alpha : float
        The amount of shrinkage (>0 to 1) to apply to the empirical covariance
        matrix. Must be greater than zero. Defaults to 0.5
    """
    def __init__(self, alpha=0.5):
        if alpha < 1E-15:
            raise ValueError('alpha must be greater than zero')
        self.alpha = alpha
        self.scale_by_var = True  # Always True for shrinkage regularization

    def compute_AB_parts(self, X):
        A_inv = 1 / self.alpha
        self.mean_var = np.sum(X ** 2) / X.shape[1]
        A_inv /= self.mean_var
        B = (1 - self.alpha) * X.T
        return A_inv, B

    def update(self, X, alpha):
        return ShrinkageKernel(alpha).fit(X)

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0.01, 1)]

    def __repr__(self):
        return 'ShrinkageKernel(alpha={})'.format(self.alpha)


class KroneckerKernel(_InversionLemma):
    """
    Kronecker shrinkage of the covariance using the kernel formulation.

    This approach is more efficient than ``Kronecker`` if #features > #samples.

    Parameters
    ----------
    outer_size : int | None
        The size of the outer (=first) matrix of the Kronecker product.
        Either ``outer_size`` or ``inner_size`` must be specified.
    inner_size : int | None
        The size of the inner (=second) matrix of the Kronecker product.
        Either ``outer_size`` or ``inner_size`` must be specified.
    alpha : float
        The amount of shrinkage (0 to 1) to apply to the outer matrix.
        Defaults to 0.5
    beta : float
        The amount of shrinkage (0 to 1) to apply to the inner matrix.
        Defaults to 0.5

    Attributes
    ----------
    outer_mat_ : ndarray, shape (outer_size, outer_size)
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

    def fit(self, X):
        n_samples, n_features = X.shape

        if self.inner_size is None:
            self.inner_size = n_features // self.outer_size
        if self.outer_size is None:
            self.outer_size = n_features // self.inner_size

        self._check_X(X)

        X_ = X.reshape(n_samples, self.outer_size, self.inner_size)
        X_ = X_.transpose(1, 0, 2).reshape(-1, n_samples * self.inner_size)
        self.outer_mat_ = X_.dot(X_.T) # / self.inner_size
        self.diag_loading = np.trace(self.outer_mat_) / (self.outer_size * self.inner_size)

        self._compute_kernel(X)
        return self

    def _compute_kernel(self, X):
        A = (1 - self.alpha) * self.beta * self.outer_mat_
        A.flat[::self.outer_size + 1] += self.alpha * self.diag_loading
        A_inv = pinv(A)
        B = (1 - self.alpha) * (1 - self.beta) * X.T
        G = self._kronecker_dot(A_inv, B)
        K = X @ G
        K.flat[::len(K) + 1] += 1

        self.A_inv = A_inv
        self.G = G
        self.K = K
        self.K_inv = pinv(K)

    def update(self, X, alpha=None, beta=None):
        if self.outer_mat_ is None:
            raise RuntimeError('First run `fit`.')

        if alpha is None:
            alpha = self.alpha  # use default value
        if beta is None:
            beta = self.beta  # use default value

        k = KroneckerKernel(self.outer_size, self.inner_size, alpha, beta)
        k.outer_mat_ = self.outer_mat_
        k.diag_loading = self.diag_loading
        k._compute_kernel(X)
        return k

    def inv_dot(self, X, P):
        """Computes inv(cov(X)) @ P"""
        A_inv_P = self._kronecker_dot(self.A_inv, P)
        return A_inv_P - multi_dot((self.G, self.K_inv, X, A_inv_P))

    def loo_inv_dot(self, X, Ps, remove_mean=False):
        """Computes inv(cov(X)) @ P in a leave-one-out scheme"""
        Xs = loo(X)
        Gs = loo(self.G, axis=1)
        K_invs = loo_kern_inv(self.K)

        for G, K_inv, X_, P in zip(Gs, K_invs, Xs, Ps):
            A_inv_P = self._kronecker_dot(self.A_inv, P)
            yield A_inv_P - multi_dot((G, K_inv, X_, A_inv_P))

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
