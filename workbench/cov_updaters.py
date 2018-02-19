import numpy as np


class CovUpdater(object):
    def __init__(self, data):
        self._data = data

    def fit(self, X, y):
        return self

    def update(self):
        return self

    def add(self, X):
        raise NotImplemented('This function must be implemented in a subclass')

    def dot(self, X):
        raise NotImplemented('This function must be implemented in a subclass')

    def inv(self):
        raise NotImplemented('This function must be implemented in a subclass')

    def get_x0(self):
        return None

    def get_bounds(self):
        return None

    def __add__(self, X):
        return self.add(X)

    def __radd__(self, X):
        return self.add(X)

    def __sub__(self, X):
        return self.add(-X)

    def __rsub__(self, X):
        return -self.add(-X)

    def __matmul__(self, X):
        return self.dot(X)


class ShrinkageUpdater(CovUpdater):
    def __init__(self, alpha=1.0, scale_by_trace=True):
        CovUpdater.__init__(self, None)
        self.alpha = alpha
        self.scale_by_trace = scale_by_trace
        if not scale_by_trace:
            self._scale = 1.

    def fit(self, X, y):
        if self.scale_by_trace:
            self._scale = np.trace(X.T.dot(X)) / X.shape[1]
        else:
            self._scale = 1.
        return self

    def update(self, alpha=None):
        if alpha is None:
            return self  # use default value
        else:
            s = ShrinkageUpdater(alpha)
            s._scale = self._scale
            return s

    def add(self, X):
        X = X.copy()
        X.flat[::len(X) + 1] += self._scale * self.alpha
        return X

    def dot(self, X):
        return self.alpha * self.scale_by_trace * X

    def inv(self):
        return ShrinkageUpdater(1 / float(self.alpha))

    def get_x0(self):
        return [self.alpha]

    def get_bounds(self):
        return [(0.1, None)]

    def __repr__(self):
        return 'ShrinkageUpdater(alpha=%f)' % self.alpha


class KroneckerUpdater(CovUpdater):
    """
    Attributes
    ----------
    outer_cov_ : ndarray, shape (outer_size, outer_size)
        The estimated outer matrix of the covariance. Becomes available after
        calling the `fit` method.
    """
    def __init__(self, outer_size=None, inner_size=None, alpha=1.0,
                 beta=1.0):
        CovUpdater.__init__(self, None)
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
        self._data = X_.dot(X_.T) / self.inner_size
        self._scale = np.trace(self._data) / self.outer_size

        return self

    def copy(self, deep=True):
        k = KroneckerUpdater(self.inner_size, self.outer_size, self.alpha,
                             self.beta)
        if self._data is None:
            k._data = None
        elif deep:
            k._data = self._data.copy()
        else:
            k._data = self._data

        if hasattr(self, '_scale'):
            k._scale = self._scale

        return k

    def update(self, alpha=None, beta=None):
        if self._data is None:
            raise RuntimeError('First run `fit`.')

        if alpha is None:
            alpha = self.alpha  # use default value
        if beta is None:
            beta = self.beta  # use default value

        k = KroneckerUpdater(self.inner_size, self.outer_size, alpha, beta)
        k._data = self._data * beta
        k._data.flat[::self.outer_size + 1] += alpha * self._scale
        return k

    def add(self, X):
        if self._data is None:
            raise RuntimeError('First run `fit`.')

        return X + np.kron(self.data, np.eye(self.inner_size))

    def dot(self, X):
        if self._data is None:
            raise RuntimeError('First run `fit`.')

        result = self._data.dot(X.reshape(self.outer_size, -1))
        result = result.reshape(X.shape)
        return result

    def inv(self):
        if self._data is None:
            raise RuntimeError('First run `fit`.')

        k = self.copy(deep=False)
        k._data = np.linalg.pinv(self._data)
        return k

    def get_x0(self):
        return [self.alpha, self.beta]

    def get_bounds(self):
        return [(0.1, None), (0, None)]

    def __repr__(self):
        return 'KroneckerUpdater(alpha={}, beta={})'.format(self.alpha,
                                                            self.beta)
