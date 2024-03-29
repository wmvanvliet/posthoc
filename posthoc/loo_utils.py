#encoding: utf-8
"""
Tools for efficient computation of various things in a leave-one-out manner.
"""
import numpy as np
from numpy.linalg import pinv, multi_dot
from sklearn.base import RegressorMixin
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
import progressbar


def _start_progress_bar(n):
    return progressbar.ProgressBar(
        maxval=n,
        widgets=[
            progressbar.widgets.Bar(),
            progressbar.widgets.SimpleProgress(sep='/'),
            '|',
            progressbar.widgets.ETA(),
        ],
    ).start()


def loo(X, axis=0):
    """Generates leave-one-out selections along a given axis.

    In the first iteration, the first item is left out. In following
    iterations, each item to leave out is replaced with the first item. This
    allows us to only make a single copy and perform all other operations
    in-place.

    Parameters
    ----------
    X : ndarray
        The array to produce leave-one-out selections of.
    axis : int (default: 0)
        The axis along which to perform the leave-one-out selections.

    Yields
    ------
    X_loo : ndarray
        A copy of X, with the n'th item left out.
    """
    # Generate an index that cuts off the top along the desired axis.
    # We use slices to avoid copying data.
    cut_top = tuple([slice(None) if i != axis else slice(1, None)
                    for i in range(X.ndim)])

    # Generate an index that returns a slice at a given index along the desired
    # axis, without copying data.
    def slice_at(ind):
        return tuple([slice(None) if i != axis else ind
                      for i in range(X.ndim)])

    X_loo = None
    for i in range(X.shape[axis]):
        if X_loo is None:
            X_loo = X[cut_top].copy()
        else:
            if i >= 2:
                X_loo[slice_at(i - 2)] = X[slice_at(i - 1)]
            X_loo[slice_at(i - 1)] = X[slice_at(0)]
        yield X_loo


def update_inv(X, X_inv, i, v):
    """Computes a rank 1 update of the the inverse of a symmetrical matrix.

    Given a symmerical matrix X and its inverse X^{-1}, this function computes
    the inverse of Y, which is a copy of X, with the i'th row&column replaced
    by given vector v.

    Parameters
    ----------
    X : ndarray, shape (N, N)
        A symmetrical matrix.
    X_inv : nparray, shape (N, N)
        The inverse of X_inv.
    i : int
        The index of the row/column to replace.
    v : ndarray, shape (N,)
        The values to replace the row/column with.

    Returns
    -------
    Y_inv : ndarray, shape (N, N)
        The inverse of Y.
    """
    U = v[:, np.newaxis] - X[:, [i]]
    mask = np.zeros((len(U), 1))
    mask[i] = 1
    U = np.hstack((U, mask))

    V = U[:, [1, 0]].T
    V[1, i] = 0

    C = np.eye(2)

    X_inv_U = X_inv.dot(U)
    V_X_inv = V.dot(X_inv)
    Y_inv = X_inv - X_inv_U.dot(pinv(C + V_X_inv.dot(U))).dot(V_X_inv)

    return Y_inv


def loo_kern_inv(K):
    """A generator for leave-one-out crossval iterations from a kernel matrix.

    Returns versions of K^{-1} where the i'th row&column are removed.

    In the first iteration, the first item of K is left out. In following
    iterations, each item to leave out is replaced with the first item. This
    allows us to only make a single copy and perform all other operations
    in-place.

    This means this function is best used in combination with :func:`loo`.

    Parameters
    ----------
    K : ndarray, shape (n_samples, n_samples)
        The kernel matrix. Needs to be symmetric.

    Yields
    ------
    K_inv : ndarray, shape (n_samples, n_samples)
        The inverse of the updated kernel matrix.
    """
    K1 = None
    K1_inv = None
    for i in range(len(K)):
        if K1 is None:  # First iteration
            K1 = K[1:, :][:, 1:]
            K1_inv = pinv(K1)
            yield K1_inv
        else:
            j = np.arange(1, len(K))
            j[i - 1] = 0
            v = K[0, j]
            yield update_inv(K1, K1_inv, i - 1, v)


def loo_mean_norm(X, return_norm=True):
    """Compute the mean() and L2 norm of a matrix in a leave-one-out manner.

    Optimizes things by computing updates to the mean and L2 norm instead of
    re-computing them for each iteration.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data.
    return_norm : bool
        Whether to also return the leave-one-out L2 norm. Defaults to True.

    Yields
    ------
    X_mean : ndarray, shape (1, n_features)
        The row-wise mean for X with the i'th row removed.
    X_norm : ndarray, shape (1, n_features) (optional)
        The row-wise L2 norm of (X - X_mean) with the i-th row removed.
        Only returned when ``return_norm=True`` is specified.

    References
    ----------
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    """
    X = np.asarray(X)
    n = len(X)

    X_mean = X.mean(axis=0, keepdims=True)
    offsets = (X - X_mean)  # Offsets from the mean
    mean_deltas = offsets / (n - 1)

    if return_norm:
        X_var = X.var(axis=0, keepdims=True)

    for mean_delta, offset, x_i in zip(mean_deltas, offsets, X):
        X_mean_ = X_mean - mean_delta
        if return_norm:
            X_norm_ = np.sqrt((n * X_var - offset * (x_i - X_mean_)))
            yield X_mean_, X_norm_
        else:
            yield X_mean_


def loo_ols_regression(X, y, normalize=False, fit_intercept=True):
    """Generate OLS regression coefficients for leave-one-out iterations.

    Employs an efficient algorithm described in [1].

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels.
    normalize : bool
        Whether to normalize the data. Defaults to False.
    fit_intercept : bool
        Whether to fit the intercept. Defaults to True.

    Yields
    ------
    coef : ndarray, shape (n_features, n_targets)
        coeficients for linear regression with the i'th sample left out.
        No intercept is provided.

    References
    ----------
    [1] George A. F. Seber and Alan J. Lee. Linear Regression Analysis
        (2nd edition, 2003), page 357.
    """
    n_samples = len(X)
    flat_y = y.ndim == 1
    if flat_y:
        y = y[:, np.newaxis]

    if normalize:
        X = (X - np.mean(X, axis=0))
        X_scale = np.linalg.norm(X, axis=0, keepdims=True)
        X /= X_scale
    if fit_intercept:
        # Fit the intercept by adding a column of ones to the data
        X = np.hstack((X, np.ones((n_samples, 1))))

    cov_inv = pinv(X.T.dot(X))
    c = cov_inv.dot(X.T)
    hat_matrix_diag = np.diag(X.dot(c))
    coef_old = c.dot(y)
    errors = y - X.dot(coef_old)

    # Compute updates to the regression weights for each leave-one-out
    # permutation.
    for i in range(n_samples):
        x_i = X[[i]].T
        e_i = errors[[i]]
        h_i = hat_matrix_diag[i]
        if h_i != 1:
            coef_update = multi_dot((cov_inv, x_i, e_i)) / (1 - h_i)
        else:
            coef_update = np.zeros_like(coef_old)
        coef = (coef_old - coef_update).T
        if fit_intercept:
            # Ignore the intercept coeficients
            coef = coef[:, :-1]
        if normalize:
            coef /= X_scale

        if flat_y:
            coef = coef.ravel()
        yield coef


def loo_kernel_regression(X, y, normalize=False, fit_intercept=True):
    """Generate kernel regression coefficients for leave-one-out iterations.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels.
    normalize : bool
        Whether to normalize the data. Defaults to False.
    fit_intercept : bool
        Whether to fit the intercept. Defaults to True.

    Yields
    ------
    coef : ndarray, shape (n_features, n_targets)
        coeficients for linear regression with the i'th sample left out.
        No intercept is provided.
    """
    n_samples = len(X)
    flat_y = y.ndim == 1
    if flat_y:
        y = y[:, np.newaxis]

    if normalize:
        X = (X - np.mean(X, axis=0))
        X_scale = np.linalg.norm(X, axis=0, keepdims=True)
        X /= X_scale
    if fit_intercept:
        # Fit the intercept by adding a column of ones to the data
        X = np.hstack((X, np.ones((n_samples, 1))))

    K = X.dot(X.T)

    for K_, X_, y_ in zip(loo_kern_inv(K), loo(X), loo(y)):
        coef = X_.T.dot(K_).dot(y_).T

        if fit_intercept:
            # Ignore the intercept coeficients
            coef = coef[:, :-1]
        if normalize:
            coef /= X_scale

        if flat_y:
            coef = coef.ravel()
        yield coef


def loo_ols_values(X, y, normalize=False, fit_intercept=True):
    """Generate OLS regression values for leave-one-out iterations.

    Employs an efficient algorithm described in [1].

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels.
    normalize : bool
        Whether to normalize the data. Defaults to False.
    fit_intercept : bool
        Whether to fit the intercept. Defaults to True.

    Yields
    ------
    y_hat_loo : ndarray, shape (n_samples, n_targets)
        For each sample, the predicted regression value computed by fitting the
        regressor on all other sample and applied to the current sample.

    References
    ----------
    [1] George A. F. Seber and Alan J. Lee. Linear Regression Analysis
        (2nd edition, 2003), page 357.
    """
    if fit_intercept:
        X = (X - np.mean(X, axis=0))
    if normalize:
        X_scale = np.linalg.norm(X, axis=0, keepdims=True)
        X /= X_scale

    cov_inv = pinv(X.T.dot(X))
    c = cov_inv.dot(X.T)
    hat_matrix_diag = np.diag(X.dot(c))
    coef = c.dot(y)
    errors = y - X.dot(coef)
    loo_errors = errors / (1 - hat_matrix_diag[:, np.newaxis])
    y_hat_loo = y - loo_errors

    return y_hat_loo


def loo_patterns_from_model(model, X, y, method='auto', verbose=False):
    """Generate patterns for leave-one-out iterations of the given model.

    Patterns are computed with the Haufe trick [1]:
        A = cov_X @ model.coef_ @ precision_y_hat

    Performs optimizations when the model is
    ``sklearn.linear_model.LinearRegression``

    Parameters
    ----------
    model : instance of `sklearn.linear_model.base.LinearModel`
        The linear model to compute the patterns for
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels.
    method : 'auto' | 'traditional' | 'kernel'
        Method of producing LOO patterns when the model is
        `sklearn.linear_model.LinearRegression`.
    verbose : bool
        Print out a progressbar. Defaults to False.

    Yields
    ------
    pattern : ndarray, shape (n_features, n_targets)
        The pattern computed for the model fitted to the data with the i'th
        sample left out. If the model performs normalization, this is reflected
        in the pattern.
    normalizer : ndarray, shape (n_targets, n_targets)
        The normalizer computed from the data with the i'th sample left out.
    """
    n_samples, n_features = X.shape
    if method == 'auto':
        if n_samples >= n_features:
            method = 'traditional'
        else:
            method = 'kernel'

    if verbose:
        print('Computing patterns for each leave-one-out iteration...')

    # Try to determine how the model normalizes the data
    normalize = hasattr(model, 'normalize') and model.normalize
    fit_intercept = hasattr(model, 'fit_intercept') and model.fit_intercept
    if verbose:
        print('Fit intercept:', 'yes' if fit_intercept else 'no')
        print('Normalize:', 'yes' if normalize else 'no')

    if fit_intercept or normalize:
        X_normalizer = loo_mean_norm(X)
        y_normalizer = loo_mean_norm(y, return_norm=False)

    if type(model) == LinearRegression:  # subclasses _not_ supported!
        if verbose:
            print('Choosing optimized code-path for LinearRegression() model.')
        if method == 'traditional':
            if verbose:
                print('Using OLS path.')
            coef_gen = loo_ols_regression(X, y, normalize, fit_intercept)
        elif method == 'kernel':
            if verbose:
                print('Using kernel path.')
            coef_gen = loo_kernel_regression(X, y, normalize, fit_intercept)
        else:
            raise ValueError('Invalid mode selected. Choose one of: '
                             "'auto', 'traditional', or 'kernel'.")

    if verbose:
        pbar = _start_progress_bar(n_samples)

    for train, _ in LeaveOneOut().split(X, y):
        X_ = X[train]
        y_ = y[train]
        if fit_intercept:
            X_offset, X_scale = next(X_normalizer)
            X_ = X_ - X_offset
            if normalize:
                X_ /= X_scale
            if isinstance(model, RegressorMixin):
                y_offset = next(y_normalizer)
                y_ = y_ - y_offset

        if type(model) == LinearRegression:  # subclasses _not_ supported!
            coef = next(coef_gen)

            # Redo normalization performed by the sklearn model to obtain
            # normalized coefficients 
            if normalize:
                coef *= X_scale
        else:
            model.fit(X_, y_)
            coef = model.coef_
            if not hasattr(model, 'coef_'):
                raise RuntimeError(
                    'Model does not have a `coef_` attribute after fitting. '
                    'This does not seem to be a linear model following the '
                    'Scikit-Learn API.'
                )

        # Compute model output
        if coef.ndim == 1:
            coef = coef[np.newaxis, :]
        y_hat = X_.dot(coef.T)

        # Compute the pattern from the base model filter weights,
        # conforming equation 6 from Haufe2014.
        normalizer = y_hat.T.dot(y_hat)
        pattern = multi_dot((X_.T, X_, coef.T, pinv(normalizer)))

        if verbose:
            pbar.update(pbar.currval + 1)

        yield pattern, normalizer
    if verbose:
        pbar.finish()
