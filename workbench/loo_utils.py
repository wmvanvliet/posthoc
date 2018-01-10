import numpy as np
from numpy.linalg import pinv, lstsq, multi_dot

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut

import progressbar


def start_progress_bar(n):
    return progressbar.ProgressBar(
        maxval=n,
        widgets=[
            progressbar.widgets.Bar(),
            progressbar.widgets.SimpleProgress(sep='/'),
            '|',
            progressbar.widgets.ETA(),
        ],
    ).start()


def update_inv(X, X_inv, i, v):
    """Computes a rank 1 update of the the inverse of a matrix.

    Given a symmerical matrix X, its inverse X^{-1}, this function computes the
    inverse of Y, which is a copy of X, with the i'th row&column replaced by
    given vector v.

    Parameters
    ----------
    X : ndarray, shape (N, N)
        A symmetrical matrix.
    X_inv : nparray, shape (N, N)
        The inverse of X_inv.

    Returns
    -------
    Y_inv : ndarray, shape (N, N)
        The inverse of Y.

    Notes
    -----
    X needs to be a symmetrical (e.g. a covariance matrix).
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

    Returns version of K and K^{-1} where the i'th row&column are removed.

    Parameters
    ----------
    K : ndarray, shape (n_samples, n_samples)
        The kernel matrix. Needs to be symmetric.

    Yields
    ------
    K : ndarrays, shape (n_samples, n_samples)
        The updated kernel matrices.
    K_inv : ndarray, shape (n_samples, n_samples)
        The inverse of the updated kernel matrices.
    """
    K1 = None
    K1_inv = None
    for i in range(len(K)):
        if K1 is None or K1_inv is None:
            K1 = K[1:, :][:, 1:]
            K1_inv = pinv(K1)
            yield K1_inv
        else:
            j = np.arange(1, len(K))
            j[i - 1] = 0
            v = K[0, j]
            yield update_inv(K1, K1_inv, i - 1, v)


def loo_linear_regression(X, y, centered=False):
    """Generate regression coefficients for leave-one-out iterations.

    Employs an efficient algorithm described in [1].

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels.
    centered : bool
        Whether X and y are already centered. Defaults to False.

    Yields
    ------
    coeff : ndarray, shape (n_features, n_targets)
        Coefficients for linear regression with the i'th sample left out.

    References
    ----------
    [1] George A. F. Seber and Alan J. Lee. Linear Regression Analysis
        (2nd edition, 2003), pp 357.
    """
    if not centered:
        X = X - X.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)

    cov_inv = pinv(X.T.dot(X))
    c = cov_inv.dot(X.T)
    hat_matrix_diag = np.diag(X.dot(c))
    coef_old = c.dot(y)

    n_samples = len(X)
    for i in range(n_samples):
        x_i = X[[i]].T
        y_i = y[i]
        update = -(cov_inv.dot(x_i).dot(y_i - x_i.dot(coef_old)) /
                   (1 - hat_matrix_diag[i]))
        yield coef_old + update


def loo_patterns_from_model(model, X, y, verbose=False):
    """Generate patterns for leave-one-out iterations of the given model.

    Patterns are computed with the Haufe trick [1]:
        A = cov_X @ model.coeff_ @ precision_y_hat


    Performs optimizations when the model is
    `sklearn.linear_model.LinearRegression` or `sklearn.linear_model.RidgeCV`.

    Parameters
    ----------
    model : instance of `sklearn.linear_model.base.LinearModel`
        The linear model to compute the patterns for
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels.
    verbose : bool
        Print out a progressbar. Defaults to False.

    Yields
    ------
    pattern : ndarray, shape (n_features, n_targets)
        The pattern computed for the model fitted to the data with the i'th
        sample left out.
    """
    if type(model) == LinearRegression:
        for train, test in LeaveOneOut().split(X, y):
            X_ = X[train]
            y_ = y[train]

            if model.fit_intercept:
                X_ = X_ - X_.mean(axis=0, keepdims=True)
                y_ = y_ - y_.mean(axis=0, keepdims=True)
            if model.normalize:
                X_ /= X.std(axis=0, keepdims=True)

            yield X_.T.dot(y_)

    else:
        n_samples = len(X)

        if verbose:
            print('Computing patterns for each leave-one-out iteration...')
            pbar = start_progress_bar(n_samples)

        # TODO: check the validity of this!
        for train, _ in LeaveOneOut().split(X, y):
            # Fit the base model
            model.fit(X[train], y[train])

            if not hasattr(model, 'coef_'):
                raise RuntimeError(
                    'Model does not have a `coef_` attribute after fitting. '
                    'This does not seem to be a linear model following the '
                    'Scikit-Learn API.'
                )

            # Compute the pattern from the base model filter weights,
            # conforming equation 6 from Haufe2014.
            y_hat = model.predict(X[train])
            if y_hat.ndim == 1:
                y_hat = y_hat[:, np.newaxis]

            X_ = X[train]
            if model.fit_intercept:
                X_ = X_ - X_.mean(axis=0, keepdims=True)
            if model.normalize:
                X_ /= X.std(axis=0, keepdims=True)
            m = LinearRegression(fit_intercept=False).fit(y_hat, X_)

            if verbose:
                pbar.update(pbar.currval + 1)

            yield m.coef_
        if verbose:
            pbar.finish()
