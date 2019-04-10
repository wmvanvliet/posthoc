#encoding: utf-8
import numpy as np
from sklearn.linear_model import LinearRegression
from .cov_estimators import Empirical


def compute_pattern(coef, X, return_y_hat=False):
    '''Derive the learned pattern from a fitted linear model.

    Applies the Haufe trick to compute patterns from weights:
    equation (6) from Haufe et al. 2014 [1].

    Optionally returns X @ W.T, which is a direct application of the weights
    to the data.

    Parameters
    ----------
    coef : ndarray, shape (n_features, n_targets)
        The weights that define the linear model. For example, all Scikit-Learn
        linear models expose the `.coef_` attribute after fitting, which is
        the intended input to this function.
    X : nparray, shape (n_samples, n_features)
        The data. No centering or normalization will be performed on it, so be
        sure this has already been properly done.
    return_y_hat : bool (Default: False)
        Whether to return X @ W.T

    Returns
    -------
    pattern : ndarray, shape (n_features, n_targets)
        The pattern learned by the linear model. This is the result of the
        Haufe trick.
    y_hat : ndarray, shape (n_samples, n_targets)
        X @ W.T. Only returned if `return_y_hat=True`.

    Notes
    -----
    Make sure to supply the actual `X` used to compute the weights. Take note
    of things like centering and z-scoring, which are sometimes performed by
    the linear model.

    References
    ----------
    [1] Haufe, S., Meinecke, F. C., Görgen, K., Dähne, S., Haynes, J. D.,
    Blankertz, B., & Bießmann, F. (2014). On the interpretation of weight
    vectors of linear models in multivariate neuroimaging. NeuroImage, 87,
    96–110. http://doi.org/10.1016/j.neuroimage.2013.10.067
    '''
    y_hat = X.dot(coef.T)
    if y_hat.ndim == 1:
        y_hat = y_hat[:, np.newaxis]

    pattern = LinearRegression(fit_intercept=False).fit(y_hat, X).coef_

    if return_y_hat:
        return pattern, y_hat
    else:
        return pattern


def disassemble_modify_reassemble(coef, X, y, cov=None,
                                  pattern_modifier=None,
                                  normalizer_modifier=None,
                                  cov_params=None,
                                  pattern_modifier_params=None,
                                  normalizer_modifier_params=None):
    '''Disassemble, modify and reassemble a fitted linear model.

    This is the meat of the post-hoc adaptation framework. The given linear
    model is disassembled, the whitener and pattern are modified, and the model
    is put together again.

    Parameters
    ----------
    coef : ndarray, shape (n_features, n_targets)
        The weights that define the linear model. For example, all Scikit-Learn
        linear models expose the `.coef_` attribute after fitting, which is
        the intended input to this function.
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels. Set to `None` if there are no labels.
    cov : instance of CovEstimator | function (cov, x, y) | None
        The method used to estimate the covariance matrix. If None, defaults to
        an emperical estimate.
    pattern_modifier : function (pattern, X, y) | None
        The user supplied function that modifies the pattern.
    normalizer_modifier : function (normalizer, X, y, pattern, coef) | None
        The user supplied function that modifies the normalizer.
    cov_params : list | None
        Extra parameters to pass to the covariance estimator.
        Defaults to None, meaning no extra parameters.
    pattern_modifier_params : list | None
        Extra parameters to pass to the pattern_modifier function. Defaults to
        None, meaning no extra parameters.
    normalizer_modifier_params : list | None
        Extra parameters to pass to the normalizer_modifier function. Defaults
        to None, meaning no extra parameters.

    Returns
    -------
    new_coef : ndarray, shape (n_targets, n_features)
        The weights of the re-assembled model.
    pattern : ndarray, shape (n_features, n_targets)
        The pattern, possibly modified by a modifier function.
    normalizer : ndarray, shape (n_targets, n_targets)
        The normalizer, possibly modified by a modifier function.

    Notes
    -----
    Make sure to supply the actual `X` that was used to compute `W`. Take note
    of things like centering and z-scoring, which are sometimes performed by a
    Scikit-Learn model.
    '''
    # Deal with defaults
    if pattern_modifier_params is None:
        pattern_modifier_params = []
    if normalizer_modifier_params is None:
        normalizer_modifier_params = []

    # Disassemble the model
    pattern, y_hat = compute_pattern(coef, X, return_y_hat=True)
    normalizer = y_hat.T.dot(y_hat)

    # Shortcut if no modifications are required
    if cov is pattern_modifier is normalizer_modifier is None:
        return coef, pattern, normalizer

    # Use default method of cov estimation
    if cov is None:
        cov = Empirical()

    # Modify the pattern
    if pattern_modifier is not None:
        pattern = pattern_modifier(pattern, X, y, *pattern_modifier_params)
    if pattern.ndim == 1:
        pattern = pattern[:, np.newaxis]

    # Compute new weights
    cov.fit(X)
    if cov_params is not None:
        cov = cov.update(X, *cov_params)
    new_coef = cov.inv_dot(X, pattern).T

    # Modify and apply the normalizer
    if normalizer_modifier is not None:
        normalizer = normalizer_modifier(normalizer, X, y, pattern, new_coef,
                                         *normalizer_modifier_params)

    if new_coef.ndim == 1:
        new_coef = new_coef[np.newaxis, :]

    normalizer = np.atleast_2d(normalizer)
    new_coef = normalizer.dot(new_coef)

    return new_coef, pattern, normalizer
