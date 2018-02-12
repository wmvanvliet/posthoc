# encoding: utf-8
from __future__ import print_function
from warnings import warn
import inspect

import numpy as np
from scipy.optimize import minimize
from numpy.linalg import multi_dot, pinv

from sklearn.base import TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.base import LinearModel
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import LeaveOneOut

from . import loo_utils
from .cov_updaters import CovUpdater


def is_updater(x):
    return isinstance(x, CovUpdater)


def compute_pattern(W, X, return_y_hat=False):
    '''Derive the learned pattern from a fitted linear model.

    Applies the Haufe trick to compute patterns from weights:
    equation (6) from Haufe et al. 2014 [1].

    Optionally returns X @ W.T, which is a direct application of the weights
    to the data.

    Parameters
    ----------
    W : ndarray, shape (n_features, n_targets)
        The weights that define the linear model. For example, all Scikit-Learn
        linear models expose the `.coeff_` attribute after fitting, which is
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
    y_hat = X.dot(W.T)
    if y_hat.ndim == 1:
        y_hat = y_hat[:, np.newaxis]

    pattern = LinearRegression(fit_intercept=False).fit(y_hat, X).coef_

    if return_y_hat:
        return pattern, y_hat
    else:
        return pattern


def disassemble_model(W, X, compute_cov_X=True):
    '''Disassemble a fitted linear model into cov_X, pattern and normalizer.

    Parameters
    ----------
    W : ndarray, shape (n_features, n_targets)
        The weights that define the linear model. For example, all Scikit-Learn
        linear models expose the `.coeff_` attribute after fitting, which is
        the intended input to this function.
    X : ndarray, shape (n_samples, n_features)
        The data. No centering or normalization will be performed on it, so be
        sure this has already been properly done.
    return_cov_X : bool (Default: True)
        Whether to compute cov_X. For the kernel formulation of the workbench,
        cov_X is not required.

    Notes
    -----
    Make sure to supply the actual `X` used to compute the weights. Take note
    of things like centering and z-scoring, which are sometimes performed by
    the linear model.

    Returns
    -------
    cov_X : ndarray, shape (n_features, n_features)
        The covariance of X. Only returned if `compute_cov_X=True`.
    pattern : ndarray, shape (n_features, n_targets)
        The pattern obtained by applying the Haufe trick.
    normalizer : ndarray, shape (n_targets, n_targets)
        The normalizer.
    '''
    pattern, y_hat = compute_pattern(W, X, return_y_hat=True)
    normalizer = y_hat.T.dot(y_hat)

    if compute_cov_X:
        cov_X = X.T.dot(X)
        return cov_X, pattern, normalizer
    else:
        return pattern, normalizer


def _compute_weights(X, y, pattern, cov_modifier=None, cov_updater=None,
                     cov_modifier_params=None, method='auto'):
    '''Computes filter weights based on the given pattern.

    This function performs part of the 'disassemble_modify_reassemble'
    procedure, namely computing the weights of the linear model given the
    pattern. The modificiation of the covariance is also handled here, since
    this depends on whether the 'traditional' or 'kernel' method is used to
    compute the weights.

    The 'traditional' method computes the (n_features x n_features) covariance
    matrix of X, while the 'kernel' method instead computes the (n_items x
    n_items) "item covariance". One method can be much more efficient than the
    other, depending on the number of features and items in the data.

    No modification of the pattern or application of a normalizer is performed
    in this function.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The data. No centering or normalization will be performed on it.
    y : ndarray, shape (n_samples, n_targets) | None
        The labels. No centering or normalization will be performed on it. Set
        to `None` if there are no labels.
    pattern : ndarray, shape (n_features, n_targets)
        The pattern of the model.
    cov_modifier : function (cov, x, y) | None
        the user supplied function that modifies the covariance matrix.
    cov_updater : function (x, y) | CovUpdater | None
        the user supplied function that updates the covariance matrix.
    cov_modifier_params : list | None
        Extra parameters to pass to the cov_modifier or cov_updater function.
        Defaults to None, meaning no extra parameters.
    method : 'auto' | 'traditional' | 'kernel'
        Whether to use the traditional formulation of the linear model, which
        computes the covariance matrix, or whether to use the kernel trick to
        avoid computing the covariance matrix. The latter is more efficient
        when `n_features > n_samples`. Using the kernel method requires using a
        `cov_updater` function instead of a `cov_modifier`. Defaults to
        `'auto'`, which attempts to find the best approach automatically.

    Returns
    -------
    coef : ndarray, shape (n_targets, n_features)
        The weights of the re-assembled model.
    cov_X : ndarray, shape (n_features, n_features) | None
        If `method == 'traditional'`, the covariance matrix of X, possibly
        modified by the function given by the user.
        Returns `None` when `method == 'kernel'`.
    '''
    # Deal with defaults
    if cov_modifier_params is None:
        cov_modifier_params = []

    n_samples, n_features = X.shape

    # Determine optimal method of solving
    if method == 'auto':
        if n_features > n_samples and cov_updater is not None:
            method = 'kernel'
        else:
            method = 'traditional'

    if method == 'traditional':
        # Compute the covariance of X
        cov_X = X.T.dot(X)

        # Modify the covariance using the function supplied by the user
        if cov_modifier is not None:
            # User supplied a modifier function
            cov_X = cov_modifier(cov_X, X, y, *cov_modifier_params)
        if cov_updater is not None:
            if is_updater(cov_updater):
                # User supplied a CovUpdater object for detailed control
                update = cov_updater.fit(X, y).update(*cov_modifier_params)
                cov_X = update.add(cov_X)
            else:
                # User supplied an updater function
                cov_X += cov_updater(X, y, *cov_modifier_params)

        # Re-assemble the linear model
        cov_X_inv = np.linalg.pinv(cov_X)

        # Normalizer that ensures W @ pattern == I
        # normalizer = multi_dot((pattern.T, cov_X_inv, pattern))
        # normalizer = np.linalg.pinv(normalizer)

        coef = cov_X_inv.dot(pattern).T
        return coef, cov_X

    elif method == 'kernel':
        # Get the covariance updater from the function supplied by the user
        if is_updater(cov_updater):
            # User supplied a CovUpdater object for detailed control
            cov_update = cov_updater.fit(X, y).update(*cov_modifier_params)
            cov_update_inv = cov_update.inv()
        else:
            # User supplied an updater function
            cov_update = cov_updater(X, y, *cov_modifier_params)
            cov_update_inv = np.linalg.pinv(cov_update)

        # Compute the weights, using the matrix inversion lemma
        G = cov_update_inv.dot(X.T)
        K = X.dot(G)
        K.flat[::n_samples + 1] += 1
        K_inv = np.linalg.pinv(K)
        GammaP = cov_update_inv.dot(pattern)
        coef = (GammaP - multi_dot((G, K_inv, X, GammaP))).T
        return coef, None

    else:
        raise ValueError('The "mode" parameter must be one of: "auto", '
                         '"traditional" or "kernel".')


def disassemble_modify_reassemble(W, X, y, cov_modifier=None, cov_updater=None,
                                  pattern_modifier=None,
                                  normalizer_modifier=None, method='auto',
                                  cov_modifier_params=None,
                                  pattern_modifier_params=None,
                                  normalizer_modifier_params=None):
    '''Disassemble, modify and reassemble a fitted linear model.

    This is the meat of the workbench approach. The linear model wrapped by
    this class is disassembled, the whitener and pattern are modified, and the
    model is put together again.

    There are two methods of computing the weights. The 'traditional' method
    computes the (n_features x n_features) covariance matrix of X, while the
    'kernel' method instead computes the (n_items x n_items) "item covariance".
    One method can be much more efficient than the other, depending on the
    number of features and items in the data. For the 'kernel' method to work,
    the `cov_updater` parameter must be used instead of the `cov_modifier`
    parameter.

    Parameters
    ----------
    W : ndarray, shape (n_features, n_targets)
        The weights that define the linear model. For example, all Scikit-Learn
        linear models expose the `.coeff_` attribute after fitting, which is
        the intended input to this function.
    X : ndarray, shape (n_samples, n_features)
        The data.
    y : ndarray, shape (n_samples, n_targets)
        The labels. Set to `None` if there are no labels.
    cov_modifier : function (cov, x, y) | none
        the user supplied function that modifies the covariance matrix.
    cov_updater : function (x, y) | CovUpdater | none
        the user supplied function that updates the covariance matrix.
    pattern_modifier : function (pattern, X, y) | None
        The user supplied function that modifies the pattern.
    normalizer_modifier : function (normalizer, X, y, pattern, coef) | None
        The user supplied function that modifies the normalizer.
    method : 'auto' | 'traditional' | 'kernel'
        Whether to use the traditional formulation of the linear model, which
        computes the covariance matrix, or whether to use the kernel trick to
        avoid computing the covariance matrix. Defaults to `'auto'`, which
        attempts to find the best approach automatically.
    cov_modifier_params : list | None
        Extra parameters to pass to the cov_modifier or cov_updater function.
        Defaults to None, meaning no extra parameters.
    pattern_modifier_params : list | None
        Extra parameters to pass to the pattern_modifier function. Defaults to
        None, meaning no extra parameters.
    normalizer_modifier_params : list | None
        Extra parameters to pass to the normalizer_modifier function. Defaults
        to None, meaning no extra parameters.

    Returns
    -------
    coef : ndarray, shape (n_targets, n_features)
        The weights of the re-assembled model.
    cov_X : ndarray, shape (n_features, n_features) | None
        If `method == 'traditional'`, the covariance matrix of X, possibly
        modified by the function given by the user.
        Returns `None` when `method == 'kernel'`.
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
    if cov_modifier_params is None:
        cov_modifier_params = []
    if pattern_modifier_params is None:
        pattern_modifier_params = []
    if normalizer_modifier_params is None:
        normalizer_modifier_params = []

    pattern, normalizer = disassemble_model(W, X, compute_cov_X=False)

    # Shortcut if no modifications are required
    if (cov_modifier is cov_updater is pattern_modifier is normalizer_modifier
            is None):
        return W, None, pattern, normalizer  # No covariance is computed

    # Modify the pattern
    if pattern_modifier is not None:
        pattern = pattern_modifier(pattern, X, y, *pattern_modifier_params)

    # Compute weights
    coef, cov_X = _compute_weights(X, y, pattern, cov_modifier=cov_modifier,
                                   cov_updater=cov_updater,
                                   cov_modifier_params=cov_modifier_params,
                                   method=method)

    # Modify and apply the normalizer
    if normalizer_modifier is not None:
        normalizer = normalizer_modifier(normalizer, X, y, pattern, coef,
                                         *normalizer_modifier_params)

    if coef.ndim == 1:
        coef = coef[np.newaxis, :]

    normalizer = np.atleast_2d(normalizer)
    coef = normalizer.dot(coef)

    return coef, cov_X, pattern, normalizer


class Workbench(LinearModel, TransformerMixin, RegressorMixin):
    '''
    Work bench for post-hoc alteration of a linear model.

    Decomposes the `.coef_` of a linear model into a whitener `pinv(cov)` and a
    pattern. The whitener and pattern can then be altered and the linear model
    can be re-assembled.

    There are two methods of computing the weights. The 'traditional' method
    computes the (n_features x n_features) covariance matrix of X, while the
    'kernel' method instead computes the (n_items x n_items) "item covariance".
    One method can be much more efficient than the other, depending on the
    number of features and items in the data. For the 'kernel' method to work,
    the `cov_updater` parameter must be used instead of the `cov_modifier`
    parameter.

    Parameters
    ----------
    model : instance of sklearn.linear_model.LinearModel
        The linear model to alter.
    cov_modifier : function | None
        Function that takes a covariance matrix (an ndarray of shape
        (n_features, n_features)) and modifies it. Must have the signature:
        `def cov_modifier(cov, X, y)`
        and return the modified covariance matrix. Defaults to `None`, which
        means no modification of the covariance matrix is performed.
        Alternatively, an updater function for the covariance may be specified.
        See the `cov_updater` parameter.
    cov_updater : function | CovUpdater | None
        Function that returns a matrix (an ndarray of shape
        (n_features, n_features)) that will be added to the covariance matrix.
        Must have the signature:
        `def cov_updater(X, y)`
        and return the matrix to be added. Defaults to `None`, which means no
        modification of the covariance matrix is performed. Using this
        parameter instead of `cov_modifier` allows the usage of
        `method='kernel'`.
    pattern_modifier : function | None
        Function that takes a pattern (an ndarray of shape (n_features,
        n_targets)) and modifies it. Must have the signature:
        `def pattern_modifier(pattern, X, y)`
        and return the modified pattern. Defaults to `None`, which means no
        modification of the pattern.
    normalizer_modifier : function | None
        Function that takes a normalizer (an ndarray of shape (n_targets,
        n_targets)) and modifies it. Must have the signature:
        `def normalizer_modifier(coef, X, y, pattern, coef)`
        and return the modified normalizer. Defaults to `None`, which means no
        modification of the normalizer.
    method : 'traditional' | 'kernel' | 'auto'
        Whether to use the traditional formulation of the linear model, which
        computes the covariance matrix, or whether to use the kernel trick to
        avoid computing the covariance matrix. Defaults to `'auto'`, which
        attempts to find the best approach automatically.

    Attributes
    ----------
    coef_ : ndarray, shape (n_targets, n_features)
        Matrix containing the filter weights.
    intercept_ : ndarray, shape (n_targets)
        The intercept of the linear model.
    cov_X_ : ndarray, shape (n_features, n_features)
        The altered covariance matrix of X.
        Only set when `method=='traditional'`.
    pattern_ : ndarray, shape (n_features, n_targets)
        The altered pattern.
    normalizer_ : ndarray, shape (n_targets, n_targets)
        The altered normalizer.
    '''
    def __init__(self, model, cov_modifier=None, cov_updater=None,
                 pattern_modifier=None, normalizer_modifier=None,
                 method='auto'):
        self.model = model
        self.cov_modifier = cov_modifier
        self.cov_updater = cov_updater
        self.pattern_modifier = pattern_modifier
        self.normalizer_modifier = normalizer_modifier
        self.method = method

        if not isinstance(model, LinearModel):
            warn(
                'The base model must be a linear model following the API of '
                'Scikit-learn. However, the model you specified is not a '
                'subclass of `sklearn.linear_model.base.LinearModel`. '
                'Proceeding under the assumption that the `coef_` attribute '
                'will be properly set after fitting the model.'
            )

        if cov_modifier is not None and cov_updater is not None:
            raise ValueError('Cannot specify both a covariance modifier and '
                             'a coviarance updater function.')

        if is_updater(cov_modifier):
            raise ValueError('Cannot specify a CovUpdater object as '
                             'cov_modifier. Use the cov_updater parameter '
                             'instead.')

        if method not in ['traditional', 'kernel', 'auto']:
            raise ValueError('Invalid value for "method" parameter. Must be '
                             'one of: "traditional", "kernel", or "auto"')

        if (method == 'kernel' and
                cov_updater is None and
                cov_modifier is not None):
            raise ValueError('When using the kernel method, please specify a '
                             'covariance updater function, rather than a '
                             'covariance modifier function.')

    def fit(self, X, y):
        """Fit the model to the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data to fit the model to.
        y : ndarray, shape (n_features, n_targets)
            The target labels.

        Returns
        -------
        self : instance of Workbench
            The fitted model.
        """
        # Fit the base model
        self.model.fit(X, y)

        if not hasattr(self.model, 'coef_'):
            raise RuntimeError(
                'Model does not have a `coef_` attribute after fitting. '
                'This does not seem to be a linear model following the '
                'Scikit-Learn API.'
            )

        # Remove the offset from X and y to compute the covariance later.
        # Also normalize X if the base model did so.
        self.fit_intercept = getattr(self.model, 'fit_intercept', False)
        self.normalize = getattr(self.model, 'normalize', False)

        X, y_, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X=X, y=y, fit_intercept=self.fit_intercept,
            normalize=self.normalize, copy=True, 
        )
        if isinstance(self.model, RegressorMixin):
            y = y_
        else:
            y_offset = 0

        # Ensure that y is a 2D array: n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T

        # The `coef_` attribute of Scikit-Learn linear models are re-scaled
        # after normalization. Undo this re-scaling.
        W = self.model.coef_ * X_scale
        # Modify the original linear model and obtain a new one
        coef, cov_X, pattern, normalizer = disassemble_modify_reassemble(
            W, X, y, self.cov_modifier, self.cov_updater,
            self.pattern_modifier, self.normalizer_modifier, self.method,
        )

        # Store the decomposed model as attributes, so the user may inspect it
        if flat_y:
            self.coef_ = coef.ravel()
        else:
            self.coef_ = coef

        if self.normalize:
            self.pattern_normalized_ = pattern
        self.pattern_ = pattern * X_scale[:, np.newaxis]

        if cov_X is not None:
            self.cov_X_ = cov_X

        self.normalizer_ = normalizer

        # Set intercept and undo normalization
        self._set_intercept(X_offset, y_offset, X_scale)

        return self

    def transform(self, X):
        """Apply the model to the data.

        Parameters
        ----------
        X : ndarray, shape (n_items, n_features)
            The data.

        Returns
        -------
        X_trans : ndarray, shape (n_items, n_targets)
            The transformed data.
        """
        return self.predict(X)


def get_args(updater):
    """Get the arguments of an updater or modifier function that can be
    optimized.

    Parameters
    ---------
    updater : function | instance of `CovUpdater`
        The updater to get the optimizable arguments for.

    Returns
    -------
    args : list of str
        The arguments that can be optimized.
    """
    if isinstance(updater, CovUpdater):
        args = inspect.getargspec(updater.update).args
        args = [arg for arg in args if arg != 'self']
    else:
        args = inspect.getargspec(updater).args
        ignore_args = {'self', 'X', 'y', 'pattern', 'normalizer', 'coef'}
        args = [arg for arg in args if arg not in ignore_args]

    return args


def _get_opt_params(modifier, x0, bounds):
    '''Get x0 and bounds for the optimization algorithm.'''
    if modifier is None:
        return [], []

    if x0 is None:
        if is_updater(modifier):
            x0 = modifier.get_x0()

        if x0 is None:  # is still None
            n_args = len(get_args(modifier))
            x0 = [0] * n_args

    if bounds is None:
        if is_updater(modifier):
            bounds = modifier.get_bounds()

        if bounds is None:  # is still None
            n_args = len(get_args(modifier))
            bounds = [(None, None)] * n_args

    return x0, bounds


class WorkbenchOptimizer(Workbench):
    """Experimental work in process. Don't use this yet."""
    def __init__(self, model, cov_modifier=None, cov_updater=None,
                 cov_param_x0=None, cov_param_bounds=None,
                 pattern_modifier=None, pattern_param_x0=None,
                 pattern_param_bounds=None, normalizer_modifier=None,
                 normalizer_param_x0=None, normalizer_param_bounds=None,
                 optimizer_options=None, method='auto', verbose=True,
                 scoring='neg_mean_squared_error'):
        Workbench.__init__(self, model, cov_modifier, cov_updater,
                           pattern_modifier, normalizer_modifier, method)

        self.cov_param_x0, self.cov_param_bounds = _get_opt_params(
            cov_modifier if cov_modifier is not None else cov_updater,
            cov_param_x0, cov_param_bounds)
        self.pattern_param_x0, self.pattern_param_bounds = _get_opt_params(
            pattern_modifier, pattern_param_x0, pattern_param_bounds)
        self.normalizer_param_x0, self.normalizer_param_bounds = _get_opt_params(
            normalizer_modifier, normalizer_param_x0, normalizer_param_bounds)

        self.verbose = verbose
        self.scoring = scoring

        self.optimizer_options = dict(maxiter=10, eps=1E-3, ftol=1E-6)
        if optimizer_options is not None:
            self.optimizer_options.update(optimizer_options)


    def fit(self, X, y):
        """Fit the model to the data and optimize all parameters.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data to fit the model to.
        y : ndarray, shape (n_features, n_targets)
            The target labels.

        Returns
        -------
        self : instance of Workbench
            The fitted model.
        """
        # Keep a copy of the original X and y
        X_orig, y_orig = X, y

        # Remove the offset from X and y to compute the covariance later.
        # Also normalize X if the base model did so.
        self.fit_intercept = getattr(self.model, 'fit_intercept', False)
        self.normalize = getattr(self.model, 'normalize', False)
        X, y_, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X=X, y=y, fit_intercept=self.fit_intercept,
            normalize=self.normalize, copy=True, 
        )

        if isinstance(self.model, RegressorMixin):
            y = y_
        else:
            y_offset = 0

        n_samples, n_features = X.shape

        # Ensure that y is a 2D array: n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T
        n_targets = y.shape[1]

        if is_updater(self.cov_updater):
            # Initialize the CovUpdater object
            self.cov_updater.fit(X, y)

        # Collect parameters to optimize
        n_cov_updater_params = len(get_args(self.cov_updater))
        if self.pattern_modifier is not None:
            n_pat_modifier_params = len(get_args(self.pattern_modifier))
        else:
            n_pat_modifier_params = 0

        # Prepare scoring functions
        scorer = check_scoring(self, scoring=self.scoring, allow_none=False)

        # The scorer wants an object that will make the predictions but
        # they are already computed. This identity_estimator will just
        # return them.
        def identity_estimator():
            pass
        identity_estimator.decision_function = lambda y_predict: y_predict
        identity_estimator.predict = lambda y_predict: y_predict

        # Determine optimal method of solving
        if self.method == 'auto':
            if n_features > n_samples and self.cov_updater is not None:
                method = 'kernel'
            else:
                method = 'traditional'
        else:
            method = self.method

        # Compute patterns and normalizers for all LOO iterations
        Ps, Ns = self._loo_patterns_normalizers(X, y)
        if method == 'traditional':
            cov_X = X.T.dot(X)

        cache = dict()
        self.log_ = []  # Keeps track of the tried parameters and their score

        def score(args):
            # Convert params to a tuple, so it can be hashed
            cov_updater_params = tuple(
                args[:n_cov_updater_params].tolist()
            )
            pattern_modifier_params = tuple(
                args[n_cov_updater_params:n_cov_updater_params +
                     n_pat_modifier_params].tolist()
            )
            normalizer_modifier_params = tuple(
                args[n_cov_updater_params+n_pat_modifier_params:].tolist()
            )

            if method == 'traditional':
                y_hat = do_loo(X, y, Ps, Ns, cov_X, self.cov_modifier,
                               self.cov_updater, cov_updater_params,
                               self.pattern_modifier, pattern_modifier_params,
                               self.normalizer_modifier,
                               normalizer_modifier_params, cache)
            elif method == 'kernel':
                y_hat = do_loo_kernel(X, y, Ps, Ns, self.cov_updater,
                                      cov_updater_params,
                                      self.pattern_modifier,
                                      pattern_modifier_params,
                                      self.normalizer_modifier,
                                      normalizer_modifier_params, cache)

            score = scorer(identity_estimator, y.ravel(), y_hat.ravel())
            self.log_.append(args.tolist() + [score])

            if self.verbose:
                print('cov_updater_params=%s, pattern_modifier_params=%s, '
                      'normalizer_modifier_params=%s score=%f' %
                      (cov_updater_params, pattern_modifier_params,
                       normalizer_modifier_params, score))
            return -score

        params = minimize(
            score,
            x0=self.cov_param_x0 + self.pattern_param_x0 + self.normalizer_param_x0,
            method='L-BFGS-B',
            bounds=self.cov_param_bounds + self.pattern_param_bounds + self.normalizer_param_bounds,
            options=self.optimizer_options,
        ).x.tolist()

        self.cov_updater_params_ = params[:n_cov_updater_params]
        self.pattern_modifier_params_ = params[n_cov_updater_params:n_cov_updater_params + n_pat_modifier_params]
        self.normalizer_modifier_params_ = params[n_cov_updater_params + n_pat_modifier_params:]

        # Compute the linear model with the optimal parameters
        W = self.model.fit(X_orig, y_orig).coef_
        W *= X_scale

        # Modify the original linear model and obtain a new one
        coef, cov_X, pattern, normalizer = disassemble_modify_reassemble(
            W, X, y,
            cov_updater=self.cov_updater,
            pattern_modifier=self.pattern_modifier,
            normalizer_modifier=self.normalizer_modifier,
            cov_modifier_params=self.cov_updater_params_,
            pattern_modifier_params=self.pattern_modifier_params_,
            normalizer_modifier_params=self.normalizer_modifier_params_,
        )

        # Store the decomposed model as attributes, so the user may inspect it
        if flat_y:
            self.coef_ = coef.ravel()
        else:
            self.coef_ = coef

        if self.normalize:
            self.pattern_normalized_ = pattern
        self.pattern_ = pattern * X_scale[:, np.newaxis]

        if cov_X is not None:
            self.cov_X_ = cov_X

        self.normalizer_ = normalizer

        # Set intercept and undo normalization
        self._set_intercept(X_offset, y_offset, X_scale)

        return self

    def _loo_patterns_normalizers(self, X, y):
        """Construct arrays of patterns and normalizers for each LOO iteration."""
        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        Ps = np.empty((n_samples, n_features, n_targets), dtype=np.float)
        Ns = np.empty((n_samples, n_targets, n_targets), dtype=np.float)
        patterns = loo_utils.loo_patterns_from_model(self.model, X, y,
                                                     verbose=self.verbose)
        for i, (pattern, normalizer) in enumerate(patterns):
            Ps[i] = pattern
            Ns[i] = normalizer

        return Ps, Ns


def do_loo(X, y, Ps, Ns, cov_X, cov_modifier, cov_updater, cov_updater_params,
           pattern_modifier, pattern_modifier_params, normalizer_modifier,
           normalizer_modifier_params, cache):

    if cov_updater_params in cache:
        # Cache hit
        cov_X_inv = cache[cov_updater_params]
    else:
        # Cache miss, compute values and store in cache
        if cov_modifier is not None:
            cov_X = cov_modifier(cov_X, X, y, *cov_updater_params)
        if is_updater(cov_updater):
            # User supplied a CovUpdater object for detailed control
            cov_update = cov_updater.update(*cov_updater_params)
            cov_X = cov_update.add(cov_X)
        else:
            # User supplied an updater function
            cov_X += cov_updater(X, y, *cov_updater_params)

        cov_X_inv = pinv(cov_X)
        cache[cov_updater_params] = cov_X_inv

    # Do leave-one-out crossvalidation
    y_hat = np.zeros_like(y)
    for train, test in LeaveOneOut().split(X, y):
        P = Ps[test[0]]
        if pattern_modifier is not None:
            P = pattern_modifier(P, X, y, *pattern_modifier_params)

        coef = cov_X_inv.dot(P).T

        N = Ns[test[0]]
        if normalizer_modifier is not None:
            N = normalizer_modifier(N, X, y, P, coef
                                    *normalizer_modifier_params)
        N = np.atleast_2d(N)
        y_hat[test] = multi_dot((X[test], coef.T, N))

    return y_hat


def do_loo_kernel(X, y, Ps, Ns, cov_updater, cov_updater_params,
                  pattern_modifier, pattern_modifier_params,
                  normalizer_modifier, normalizer_modifier_params, cache):
    n_samples = len(X)

    if cov_updater_params in cache:
        # Cache hit
        cov_update_inv, G, K = cache[cov_updater_params]
    else:
        # Cache miss, compute values and store in cache
        if is_updater(cov_updater):
            # User supplied a CovUpdater object for detailed control
            cov_update = cov_updater.update(*cov_updater_params)
            cov_update_inv = cov_update.inv()
        else:
            # User supplied an updater function
            cov_update = cov_updater(X, y)
            cov_update_inv = np.linalg.pinv(cov_update)

        G = cov_update_inv.dot(X.T)
        K = X.dot(G)
        K.flat[::n_samples + 1] += 1
        cache[cov_updater_params] = (cov_update_inv, G, K)

    # Do efficient leave-one-out crossvalidation
    y_hat = np.zeros_like(y)
    G1 = None
    X1 = None
    y1 = None
    for K_i, test in zip(loo_utils.loo_kern_inv(K), range(n_samples)):
        if G1 is None or X1 is None:
            G1 = G[:, 1:].copy()
            X1 = X[1:].copy()
            y1 = y[1:].copy()
        else:
            if test >= 2:
                G1[:, test - 2] = G[:, test - 1]
                X1[test - 2] = X[test - 1]
                y1[test - 2] = y[test - 1]
            G1[:, test - 1] = G[:, 0]
            X1[test - 1] = X[0]
            y1[test - 1] = y[0]

        P = Ps[test]
        if pattern_modifier is not None:
            P = pattern_modifier(P, X, y, *pattern_modifier_params)
        GammaP = cov_update_inv.dot(P)

        coef = (GammaP - G1.dot(K_i.dot(X1.dot(GammaP)))).T

        N = Ns[test]
        if normalizer_modifier is not None:
            N = normalizer_modifier(N, X, y, P, coef,
                                    *normalizer_modifier_params)
        N = np.atleast_2d(N)

        if coef.ndim == 1:
            coef = coef[np.newaxis, :]

        y_hat[test] = multi_dot((X[[test]], coef.T, N))

    return y_hat
