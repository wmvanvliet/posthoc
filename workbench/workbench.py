# encoding: utf-8
from __future__ import print_function
from warnings import warn
import inspect

import numpy as np
from scipy.optimize import minimize
from numpy.linalg import multi_dot

from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.base import LinearModel
from sklearn.metrics.scorer import check_scoring

from . import loo_utils
from .cov_updaters import CovUpdater


def is_updater(x):
    return isinstance(x, CovUpdater)


def patterns_from_weights(coef, X):
    '''Derive the learned pattern from a fitted linear model.

    Applies the Haufe trick to compute patterns from weights:
    equation (6) from Haufe et al. 2014 [1].

    Parameters
    ----------
    coef : ndarray, shape (n_features, n_targets)
        The weights that define the linear model. For example, all Scikit-Learn
        linear models expose the `.coeff_` attribute after fitting, which is
        the intended input to this function.
    X : nparray, shape (n_samples, n_features)
        The data.

    Returns
    -------
    pattern : ndarray, shape (n_features, n_targets)
        The pattern learned by the linear model. This is the result of the
        Haufe trick.

    Notes
    -----
    Make sure to supply the actual `X` used to compute the `coeff`. Take note
    of thinks like centering and z-scoring, which are sometimes performed by
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

    return LinearRegression(fit_intercept=False).fit(y_hat, X).coef_


class Workbench(LinearModel, TransformerMixin, RegressorMixin):
    '''
    Work bench for post-hoc alteration of a linear model.

    Decomposes the `.coef_` of a linear model into a whitener `pinv(cov)` and a
    pattern. The whitener and pattern can then be altered and the linear model
    can be re-assembled.

    Parameters
    ----------
    model : instance of sklearn.linear_model.LinearModel
        The linear model to alter.

    cov_modifier : function | None
        Function that takes a covariance matrix (an ndarray of shape
        (n_features, n_features)) and modifies it.
        Must have the signature `def cov_modifier(cov, X, y)` and return the
        modified covariance matrix. Defaults to `None`, which means no
        modification of the covariance matrix is performed. Alternatively, an
        updater function for the covariance may be specified. See the
        `cov_updater` parameter.

    cov_updater : function | None
        Function that returns a matrix (an ndarray of shape
        (n_features, n_features)) that will be added to the covariance matrix.
        Must have the signature `def cov_updater(X, y)` and return the matrix
        to be added. Defaults to `None`, which means no modification of the
        covariance matrix is performed. Using this parameter instead of
        `cov_modifier` allows the usage of `method='kernel'`.

    pattern_modifier : function | None
        Function that takes a pattern (an ndarray of shape (n_features,
        n_targets)) and modifies it.
        Must have the signature `def cov_pattern(pattern, X, y)` and return the
        modified pattern. Defaults to `None`, which means no modification of
        the pattern.

    method : 'traditional' | 'kernel' | 'auto'
        Whether to use the traditional formulation of the linear model, which
        computes the covariance matrix, or whether to use the kernel trick to
        avoid computing the covariance matrix. The latter is more efficient
        when `n_features > n_samples`. Using the kernel method requires using a
        `cov_updater` function instead of a `cov_modifier`. Defaults to
        `'auto'`, which means using the kernel method if `n_features >
        n_samples` and the `cov_updater` parameter is used, otherwise using the
        traditional method.

    Attributes
    ----------
    coef_ : ndarray, shape (n_features, n_targets)
        Matrix containing the filter weights.
    intercept_ : ndarray, shape (n_targets)
        The intercept of the linear model.
    cov_ : ndarray, shape (n_features, n_features)
        When using the traditional method, the altered covariance matrix.
    pattern_ : ndarray, shape (n_features, n_targets)
        The altered pattern.
    '''
    def __init__(self, model, cov_modifier=None, cov_updater=None,
                 pattern_modifier=None, method='auto', x0=None, bounds=None):

        self.model = model
        self.cov_modifier = cov_modifier
        self.cov_updater = cov_updater
        self.pattern_modifier = pattern_modifier
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

        if method not in ['traditional', 'kernel', 'auto']:
            raise ValueError('Invalid value for "method" parameter. Must be '
                             'one of: "traditional", "kernel", or "auto"')

        if (method == 'kernel' and
                cov_updater is None and
                cov_modifier is not None):
            raise ValueError('When using the kernel method, please specify a '
                             'covariance updater function, rather than a '
                             'covariance modifier function.')

    def _disassemble_modify_reassemble(self, W, X, y, cov_modifier=None,
                                       cov_updater=None,
                                       pattern_modifier=None):
        '''Disassemble, modify and reassemble a fitted linear model.

        This is the meat of the workbench approach. The linear model wrapped by
        this class is disassembled, the whitener and pattern are modified, and
        the model is put together again.

        Parameters
        ----------
        W : ndarray, shape (n_features, n_targets)
            The weights that define the linear model. For example, all
            Scikit-Learn linear models expose the `.coeff_` attribute after
            fitting, which is the intended input to this function.
        X : ndarray, shape (n_samples, n_features)
            The data.
        y : ndarray, shape (n_samples, n_targets)
            The labels.
        cov_modifier : function (cov, X, y) | None
            The user supplied function that modifies the covariance matrix.
        cov_updater : function (X, y) | None
            The user supplied function that updates the covariance matrix.
        pattern_modifier : function (pattern, X, y) | None
            The user supplied function that modifies the pattern.

        Returns
        -------
        coef : ndarray, shape (n_targets, n_features)
            The weights of the re-assembled model.
        pattern : ndarray, shape (n_features, n_targets)
            The pattern, possibly modified by the function given by the user.
        cov_X : ndarray, shape (n_features, n_features) | None
            If `method == 'traditional'`, the covariance matrix of X, possibly
            modified by the function given by the user. Returns `None` when
            `method == 'kernel'`.

        Notes
        -----
        Make sure to supply the actual `X` that was used to compute `W`. Take
        note of things like centering and z-scoring, which are sometimes
        performed by a Scikit-Learn model.
        '''
        n_samples, n_features = X.shape

        y_hat = X.dot(W.T)
        if y_hat.ndim == 1:
            y_hat = y_hat[:, np.newaxis]

        # Compute the pattern from the base model filter weights, conforming
        # equation 6 from Haufe2014.
        pattern = LinearRegression(fit_intercept=False).fit(y_hat, X).coef_
        #pattern = X.T.dot(y_hat)
        scaler = y_hat.T.dot(y_hat)
        #pattern = X.T.dot(X).dot(W[:, None]).dot(np.linalg.pinv(scaler))

        # Modify the pattern using the function supplied by the user
        if self.pattern_modifier is not None:
            pattern = self.pattern_modifier(pattern, X, y)

        # Shortcut if no modifications are required
        if cov_modifier is cov_updater is pattern_modifier is None:
            return W, pattern, None  # No covariance is computed

        # Determine optimal method of solving
        if self.method == 'auto':
            if n_features > n_samples and cov_updater is not None:
                method = 'kernel'
            else:
                method = 'traditional'
        else:
            method = self.method

        if method == 'traditional':
            # Compute the covariance of X
            cov_X = X.T.dot(X)

            # Modify the covariance using the function supplied by the user
            if cov_modifier is not None:
                # User supplied a modifier function
                cov_X = cov_modifier(cov_X, X, y)
            if cov_updater is not None:
                if is_updater(cov_updater):
                    # User supplied a CovUpdater object for detailed control
                    update = cov_updater.fit(X, y).update()
                    cov_X = update.add(cov_X)
                else:
                    # User supplied an updater function
                    cov_X += cov_updater(X, y)

            # Re-assemble the linear model
            cov_X_inv = np.linalg.pinv(cov_X)

            if self.pattern_modifier is None:
                # Scaler that will reconstruct the original weights perfectly
                scaler = y_hat.T.dot(y_hat)
            else:
                # Scaler that re-estimates the weights now that the pattern
                # has been modified.
                scaler = multi_dot((pattern.T, cov_X_inv, pattern))
                scaler = np.linalg.pinv(scaler)

            coef = multi_dot((cov_X_inv, pattern, scaler)).T

        elif method == 'kernel':
            # Get the covariance updater from the function supplied by the user
            if is_updater(cov_updater):
                # User supplied a CovUpdater object for detailed control
                cov_update = cov_updater.fit(X, y).update()
                cov_update_inv = cov_update.inv()
            else:
                # User supplied an updater function
                cov_update = cov_updater(X, y)
                cov_update_inv = np.linalg.pinv(cov_update)

            # Re-assemble the linear model, using the matrix inversion lemma
            G = cov_update_inv.dot(X.T)
            K = X.dot(G)
            K.flat[::n_samples + 1] += 1
            K_inv = np.linalg.pinv(K)
            GammaP = cov_update_inv.dot(pattern)
            coef = (GammaP - multi_dot((G, K_inv, X, GammaP))).T

            scaler = y_hat.T.dot(y_hat)
            if coef.ndim == 1:
                coef = coef[np.newaxis, :]
            coef = scaler.dot(coef)

            cov_X = None  # No actual covariance is computed using the method.

        return coef, pattern, cov_X

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

        n_samples, n_features = X.shape

        # Remove the offset from X and y to compute the covariance later.
        # Also normalize X if the base model did so.
        self.fit_intercept = getattr(self.model, 'fit_intercept', False)
        self.normalize = getattr(self.model, 'normalize', False)
        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X=X, y=y, fit_intercept=self.fit_intercept,
            normalize=self.normalize, copy=True,
        )

        # Ensure that y is a 2D array: n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T

        # The `coef_` attribute of Scikit-Learn linear models are re-scaled
        # after normalization. Undo this re-scaling.
        W = self.model.coef_ * X_scale

        # Modify the original linear model and obtain a new one
        self.coef_, pattern, cov_X = self._disassemble_modify_reassemble(
            W, X, y, self.cov_modifier, self.cov_updater, self.pattern_modifier
        )

        # Store the pattern as an attribute, so the user may inspect it
        if self.normalize:
            self.pattern_normalized_ = pattern
        self.pattern_ = pattern * X_scale[:, np.newaxis]

        # Store the covariance as an attribute, so the user may inspect it
        if cov_X is not None:
            self.cov_ = cov_X

        if flat_y:
            self.coef_ = self.coef_.ravel()

        # Set intercept and undo normalization
        self._set_intercept(X_offset, y_offset, X_scale)

        return self


def get_args(updater):
    """Get the arguments of an updater or modifier function that can be optimized.

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
        ignore_args = {'self', 'X', 'y', 'pattern'}
        args = [arg for arg in args if arg not in ignore_args]
        print('optimizable args:', args)

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
    def __init__(self, model,
                 cov_modifier=None, cov_updater=None,
                 cov_param_x0=None, cov_param_bounds=None,
                 pattern_modifier=None,
                 pattern_param_x0=None, pattern_param_bounds=None,
                 method='auto', verbose=True,
                 scoring='neg_mean_squared_error'):
        Workbench.__init__(self, model, cov_modifier, cov_updater,
                           pattern_modifier, method)

        self.cov_param_x0, self.cov_param_bounds = _get_opt_params(
            cov_modifier if cov_modifier is not None else cov_updater,
            cov_param_x0, cov_param_bounds)
        self.pattern_param_x0, self.pattern_param_bounds = _get_opt_params(
            pattern_modifier, pattern_param_x0, pattern_param_bounds)

        self.verbose = verbose
        self.scoring = scoring

    def fit(self, X, y, sample_weight=None):
        """Fit the model to the data and optimize all parameters.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data to fit the model to.
        y : ndarray, shape (n_features, n_targets)
            The target labels.
        sample_weight : nparray, shape (n_samples) | None
            Weighting factors for each feature. If None, all samples are
            equally weighted.

        Returns
        -------
        self : instance of Workbench
            The fitted model.
        """
        # Remove the offset from X and y to compute the covariance later.
        # Also normalize X if the base model did so.
        self.fit_intercept = getattr(self.model, 'fit_intercept', False)
        self.normalize = getattr(self.model, 'normalize', False)
        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X=X, y=y, fit_intercept=self.fit_intercept,
            normalize=self.normalize, copy=True, sample_weight=sample_weight,
        )

        n_samples, n_features = X.shape

        # Ensure that y is a 2D array: n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T
        n_targets = y.shape[1]

        Ps = np.empty((n_samples, n_features, n_targets), dtype=np.float)
        patterns = loo_utils.loo_patterns_from_model(
            self.model, X, y, verbose=self.verbose)
        for i, pattern in enumerate(patterns):
            Ps[i] = pattern

        if is_updater(self.cov_updater):
            self.cov_updater.fit(X, y)

        n_cov_updater_params = len(get_args(self.cov_updater))
        cov_param_cache = dict()

        scorer = check_scoring(self, scoring=self.scoring, allow_none=False)

        # The scorer wants an object that will make the predictions but
        # they are already computed. This identity_estimator will just
        # return them.
        def identity_estimator():
            pass
        identity_estimator.decision_function = lambda y_predict: y_predict
        identity_estimator.predict = lambda y_predict: y_predict

        def compute_GK(cov_update_inv, X):
            G = cov_update_inv.dot(X.T)
            K = X.dot(G)
            K.flat[::n_samples + 1] += 1

            return G, K

        def score(args):
            # Convert params to a tuple, so it can be hashed
            cov_updater_params = tuple(
                args[:n_cov_updater_params].tolist()
            )
            pattern_modifier_params = tuple(
                args[n_cov_updater_params:].tolist()
            )

            if cov_updater_params in cov_param_cache:
                # Cache hit
                cov_update_inv, G, K = cov_param_cache[cov_updater_params]
            else:
                # Cache miss, compute values and store in cache
                if is_updater(self.cov_updater):
                    # User supplied a CovUpdater object for detailed control
                    cov_update = self.cov_updater.update(*cov_updater_params)
                    cov_update_inv = cov_update.inv()
                else:
                    # User supplied an updater function
                    cov_update = self.cov_updater(X, y)
                    cov_update_inv = np.linalg.pinv(cov_update)
                G, K = compute_GK(cov_update_inv, X)
                cov_param_cache[cov_updater_params] = (cov_update_inv, G, K)

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
                if self.pattern_modifier is not None:
                    P = self.pattern_modifier(P, X, y,
                                              *pattern_modifier_params)
                GammaP = cov_update_inv.dot(P)

                y_hat[test] = X[test].dot(GammaP -
                                          G1.dot(K_i.dot(X1.dot(GammaP))))
            score = scorer(identity_estimator, y.ravel(), y_hat.ravel())

            if self.verbose:
                print('cov_updater_params=%s, pattern_modifier_params=%s, '
                      'score=%f' %
                      (cov_updater_params, pattern_modifier_params, score))
            return -score

        params = minimize(
            score,
            x0=self.cov_param_x0 + self.pattern_param_x0,
            method='L-BFGS-B',
            bounds=self.cov_param_bounds + self.pattern_param_bounds,
            options=dict(
                maxiter=10,
                eps=1E-3,
                ftol=1E-6,
            ),
        ).x.tolist()
        #params = self.cov_param_x0
        #score(np.asarray(params))
        self.cov_updater_params_ = params[:n_cov_updater_params]
        self.pattern_modifier_params_ = params[n_cov_updater_params:]

        # Construct wrapper methods that call the user supplied functions with
        # the optimal parameters.
        if is_updater(self.cov_updater):
            cov_updater = self.cov_updater.update(*self.cov_updater_params_)
        elif self.cov_updater is not None:
            def cov_updater(X, y):
                return self.cov_updater(X, y, *self.cov_updater_params_)
        else:
            cov_updater = None

        if self.pattern_modifier is not None:
            def pattern_modifier(pattern, X, y):
                return self.pattern_modifier(X, y,
                                             *self.pattern_modifier_params_)
        else:
            pattern_modifier = None

        # Compute the linear model with the optimal parameters
        W = self.model.fit(X, y).coef_ * X_scale
        self.coef_, pattern, cov_X = self._disassemble_modify_reassemble(
            W, X, y,
            cov_updater=cov_updater,
            pattern_modifier=pattern_modifier
        )

        # Store the pattern as an attribute, so the user may inspect it
        if self.normalize:
            self.pattern_normalized_ = pattern
        self.pattern_ = pattern * X_scale[:, np.newaxis]

        if flat_y:
            self.coef_ = self.coef_.ravel()

        # Set intercept and undo normalization
        self._set_intercept(X_offset, y_offset, X_scale)

        return self
