# encoding: utf-8
from __future__ import print_function
import inspect

import numpy as np
from scipy.optimize import minimize

from sklearn.base import TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import check_scoring

from . import loo_utils
from .cov_estimators import CovEstimator, Empirical
from .linear_model import disassemble_modify_reassemble


def is_estimator(x):
    return isinstance(x, CovEstimator)


class Workbench(LinearModel, TransformerMixin, RegressorMixin):
    '''
    Work bench for post-hoc alteration of a linear model.

    Decomposes the ``.coef_`` of a linear model into a covariance matrix, a
    pattern and a normalizer. These components are then altered by user
    speficied functions and the linear model is re-assembled.

    Parameters
    ----------
    model : instance of sklearn.linear_model._base.LinearModel
        The linear model to alter.
    cov : instance of CovEstimator | function | None
        The method used to estimate the covariance. Can either be one of the
        predefined CovEstimator objects, or a function. If a function is used,
        it must have the signature: ``def cov_modifier(cov, X, y)``, take the
        training data as input and return a matrix that will be added to the
        emperical covariance matrix. Defaults to `None`, which means the
        default empirical estimator of the covariance matrix is used.
    pattern_modifier : function | None
        Function that takes a pattern (an ndarray of shape (n_features,
        n_targets)) and modifies it. Must have the signature:
        ``def pattern_modifier(pattern, X, y)``
        and return the modified pattern. Defaults to ``None``, which means no
        modification of the pattern.
    normalizer_modifier : function | None
        Function that takes a normalizer (an ndarray of shape (n_targets,
        n_targets)) and modifies it. Must have the signature:
        ``def normalizer_modifier(coef, X, y, pattern, coef)``
        and return the modified normalizer. Defaults to ``None``, which means
        no modification of the normalizer.

    Attributes
    ----------
    coef_ : ndarray, shape (n_targets, n_features)
        Matrix containing the filter weights.
    intercept_ : ndarray, shape (n_targets)
        The intercept of the linear model.
    pattern_ : ndarray, shape (n_features, n_targets)
        The altered pattern.
    normalizer_ : ndarray, shape (n_targets, n_targets)
        The altered normalizer.
    '''
    def __init__(self, model, cov=None, pattern_modifier=None,
                 normalizer_modifier=None):
        ## Morph our class to be that of the base model
        #new_class = (self.__class__, LinearModel, TransformerMixin)
        #if isinstance(model, RegressorMixin):
        #    new_class += (RegressorMixin,)
        #elif isinstance(model, ClassifierMixin):
        #    new_class += (ClassifierMixin,)
        #else:
        #    raise ValueError('Base model must identify as being a regression '
        #                     'or classification model by inheriting from '
        #                     'either sklearn.base.RegressorMixin or '
        #                     'sklearn.base.ClassifierMixin')
        #self.__class__ = type(self.__class__.__name__, new_class, {})
        if not isinstance(model, (RegressorMixin, ClassifierMixin)):
            raise ValueError('Base model must identify as being a regression '
                             'or classification model by inheriting from '
                             'either sklearn.base.RegressorMixin or '
                             'sklearn.base.ClassifierMixin')
        self.model = model
        self.cov = cov
        self.pattern_modifier = pattern_modifier
        self.normalizer_modifier = normalizer_modifier

    def fit(self, X, y, fit_intercept=False, normalize=False):
        """Fit the model to the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data to fit the model to.
        y : ndarray, shape (n_features, n_targets)
            The target labels.
        fit_intercept : bool
            Whether to fit and remove the mean of the data before passing it to
            the model. Defaults to False.
        normalize : bool
            Whether to fit and remove the standard dev. of the data before
            passing it to the model. Can only be used when `fit_intercept=True`.
            Defaults to False.

        Returns
        -------
        self : instance of Workbench
            The fitted model.
        """
        # Store the unique class labels, we need them to predict the exact
        # class labels as were given.
        if isinstance(self.model, ClassifierMixin):
            self.classes_ = np.unique(y)

        # Remove the offset from X and y to compute the covariance later.
        # Also normalize X if the base model did so.
        self.fit_intercept = getattr(self.model, 'fit_intercept', fit_intercept)
        self.normalize = getattr(self.model, 'normalize', normalize)

        X, y_, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X=X, y=y, fit_intercept=self.fit_intercept,
            normalize=self.normalize, copy=True,
        )
        if isinstance(self.model, RegressorMixin):
            y = y_
        else:
            y_offset = 0.

        # Fit the base model
        self.model.fit(X, y)

        if not hasattr(self.model, 'coef_'):
            raise RuntimeError(
                'Model does not have a `coef_` attribute after fitting. '
                'This does not seem to be a linear model following the '
                'Scikit-Learn API.'
            )

        # Get the weight matrix
        W = self.model.coef_

        # For the next computations, ensure that y is a 2D array:
        # n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T

        # Modify the original linear model and obtain a new one
        coef, pattern, normalizer = disassemble_modify_reassemble(
            W, X, y, self.cov, self.pattern_modifier, self.normalizer_modifier
        )

        # Store the decomposed model as attributes, so the user may inspect it
        if flat_y:
            self.coef_ = coef.ravel()
        else:
            self.coef_ = coef

        self.pattern_normalized_ = pattern
        self.pattern_ = pattern * X_scale[:, np.newaxis]

        self.normalizer_ = normalizer

        # Set intercept and undo normalization
        self._set_intercept(X_offset, y_offset, X_scale)
        self.inverse_intercept_ = X_offset - np.dot(y_offset, self.pattern_.T)

        return self

    def transform(self, X):
        """Apply the model to the data.

        Parameters
        ----------
        X : ndarray, shape (n_items, n_features)
            The data to apply the model to.

        Returns
        -------
        X_trans : ndarray, shape (n_items, n_targets)
            The transformed data.
        """
        return self.predict(X)

    def inverse_predict(self, y):
        """Apply the inverse of the linear model to the targets.

        If a linear model predicts y from X: ``y_hat = X @ self.coef_.T``
        then this function predicts X from y: ``X_hat = y @ self.pattern_.T``

        Parameters
        ----------
        y : ndarray, shape (n_items, n_targets)
            The targets to predict the original data for.

        Returns
        -------
        X_hat : ndarray, shape (n_items, n_features)
            The predicted data.
        """
        return y @ self.pattern_.T + self.inverse_intercept_

    def predict(self, X):
        """Predict using the post-hoc modified linear model.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        if isinstance(self.model, ClassifierMixin):
            scores = self._decision_function(X)
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(np.int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]
        else:
            return self._decision_function(X)

    def score(self, X, y, sample_weight=None):
        """Evaluate performence of the model on the given test data and labels.

        Depending on whether the original model was a regression or
        classification model, return either the mean R^2 or the accuracy on the
        given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Mean R^2 or accuracy of ``self.predict(X)`` wrt. `y`.
        """
        if isinstance(self.model, RegressorMixin):
            from sklearn.metrics import r2_score
            score_function = r2_score
        else:
            from sklearn.metrics import accuracy_score
            score_function = accuracy_score
        return score_function(y, self.predict(X), sample_weight=sample_weight)


def get_args(inst):
    """Get the arguments of an updater or modifier function that can be
    optimized.

    Parameters
    ---------
    inst : function | instance of `CovEstimator`
        The instance to get the optimizable arguments for.

    Returns
    -------
    args : list of str
        The arguments that can be optimized.
    """
    if is_estimator(inst):
        args = inspect.getargspec(inst.update).args
        args = [arg for arg in args if arg != 'self' and arg != 'X']
    else:
        args = inspect.getargspec(inst).args
        ignore_args = {'self', 'X', 'y', 'pattern', 'normalizer', 'coef'}
        args = [arg for arg in args if arg not in ignore_args]

    return args


def _get_opt_params(inst, x0, bounds):
    '''Get x0 and bounds for the optimization algorithm.'''
    if x0 is None:
        if inst is None:
            x0 = []
        elif is_estimator(inst):
            x0 = inst.get_x0()

        if x0 is None:  # is still None
            n_args = len(get_args(inst))
            x0 = [0] * n_args

    if bounds is None:
        if inst is None:
            bounds = []
        elif is_estimator(inst):
            bounds = inst.get_bounds()

        if bounds is None:  # is still None
            n_args = len(get_args(inst))
            bounds = [(None, None)] * n_args

    return x0, bounds


class WorkbenchOptimizer(Workbench):
    '''
    Work bench for post-hoc alteration of a linear model, with optimization.

    Decomposes the ``.coef_`` of a linear model into a covariance matrix, a
    pattern and a normalizer. These components are then altered by user
    speficied functions and the linear model is re-assembled.

    In this optimizing version of the workbench, the functions speficied to
    alter the components of the model can have free parameters. These
    parameters will be optimized using an inner leave-one-out cross validation
    loop, using a general purpose convex optimization algorithm (L-BFGS-B).

    Parameters
    ----------
    model : instance of sklearn.linear_model._base.LinearModel
        The linear model to alter.
    cov : instance of CovEstimator
        The method used to estimate the covariance. Can either be one of the
        predefined CovEstimator objects, or a function. If a function is used,
        it must have the signature: ``def cov_modifier(cov, X, y)``, take the
        training data as input and return a matrix that will be added to the
        emperical covariance matrix. Defaults to `None`, which means the
        default empirical estimator of the covariance matrix is used.
    cov_param_x0 : tuple | None
        The initial parameters for the covariance estimator. These parameters
        will be optimized. Defaults to ``None``, which means the settings
        defined in the ``CovEstimator`` object will be used.
    cov_param_bounds : list of tuple | None
        For each parameter for the covariance estimator, the (min, max) bounds.
        You can set a boundary to ``None`` to indicate the parameter is
        unbounded in that direction. By default, the settings defined in the
        ``CovEstimator`` object will be used.
    pattern_modifier : function | None
        Function that takes a pattern (an ndarray of shape (n_features,
        n_targets)) and modifies it. Must have the signature:
        ``def pattern_modifier(pattern, X, y)``
        and return the modified pattern. Defaults to ``None``, which means no
        modification of the pattern.
    pattern_param_x0 : tuple | None
        The initial parameters for the pattern modifier function. These
        parameters will be optimized. Defaults to ``None``, which means no
        parameters of the pattern modifier function will be optimized.
    pattern_param_bounds : list of tuple | None
        For each parameter for the pattern modifier function, the (min, max)
        bounds. You can set a boundary to ``None`` to indicate the parameter
        is unbounded in that direction. By default, all parameters are
        considered unbounded.
    normalizer_modifier : function | None
        Function that takes a normalizer (an ndarray of shape (n_targets,
        n_targets)) and modifies it. Must have the signature:
        ``def normalizer_modifier(normalizer, X, y, pattern, coef)``
        and return the modified normalizer. Defaults to ``None``, which means
        no modification of the normalizer.
    normalizer_param_x0 : tuple | None
        The initial parameters for the normalizer modifier function. These
        parameters will be optimized. Defaults to ``None``, which means no
        parameters of the normalizer modifier function will be optimized.
    normalizer_param_bounds : list of tuple | None
        For each parameter for the normalizer modifier function, the (min, max)
        bounds. You can set a boundary to ``None`` to indicate the parameter
        is unbounded in that direction. By default, all parameters are
        considered unbounded.
    random_search : int
        Number of random sets of parameters to try, before picking the best set
        to use as starting point for the L-BFGS-S algorithm. Set to 0 to
        disable the random search and use the specified x0 instead.
        Defaults to 0.
    optimizer_options : dict | None
        A dictionary with extra options to supply to the L-BFGS-S algorithm. See
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.http://scikit-learn.org/stable/modules/model_evaluation.html
        for a list of parameters.
    loo_patterns_method : 'auto' | 'traditional' | 'kernel'
        Method for computing the pattern matrix for the leave-one-out inner
        crossvalidation loop. The 'traditional' setting will compute it by
        performing a linear regression between the output of the initial model
        and the weight matrix, as described in [1]. The 'kernel' setting
        will use the matrix inversion lemma to compute the patterns, which can
        be much faster when the number of features is larger than the number of
        training samples, at the possible loss of some precision. The default
        'auto' setting will switch between the 'traditional' and 'kernel'
        approaches based on the ratio between the number of features and
        training samples.
    scoring : str | function
        The scoring function to use while optimizating the parameters. Can be
        any scikit-learn scorer, see:
        http://scikit-learn.org/stable/modules/model_evaluation.html.
        Defaults to ``'neg_mean_squared_error'`` which causes the optimization
        algorithm to attempt to minimize the mean-squared-error (MSE) between
        the model output and training labels.
    random_state : :class:`numpy.random.RandomState` | int | None
        Random state to use. Can either be a NumPy
        :class:`numpy.random.RandomState` object, an integer seed or ``None``
        to use a different seed every time. Defaults to ``None``.
    verbose : bool
        Whether to be verbose or not. Defaults to ``True``.

    Attributes
    ----------
    coef_ : ndarray, shape (n_targets, n_features)
        Matrix containing the filter weights.
    intercept_ : ndarray, shape (n_targets)
        The intercept of the linear model.
    pattern_ : ndarray, shape (n_features, n_targets)
        The altered pattern.
    normalizer_ : ndarray, shape (n_targets, n_targets)
        The altered normalizer.
    '''  # noqa
    def __init__(self, model, cov=None, cov_param_x0=None,
                 cov_param_bounds=None, pattern_modifier=None,
                 pattern_param_x0=None, pattern_param_bounds=None,
                 normalizer_modifier=None, normalizer_param_x0=None,
                 normalizer_param_bounds=None, random_search=0,
                 optimizer_options=None,
                 loo_patterns_method='auto', scoring='neg_mean_squared_error',
                 random_state=None, verbose=True):
        Workbench.__init__(self, model, cov, pattern_modifier,
                           normalizer_modifier)

        if cov is None:
            cov = Empirical()

        self.cov_param_x0, self.cov_param_bounds = _get_opt_params(
            cov, cov_param_x0, cov_param_bounds)
        self.pattern_param_x0, self.pattern_param_bounds = _get_opt_params(
            pattern_modifier, pattern_param_x0, pattern_param_bounds)
        self.normalizer_param_x0, self.normalizer_param_bounds = _get_opt_params(
            normalizer_modifier, normalizer_param_x0, normalizer_param_bounds)

        self.verbose = verbose
        self.scoring = scoring
        self.loo_patterns_method = loo_patterns_method
        self.random_search = random_search

        default_options = dict(maxiter=10, eps=1E-3, ftol=1E-6)
        if optimizer_options is None:
            optimizer_options = default_options
        elif not all([key in optimizer_options for key in default_options.keys()]):
            optimizer_options = dict(optimizer_options).update(default_options)
        self.optimizer_options = optimizer_options

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

    def fit(self, X, y, fit_intercept=False, normalize=False):
        """Fit the model to the data and optimize all parameters.

        After fitting, the optimal parameters are available as
        ``.cov_params_``, ``.pattern_modifier_params_`` and
        ``.normalizer_modifier_params_``.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data to fit the model to.
        y : ndarray, shape (n_features, n_targets)
            The target labels.
        fit_intercept : bool
            Whether to fit and remove the mean of the data before passing it to
            the model. Defaults to False.
        normalize : bool
            Whether to fit and remove the standard dev. of the data before
            passing it to the model. Can only be used when `fit_intercept=True`.
            Defaults to False.

        Returns
        -------
        self : instance of Workbench
            The fitted model.
        """
        # Keep a copy of the original X and y
        X_orig, y_orig = X, y

        # Store the unique class labels, we need them to predict the exact
        # class labels as were given.
        if isinstance(self.model, ClassifierMixin):
            self.classes_ = np.unique(y)

        # Remove the offset from X and y to compute the covariance later.
        # Also normalize X if the base model did so.
        self.fit_intercept = getattr(self.model, 'fit_intercept', fit_intercept)
        self.normalize = getattr(self.model, 'normalize', normalize)
        X, y_, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X=X, y=y, fit_intercept=self.fit_intercept,
            normalize=self.normalize, copy=True,
        )

        if isinstance(self.model, RegressorMixin):
            y = y_
        else:
            y_offset = 0

        self.X_offset = X_offset
        self.y_offset = y_offset

        n_samples, n_features = X.shape

        # Initialize the CovEstimator object
        self.cov.fit(X)

        # Collect parameters to optimize
        n_cov_params = len(get_args(self.cov))
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

        # Compute patterns and normalizers for all LOO iterations
        Ps, Ns = self._loo_patterns_normalizers(
            X, y, method=self.loo_patterns_method)

        # For the next computations ensure that y is a 2D array:
        # n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T

        cache = dict()  # Cache previously computed covariances and patterns
        self.log_ = []  # Keeps track of the tried parameters and their score

        def score(args):
            # Convert params to a tuple, so it can be hashed
            cov_params = tuple(args[:n_cov_params].tolist())
            pattern_modifier_params = tuple(
                args[n_cov_params:n_cov_params + n_pat_modifier_params].tolist()
            )
            normalizer_modifier_params = tuple(
                args[n_cov_params + n_pat_modifier_params:].tolist()
            )

            if cov_params in cache:
                # Cache hit
                cov = cache[cov_params]
            else:
                # Cache miss, compute values and store in cache
                cov = self.cov.update(X, *cov_params)
                cache[cov_params] = cov

            y_hat = self._loo(
                X, y, Ps, Ns, cov, pattern_modifier_params,
                normalizer_modifier_params,
            )

            score = scorer(identity_estimator, y_hat.ravel(), y.ravel())
            self.log_.append(args.tolist() + [score])

            if self.verbose:
                print('cov_params=%s, pattern_modifier_params=%s, '
                      'normalizer_modifier_params=%s score=%f' %
                      (cov_params, pattern_modifier_params,
                       normalizer_modifier_params, score))
            return -score

        def random_x0():
            x0 = (self.cov_param_x0 + self.pattern_param_x0 +
                  self.normalizer_param_x0)
            bounds = (self.cov_param_bounds + self.pattern_param_bounds +
                      self.normalizer_param_bounds)
            for i, b in enumerate(bounds):
                if b[0] is not None and b[1] is not None:
                    x0[i] = self.random_state.uniform(*b)
            return np.array(x0)

        # Try different initial parameters
        x0s = [np.array(self.cov_param_x0 + self.pattern_param_x0 + self.normalizer_param_x0)]
        x0s += [random_x0() for _ in range(self.random_search)]
        x0s_perf = [score(x0) for x0 in x0s]

        # Pick best initial parameters
        x0 = x0s[np.argmin(x0s_perf)]

        params = minimize(
            score,
            x0=x0,
            method='L-BFGS-B',
            bounds=(self.cov_param_bounds + self.pattern_param_bounds +
                    self.normalizer_param_bounds),
            options=self.optimizer_options,
        ).x.tolist()

        # Store optimal parameters
        self.cov_params_ = params[:n_cov_params]
        self.pattern_modifier_params_ = params[n_cov_params:n_cov_params + n_pat_modifier_params]
        self.normalizer_modifier_params_ = params[n_cov_params + n_pat_modifier_params:]

        # Compute the linear model with the optimal parameters
        W = self.model.fit(X_orig, y_orig).coef_
        W *= X_scale

        # Modify the original linear model and obtain a new one
        coef, pattern, normalizer = disassemble_modify_reassemble(
            W, X, y,
            cov=self.cov,
            pattern_modifier=self.pattern_modifier,
            normalizer_modifier=self.normalizer_modifier,
            cov_params=self.cov_params_,
            pattern_modifier_params=self.pattern_modifier_params_,
            normalizer_modifier_params=self.normalizer_modifier_params_,
        )

        # Store the decomposed model as attributes, so the user may inspect it
        self.coef_ = coef
        if self.normalize:
            self.pattern_normalized_ = pattern
        self.pattern_ = pattern * X_scale[:, np.newaxis]
        self.normalizer_ = normalizer

        # Re-flatten if the original y was a 1-dimensional matrix
        if flat_y:
            self.coef_ = coef.ravel()
            self.pattern_ = self.pattern_.ravel()

        # Set intercepts
        self._set_intercept(X_offset, y_offset, X_scale)
        self.inverse_intercept_ = X_offset - np.dot(y_offset, self.pattern_.T)

        return self

    def _loo_patterns_normalizers(self, X, y, method):
        """Construct arrays of patterns and normalizers for each LOO iteration.
        """
        n_samples, n_features = X.shape
        if y.ndim == 1:
            n_targets = 1
        else:
            n_targets = y.shape[1]

        Ps = np.empty((n_samples, n_features, n_targets), dtype=float)
        Ns = np.empty((n_samples, n_targets, n_targets), dtype=float)
        patterns = loo_utils.loo_patterns_from_model(
            self.model, X, y, method=method, verbose=self.verbose)
        for i, (pattern, normalizer) in enumerate(patterns):
            Ps[i] = pattern
            Ns[i] = normalizer

        return Ps, Ns

    def _loo(self, X, y, Ps, Ns, cov, pattern_modifier_params,
             normalizer_modifier_params):
        """Compute leave-one-out values."""
        # Do leave-one-out crossvalidation
        y_hat = np.zeros_like(y, dtype=float)
        if self.pattern_modifier is not None:
            Ps = [self.pattern_modifier(P, X_, y_, *pattern_modifier_params)
                  for P, X_, y_ in zip(Ps, loo_utils.loo(X), loo_utils.loo(y))]
            Ps = np.array(Ps)

        parameters = zip(
            cov.loo_inv_dot(X, Ps),
            loo_utils.loo(X), loo_utils.loo(y), Ps, Ns
        )
        for test, (coef, X_, y_, P, N) in enumerate(parameters):
            if coef.ndim == 1:
                coef = coef[np.newaxis, :]
            else:
                coef = coef.T
            if P.ndim == 1:
                P = P[:, np.newaxis]
            N = Ns[test]
            if self.normalizer_modifier is not None:
                N = self.normalizer_modifier(N, X_, y_, P, coef,
                                             *normalizer_modifier_params)
            N = np.atleast_2d(N)
            new_coef = coef.T.dot(N)

            # LOO intercept correction
            Xtest = X[[test]]
            new_offset = 0
            if self.fit_intercept:
                new_offset = (X_ + self.X_offset).mean(axis=0)
                Xtest = Xtest + self.X_offset
                new_offset = -np.dot(new_offset, new_coef)
                if isinstance(self.model, RegressorMixin):
                    new_offset += (y_ + self.y_offset).mean(axis=0)
            y_hat[test] = np.dot(Xtest, new_coef) + new_offset
        return y_hat
