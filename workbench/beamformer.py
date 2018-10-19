# encoding: utf-8
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel

from .cov_estimators import Empirical


class Beamformer(LinearModel, TransformerMixin, RegressorMixin):
    '''A beamformer filter.

    A beamformer filter attempts to isolate a specific signal in the data. The
    signal of interest is specified as an activation pattern.

    By default, a linear constrained minimum variance (LCMV) beamformer is
    used. This beamformer passes a signal conforming to the given template with
    unit gain (self.coef_ @ template == I), while minimizing overall output.
    Other types of beamformers can be constructed by using the
    `normalizer_modifier` parameter.

    Parameters
    ----------
    template : ndarray, shape (n_features,) | (n_signals, n_features)
       Activation template(s) of the signal(s) to extract.
    center : bool (default: True)
        Whether to remove the data mean before applying the filter.
        WARNING: only set to False if the data has been pre-centered. Applying
        the filter to un-normalized data may result in inaccuracies.
    normalize : bool (default: True)
        Whether to normalize (std. dev = 1) the data before fitting the
        beamformer. Can make the filter more robust.
    cov : instance of CovEstimator | function | None
        The method used to estimate the covariance. Can either be one of the
        predefined CovEstimator objects, or a function that takes the empirical
        covariance matrix (an ndarray of shape (n_features, n_features)) as
        input and modifies it. If such a function is used, it must have the
        signature: `def cov_modifier(cov, X, y)` and return the modified
        covariance matrix. Defaults to `None`, which means the default
        empirical estimator of the covariance matrix is used.
    normalizer_modifier : function | None
        Function that takes a normalizer (an ndarray of shape (n_targets,
        n_targets)) and modifies it. Must have the signature:
        `def normalizer_modifier(normalizer, X, y, template, coef)`
        and return the modified normalizer. Defaults to `None`, which means no
        modification of the normalizer.

    Attributes
    ----------
    coef_ : ndarray, shape (n_channels * n_samples, n_signals)
        The filter weights.
    '''
    def __init__(self, template, center=True, normalize=False,
                 cov=None, normalizer_modifier=None,
                 method='auto'):
        template = np.asarray(template)
        if template.ndim == 1:
            self.template = template[np.newaxis, :]
        else:
            self.template = template
        self.center = center
        self.fit_intercept = self.center
        self.normalize = normalize
        if cov is None:
            self.cov = Empirical()
        else:
            self.cov = cov
        self.normalizer_modifier = normalizer_modifier
        self.method = method

    def fit(self, X, y=None):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data.
        y : None
            Unused.
        """
        n_samples, n_features = X.shape

        X, _, X_offset, _, X_scale = LinearModel._preprocess_data(
            X=X, y=np.zeros(n_samples),
            fit_intercept=self.center,
            normalize=self.normalize,
            copy=True
        )

        # Compute weights
        coef = self.cov.fit(X).inv_dot(X, self.template.T).T

        # The default normalizer constructs an LCMV beamformer
        normalizer = np.linalg.pinv(coef.dot(self.template.T))

        # Modify the normalizer with the user specified function
        if self.normalizer_modifier is not None:
            normalizer = self.normalizer_modifier(normalizer, X, None,
                                                  self.template.T, coef)

        # Apply the normalizer
        self.coef_ = normalizer.dot(coef)

        # Undo scaling if self.normalize == True
        self._set_intercept(X_offset, 0, X_scale)

        return self

    def transform(self, X):
        """Apply the beamformer to the data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data.

        Returns
        -------
        X_trans : ndarray, shape (n_samples, n_signals)
            The transformed data.
        """
        return self.predict(X)
