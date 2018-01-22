# encoding: utf-8
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel

from .workbench import _compute_weights


class LCMV(LinearModel, TransformerMixin, RegressorMixin):
    '''An LCMV beamformer filter.

    A beamformer filter attempts to isolate a specific signal in the data. The
    signal of interest is specified as an activation template. The linearly
    constrained minimum variance (LCMV) beamformer attemps to minimize the
    overall variance of the output while not reducing the variance of the
    signal of interest.

    By default, a 'unit-gain' LCMV beamformer is used, that passes a signal
    conforming to the given template with unit gain (self.coef_ @ template ==
    I). Other types of beamformers can be constructed by using the
    `normalizer_modifier` parameter.

    Shrinkage of the covariance matrix can improve the stability of the filter
    when only few data are available. This shrinkage can be performed by
    using either a `cov_modifier` or `cov_updater` parameter.

    There are two methods of computing the weights. The 'traditional' method
    computes the (n_features x n_features) covariance matrix of X, while the
    'kernel' method instead computes the (n_items x n_items) "item covariance".
    One method can be much more efficient than the other, depending on the
    number of features and items in the data. For the 'kernel' method to work
    in combination with shrinkage, the `cov_updater` parameter must be used
    instead of the `cov_modifier` parameter.

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
    normalizer_modifier : function | None
        Function that takes a normalizer (an ndarray of shape (n_targets,
        n_targets)) and modifies it. Must have the signature:
        `def normalizer_modifier(coef, X, y, template, coef)`
        and return the modified normalizer. Defaults to `None`, which means no
        modification of the normalizer.
    method : 'traditional' | 'kernel' | 'auto'
        Whether to use the traditional formulation of the linear model, which
        computes the covariance matrix, or whether to use the kernel trick to
        avoid computing the covariance matrix. Defaults to `'auto'`, which
        attempts to find the best approach automatically.

    Attributes
    ----------
    coef_ : ndarray, shape (n_channels * n_samples, n_signals)
        The filter weights.
    '''
    def __init__(self, template, center=True, normalize=False,
                 cov_modifier=None, cov_updater=None, normalizer_modifier=None,
                 method='auto'):
        template = np.asarray(template)
        if template.ndim == 1:
            self.template = template[np.newaxis, :]
        else:
            self.template = template
        self.center = center
        self.fit_intercept = self.center
        self.normalize = normalize
        self.cov_modifier = cov_modifier
        self.cov_updater = cov_updater
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
        coef, _ = _compute_weights(X, None, self.template.T, self.cov_modifier,
                                   self.cov_updater, self.method)

        # The default normalizer constructs a unit-gain LCMV beamformer
        normalizer = [c.dot(p) for c, p in zip(coef, self.template)]
        normalizer = np.diag(normalizer)

        # Modify and apply the normalizer
        if self.normalizer_modifier is not None:
            normalizer = self.normalizer_modifier(normalizer, X, None,
                                                  self.template.T, coef)
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
