# encoding: utf-8
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel


class LCMV(LinearModel, TransformerMixin, RegressorMixin):
    '''
    LCMV beamformer operating on a template.

    Parameters
    ----------
    template : 1D array (n_features,)
       Activation pattern of the component to extract.
    shrinkage : float (default: 0)
        Shrinkage to apply to the covariance matrix before computing inverse.
    center : bool (default: True)
        Whether to remove the data mean before applying the filter.
        WARNING: only set to False if the data has been pre-centered. Applying
        the filter to un-normalized data may result in inaccuracies.
    normalize : bool (default: True)
        Whether to normalize (std. dev = 1) the data before fitting the
        beamformer. Can make the filter more robust.

    Attributes
    ----------
    coef_ : 1D array (n_channels * n_samples,)
        Vector containing the filter weights.
    '''
    def __init__(self, template, shrinkage=0, center=True, normalize=True):
        if template.ndim == 1:
            self.template = template[:, np.newaxis]
        else:
            self.template = template
        if self.template.shape[1] != 1:
            raise ValueError('The template should have shape (n_features,) or '
                             '(n_features, 1).')

        self.shrinkage = shrinkage
        self.center = center
        self.normalize = normalize
        self.fit_intercept = self.center

    def fit(self, X, y=None):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : 2D array (n_samples, n_features)
            The trials.
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

        cov = X.T.dot(X)
        scale = np.trace(cov) / len(cov)
        cov *= 1 - self.shrinkage
        cov.flat[::n_features + 1] += self.shrinkage * scale
        cov_i = np.linalg.pinv(cov)
        self.coef_ = cov_i.dot(self.template).ravel()
        self.coef_ = self.coef_.ravel()

        # Undo scaling if self.normalize == True
        self._set_intercept(X_offset, 0, X_scale)

        return self
