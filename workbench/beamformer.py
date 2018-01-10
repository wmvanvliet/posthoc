# encoding: utf-8
import numpy as np
from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import minimize
from scipy.stats import zscore
from . import utils


class stLCMV(LinearModel, TransformerMixin, RegressorMixin):
    '''
    Spatio-temporal LCMV beamformer operating on a spatio-temporal template.

    Parameters
    ----------
    template : 2D array (n_channels, n_samples)
       Spatio-temporal activation pattern of the component to extract.

    shrinkage : str | float (default: 'oas')
        Shrinkage parameter for the covariance matrix inversion. This can
        either be speficied as a number between 0 and 1, or as a string
        indicating which automated estimation method to use:

        'none': No shrinkage: emperical covariance
        'oas': Oracle approximation shrinkage
        'lw': Ledoit-Wolf approximation shrinkage

    center : bool (default: True)
        Whether to remove the data mean before applying the filter.
        WARNING: only set to False if the data has been pre-centered. Applying
        the filter to un-normalized data may result in inaccuracies.

    normalize : bool (default: True)
        Whether to normalize (std. dev = 1) the data before fitting the
        beamformer. Can make the filter more robust.

    cov_i : 2D array (n_channels, n_channels) | None
        The inverse spatio-temporal covariance matrix of the data. Use this to
        avoid re-computing it during fitting. When this parameter is set, the
        shrinkage parameter is ignored.

    Attributes
    ----------
    coef_ : 1D array (n_channels * n_samples,)
        Vector containing the filter weights.
    '''
    def __init__(self, template, spacing, alpha=0, beta=0, center=True,
                 normalize=True, solver='auto'):
        if template.ndim == 1:
            self.template = template[:, np.newaxis]
        else:
            self.template = template

        self.alpha = alpha
        self.beta = beta
        self.spacing = spacing
        self.center = center
        self.normalize = normalize
        self.solver = solver
        self.fit_intercept=self.center

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
        if n_features % self.spacing != 0:
            raise ValueError(
                'Spacing parameter (%d) is incompatible with the number of '
                'features (%d)' % (self.spacing, n_features)
            )
        n_blocks = n_features // self.spacing

        X, _, X_offset, _, X_scale = LinearModel._preprocess_data(
            X=X, y=np.zeros(n_samples),
            fit_intercept=self.center,
            normalize=self.normalize,
            copy=True
        )

        # Determine optimal method of solving
        if self.solver == 'auto':
            if n_features > n_samples:
                solver = 'kernel'
            else:
                solver = 'traditional'
        else:
            solver = self.solver

        # Compute outer cov matrix
        X_ = X.reshape(n_samples, -1, self.spacing)
        X_ = X_.transpose(1, 0, 2).reshape(-1, n_samples * self.spacing)
        outer_cov = X_.dot(X_.T) / self.spacing

        # Shrink the outer cov matrix
        scale = np.trace(outer_cov) / n_blocks
        P = self.template

        if solver == 'traditional':
            cov = X.T.dot(X)
            cov *= (1 - self.beta) * (1 - self.alpha)
            cov += np.kron((1 - self.beta) * self.alpha * outer_cov, np.eye(self.spacing))
            cov.flat[::n_features + 1] += self.beta * scale
            cov_i = np.linalg.pinv(cov)
            self.coef_ = cov_i.dot(P).ravel()

        elif solver == 'kernel':
            # Use matrix inversion lemma
            outer_cov_shrunk = outer_cov * (1 - self.beta) * self.alpha
            outer_cov_shrunk.flat[::n_blocks + 1] += self.beta * scale
            outer_cov_shrunk_i = np.linalg.pinv(outer_cov_shrunk)
            G = outer_cov_shrunk_i.dot(X.T.reshape(n_blocks, -1))
            G = G.reshape(n_features, n_samples)
            G *= (1 - self.alpha) * (1 - self.beta)
            K = X.dot(G)
            K.flat[::n_samples + 1] += 1
            K_i = np.linalg.pinv(K)
            GammaP = outer_cov_shrunk_i.dot(P.reshape(n_blocks, -1))
            GammaP = GammaP.reshape(n_features, -1)
            self.coef_ = GammaP - G.dot(K_i.dot(X.dot(GammaP)))

        self.coef_ = self.coef_.ravel()

        # Undo scaling if self.normalize == True
        self._set_intercept(X_offset, 0, X_scale)

        return self


class stLCMVCV(stLCMV):
    def __init__(self, spacing, fit_sigma=True, grand_average=None,
                 fit_intercept=True, copy_X=True,
                 normalize=True, verbose=True,
                 scoring='neg_mean_squared_error', init_alpha=0.5,
                 init_beta=0.5, init_sigma=10, init_rho=0.5):
        self.spacing = spacing
        self.fit_intercept = fit_intercept
        self.fit_sigma = fit_sigma
        self.normalize = normalize
        self.verbose = verbose
        self.scoring = scoring
        self.copy_X = copy_X
        self.grand_average = grand_average
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.init_sigma = init_sigma
        self.init_rho = init_rho

    #@profile
    def fit(self, X, y, sample_weight=None):
        """Fit the beamformer to the data.

        Parameters
        ----------
        X : 2D array (n_samples, n_features)
            The trials.
        y : 2D array (n_samples, n_targets)
            The training labels.
        """
        n_samples, n_features = X.shape
        if n_features % self.spacing != 0:
            raise ValueError(
                'Spacing parameter (%d) is incompatible with the number of '
                'features (%d)' % (self.spacing, n_features)
            )
        n_blocks = n_features // self.spacing

        # Ensure that y is a 2D array: n_samples x n_targets
        flat_y = y.ndim == 1
        if flat_y:
            y = np.atleast_2d(y).T
        n_targets = y.shape[1]

        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        def compute_outer_cov_inv(outer_cov, alpha, beta, scale):
            outer_cov_shrunk = outer_cov * alpha * (1 - beta)
            outer_cov_shrunk.flat[::n_blocks + 1] += beta * scale
            outer_cov_inv = np.linalg.pinv(outer_cov_shrunk)
            return outer_cov_inv

        def compute_GK(outer_cov_i, X, y, alpha, beta):
            G = outer_cov_i.dot(X.T.reshape(n_blocks, -1))
            G = G.reshape(n_features, n_samples)
            G *= (1 - alpha) * (1 - beta)
            K = X.dot(G)
            K.flat[::n_samples + 1] += 1
            return G, K

        def compute_dual_coef(K, y):
            Kinv = np.linalg.pinv(K)
            return Kinv.dot(y)

        def compute_GammaP(outer_cov_inv, P):
            GammaP = outer_cov_inv.dot(P.reshape(n_blocks, -1))
            GammaP = GammaP.reshape(n_features, -1)
            return GammaP

        # Compute outer cov matrix
        X_ = X.reshape(n_samples, -1, self.spacing)
        X_ = X_.transpose(1, 0, 2).reshape(-1, n_samples * self.spacing)
        outer_cov = X_.dot(X_.T) / self.spacing
        scale = np.trace(outer_cov) / n_blocks

        scorer = check_scoring(self, scoring=self.scoring, allow_none=False)

        # The scorer wants an object that will make the predictions but
        # they are already computed. This identity_estimator will just
        # return them.
        def identity_estimator():
            pass
        identity_estimator.decision_function = lambda y_predict: y_predict
        identity_estimator.predict = lambda y_predict: y_predict

        G_cache = dict()
        K_cache = dict()
        outer_cov_inv_cache = dict()
        kernel_cache = dict()
        Ps = [zscore(X[train], axis=0).T.dot(y[train] - y[train].mean(axis=0)) for train, _ in LeaveOneOut().split(X, y)]
        # Ps = [X[train].T.dot(y[train]) for train, _ in LeaveOneOut().split(X, y)]
        self.pattern = Ps[0]
        #return self
        #@profile
        def score(args):
            args = args.tolist()
            alpha = args.pop(0)
            beta = args.pop(0)

            rho, sigma = np.inf, np.inf
            if self.grand_average is not None:
                rho = args.pop(0)
            if self.fit_sigma:
                sigma = args.pop(0)

            if (alpha, beta) not in outer_cov_inv_cache:
                outer_cov_inv = compute_outer_cov_inv(outer_cov, alpha, beta, scale)
                G, K = compute_GK(outer_cov_inv, X, y, alpha, beta)
                outer_cov_inv_cache[(alpha, beta)] = outer_cov_inv
                G_cache[(alpha, beta)] = G
                K_cache[(alpha, beta)] = K
            else:
                outer_cov_inv = outer_cov_inv_cache[(alpha, beta)]
                G = G_cache[(alpha, beta)]
                K = K_cache[(alpha, beta)]

            if self.fit_sigma:
                if sigma not in kernel_cache:
                    kernel = utils.time_kernel(sigma, n_samples=self.spacing)
                    #kernel = utils.time_kernel(sigma, n_samples=n_blocks) 
                    kernel_cache[sigma] = kernel
                else:
                    kernel = kernel_cache[sigma]

            # Do efficient leave-one-out crossvalidation
            y_hat = np.zeros_like(y)
            G1 = None
            X1 = None
            y1 = None
            for K_i, test in zip(utils.loo_inv(K), range(n_samples)):
                if G1 is None or X1 is None:
                    G1 = G[:, 1:].copy()
                    X1 = X[1:].copy()
                    y1 = y[1:].copy()
                else:
                    if test >= 2:
                        G1[:, test - 2] = G[:, test - 1]
                        X1[test - 2] = X[test -1]
                        y1[test - 2] = y[test -1]
                    G1[:, test - 1] = G[:, 0]
                    X1[test - 1] = X[0]
                    y1[test - 1] = y[0]

                P = Ps[test].T
                if self.grand_average is not None:
                    P = (1 - rho) * P + rho * self.grand_average
                if self.fit_sigma:
                    P = P.reshape(n_blocks, self.spacing, n_targets)
                    P = P * kernel[np.newaxis, :, np.newaxis]
                    P = P.ravel()
                GammaP = compute_GammaP(outer_cov_inv, P)
                y_hat[test] = X[test].dot(GammaP - G1.dot(K_i.dot(X1.dot(GammaP))))
            score = scorer(identity_estimator, y.ravel(), y_hat.ravel())

            if self.verbose:
                print('alpha=%f, beta=%f, sigma=%f, rho=%f, score=%f' % (alpha, beta, sigma, rho, score))
            return -score

        x0 = [self.init_alpha, self.init_beta]
        bounds = [(0.0, 1.0), (0.1, 1.0)]
        if self.grand_average is not None:
            x0.append(self.init_rho)
            bounds.append((0.0, 1.0))
        if self.fit_sigma:
            x0.append(self.init_sigma)
            bounds.append((1.0, self.spacing))

        params = minimize(
            score,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options=dict(
                maxiter=10,
                eps=1E-3,
                ftol=1E-6,
            ),
        ).x.tolist()

        
        self.alpha_ = params.pop(0)
        self.beta_ = params.pop(0)

        P = X.T.dot(y).T

        if self.grand_average is not None:
            self.rho_ = params.pop(0)
            P = (1 - self.rho_) * P + self.rho_ * self.grand_average
        if self.fit_sigma:
            self.sigma_ = params.pop(0)
            P = utils.refine_pat(P, n_blocks, self.spacing, sigma=self.sigma_)

        self.pattern_ = P
        outer_cov_inv = compute_outer_cov_inv(outer_cov, self.alpha_, self.beta_, scale)
        G, K = compute_GK(outer_cov_inv, X, y, self.alpha_, self.beta_)
        GammaP = compute_GammaP(outer_cov_inv, P)
        self.coef_ = GammaP - G.dot(np.linalg.pinv(K).dot(X.dot(GammaP)))
        self.coef_ = self.coef_.T

        if flat_y:
            self.coef_ = self.coef_.ravel()

        self.normalized_coef_ = self.coef_.copy()

        self._set_intercept(X_offset, y_offset, X_scale)
        return self
