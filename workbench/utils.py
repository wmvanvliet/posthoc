import numpy as np

def gen_data(noise_scale=2, zero_mean=False, N=1000):
    """Generate some testing data.

    Parameters
    ----------
    noise_scale : float
        The amount of noise (in standard deviations) to add to the data.
    zero_mean : bool
        Whether X and y should be zero-mean (across samples) or not.
        Defaults to False.
    N : int
        Number of samples to generate. Defaults to 1000.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The measured data.
    Y : ndarray, shape (n_samples, n_targets)
        The latent variables generating the data.
    A : ndarray, shape (n_features, n_targets)
        The forward model, mapping the latent variables (=Y) to the measured
        data (=X).
    """
    # Fix random seed for consistent tests
    random = np.random.RandomState(42)

    M = 5  # Number of features

    # Y has 3 targets and the following covariance:
    cov_Y = np.array([
        [10, 1, 2],
        [1,  5, 1],
        [2,  1, 3],
    ]).astype(float)
    mean_Y = np.array([1, -3, 7])
    Y = random.multivariate_normal(mean_Y, cov_Y, size=N)
    Y -= Y.mean(axis=0)

    # The pattern (=forward model)
    A = np.array([
        [1, 10, -3],
        [4,  1,  8],
        [3, -2,  4],
        [1,  1,  1],
        [7,  6,  0],
    ]).astype(float)

    # The noise covariance matrix
    cov_noise = np.array([
        [1.25,  0.89,  1.06,  0.99,  1.27],
        [0.89,  1.10,  1.17,  1.08,  1.14],
        [1.06,  1.17,  1.32,  1.28,  1.36],
        [0.99,  1.08,  1.28,  1.37,  1.34],
        [1.27,  1.14,  1.36,  1.34,  1.60],
    ])
    mean_noise = np.zeros(M)
    noise = random.multivariate_normal(mean_noise, cov_noise, size=N)

    # Y = Y[:, :1]
    # A = A[:1, :1]
    # noise = noise[:, :1]

    # Construct X
    X = Y.dot(A.T)
    X += noise_scale * noise

    if zero_mean:
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)

    return X, Y, A
