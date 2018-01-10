from __future__ import print_function
from scipy.stats import pearsonr
import pandas
import psychic
import mne
import progressbar
import numpy as np
import erp_beamformer
from scipy.linalg import toeplitz
from scipy.stats import norm

def scorer(model, X, y):
    return pearsonr(model.predict(X), y)[0]

def scorer_cl(model, X, y):
    y_hat = model.predict(X)
    hits = np.sum((y_hat > y.mean()) == (y > y.mean()))
    return hits / float(len(y))

good_subjects = {
    'dedeyne_levels': list(range(13)), #[0, 4, 6, 7, 8, 9],
    'television_commercials': list(range(21)), #[0, 1, 3, 5, 7, 9, 11],
    'manypairs': list(range(3)),
}

ch_names = [
    'Fp1',
    'AF3',
    'F7',
    'F3',
    'FC1',
    'FC5',
    'T7',
    'C3',
    'CP1',
    'CP5',
    'P7',
    'P3',
    'Pz',
    'PO3',
    'O1',
    'Oz',
    'O2',
    'PO4',
    'P4',
    'P8',
    'CP6',
    'CP2',
    'C4',
    'T8',
    'FC6',
    'FC2',
    'F4',
    'F8',
    'AF4',
    'Fp2',
    'Fz',
    'Cz',
]

times = (np.arange(50)/50. - 0.1).tolist()

def start_progress_bar(n):
    return progressbar.ProgressBar(
        maxval = n,
        widgets = [
            progressbar.widgets.Bar(),
            progressbar.widgets.SimpleProgress(sep='/'),
            '|',
            progressbar.widgets.ETA(),
        ],
        term_width=100,
    ).start()

def load_set(name='dedeyne_levels', subjects=None):
    """Load an EEG data set.

    Parameters
    ----------
    name : 'dedeyne_levels' | 'television_commercials' | 'manypairs'
        The dataset to load

    subjects : list of int | None
        If given, only load the data for these subjects. Defaults to None.

    Returns
    -------
    ds : list of psychic.DataSet
        The loaded EEG datasets
    labels : list of lists
        For each EEG dataset, the corresponding FAS values.
    """
    # This file contains semantic norms for many word pairs.
    # Also includes information such as length, frequency, etc.
    with pandas.HDFStore('datasets/assoCountMatricesNormedAnnotated.h5') as store:
        A_123 = store['/A_123']

    ds = []
    labels = []

    if name == 'dedeyne_levels':
        all_subjects = ['anne', 'cedric', 'daniel', 'dieter', 'heleen', 'koen', 'l', 'marijn', 'me', 'ph', 'tom', 'zander', 'zeger']
    elif name == 'television_commercials':
        #all_subjects = ['subject02', 'subject04', 'subject08', 'subject09', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15', 'subject18', 'subject19', 'subject20', 'subject21', 'subject22', 'subject23', 'subject24']
        all_subjects = ['subject02', 'subject04', 'subject05', 'subject06', 'subject07', 'subject08', 'subject09', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15', 'subject16', 'subject17', 'subject18', 'subject19', 'subject20', 'subject21', 'subject22', 'subject23', 'subject24']
    elif name == 'manypairs':
        all_subjects = ['do', 'bram', 'marijn']
    else:
        raise ValueError("name parameter must be one of ['dedeyne_levels', 'television_commercials', 'manypairs']")

    if subjects is None:
        subjects = all_subjects
    else:
        subjects = [all_subjects[subject] for subject in subjects]

    if len(subjects) > 1: pb = start_progress_bar(len(subjects))
    for subject in subjects:
        # Load evoked data
        if name == 'dedeyne_levels':
            #d = psychic.DataSet.load('datasets/dedeyne_levels/%s-dedeyne_levels_norm_delayed_button-trials.dat' % subject)
            #to_keep = find_outliers(d.X)
            #d = d[to_keep]
            epochs = mne.read_epochs('datasets/dedeyne_levels/%s-epo.fif' % subject)
            to_keep = ~np.array(['AUTOREJECT' in reason for reason in epochs.drop_log if 'IGNORED' not in reason])
            d = epochs_to_dataset(epochs, [2, 3, 4], ['unrelated', 'intermediate', 'related'])
        elif name == 'television_commercials':
            #d = psychic.DataSet.load('datasets/television_commercials/%s-television_commercials-trials.dat' % subject)
            epochs = mne.read_epochs('datasets/television_commercials/%s-epo.fif' % subject)
            to_keep = ~np.array(['AUTOREJECT' in reason for reason in epochs.drop_log if 'IGNORED' not in reason])
            d = epochs_to_dataset(epochs, [6, 7], ['related', 'unrelated'])
        elif name == 'manypairs':
            d = psychic.DataSet.load('datasets/manypairs/%s-manypairs-trials.dat' % subject)
            d = d.get_class([0, 1], drop_others=True)
            to_keep = find_outliers(d.X)
            d = d[to_keep]

        d = d.lix[:'Cz', -0.1:0.9, :]
        d = psychic.nodes.Resample(50.).train_apply(d)

        #bads = psychic.faster.bad_channels(psychic.concatenate_trials(d), use_metrics=['variance', 'correlation'])
        #if len(bads) > 0:
        #    d = psychic.faster.interpolate_channels(d, bads)[0]

        # Load FAS values
        if name == 'dedeyne_levels':
            df = pandas.read_table('datasets/dedeyne_levels/%s-dedeyne_levels_norm_delayed_button-02.log' % subject, header=0, skiprows=4, index_col=[1, 2], encoding='utf8')
            df = df[df['label'] > 1]
            y = np.log(1 + A_123.reindex(df.index).fillna(0).values)
        elif name == 'television_commercials':
            df = pandas.read_table('datasets/television_commercials/%s.log' % subject, header=0, skiprows=18, index_col=[1, 2], encoding='utf8')
            df = df[df['label'] > 5]
            y = np.log(1 + A_123.reindex(df.index).fillna(0).values)
        elif name == 'manypairs':
            df = pandas.read_table('datasets/manypairs/%s-manypairs-02.log' % subject, header=0, skiprows=4, index_col=[1, 2], encoding='utf8')
            df = df[df['label'] < 3]
            y = np.log(1 + A_123.reindex(df.index).fillna(0).values)
        y = y[to_keep]

        ds.append(d)
        labels.append(y)
        #ds.append(d)
        #labels.append(y)

        if len(subjects) > 1: pb.update(pb.value + 1)
    if len(subjects) > 1: pb.finish()
    return ds, labels

def epochs_to_dataset(epochs, codes, cl_lab):
    labels = np.array([epochs.events[:, 2] == c for c in codes])
    return psychic.DataSet(data=epochs.get_data().transpose(1, 2, 0), 
                           labels=labels,
                           feat_lab=[epochs.ch_names, epochs.times.tolist()],
                           cl_lab=cl_lab)

def find_outliers(X):
    return np.ones(len(X), dtype=np.bool)
    scores = np.abs(np.mean(X.reshape(X.shape[0], -1), axis=1))
    to_keep = np.flatnonzero(scores < 5 * np.std(scores))
    return to_keep

def remove_outliers(X, y):
    scores = np.abs(np.mean(X, axis=1))
    to_keep = np.flatnonzero(scores < 5 * np.std(scores))
    #X = X[to_keep]
    #y = y[to_keep]
    return X, y

def time_kernel(sigma, n_samples=50, mu=25):
    if sigma == 'none':
        return np.ones(n_samples)
    else:
        kernel = norm(mu, sigma).pdf(np.arange(n_samples))
        kernel /= kernel.max()
        return kernel

def refine_pat(spat_temp_pat, n_channels, n_samples, sigma=15):
    if sigma == 'none':
        return spat_temp_pat

    spat_temp_pat = spat_temp_pat.reshape(n_channels, n_samples)
    return (spat_temp_pat * time_kernel(sigma, mu=25, n_samples=n_samples)[None, :]).ravel()

def fast_inv(cov):
    """Computes the inverse of the given matrix on the GPU."""
    try:
        if len(cov) > 5000:
            raise ValueError('Covariance matrix too big for GPU memory')

        # Fast, GPU implementation
        import torch
        if type(cov) != torch.LongTensor:
            cov_ = torch.from_numpy(cov)
        cov_ = cov_.cuda()
        inv_ = torch.inverse(cov_)
        return inv_.cpu().numpy()
    except Exception as e:
        print(e)
        # Slow, CPU implementation
        return np.linalg.pinv(cov)

def fast_cov(X, assume_centered=False, return_inv=False):
    """Computes the covariance matrix on the GPU.
    
    Parameters
    ----------
    X : 2D-array (n_items, n_features)
        The items to compute the covariance for.
    assume_centered : bool
        Whether to assume the features are zero-centered. Defaults to False. 
    return_inv : bool
        Whether to also return the inverse of the covariance matrix. Defaults
        to False.
    """
    try:
        if len(X) > 5000:
            raise ValueError('Covariance matrix too big for GPU memory')

        # Fast GPU implementation
        import torch
        if type(X) != torch.LongTensor:
            X_ = torch.from_numpy(X)
        X_ = X_.cuda()
        if not assume_centered:
            X_ = X_.sub((X_.sum(0) / len(X_)).expand_as(X_))
        cov_ = torch.mm(X_.transpose(1, 0), X_)
        if return_inv:
            cov_.cpu().numpy(), fast_inv(cov_)
        else:
            return cov_.cpu().numpy()
    except:
        # Slow CPU implementation
        if not assume_centered:
            X = X - X.mean(axis=0, keepdims=True)
        cov = X.T.dot(X)
        if return_inv:
            return cov, np.linalg.pinv(cov)
        else:
            return cov

def block_view(A, block_shape):
    """Provide a 2D block view to 2D array.

    No error checking is performed, so this is only meaningful for blocks
    strictly compatible with the shape of A.

    Parameters
    ----------
    A : 2D array (height, width)
        The matrix to expose in a blocked view
    block_shape : tuple (block_height, block_width)
        Shape of the blocks

    Returns
    -------
    view : 4D array (height / block_height, width / block_width, block_height, block_width)  # noqa
        A 2D view where each cell contains a 2D block.

    Examples
    --------
    >>> import numpy as np
    ... a = np.arange(16).reshape(4, 4)
    ... blocks = block_view(a, (2, 2))
    ... blocks[0, 0]
    array([[0, 1],
           [4, 5]])

    Notes
    -----
    Implementation adepted from:
    http://stackoverflow.com/questions/5073767
    """
    # Mind the tuple additions on this code.
    height, width = A.shape
    block_height, block_width = block_shape
    shape = (height / block_height, width / block_width) + block_shape
    strides = (block_height * A.strides[0], block_width * A.strides[1]) + A.strides
    return np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)

def optimized_mu(X, spacing):
    """Efficiently obtain the sub-matrix traces."""
    n_samples, n_features = X.shape
    X = X - X.mean(axis=0)
    X = X.reshape(n_samples, -1, spacing)
    X = X.transpose(1, 0, 2).reshape(-1, n_samples * spacing)

    mu = X.dot(X.T) / (spacing * n_samples)
    #mu = fast_cov(X, assume_centered=True)
    return mu

def toeplitz_shrink(n_channels, n_samples):
    """Performs shrinkage not only towards the diagonal, but also towards a
    Toeplitz matrix."""

    # Compute Toeplitz matrix
    first_row = np.zeros(n_channels * n_samples)
    first_row[::n_samples] = 1

    def shrink_func(cov, alpha):
        spacing = n_samples
        n_features = len(cov)
        assert n_features % spacing == 0
        n_blocks = n_features // spacing

        if type(alpha) == tuple:
            alpha, beta = alpha
        else:
            alpha, beta = alpha, alpha

        cov = cov.copy()

        if beta < 1E10:
            # Efficiently obtain the sub-matrix diagonals
            cov_blocks = block_view(cov, (spacing, spacing))
            diagonals = cov_blocks.reshape(n_blocks, n_blocks, -1)[:, :, ::spacing + 1]

            # Shrink towards the diagonal of each sub-matrix
            mu = np.mean(diagonals, axis=2)
            cov *= 1 - alpha
            I = np.eye(spacing)
            for i in range(n_blocks):
                for j in range(n_blocks):
                    cov_blocks[i, j] += I * mu[i, j] * alpha

        # Shrink towards the diagonal of the whole matrix
        mu = np.trace(cov) / n_features
        cov *= 1 - beta
        cov.flat[::n_features + 1] += beta * mu

        cov_i = fast_inv(cov)
        return cov, cov_i
    return shrink_func


def shrink(cov, alpha):
    """Performs shrinkage towards the diagonal"""
    n_features = len(cov)
    mu = np.trace(cov) / n_features
    cov_shrunk = (1 - alpha) * cov
    cov_shrunk.flat[::n_features + 1] += mu * alpha
    cov_shrunk_i = fast_inv(cov_shrunk)
    return cov_shrunk, cov_shrunk_i

def pattern_from_model(model, spat_temp_cov=None):
    if spat_temp_cov is None:
        spat_temp_cov = model.cov_
    spat_temp_pat = model.normalized_coef_.dot(spat_temp_cov).ravel()
    spat_temp_pat /= np.std(spat_temp_pat)
    return spat_temp_pat

def visualize_pattern(pat, vspace=None):
    from matplotlib import pyplot as plt
    pat = pat.reshape(32, 50, 1)
    psychic.plot_erp(psychic.DataSet(pat, feat_lab=[ch_names, times]), vspace=vspace)
    plt.axvline(0.4, color='k', alpha=0.2)

def normalize(X):
    """Normalize the features of X"""
    X = X - np.mean(X, axis=0, keepdims=True)
    scale = np.std(X, axis=0, keepdims=True)
    scale[scale == 0] = 1
    X /= scale
    return X

def update_inv(X, X_inv, i, v):
    """Computes Y^-1 from X^-1 and a row/column update vector.

    X and Y need to be symmetrical matrices (e.g. covariance matrices).
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
    Y_inv = X_inv - X_inv_U.dot(np.linalg.pinv(C + V_X_inv.dot(U))).dot(V_X_inv)

    return Y_inv

def loo_inv(cov):
    """Compute leave-one-out crossvalidation iterations from a covariance matrix."""
    cov1 = None
    cov1_inv = None
    for test in range(len(cov)):
        if cov1 is None or cov1_inv is None:
            cov1 = cov[1:, :][:, 1:]
            cov1_inv = np.linalg.pinv(cov1)
            yield cov1_inv
        else:
            j = np.arange(1, len(cov))
            j[test - 1] = 0
            v = cov[0, j]
            yield update_inv(cov1, cov1_inv, test - 1, v)
