"""
Some ways of modifying the pattern matrix, useful in EEG/MEG use cases.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from copy import deepcopy

import numpy as np


class PatternModifier(object):
    """Abstract base class for pattern modifier methods."""
    def update(self):
        return self.copy()

    def copy(self):
        return deepcopy(self)

    def __call__(self, pattern, X, y):
        raise NotImplementedError('This function must be implemented in a '
                                  'subclass')


class GaussianKernel(PatternModifier):
    """
    Multiplies the pattern with a Gaussian kernel in the time dimension.

    The data that is given to the model (X) is assumed to be a flattened
    version of a (channels x time) matrix. This modifier will reshape X and
    apply a Gaussian kernel across the time dimension.

    Parameters
    ----------
    n_samples : int
        Number of time samples in the data. This is used to reshape X.
    center : int | float
        Time sample on which the Gaussian kernel will be centered. Can be a
        value between two samples.
    width : int | float
        Width of the Gaussian kernel, measured in samples. Value does not have
        to be an integer number of samples.
    """
    def __init__(self, n_samples, center, width):
        self.n_samples = n_samples
        self.center = center
        self.width = width
        self.update(center, width)

    def update(self, center, width):
        time = np.arange(self.n_samples)
        self.kernel = np.exp(-0.5 * ((time - center) / width) ** 2)
        return self

    def __call__(self, pattern, X, y):
        mod_pattern = pattern.reshape(-1, self.n_samples)
        mod_pattern = mod_pattern * self.kernel[np.newaxis, :]
        return mod_pattern.reshape(pattern.shape)

    def __repr__(self):
        return 'GaussianKernel(n_samples={}, center={}, width={})'.format(
            self.n_samples, self.center, self.width)
