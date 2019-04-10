# encoding: utf-8
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
