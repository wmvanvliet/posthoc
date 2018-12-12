"""
Optimizing hyperparameters using L-BFGS-B
=========================================

This example will demonstrate how a simple linear decoder can be enhanced
through post-hoc adaptation. This example contains the core ideas that are
presented in the main paper [1]_.

We will start with a logistic regressor as a base model. Then, we will modify
the covariance matrix by applying shrinkage, modify the pattern with a Gaussian
kernel and modify the normalizer to be "unit noise gain", meaning the weights
all sum to 1.

All parameters will be tuned using a general purpose convex minimizer
(L-BFGS-B) with an inner leave-one-out crossvalidation loop.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
# Required imports
import numpy as np
from scipy.stats import zscore, norm
import mne
from workbench import (Workbench, WorkbenchOptimizer, cov_estimators,
                       normalizers)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import log_logistic
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter('ignore')

###############################################################################
# We will use the MNE sample dataset. It is an MEG recording of a participant
# listening to auditory beeps and looking at visual stimuli. For this example,
# we attempt to discriminate between auditory beeps presented to the left
# versus the right of the head. The following code reads in the sample dataset.
path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(path + '/MEG/sample/sample_audvis_raw.fif',
                          preload=True)
events = mne.find_events(raw)
event_id = dict(left=1, right=2)
raw.pick_types(meg='grad')
raw.filter(None, 20)
raw, events = raw.resample(50, events=events)

# Create epochs
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                    baseline=(-0.2, 0), preload=True)
n_epochs, n_channels, n_samples = epochs.get_data().shape

###############################################################################
# The data is now loaded as an :class:`mne.Epochs` object. array. In order to
# use ``sklearn`` and ``workbench`` packages effectively, we need to shape this
# data into a (observations x features) matrix ``X`` and corresponding
# (observations x targets) ``y`` matrix.
X = epochs.get_data().reshape(len(epochs), -1)

# Normalize the data
X = zscore(X, axis=0)
#X *= 1E15

# Create training labels, based on the event codes during the experiment.
y = epochs.events[:, [2]]
y = y - 1.5
y *= 2
y = y.astype(int)

###############################################################################
# Now, we are ready to define a logistic regression model and apply it to the
# data. We split the data 50/50 into a training and test set. We present the
# training data to the model to learn from and test its performance on the test
# set.

# Split the data 50/50, but make sure the number of left/right epochs are
# balanced.
folds = StratifiedKFold(n_splits=2)
train_index, test_index = next(folds.split(X, y))
X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]

# The logistic regression model ignores observations that are close to the
# decision boundary. The parameter `C` controls how far away observations have
# to be in order to not be ignored. A setting of 25 means "quite far". We also
# specify the seed for the random number generator, so that this example
# replicates exactly every time.
base_model = LogisticRegression(C=25, solver='lbfgs', random_state=0,
                                fit_intercept=False)

# Train on the training data and predict the test data.
base_model.fit(X_train, y_train)
y_hat = base_model.predict(X_test)

# How many epochs did we decode correctly?
base_model_accuracy = accuracy_score(y_test, y_hat)
print('Base model accuracy:', base_model_accuracy)

###############################################################################
# To inspect the pattern that the model has learned, we wrap the model in a
# :class:`workbench.Workbench` object. After fitting, this object exposes the
# `.pattern_` attribute.
base_model = Workbench(base_model).fit(X_train, y_train)

# Plot the pattern
plt.figure()
plt.plot(epochs.times, base_model.pattern_.reshape(n_channels, n_samples).T,
         color='black', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Signal (normalized units)')
plt.title('Pattern learned by the base model')


###############################################################################
# Post-hoc adaptation can be used to improve the model somewhat.
#
# For starters, the template is quite noisy. The main distinctive feature
# between the conditions should be the auditory evoked potential around 0.05
# seconds.  Let's apply post-hoc adaptation to inform the model of this, by
# multiplying the pattern with a Gaussian kernel to restrict it to a specific
# time interval.

# The function that modifies the pattern takes as input the original pattern,
# the training data, and two parameters that define the center and width of the
# Gaussian kernel.
cache = dict()
def pattern_modifier(pattern, X, y, center, width):
    """Multiply the pattern with a Gaussian kernel."""
    mod_pattern = pattern.reshape(n_channels, n_samples)
    key = (center, width)
    if key in cache:
        kernel = cache[key]
    else:
        kernel = norm(center, width).pdf(np.arange(n_samples))
        kernel /= kernel.max()
        cache[key] = kernel
    mod_pattern = mod_pattern * kernel[np.newaxis, :]
    return mod_pattern.reshape(pattern.shape)


###############################################################################
# We will search for the optimal ``center`` and ``width`` parameters by using
# an optimization algorithm. In order to select the best parameter, we must
# define a scoring function. Let's use the same scoring function as the
# logistic regression model: logistic loss.
def logistic_loss_score(model, X, y):
    """Logistic loss scoring function."""
    y_hat = model.predict(X)
    #return np.sum(log_logistic(y * y_hat))
    return roc_auc_score(y, y_hat)


###############################################################################
# Now we can assemble the post-hoc model. The covariance matrix is computed
# using a shrinkage estimator. Since the number of features far exceeds the
# number of training observations, the kernel version of the estimator is much
# faster. We modify the pattern using the ``pattern_modifier`` function that we
# defined earlier, but modifying the pattern like this will affect the scaling
# of the output. To obtain a result with a consistant scaling, we modify the
# normalizer such that the modified pattern passes through our model with unit
# gain.

# Define initial values for the parameters. The optimization algorithm will
# use gradient descend using these as starting point.
initial_center = np.searchsorted(epochs.times, 0.05)
initial_width = 5

# Define the allowed range for the parameters. The optimizer will not exceed
# these.
center_bounds = (5, 25)
width_bounds = (1, 50)

# Define the post-hoc model using an optimizer to fine-tune the parameters.
optimized_model = WorkbenchOptimizer(
    base_model,
    cov=cov_estimators.ShrinkageKernel(1.0),
    cov_param_bounds=[(0.9, 1.0)],
    pattern_modifier=pattern_modifier,
    pattern_param_x0=[initial_center, initial_width],
    pattern_param_bounds=[center_bounds, width_bounds],
    normalizer_modifier=normalizers.unit_gain,
    scoring=logistic_loss_score,
    verbose=True,
    random_search=20,
).fit(X, y)

# Decode the test data
y_hat = optimized_model.predict(X_test).ravel()

# Assign the 'left' class to values above 0 and 'right' to values below 0
y_bin = np.zeros(len(y_hat), dtype=np.int)
y_bin[y_hat >= 0] = 1
y_bin[y_hat < 0] = -1

# How many epochs did we decode correctly?
optimized_model_accuracy = accuracy_score(y_test, y_bin)
print('Base model accuracy:', base_model_accuracy)
print('Optimized model accuracy:', optimized_model_accuracy)

###############################################################################
# The post-hoc model performs better. Let's visualize the optimized pattern.
# sphinx_gallery_thumbnail_number = 2
plt.figure()
plt.plot(epochs.times,
         optimized_model.pattern_.reshape(n_channels, n_samples).T,
         color='black', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Signal (normalized units)')
plt.title('Optimized pattern')

###############################################################################
# References
# ----------
# .. [1] Marijn van Vliet and Riitta Salmelin. Post-hoc modification of linear
#        models: combining machine learning with domain information to make
#        solid inferences from noisy data. In preparation.
