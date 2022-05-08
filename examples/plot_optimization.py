"""
Automatic post-hoc optimization of linear models
=================================================

This example will demonstrate how to define custom modifications to a linear
model that introduce new hyperparameters. We will then use post-hoc's optimizer
to find the optimal values for these hyperparameters.

We will start with ordinary linear regression as a base model. Then, we will
modify the covariance matrix by applying shrinkage, modify the pattern with a
Gaussian kernel and modify the normalizer to be "unit noise gain", meaning the
weights all sum to 1.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
# Required imports
from matplotlib import pyplot as plt
from posthoc import Workbench, WorkbenchOptimizer, cov_estimators, normalizers
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import normalize
from functools import partial
import mne
import numpy as np
import warnings
warnings.simplefilter('ignore')

###############################################################################
# We will use some data from the original publication [1]_. A participant was
# silently reading word-pairs. In these pairs, the two words had a varying
# forward association strength between them. For example: ``locomotiv ->
# train`` has a high association strength, and ``dog -> submarine`` has not. In
# the case of word-pairs with high association strength, the brain will process
# second word is faster, since it has been semantically primed by the first
# word.
#
# We are going to deduce the memory priming effect from epochs of EEG data and
# use that to predict what the forward association strength was for a given
# word-pair.
#
# Let's first load the data and plot a contrast between word-pairs with a high
# versus low association strength, so we can observe how the memory priming
# effect manifests in the EEG data.
epochs = mne.read_epochs('subject04-epo.fif')
related = epochs['FAS > 0.2'].average()
related.comment = 'related'
unrelated = epochs['FAS < 0.2'].average()
unrelated.comment = 'unrelated'
mne.viz.plot_evoked_topo([related, unrelated])

###############################################################################
# Around 400ms after the presentation of the second word, there is a negative
# peak named the N400 potential. We can clearly observe the semantic priming
# effect as the N400 is more prominent in cases where the words have a low
# forward associative strength.
#
# A naive approach to deduce the forward association strength from a word pair
# is to take the average signal around 400ms at some sensors that show the N400
# well:
ROI = epochs.copy()
ROI.pick_channels(['P3', 'Pz', 'P4'])
ROI.crop(0.3, 0.47)
FAS_pred = ROI.get_data().mean(axis=(1, 2))

perf_naive, _ = pearsonr(epochs.metadata['FAS'], FAS_pred)
print(f'Performance: {perf_naive:.2f}')

###############################################################################
# Let's try ordinary linear regression next, using 10-fold cross-validation.
X = normalize(epochs.get_data().reshape(200, 32 * 60))
y = epochs.metadata['FAS'].values
ols = LinearRegression()
FAS_pred = cross_val_predict(ols, X, y, cv=10)
perf_ols, _ = pearsonr(epochs.metadata['FAS'], FAS_pred)
print(f'Performance: {perf_ols:.2f} (to beat: {perf_naive:.2f})')

###############################################################################
# Feeding all data into a linear regression model performs worse than taking
# the average signal in a well chosen sensors. That is because the model is
# overfitting. We could restrict the data going into the model to the same
# sensors and time window as we did when averaging the signal, but we can do so
# much better.
#
# Let's use the post-hoc framework to modify the linear regression model and
# incorporate some information about the nature of the data and the N400
# potential.
#
# First, let's try to reduce overfitting by applying some shrinkage to the
# covariance matrix. The data consists of 32 EEG electrodes, each recording 60
# samples of data. This causes a clear pattern to appear in the covariance
# matrix:
plt.figure()
plt.matshow(np.cov(X.T), cmap='magma')

###############################################################################
# The covariance matrix is build up from 32x32 squares, each square being
# 60x60. The ``KroneckerShrinkage`` class can make use of this information and
# apply different amounts of shrinkage to the diagonal of each square and the
# covariance matrix overall.
cov = cov_estimators.KroneckerKernel(outer_size=32, inner_size=60)

###############################################################################
# To determine the optimal amount of shrinkage to apply, we can wrap our linear
# regression model in the ``WorkbenchOptimizer`` class. By default, this uses
# heavily optimized leave-one-out cross-validation with a gradient descent
# algorithm to find the best values.

# We're optimizing for correlation between model prediction and true FAS
def scorer(model, X, y):
    return pearsonr(model.predict(X), y)[0]

# Construct the post-hoc workbench, tell it to modify the model by applying
# Kronecker shrinkage.
model = WorkbenchOptimizer(ols, cov=cov, scoring=scorer).fit(X, y)

shrinkage_params = model.cov_params_
print('Optimal shrinkage parameters:', shrinkage_params)

###############################################################################
# Let's inspect the pattern that the model has learned:
plt.figure()
plt.plot(epochs.times, model.pattern_.reshape(32, 60).T, color='black', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Signal (normalized units)')
plt.title('Pattern learned by the model using Kronecker shrinkage')

###############################################################################
# We can clearly see that the model is picking up on the N400. Let's fine-tune
# the pattern a bit by multiplying it with a Guassian kernel, centered around
# 400 ms.
def pattern_modifier(pattern, X, y, mean, std):
    """Multiply the pattern with a Gaussian kernel."""
    n_channels, n_samples = 32, 60
    kernel = norm(mean, std).pdf(np.arange(n_samples))
    kernel /= kernel.max()
    mod_pattern = pattern.reshape(n_channels, n_samples)
    mod_pattern = mod_pattern * kernel[np.newaxis, :]
    return mod_pattern.reshape(pattern.shape)

###############################################################################
# Now the optimizer has four hyperparameters to tune: two shrinkage values and
# two values dictating the shape of the Gaussian kernel.
model_opt = WorkbenchOptimizer(
    ols,
    cov=cov,
    pattern_modifier=pattern_modifier,
    pattern_param_x0=[30, 5],  # Initial guess for decent kernel shape
    pattern_param_bounds=[(0, 60), (2, None)],  # Boundaries for what values to try
    normalizer_modifier=normalizers.unit_gain,
    scoring=scorer,
).fit(X, y)
shrinkage_params = model_opt.cov_params_
pattern_params = model_opt.pattern_modifier_params_
print('Optimal shrinkage parameters:', shrinkage_params)
print('Optimal pattern parameters:', pattern_params)

###############################################################################
# We can now freeze the optimal parameters and evaluate the performance of our
# tuned model.
model = Workbench(
    ols,
    cov=cov_estimators.ShrinkageKernel(alpha=shrinkage_params[0]),
    pattern_modifier=partial(pattern_modifier, mean=pattern_params[0], std=pattern_params[1]),
    normalizer_modifier=normalizers.unit_gain,
)
FAS_pred = cross_val_predict(model, X, y, cv=10)
perf_opt, _ = pearsonr(epochs.metadata['FAS'], FAS_pred)
print(f'Performance: {perf_opt:.2f} (to beat: {perf_naive:.2f})')

###############################################################################
# Here is the final pattern:
model.fit(X, y)
plt.figure()
plt.plot(epochs.times, model.pattern_.reshape(32, 60).T, color='black', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Signal (normalized units)')
plt.title('Pattern learned by the post-hoc model')

###############################################################################
# References
# ----------
# .. [1] Marijn van Vliet and Riitta Salmelin (2020). Post-hoc modification
#        of linear models: combining machine learning with domain information
#        to make solid inferences from noisy data. Neuroimage, 204, 116221.
#        https://doi.org/10.1016/j.neuroimage.2019.116221
#
# sphinx_gallery_thumbnail_number = 5
