"""
Post-hoc adaptation of linear models
====================================

This example will demonstrate how a simple linear decoder can be enhanced
through post-hoc adaptation.

We will modify the covariance matrix with Kronecker-shrinkage, modify the
pattern with a Hanning window and modify the normalizer to be "unit noise
gain", meaning the weights all sum to 1.

All parameters will be tuned using a general purpose convex minimizer
(L-BFGS-B) with an inner leave-one-out crossvalidation loop.
"""
import numpy as np
from scipy.stats import zscore, norm, pearsonr
import mne
from workbench import WorkbenchOptimizer, cov_estimators, Workbench
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.utils.extmath import log_logistic
from matplotlib import pyplot as plt
from functools import partial

path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(path + '/MEG/sample/sample_audvis_raw.fif',
                          preload=True)
events = mne.find_events(raw)
event_id = dict(left=1, right=2)
raw.pick_types(meg='grad')
raw.filter(None, 20)
raw, events = raw.resample(50, events=events)

epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                    baseline=(-0.2, 0), preload=True)
n_epochs, n_channels, n_samples = epochs.get_data().shape

# Create scikit-learn compatible data matrices
X = epochs.get_data().reshape(len(epochs), -1)
y = epochs.events[:, 2]

# Normalize the data
X = zscore(X, axis=0)
y = y - 1.5
y *= 2
y = y.astype(int)

# Split the data in a train and test set
folds = StratifiedKFold(n_splits=2)
train_index, test_index = next(folds.split(X, y))
X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]

def pattern_modifier(pattern, X, y, center, width=10):
    pattern = pattern.reshape(n_channels, n_samples)
    kernel = norm(center, width).pdf(np.arange(n_samples))
    kernel /= kernel.max()
    pattern = pattern * kernel[np.newaxis, :]
    return pattern.ravel()

def normalizer_modifier(normalizer, X, y, pattern, coef):
    return 1 / coef.dot(pattern)

def scorer(model, X, y):
    # Logistic loss function
    y_hat = model.predict(X)
    perf = np.sum(log_logistic(y * y_hat))
    return perf

model = WorkbenchOptimizer(
    LogisticRegression(C=25, solver='lbfgs', max_iter=1000, random_state=0, fit_intercept=False),
    cov=cov_estimators.ShrinkageKernel(),
    pattern_modifier=pattern_modifier,
    pattern_param_x0=[np.searchsorted(epochs.times, 0.05), 5],
    pattern_param_bounds=[(5, 25), (1, 50)],
    normalizer_modifier=normalizer_modifier,
    scoring=scorer,
    verbose=True,
).fit(X_train, y_train)

# Decode the test data
y_hat = model.predict(X_test).ravel()

# Visualize the output of the model
y_left = y_hat[y_test == -1]
y_right = y_hat[y_test == 1]
lim = np.max(np.abs(y_hat))
plt.figure()
plt.scatter(np.arange(len(y_left)), y_left)
plt.scatter(np.arange(len(y_left), len(y_left) + len(y_right)), y_right)
plt.legend(['left', 'right'])
plt.axhline(0, color='444')
plt.ylim(-lim, lim)
plt.xlabel('Epochs')
plt.ylabel('Model output')

# Assign the 'left' class to values above 0 and 'right' to values below 0
y_bin = np.zeros(len(y_hat), dtype=np.int)
y_bin[y_hat >= 0] = 1
y_bin[y_hat < 0] = -1 

# How many epochs did we decode correctly?
print('Model accuracy:', accuracy_score(y_test, y_bin))
print('Model ROC:', roc_auc_score(y_test, y_hat))
