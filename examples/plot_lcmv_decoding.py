"""
Decoding with a spatio-temporal LCMV beamformer
===============================================

This example will demonstrate a simple decoder based on an LCMV beamformer
applied to the MNE-Sample dataset. This dataset contains MEG recordings of a
subject being presented with audio beeps on either the left or right side of
the head.

We will use some of the epochs to construct a template of the difference
between the 'left' and 'right' reponses. Then, a spatio-temporal LCMV
beamformer is created based on this template and applied to the rest of the
epochs.

Some shrinkage of the covariance matrix is applied in order to create a stable
filter, given the limited amount of training data.

As we will see, the difference between the left and right auditory beeps is
quite large and can be reliably decoded.
"""
import numpy as np
import mne
from workbench import LCMV, ShrinkageUpdater
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

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

# Create scikit-learn compatible data matrices
X = epochs.get_data().reshape(len(epochs), -1)
y = epochs.events[:, 2]

# Split the data in a train and test set
folds = StratifiedKFold(n_splits=2)
train_index, test_index = next(folds.split(X, y))
X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]

# Create a template for the beamformer. For this, we compute the "evoked
# potential" for the left and right beeps. The contrast between these
# conditions will serve as our template.
evoked_left = X_train[y_train == 1].mean(axis=0)
evoked_right = X_train[y_train == 2].mean(axis=0)
template = evoked_left - evoked_right

# This creates a (channels x time) view of the template
template_ch_time = template.reshape(epochs.info['nchan'], -1)

# Plot the template
plt.figure()
plt.plot(epochs.times, template_ch_time.T, color='black', alpha=0.2)
plt.xlabel('Time (s)')
plt.title('Original template')

# The template is quite noisy. The main distinctive feature between the
# conditions should be the auditory evoked potential around 0.05 seconds.
# Let's create a Hanning window to limit our template to just the evoked
# potential.
center = np.searchsorted(epochs.times, 0.05)
width = 10
window = np.zeros(len(epochs.times))
window[center - width // 2: center + width // 2] = np.hanning(width)
template_ch_time *= window[np.newaxis, :]

# Plot the refined template
plt.figure()
plt.plot(epochs.times, template_ch_time.T, color='black', alpha=0.2)
plt.xlabel('Time (s)')
plt.title('Refined template')

# Make an LCMV beamformer based on the template. We apply a bit of
# regularization to the covariance matrix.
reg = np.linalg.norm(X_train) * 0.1
beamformer = LCMV(template, cov_updater=ShrinkageUpdater(reg)).fit(X_train)

# Decode the test data
y_hat = beamformer.predict(X_test).ravel()

# Visualize the output of the LCMV beamformer
y_left = y_hat[y_test == 1]
y_right = y_hat[y_test == 2]
lim = np.max(np.abs(y_hat))
plt.figure()
plt.scatter(np.arange(len(y_left)), y_left)
plt.scatter(np.arange(len(y_left), len(y_left) + len(y_right)), y_right)
plt.legend(['left', 'right'])
plt.axhline(0, color='444')
plt.ylim(-lim, lim)
plt.xlabel('Epochs')
plt.ylabel('Beamformer output')

# Assign the 'left' class to values above 0 and 'right' to values below 0
y_bin = np.zeros(len(y_hat), dtype=np.int)
y_bin[y_hat >= 0] = 1
y_bin[y_hat < 0] = 2

# How many epochs did we decode correctly?
print('LCMV accuracy:', metrics.accuracy_score(y_test, y_bin))
