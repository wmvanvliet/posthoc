import mne
from workbench import LCMV
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

left = epochs['left'].average()
right = epochs['right'].average()
contrast = mne.combine_evoked([left, right], [1, -1])
evokeds = [left, right, contrast]

# Create template
template_left = left.copy().crop(0.05, 0.15).data.mean(axis=1)
template_right = right.copy().crop(0.05, 0.15).data.mean(axis=1)
template_contrast = template_left - template_right


def make_X(epochs):
    X = epochs.get_data().transpose(0, 2, 1).reshape(-1, epochs.info['nchan'])
    return X


X = make_X(epochs)
X_left = make_X(epochs['left'])
X_right = make_X(epochs['right'])

filter_left = LCMV(template_left).fit(X)
filter_right = LCMV(template_right).fit(X)
filter_contrast = LCMV(template_contrast).fit(X)

X_left_left = filter_left.predict(X_left)
X_left_right = filter_left.predict(X_right)
X_right_left = filter_right.predict(X_left)
X_right_right = filter_right.predict(X_right)
X_contrast_left = filter_contrast.predict(X_left)
X_contrast_right = filter_contrast.predict(X_right)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].plot(X_left_left.reshape(len(epochs['left']), -1).mean(axis=0))
axes[0].plot(X_left_right.reshape(len(epochs['right']), -1).mean(axis=0))
axes[0].set_title('Filtered for left')

axes[1].plot(X_right_left.reshape(len(epochs['left']), -1).mean(axis=0))
axes[1].plot(X_right_right.reshape(len(epochs['right']), -1).mean(axis=0))
axes[1].set_title('Filtered for right')

axes[2].plot(X_contrast_left.reshape(len(epochs['left']), -1).mean(axis=0))
axes[2].plot(X_contrast_right.reshape(len(epochs['right']), -1).mean(axis=0))
axes[2].set_title('Filtered for contrast')
