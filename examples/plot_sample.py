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

# Filter the data to extract only the left beeps
X = epochs.get_data().transpose(0, 2, 1).reshape(-1, epochs.info['nchan'])
filter_left = LCMV(template_left).fit(X)

X_left = epochs['left'].get_data().transpose(0, 2, 1).reshape(-1, epochs.info['nchan'])
X_right = epochs['right'].get_data().transpose(0, 2, 1).reshape(-1, epochs.info['nchan'])

X_left_filt = filter_left.predict(X_left)
X_right_filt = filter_left.predict(X_right)

plt.figure()
plt.plot(X_left_filt.reshape(len(epochs['left']), -1).mean(axis=0))
plt.plot(X_right_filt.reshape(len(epochs['right']), -1).mean(axis=0))
