import mne
import matplotlib.pyplot as plt

# AUTOMATIC DOWNLOAD - No need to manually download files!
print("📥 Downloading data from PhysioNet...")
print("   (First time will download, then it's cached locally)")

# Correct syntax: first argument is subject, second is runs
files = mne.datasets.eegbci.load_data(1, [5])  # Subject 1, Run 5

# Load the downloaded file
print(f"\n✓ Data downloaded to: {files[0]}")
raw = mne.io.read_raw_edf(files[0], preload=True)

print("=" * 50)
print("FILE INFORMATION")
print("=" * 50)
print(f"Channels: {raw.info['nchan']}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")

print("\n" + "=" * 50)
print("EVENTS/ANNOTATIONS")
print("=" * 50)
events, event_dict = mne.events_from_annotations(raw)
print(f"Event types: {event_dict}")
for event_name, event_id in event_dict.items():
    count = len(events[events[:, 2] == event_id])
    print(f"  {event_name}: {count} times")

# Find motor channels
motor_channels = [ch for ch in raw.ch_names if 'C3' in ch or 'Cz' in ch or 'C4' in ch]
print(f"\nMotor channels: {motor_channels}")

# Show PSD BEFORE filtering
print("\n📊 BEFORE filtering - showing frequency content...")
fig_before = raw.plot_psd(picks=motor_channels, fmax=50)
fig_before.suptitle('BEFORE Filtering - Motor Cortex (0-50 Hz)')

# Apply bandpass filter (8-30 Hz)
print("\n🔧 Applying bandpass filter: 8-30 Hz...")
raw_filtered = raw.copy()
raw_filtered.filter(l_freq=8.0, h_freq=30.0, method='fir')
print("✓ Filtering complete!")

# Show PSD AFTER filtering
print("\n📊 AFTER filtering - showing frequency content...")
fig_after = raw_filtered.plot_psd(picks=motor_channels, fmax=50)
fig_after.suptitle('AFTER Filtering - Motor Cortex (only 8-30 Hz remains)')

plt.show()
