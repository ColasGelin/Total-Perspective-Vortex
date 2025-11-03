import mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from mne.decoding import CSP
import matplotlib.pyplot as plt

print("=" * 60)
print("EEG MOTOR IMAGERY CLASSIFICATION PIPELINE")
print("CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis)")
print("=" * 60)

# STEP 1: LOAD AND PREPARE DATA

files = mne.datasets.eegbci.load_data(1, [3, 7, 11])  # Subject 1, multiple runs
raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])
print(f"✓ Loaded {len(files)} runs")

# STEP 2: PREPROCESSING
print("\n🔧 Preprocessing...")
# Bandpass filter for motor imagery (mu and beta bands)
raw.filter(l_freq=8.0, h_freq=30.0, method='fir')
print("✓ Bandpass filter applied: 8-30 Hz")

# Extract events
events, event_dict = mne.events_from_annotations(raw)
print(f"\n📊 Events found: {event_dict}")

# We want to classify T1 (left fist) vs T2 (right fist)
# Remove T0 (rest) events
event_id = {'T1': event_dict['T1'], 'T2': event_dict['T2']}

# We want to classify T0 (rest) vs T1 (left fist) vs T2 (right fist)
# event_id = {'T0': event_dict['T0'], 'T1': event_dict['T1'], 'T2': event_dict['T2']}
print(f"✓ Using events: {event_id}")

# STEP 3: CREATE EPOCHS (DATA CHUNKS)
print("\n✂️ Creating epochs (data chunks around events)...")
tmin, tmax = 0.0, 4.0  # 0-4 seconds after event (motor imagery period)
epochs = mne.Epochs(
    raw, 
    events, 
    event_id=event_id,
    tmin=tmin, 
    tmax=tmax,
    baseline=None,  # No baseline correction for CSP
    preload=True,
    reject=None  # You can add artifact rejection here if needed
)
print(f"✓ Created {len(epochs)} epochs")
print(f"  - T1 (left fist): {len(epochs['T1'])} epochs")
print(f"  - T2 (right fist): {len(epochs['T2'])} epochs")

# Get data in sklearn format
X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, 2]  # Labels
print(f"\n📐 Data shape: {X.shape}")
print(f"   (n_epochs={X.shape[0]}, n_channels={X.shape[1]}, n_times={X.shape[2]})")

# STEP 4: CREATE SKLEARN PIPELINE
print("\n🔨 Building sklearn pipeline...")
print("   Stage 1: CSP (Common Spatial Patterns) - dimensionality reduction")
print("   Stage 2: LDA (Linear Discriminant Analysis) - classification")

# Create the pipeline
pipeline = Pipeline([
    ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ('LDA', LinearDiscriminantAnalysis())
])

print("✓ Pipeline created!")
print("\nPipeline structure:")
print(pipeline)

# STEP 5: TRAIN/TEST SPLIT
print("\n📊 Splitting data into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} epochs")
print(f"✓ Test set: {X_test.shape[0]} epochs")

# STEP 6: TRAIN THE PIPELINE
print("\n🎓 Training the pipeline...")
pipeline.fit(X_train, y_train)
print("✓ Training complete!")

# STEP 7: EVALUATE
print("\n📈 EVALUATION")
print("=" * 60)

# Test accuracy
test_score = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_score * 100:.2f}%")

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"\nCross-Validation (5-fold):")
print(f"  Mean: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# STEP 8: SAVE THE MODEL
import joblib
joblib.dump(pipeline, 'eeg_model.pkl')
print("\n✅ Model saved to 'eeg_model.pkl'")
print("Ready for real-time prediction!")