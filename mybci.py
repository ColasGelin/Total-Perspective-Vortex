import mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pca import PCA
from wavelet_transformer import WaveletTransformer
import argparse
mne.set_log_level('WARNING')

# Define 6 experiments
EXPERIMENTS = {
    0: {'runs': [3, 7, 11], 'events': ['T1', 'T2'], 'desc': 'Real left vs right fist'},
    1: {'runs': [5, 9, 13], 'events': ['T1', 'T2'], 'desc': 'Real both fists vs feet'},
    2: {'runs': [4, 8, 12], 'events': ['T1', 'T2'], 'desc': 'Imagined left vs right fist'},
    3: {'runs': [6, 10, 14], 'events': ['T1', 'T2'], 'desc': 'Imagined both fists vs feet'},
    4: {'runs': [4, 8, 12], 'events': ['T0', 'T1'], 'desc': 'Rest vs left fist'},
    5: {'runs': [5, 9, 13], 'events': ['T0', 'T2'], 'desc': 'Rest vs both feet'}
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description='EEG Motor Imagery Classification Pipeline')
parser.add_argument('--n_components', type=float, default=0.95, 
                    help='PCA n_components (float 0-1 for variance ratio, int for fixed count). Default: 0.99')
parser.add_argument('--freq_min', type=int, default=8, 
                    help='Minimum wavelet frequency (Hz). Default: 8')
parser.add_argument('--freq_max', type=int, default=30, 
                    help='Maximum wavelet frequency (Hz). Default: 30')
parser.add_argument('--freq_step', type=int, default=2, 
                    help='Wavelet frequency step (Hz). Default: 2')
parser.add_argument('--subjects', type=int, default=50,
                    help='Subject IDs to load. Default: 1 2 3 4 5')
args = parser.parse_args()

print("=" * 60)
print("EEG MOTOR IMAGERY CLASSIFICATION PIPELINE")
print("Wavelet + PCA + LDA")
print("=" * 60)
print(f"\n⚙️  Configuration:")
print(f"   PCA n_components: {args.n_components}")
print(f"   Wavelet frequencies: {args.freq_min}-{args.freq_max} Hz (step {args.freq_step})")
print(f"   Subjects: {args.subjects}")

# STEP 1: LOAD AND PREPARE DATA FROM MULTIPLE SUBJECTS
print("\n📥 Loading EEG motor imagery dataset...")
subjects = np.arange(1, args.subjects + 1)
all_X = []
all_y = []

subjects = np.arange(1, args.subjects + 1)
all_subject_scores = []
all_experiment_accuracies = []

for exp_num in range(6):
    exp_config = EXPERIMENTS[exp_num]
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT {exp_num}: {exp_config['desc']}")
    print(f"   Runs: {exp_config['runs']}, Events: {exp_config['events']}")
    print('='*60)
    
    experiment_subject_accuracies = []
    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"📂 Processing Subject {subject}")
        print('='*60)
        
        # Load THIS subject only
        try:
            files = mne.datasets.eegbci.load_data(subject, exp_config['runs'])
            raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])
        except Exception as e:
            print(f"⚠️  Skipping subject {subject}: {e}")
            continue
        
        # Preprocessing (same as before)
        motor_channels = ['C3..', 'Cz..', 'C4..', 'Fc3.', 'Fcz.', 'Fc4.', 'Cp3.', 'Cpz.', 'Cp4.']
        raw.pick(motor_channels)
        raw.set_eeg_reference('average', projection=True)
        raw.filter(l_freq=8.0, h_freq=30.0, method='fir')
        raw.notch_filter(freqs=60)  # ADD THIS
        
        events, event_dict = mne.events_from_annotations(raw)
        event_id = {event: event_dict[event] for event in exp_config['events'] if event in event_dict}
        
        if len(event_id) != 2:
            print(f"⚠️  Skipping subject {subject}: Not enough events found")
            continue
        
        epochs = mne.Epochs(raw, events, event_id=event_id, 
                            tmin=0.0, tmax=3.0, baseline=None, 
                            preload=True, reject=None)
        epochs.drop_bad()
        
        epochs = epochs[exp_config['events']]
        
        if len(epochs) < 10:  # Sanity check
            print(f"⚠️  Skipping subject {subject}: Too few epochs ({len(epochs)})")
            continue
        
        X = epochs.get_data()
        y = epochs.events[:, 2]
        
        print(f"✓ Subject {subject}: {len(epochs)} epochs")
        
        # Split THIS subject's data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build pipeline for THIS subject
        wavelet_freqs = np.arange(args.freq_min, args.freq_max + 1, args.freq_step)
        
        pipeline = Pipeline([
            ('wavelet', WaveletTransformer(frequencies=wavelet_freqs)),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('lda', LinearDiscriminantAnalysis(solver="lsqr"))
        ])
        
        # GridSearchCV for THIS subject
        from sklearn.model_selection import GridSearchCV, ShuffleSplit
        
        param_grid = {
            'pca__n_components': [2, 4, 6, 8, 10, 15, 20],
            'lda__shrinkage': ['auto', 0.1, 0.3]
        }
        
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, 
                        scoring='accuracy', n_jobs=-1, verbose=0)
        
        print(f"🔍 Training and optimizing for subject {subject}...")
        grid.fit(X_train, y_train)

        # Test on held-out data with playback simulation
        predictions = []
        for i in range(len(X_test)):
            single_epoch = X_test[i:i+1]  # One epoch at a time
            prediction = grid.predict(single_epoch)
            predictions.append(prediction[0])  # Store prediction

        # Calculate accuracy for this subject
        test_score = np.mean(np.array(predictions) == y_test)
        all_subject_scores.append(test_score)

        print(f"✅ Subject {subject} - Best params: {grid.best_params_}")
        print(f"✅ Subject {subject} - Test accuracy: {test_score*100:.2f}%")

# Final results
print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)
print(f"Mean accuracy across {len(subjects)} subjects: {np.mean(all_subject_scores)*100:.2f}%")
print(f"Std: +/- {np.std(all_subject_scores)*100:.2f}%")
print(f"Individual scores: {[f'{s*100:.1f}%' for s in all_subject_scores]}")

if np.mean(all_subject_scores) >= 0.60:
    print("\n✅ REQUIREMENT MET: Mean accuracy ≥ 60%")
else:
    print(f"\n❌ REQUIREMENT NOT MET: {np.mean(all_subject_scores)*100:.2f}% < 60%")