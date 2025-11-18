from pathlib import Path
import mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
from pca import PCA
from wavelet_transformer import WaveletTransformer
import matplotlib.pyplot as plt
import warnings
import time

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*annotation.*expanding outside.*')

MOTOR_CHANNELS = [
    'C3..','Cz..','C4..','Fc3.','Fcz.','Fc4.','Cp3.','Cpz.','Cp4.'
]

WAVELET_FREQS = np.arange(8, 31, 2)

EXPERIMENTS = {
    0: {'runs': [3, 7, 11], 'events': ['T1', 'T2'], 'desc': 'Real left vs right fist'},
    1: {'runs': [5, 9, 13], 'events': ['T1', 'T2'], 'desc': 'Real both fists vs feet'},
    2: {'runs': [4, 8, 12], 'events': ['T1', 'T2'], 'desc': 'Imagined left vs right fist'},
    3: {'runs': [6, 10, 14], 'events': ['T1', 'T2'], 'desc': 'Imagined both fists vs feet'},
    4: {'runs': [4, 8, 12], 'events': ['T0', 'T1'], 'desc': 'Rest vs left fist'},
    5: {'runs': [5, 9, 13], 'events': ['T0', 'T2'], 'desc': 'Rest vs both feet'}
}

DATA_BASE_PATH = Path("./files/")

def load_raw(subject, runs):
    subject_dir = DATA_BASE_PATH / f"S{subject:03d}"
    
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    
    raw_list = []
    for run in runs:
        edf_file = subject_dir / f"S{subject:03d}R{run:02d}.edf"
        
        if not edf_file.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_file}")
        
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        raw_list.append(raw)
    
    return mne.concatenate_raws(raw_list)


def preprocess_raw(raw):
    """Common preprocessing pipeline for all modes."""
    raw.pick(MOTOR_CHANNELS)
    raw.set_eeg_reference('average', projection=True)
    raw.filter(l_freq=8.0, h_freq=30.0, method='fir')
    raw.notch_filter(freqs=60)
    return raw


def epoch_data(raw, event_id, tmin=0.0, tmax=3.0):
    """Create epochs from raw data."""
    events, raw_event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, reject=None
    )
    epochs.drop_bad()
    return epochs


def build_pipeline():
    """Build the feature-extraction and classification pipeline."""
    return Pipeline([
        ('wavelet', WaveletTransformer(frequencies=WAVELET_FREQS)),
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('lda', LinearDiscriminantAnalysis(solver="lsqr"))
    ])

def train_mode(args):
    if args.subject is None or args.run is None:
        print("❌ train mode requires <subject> <run>")
        exit(1)

    # Load + preprocess
    raw = preprocess_raw(load_raw(args.subject, [args.run]))

    # Epoch
    _, event_dict = mne.events_from_annotations(raw)
    epochs = epoch_data(raw, event_dict)
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Split train/test BEFORE any training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train CV on TRAINING SET ONLY
    pipeline = build_pipeline()
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

    print("Cross-validation scores (on training set):")
    print(scores)
    print(f"Cross-val mean: {np.mean(scores):.4f}")

    # Fit on TRAINING SET ONLY
    pipeline.fit(X_train, y_train)

    # Save model + test indices
    fname = f"model_s{args.subject}_r{args.run}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump({
            'pipeline': pipeline,
            'subject': args.subject,
            'run': args.run,
            'event_dict': event_dict,
            'cv_scores': scores,
            'mean_cv_score': np.mean(scores),
            'X_test': X_test,
            'y_test': y_test
        }, f)

    print(f"✔ Model saved to {fname}")
    
def predict_mode(args):
    if args.subject is None or args.run is None:
        print("❌ predict mode requires <subject> <run>")
        exit(1)

    # Load trained model
    fname = f"model_s{args.subject}_r{args.run}.pkl"
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Model file not found: {fname}")
        exit(1)

    pipeline = data['pipeline']
    X_test = data['X_test']
    y_test = data['y_test']

    # Playback simulation: process one epoch at a time
    print("epoch nb: [prediction] [truth] equal?")
    predictions = []
    processing_times = []
    
    for i in range(len(X_test)):
        start = time.time()
        pred = pipeline.predict(X_test[i:i+1])[0]
        proc_time = time.time() - start
        time.sleep(1.0)
        processing_times.append(proc_time)
        predictions.append(pred)
        
        print(f"epoch {i:02d}: [{pred}] [{y_test[i]}] {pred == y_test[i]}")
    
    accuracy = np.mean(np.array(predictions) == y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Max processing time: {max(processing_times)*1000:.1f}ms")
    
def full_evaluation_mode(args):
    subjects = np.arange(1, args.subject + 1)
    all_scores = []

    for exp_index, exp_cfg in EXPERIMENTS.items():
        experiment_scores = []
        print("\n\nExperiment {}: {}".format(exp_index + 1, exp_cfg['desc']))
        
        for subject in subjects:
            print(f"Experiment {exp_index + 1} | Subject {subject}... ", end='', flush=True)

            # Load
            try:
                raw = load_raw(subject, exp_cfg['runs'])
            except:
                print("Skipping (load error)")
                continue

            raw = preprocess_raw(raw)

            # Epoch
            events, event_dict = mne.events_from_annotations(raw)
            event_id = {e: event_dict[e] for e in exp_cfg['events'] if e in event_dict}
            if len(event_id) != 2:
                print("Skipping (missing events)")
                continue

            epochs = epoch_data(raw, event_id)
            epochs = epochs[exp_cfg['events']]

            if len(epochs) < 10:
                print("Skipping (too few epochs)")
                continue

            X = epochs.get_data()
            y = epochs.events[:, 2]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Build pipeline
            pipeline = build_pipeline()

            pipeline.fit(X_train, y_train)

            preds = [pipeline.predict(X_test[i:i+1])[0] for i in range(len(X_test))]
            acc = np.mean(preds == y_test)
            print(f"✔ {acc*100:.2f}%")
            experiment_scores.append(acc)

        print(f"Experiment {exp_index} Accuracy: {np.mean(experiment_scores)*100:.2f}%")
        all_scores.append(np.mean(experiment_scores)*100)

    return all_scores

def visualize_mode(args):
    # Load and filter
    raw_original = load_raw(args.subject, [args.run])
    raw_original.pick(MOTOR_CHANNELS)
    raw_filtered = preprocess_raw(load_raw(args.subject, [args.run]))
    
    # Figure 1: Filtered EEG
    raw_filtered.plot(duration=10, n_channels=len(MOTOR_CHANNELS),
                      scalings='auto', title='Filtered EEG Data (8-30 Hz)')
    
    # Figure 2: PSD comparison
    plt.figure(figsize=(14, 6))
    psds_raw, freqs = raw_original.compute_psd(fmax=50).get_data(return_freqs=True)
    psds_filtered, _ = raw_filtered.compute_psd(fmax=50).get_data(return_freqs=True)
    
    plt.semilogy(freqs, psds_raw.mean(axis=0), label='Raw', alpha=0.7, linewidth=2)
    plt.semilogy(freqs, psds_filtered.mean(axis=0), label='Filtered (8-30 Hz)', alpha=0.7, linewidth=2)
    plt.axvspan(8, 30, alpha=0.2, color='green', label='Kept frequencies (8-30 Hz)')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power Spectral Density (V²/Hz)', fontsize=12)
    plt.title('Power Spectral Density - Raw vs Filtered', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 50])
    plt.tight_layout()
    plt.show()

start_time = time.time()
parser = argparse.ArgumentParser(description='EEG Motor Imagery Classification Pipeline')
parser.add_argument('subject', type=int, nargs='?', default=109)
parser.add_argument('run', type=int, nargs='?', default=None)
parser.add_argument('mode', type=str, nargs='?', default='eval',
                    choices=['train', 'predict', 'eval', 'visualize'])
parser.add_argument('-b', action='store_true', dest='b_flag', 
                    help='Set b_flag variable to True')
args = parser.parse_args()

if args.mode == 'train':
    train_mode(args)
elif args.mode == 'predict':
    predict_mode(args)
elif args.mode == 'visualize':
    visualize_mode(args)
else:
    all_experiment_scores = full_evaluation_mode(args)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"All Experiment Accuracies: {[f'{score:.2f}%' for score in all_experiment_scores]}")
    print(f"Mean Accuracy: {np.mean(all_experiment_scores):.2f}%")
    print(f"Std: {np.std(all_experiment_scores)*100:.2f}%")
    print(f"Time of Execution: {time.time() - start_time:.2f} seconds")
