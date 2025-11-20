from pathlib import Path
import mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
from pca import PCA
from wavelet_transformer import WaveletTransformer
import matplotlib.pyplot as plt
import warnings
import time
from lda import LDA

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*annotation.*expanding outside.*')
warnings.filterwarnings("ignore", message="Channel names are not unique")

MOTOR_CHANNELS = [
    'C3..','Cz..','C4..','Fc3.','Fcz.','Fc4.','Cp3.','Cpz.','Cp4.'
]

BCIC_CHANNELS = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8',
    'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-Pz',
    'EEG-14', 'EEG-15', 'EEG-16'
] 

BCIC_EVENT_MAP = {
    'left_right': {'769': 'left', '770': 'right'},  # classes 1,2
    'foot_tongue': {'771': 'foot', '772': 'tongue'}  # classes 3,4
}

WAVELET_FREQS = np.arange(8, 31, 2)

EXPERIMENTS = {
    0: {'runs': [3, 7, 11], 'events': ['T1', 'T2'], 'desc': 'Real left vs right fist'},
    1: {'runs': [5, 9, 13], 'events': ['T1', 'T2'], 'desc': 'Real both fists vs feet'},
    2: {'runs': [4, 8, 12], 'events': ['T1', 'T2'], 'desc': 'Imagined left vs right fist'},
    3: {'runs': [6, 10, 14], 'events': ['T1', 'T2'], 'desc': 'Imagined both fists vs feet'},
    4: {'runs': [4, 8, 12], 'events': ['T0', 'T1'], 'desc': 'Rest vs left fist'},
    5: {'runs': [5, 9, 13], 'events': ['T0', 'T2'], 'desc': 'Rest vs both feet'}
}

# Add this new constant


DATA_BASE_PATH = Path("./files/")

def load_raw_bcic(subject, session='T'):
    """Load BCI Competition IV 2a dataset (GDF format)."""
    bcic_path = Path("./bcic_files/")
    gdf_file = bcic_path / f"A{subject:02d}{session}.gdf"
    
    if not gdf_file.exists():
        raise FileNotFoundError(f"GDF file not found: {gdf_file}")
    raw = mne.io.read_raw_gdf(gdf_file, preload=True)
    
    # Debug: check channels and events
    events, event_dict = mne.events_from_annotations(raw)
    return raw

def normalize_bcic(raw):
    events, ann_dict = mne.events_from_annotations(raw)
    
    # Detect whether BCIC labels exist
    if not any(k in ann_dict for k in ["769", "770", "771", "772"]):
        print("⚠ normalize_bcic(): No BCIC labels found → skipping BCIC mapping.")
        return events, {}

    # BCIC labels (string → int)
    bcic_map = {
        "769": 1,  # left
        "770": 2,  # right
        "771": 3,  # foot
        "772": 4,  # tongue
    }

    # Apply mapping to events array
    new_events = events.copy()
    for old, new in bcic_map.items():
        if old in ann_dict:
            old_code = ann_dict[old]
            new_events[new_events[:, 2] == old_code, 2] = new

    # Build PhysioNet-like event_id dict
    event_id = {
        "left":   1,
        "right":  2,
        "foot":   3,
        "tongue": 4,
    }

    return new_events, event_id


def test_bcic_load(subject=1):
    """Quick test to verify BCIC data loads correctly."""
    print("="*60)
    print(f"TESTING BCIC DATASET - Subject A{subject:02d}")
    print("="*60)
    
    raw = load_raw_bcic(subject, 'T')
    raw = preprocess_raw(raw, BCIC_CHANNELS)
    
    # Get events
    events, event_dict = mne.events_from_annotations(raw)
    event_id = {
        'left': event_dict['769'],
        'right': event_dict['770'],
        'foot': event_dict['771'],
        'tongue': event_dict['772']
    }


    
    print(f"\n✓ Event mapping: {event_id}")
    
    epochs = epoch_data(raw, event_id, tmin=2.0, tmax=6.0)  # BCIC uses 2-6s window
    print(f"✓ Created {len(epochs)} epochs")
    print(f"✓ Epoch shape: {epochs.get_data().shape}")
    
    return epochs

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


def preprocess_raw(raw, channels=None):
    if channels is not None:
        raw.pick([ch for ch in channels if ch in raw.ch_names])
    else:
        raw.pick(MOTOR_CHANNELS)
    
    raw.set_eeg_reference('average', projection=True)
    raw.filter(l_freq=8.0, h_freq=30.0, method='fir')
    raw.notch_filter(freqs=60)
    return raw



def epoch_data(raw, event_id, tmin=0.0, tmax=3.0, events=None):
    """Create epochs from raw data."""
    
    if len(event_id) != 2:
        raise ValueError(f"❌ Cannot train: expected 2 classes, found {event_id}")

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
        ('lda', LDA(shrinkage=1e-4))
    ])

def train_mode(args):
    if args.subject is None or args.run is None:
        print("❌ train mode requires <subject> <run>")
        exit(1)

    # Load + preprocess

    if args.b:
        raw = preprocess_raw(load_raw_bcic(args.subject, 'T'), BCIC_CHANNELS)
        events, event_id = normalize_bcic(raw)
        # Keep only left vs right
        mask = np.isin(events[:, 2], [event_id["left"], event_id["right"]])
        events = events[mask]
        event_id = {"left": event_id["left"], "right": event_id["right"]}
    else:
        raw = preprocess_raw(load_raw(args.subject, [args.run]))
        events, event_id = mne.events_from_annotations(raw)
        # Keep only T1 vs T2 (like experiment 0: left vs right fist)
        mask = np.isin(events[:, 2], [event_id["T1"], event_id["T2"]])
        events = events[mask]
        event_id = {"T1": event_id["T1"], "T2": event_id["T2"]}
        
    
    epochs = epoch_data(raw, event_id, events=events)
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Split train/test BEFORE any training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train CV on TRAINING SET ONLY
    pipeline = build_pipeline()
    print(f"Loaded subject {args.subject}, run {args.run}")
    
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    # print number of training samples
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

    print("Cross-validation scores (on training set):")
    print([f"{score:.4f}" for score in scores])
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
            'event_id': event_id,
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
    if (args.b):
        subjects = np.arange(1, 10) 
    all_scores = []
    
    if args.b:
        experiments = [
            {"name": "Left vs Right", "classes": ["left", "right"]},
            {"name": "Tongue vs Foot", "classes": ["tongue", "foot"]}
        ]
    else:
        experiments = [{"name": exp_cfg["desc"], "runs": exp_cfg["runs"], "events": exp_cfg["events"]}
                       for exp_cfg in EXPERIMENTS.values()]

    for exp_index, exp_cfg in enumerate(experiments):
        experiment_scores = []
        print(f"\n\nExperiment {exp_index + 1}: {exp_cfg['name']}")

        for subject in subjects:
            print(f"Subject {subject}... ", end='', flush=True)

            try:
                if args.b:
                    # Load BCIC raw data
                    raw = preprocess_raw(load_raw_bcic(subject, 'T'), BCIC_CHANNELS)
                    events, event_id = normalize_bcic(raw)

                    # Keep only relevant classes
                    selected_labels = [event_id[c] for c in exp_cfg["classes"]]
                    mask = np.isin(events[:, 2], selected_labels)
                    events = events[mask]

                    filtered_event_id = {c: event_id[c] for c in exp_cfg["classes"]}

                else:
                    # Motor dataset
                    raw = preprocess_raw(load_raw(subject, exp_cfg['runs']))
                    events, event_dict = mne.events_from_annotations(raw)
                    # Keep only specified events
                    selected_labels = [event_dict[e] for e in exp_cfg["events"] if e in event_dict]
                    mask = np.isin(events[:, 2], selected_labels)
                    events = events[mask]

                    filtered_event_id = {e: event_dict[e] for e in exp_cfg["events"] if e in event_dict}

                if len(filtered_event_id) != 2 or len(events) < 10:
                    print("Skipping (insufficient data)")
                    continue

                # Epoch data
                epochs = epoch_data(raw, filtered_event_id, events=events)

                X = epochs.get_data()
                y = epochs.events[:, 2]

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )

                # Build pipeline
                pipeline = build_pipeline()
                pipeline.fit(X_train, y_train)

                # Predict and compute accuracy
                preds = [pipeline.predict(X_test[i:i+1])[0] for i in range(len(X_test))]
                acc = np.mean(preds == y_test)
                print(f"✔ {acc*100:.2f}%")
                experiment_scores.append(acc)

            except Exception as e:
                print(f"Skipping (error: {e})")
                continue

        mean_exp_score = np.mean(experiment_scores) if experiment_scores else 0.0
        print(f"Experiment {exp_index + 1} Accuracy: {mean_exp_score*100:.2f}%")
        all_scores.append(mean_exp_score*100)

    return all_scores

def visualize_mode(args):
    
    if args.b:
        raw_original = load_raw_bcic(args.subject, 'T')
        raw_original.pick(BCIC_CHANNELS)
        raw_filtered = preprocess_raw(raw_original, BCIC_CHANNELS)
    else:
        raw_original = load_raw(args.subject, [args.run])
        raw_original.pick(MOTOR_CHANNELS)
        raw_filtered = preprocess_raw(raw_original)
    
    # Figure 1: Filtered EEG
    raw_filtered.plot(duration=10, n_channels=len(raw_filtered.ch_names),
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
parser.add_argument('--bcic', '-b', action='store_true', dest='b', default=False, 
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
    print(f"Time of Execution: {time.time() - start_time:.2f} seconds")
