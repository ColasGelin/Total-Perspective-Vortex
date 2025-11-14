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

mne.set_log_level('WARNING')

# ---------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------

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

# ---------------------------------------------------------
#  COMMON FUNCTIONS TO REMOVE REDUNDANCY
# ---------------------------------------------------------

def load_raw(subject, runs):
    """Load EEG data for a subject and given list of runs."""
    files = mne.datasets.eegbci.load_data(subject, runs)
    return mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])


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


# ---------------------------------------------------------
#  MAIN LOGIC (TRAIN / PREDICT / EVAL)
# ---------------------------------------------------------

parser = argparse.ArgumentParser(description='EEG Motor Imagery Classification Pipeline')
parser.add_argument('subject', type=int, nargs='?', default=109)
parser.add_argument('run', type=int, nargs='?', default=None)
parser.add_argument('mode', type=str, nargs='?', default='eval',
                    choices=['train', 'predict', 'eval'])
args = parser.parse_args()

# ---------------------------------------------------------
#  TRAIN MODE
# ---------------------------------------------------------

if args.mode == 'train':
    if args.subject is None or args.run is None:
        print("❌ train mode requires <subject> <run>")
        exit(1)

    # Load + preprocess
    raw = preprocess_raw(load_raw(args.subject, [args.run]))

    # Epoch
    events, event_dict = mne.events_from_annotations(raw)
    epochs = epoch_data(raw, event_dict)
    X = epochs.get_data()
    y = epochs.events[:, 2]

    # Train CV
    pipeline = build_pipeline()
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

    print(scores)
    print(f"Cross-val mean: {np.mean(scores):.4f}")

    # Fit full
    pipeline.fit(X, y)

    # Save
    fname = f"model_s{args.subject}_r{args.run}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump({
            'pipeline': pipeline,
            'subject': args.subject,
            'run': args.run,
            'event_dict': event_dict,
            'cv_scores': scores,
            'mean_cv_score': np.mean(scores)
        }, f)

    print(f"✔ Model saved to {fname}")


# ---------------------------------------------------------
#  PREDICT MODE
# ---------------------------------------------------------

elif args.mode == 'predict':
    if args.subject is None or args.run is None:
        print("❌ predict mode requires <subject> <run>")
        exit(1)

    # Load trained model
    fname = f"model_s{args.subject}_r{args.run}.pkl"
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    pipeline = data['pipeline']
    event_id = data['event_dict']

    # Reload + preprocess
    raw = preprocess_raw(load_raw(args.subject, [args.run]))

    # Epoch
    epochs = epoch_data(raw, event_id)
    X = epochs.get_data()
    y_true = epochs.events[:, 2]

    # Predict
    y_pred = pipeline.predict(X)

    # Report
    print("\nepoch: pred | true | match")
    correct = 0
    for i, (p, t) in enumerate(zip(y_pred, y_true)):
        print(f"{i:02d}: [{p}] [{t}] {p==t}")
        correct += (p == t)

    print(f"\nAccuracy: {correct / len(y_true):.4f}")


# ---------------------------------------------------------
#  EVAL MODE (MULTI-SUBJECT)
# ---------------------------------------------------------

else:
    print("=" * 60)
    print("FULL EVAL MODE")
    print("=" * 60)

    subjects = np.arange(1, args.subject + 1)
    all_scores = []

    for exp_index, exp_cfg in EXPERIMENTS.items():
        print("\n" + "="*60)
        print(f"EXPERIMENT {exp_index}: {exp_cfg['desc']}")
        print("="*60)

        for subject in subjects:
            print(f"\n→ Subject {subject}")

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

            param_grid = {
                'pca__n_components': [2,4,6,8,10,15,20],
                'lda__shrinkage': ['auto', 0.1, 0.3]
            }

            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            grid = GridSearchCV(pipeline, param_grid, cv=cv,
                                scoring='accuracy', n_jobs=-1, verbose=0)

            grid.fit(X_train, y_train)

            preds = [grid.predict(X_test[i:i+1])[0] for i in range(len(X_test))]
            acc = np.mean(preds == y_test)
            print(f"✔ Accuracy: {acc*100:.2f}%")
            all_scores.append(acc)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Mean: {np.mean(all_scores)*100:.2f}%")
    print(f"Std: {np.std(all_scores)*100:.2f}%")
