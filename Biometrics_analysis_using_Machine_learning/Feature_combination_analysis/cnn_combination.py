import os
import numpy as np
import h5py
import joblib
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

###############################################################################
# 1. Data Loading (Unchanged)
###############################################################################
def load_features_from_hdf5(filename):
    """
    Loads EEG features from an HDF5 file in (channels × features) format.
    """
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subj = class_key.split('_', 1)[-1]
            for run_key in h5f[class_key].keys():
                for epoch_key in h5f[class_key][run_key].keys():
                    feats = h5f[class_key][run_key][epoch_key]['features'][()]
                    X.append(feats)
                    y.append(subj)
                    runs.append(run_key)
                    try:
                        ep = int(epoch_key.split('_')[-1])
                    except:
                        ep = 0
                    epoch_nums.append(ep)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)


###############################################################################
# 2. Hypermodel for 1D-CNN (Unchanged)
###############################################################################
def build_tunable_1d_cnn_model(hp):
    global input_shape, num_classes

    f1    = hp.Int('filters_1',    16, 64, step=16, default=32)
    k1    = hp.Choice('kernel_size_1',[3,5,7], default=5)
    f2    = hp.Int('filters_2',    32, 128, step=32, default=64)
    k2    = hp.Choice('kernel_size_2',[3,5], default=3)
    du    = hp.Int('dense_units',  64, 256, step=64, default=128)
    dr    = hp.Float('dropout_rate',0.0,0.5,step=0.1,default=0.2)
    lr    = hp.Float('learning_rate',1e-4,1e-2,sampling='LOG',default=1e-3)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=f1, kernel_size=k1, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(filters=f2, kernel_size=k2, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(du, activation='relu'),
        layers.Dropout(dr),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


###############################################################################
# 3. Main Function (Fully Updated)
###############################################################################
def main():
    # --- 1. Path and Directory Setup (UPDATED) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the script's location for portability
    H5_FILE = os.path.join(SCRIPT_DIR, 'data', 'all_subjects_merged_new_full_epochs.h5')
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'results', 'models')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'cnn_results')
    KT_DIR = os.path.join(SCRIPT_DIR, 'results', 'keras_tuner') # Central directory for tuner files

    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(KT_DIR, exist_ok=True)

    TRANSFORMER_FP = os.path.join(MODELS_DIR, 'power_transformer.pkl')
    # --- End Path Setup ---

    # 3.1 Load raw data
    X_raw, y, runs, epoch_nums = load_features_from_hdf5(H5_FILE)
    n_samples, n_channels, n_feats = X_raw.shape
    print(f"Loaded X_raw with shape = (n={n_samples}, channels={n_channels}, feats={n_feats})")

    # 3.2 Flatten & Yeo-Johnson transform
    X_flat = X_raw.reshape(n_samples, -1)
    if os.path.exists(TRANSFORMER_FP):
        print(f"Loading existing transformer from {TRANSFORMER_FP}")
        transformer = joblib.load(TRANSFORMER_FP)
    else:
        print("Fitting a new power transformer...")
        transformer = PowerTransformer(method='yeo-johnson')
        transformer.fit(X_flat)
        joblib.dump(transformer, TRANSFORMER_FP)
        print(f"Saved new transformer to {TRANSFORMER_FP}")
    X_tf = transformer.transform(X_flat)

    # 3.3 Reshape back and encode labels
    X_tf2 = X_tf.reshape(n_samples, n_channels, n_feats)
    global num_classes
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    num_classes = len(le.classes_)

    # 3.4 Define feature-subset indices
    feature_combinations = {
        "Set_1_Time":                     list(range(0, 5)),
        "Set_2_Frequency":                list(range(5, 10)),
        "Set_3_TimeFrequency":            list(range(14, 28)),
        "Set_4_NonLinear":                list(range(10, 14)),
        "Set_5_Time+Frequency":           list(range(0, 10)),
        "Set_6_Time+TimeFrequency":       list(range(0, 5))  + list(range(14, 28)),
        "Set_7_Frequency+TimeFrequency":  list(range(5, 10)) + list(range(14, 28)),
        "Set_8_Time+Frequency+NonLinear": list(range(0, 14)),
        "Set_9_Time+TimeFrequency+NonLinear": (
            list(range(0, 5)) + list(range(10, 14)) + list(range(14, 28))
        ),
        "Set_10_Time+Frequency+NonLinear (No TimeFreq)": list(range(0, 14)),
        "Set_11_All":                     list(range(0, n_feats))
    }

    summary = []

    # 3.5 Loop through subsets
    for set_name, feat_idx in feature_combinations.items():
        s = len(feat_idx)
        print(f"\n>>> Subset {set_name}: {s} features × {n_channels} channels")

        # Slice, transpose, and set global input shape
        X_sub = X_tf2[:, :, feat_idx]
        X_cnn = np.transpose(X_sub, (0, 2, 1))
        global input_shape
        input_shape = (s, n_channels)

        # Split data
        tr_idx, te_idx = np.where(runs == "Run_1")[0], np.where(runs == "Run_2")[0]
        X_train, y_train = X_cnn[tr_idx], y_enc[tr_idx]
        X_test, y_test = X_cnn[te_idx], y_enc[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )
        y_tr_cat = utils.to_categorical(y_tr, num_classes)
        y_val_cat = utils.to_categorical(y_val, num_classes)

        # 3.6 Hyperparameter search
        tuner_dir_for_set = os.path.join(KT_DIR, set_name)
        tuner = kt.RandomSearch(
            build_tunable_1d_cnn_model,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=1,
            directory=tuner_dir_for_set,
            project_name='cnn_tune',
            overwrite=True
        )
        tuner.search(
            X_tr, y_tr_cat, validation_data=(X_val, y_val_cat), epochs=10,
            callbacks=[callbacks.EarlyStopping('val_accuracy', patience=2, restore_best_weights=True)],
            verbose=0
        )

        # 3.7 Evaluate best model and log metrics
        best_model = tuner.get_best_models(num_models=1)[0]
        preds_prob = best_model.predict(X_test, verbose=0)
        preds_enc = np.argmax(preds_prob, axis=1)
        acc = accuracy_score(y_test, preds_enc)
        f1 = f1_score(y_test, preds_enc, average='weighted', zero_division=0)
        prec  = precision_score(y_test, preds_enc, average='weighted', zero_division=0)
        rec   = recall_score(y_test, preds_enc, average='weighted', zero_division=0)
        print(f"  → Acc={acc:.3f}, F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")

        # 3.8 Save results and summary
        df_out = pd.DataFrame({
            'True_Label': le.inverse_transform(y_test),
            'Predicted_Label': le.inverse_transform(preds_enc)
        })
        out_fp = os.path.join(RESULTS_DIR, f'preds_1dcnn_{set_name}.csv')
        df_out.to_csv(out_fp, index=False)
        summary.append({
            'Subset':    set_name,
            'Accuracy':  acc,
            'F1_Score':  f1,
            'Precision': prec,
            'Recall':    rec
        })

    # 3.9 Save summary of all subsets
    summary_path = os.path.join(RESULTS_DIR, 'cnn_subsets_summary.csv')
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == '__main__':
    main()