import os
import glob
import numpy as np
import h5py
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras_tuner as kt
import argparse # Added for command-line arguments

###############################################################################
# 1. Data Loading
###############################################################################
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
    X, y, runs = [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            for run_key in h5f[class_key].keys():
                for epoch_key in h5f[class_key][run_key].keys():
                    feats = h5f[class_key][run_key][epoch_key]['features'][()]
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
    return np.array(X), np.array(y), np.array(runs)

###############################################################################
# 2. Hypermodel for 1D CNN
###############################################################################
def build_tunable_1d_cnn_model(hp, input_shape, num_classes):
    """Builds a 1D CNN model with hyperparameters for keras_tuner."""
    model = models.Sequential()
    model.add(layers.Conv1D(
        filters=hp.Int('filters_1', 16, 64, step=16),
        kernel_size=hp.Choice('kernel_size_1', [3, 5]),
        activation='relu', input_shape=input_shape
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(
        filters=hp.Int('filters_2', 32, 128, step=32),
        kernel_size=hp.Choice('kernel_size_2', [3, 5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', 64, 256, step=64),
        activation='relu'
    ))
    model.add(layers.Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

###############################################################################
# 3. Main Processing Logic for a SINGLE file
###############################################################################
def process_file(filepath, output_dir):
    """Runs the entire pipeline for a single HDF5 file."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join(output_dir, base_name) # Create a subdirectory for each file's outputs
    os.makedirs(file_output_dir, exist_ok=True)
    print(f"--- Starting analysis for: {base_name} ---")
    print(f"--- Outputs will be saved in: {file_output_dir} ---")

    try:
        X, y, runs = load_features_from_hdf5(filepath)
        if X.shape[0] == 0:
            print(f"[WARNING] No data found in {filepath}. Skipping.")
            return
    except Exception as e:
        print(f"[ERROR] Could not load data from {filepath}: {e}. Skipping.")
        return

    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    joblib.dump(transformer, os.path.join(file_output_dir, "power_transformer.pkl"))

    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("[WARNING] Run_1/Run_2 split not possible. Using random 80/20 split.")
        y_temp_enc = LabelEncoder().fit_transform(y)
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y_temp_enc, random_state=42)

    X_train_flat, X_test_flat = X_tf[train_idx], X_tf[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    input_shape = (X_train_flat.shape[1], 1)
    X_train_cnn = X_train_flat.reshape(X_train_flat.shape[0], input_shape[0], 1)
    X_test_cnn = X_test_flat.reshape(X_test_flat.shape[0], input_shape[0], 1)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(np.unique(y_train_enc))
    y_train_cat = utils.to_categorical(y_train_enc, num_classes=num_classes)
    
    X_train_sub, X_val, y_train_sub_cat, y_val_cat = train_test_split(
        X_train_cnn, y_train_cat, test_size=0.15, stratify=y_train_enc, random_state=42
    )
    
    tuner = kt.RandomSearch(
        lambda hp: build_tunable_1d_cnn_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory=os.path.join(output_dir, 'kt_meta'), # Centralized tuner directory
        project_name=f'tuning_{base_name}'
    )

    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    tuner.search(
        X_train_sub, y_train_sub_cat,
        validation_data=(X_val, y_val_cat),
        epochs=40, callbacks=[early_stop], verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        X_train_cnn, y_train_cat, # Retrain on full training data
        epochs=40, validation_split=0.1,
        callbacks=[early_stop], verbose=1
    )
    
    preds_prob = best_model.predict(X_test_cnn)
    preds_enc = np.argmax(preds_prob, axis=1)
    
    predictions_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': le.inverse_transform(preds_enc)
    })
    predictions_df.to_csv(os.path.join(file_output_dir, 'predictions.csv'), index=False)
    
    report = classification_report(y_test_enc, preds_enc, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(os.path.join(file_output_dir, 'classification_report.csv'))

    best_model.save(os.path.join(file_output_dir, "best_model.h5"))
    joblib.dump(le, os.path.join(file_output_dir, "label_encoder.pkl"))

    print(f"--- Finished analysis for: {base_name}. Artifacts saved. ---")

###############################################################################
# 4. Main script execution loop
###############################################################################
def main(args):
    """Finds all .h5 files in the specified directory and processes them."""
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")

    h5_files = glob.glob(os.path.join(args.data_dir, '*.h5'))

    if not h5_files:
        print(f"No .h5 files found in '{args.data_dir}'. Exiting.")
        return

    print(f"Found {len(h5_files)} HDF5 files to process.")
    
    for h5_file in h5_files:
        print(f"\n{'='*80}")
        print(f"PROCESSING FILE: {os.path.basename(h5_file)}")
        print(f"{'='*80}\n")
        try:
            process_file(h5_file, args.output_dir)
            print(f"\n[SUCCESS] Completed processing for {os.path.basename(h5_file)}.")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred while processing {os.path.basename(h5_file)}: {e}")
            import traceback
            traceback.print_exc()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 1D CNN hyperparameter tuning on EEG data.")
    parser.add_argument('--data-dir', type=str, default='../data', help='Directory containing the HDF5 data files.')
    parser.add_argument('--output-dir', type=str, default='../results/cnn_tuning', help='Directory to save analysis results and tuning artifacts.')
    args = parser.parse_args()
    main(args)