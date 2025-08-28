#!/usr/bin/env python3
"""
Tunes and trains a 1D CNN for EEG-based subject identification.

This script processes all HDF5 files in the current directory. For each file,
it performs the following steps:
1. Loads and preprocesses the data (PowerTransformer, train/test split by run).
2. Uses Keras Tuner (RandomSearch) to find optimal hyperparameters for a 1D CNN
   architecture. The search is performed on a subset of the training data.
3. Trains the best model on the full training data with early stopping.
4. Evaluates the final model on the test set.
5. Saves all artifacts, including the trained model, label encoder, performance
   reports (classification report, predictions CSV), and plots.

Each processed HDF5 file gets its own dedicated output subdirectory.
"""
import os
import glob
import argparse
import numpy as np
import h5py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import keras_tuner as kt

plt.switch_backend('Agg')

# =============================================================================
# DATA LOADING
# =============================================================================
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
    X, y, runs = [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            for run_key in h5f[class_key].keys():
                for epoch_key, epoch_group in h5f[class_key][run_key].items():
                    X.append(epoch_group['features'][()])
                    y.append(subject_code)
                    runs.append(run_key)
    return np.array(X), np.array(y), np.array(runs)

# =============================================================================
# HYPERMODEL DEFINITION
# =============================================================================
def build_tunable_1d_cnn_model(hp, input_shape, num_classes):
    """Builds a tunable 1D CNN model for Keras Tuner."""
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Tunable Conv Block 1
    model.add(layers.Conv1D(
        filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())

    # Tunable Conv Block 2
    model.add(layers.Conv1D(
        filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())

    # Tunable Dense Block
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
        activation='relu'
    ))
    model.add(layers.Dropout(
        rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    ))
    
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Tunable learning rate
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_training_history(history, save_path):
    """Plots and saves training & validation accuracy and loss."""
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.1)
    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# MAIN PROCESSING LOGIC FOR A SINGLE FILE
# =============================================================================
def process_file(filepath, output_dir):
    """Runs the entire pipeline for a single HDF5 file."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    file_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)
    
    print(f"\n--- Starting analysis for: {base_name} ---")

    # Load data
    try:
        X, y, runs = load_features_from_hdf5(filepath)
        if X.shape[0] == 0:
            print(f"[WARNING] No data found in {filepath}. Skipping.")
            return
    except Exception as e:
        print(f"[ERROR] Could not load data from {filepath}: {e}. Skipping.")
        return

    # Preprocessing
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Split data by run
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("[WARNING] Run_1/Run_2 split failed. Using random stratified split.")
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y_enc, random_state=42)

    X_train_cnn = X_tf[train_idx].reshape(len(train_idx), X.shape[1], X.shape[2])
    X_test_cnn = X_tf[test_idx].reshape(len(test_idx), X.shape[1], X.shape[2])
    y_train_cat = utils.to_categorical(y_enc[train_idx], num_classes)
    y_test_cat = utils.to_categorical(y_enc[test_idx], num_classes)
    y_test_labels = y[test_idx]

    # Use a subset for hyperparameter tuning to speed up the process
    X_train_sub, _, y_train_sub_cat, _ = train_test_split(
        X_train_cnn, y_train_cat, train_size=0.8, stratify=y_enc[train_idx], random_state=42)

    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])

    # Keras Tuner setup
    model_builder = lambda hp: build_tunable_1d_cnn_model(hp, input_shape, num_classes)
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory=os.path.join(file_output_dir, 'kt_tuning'),
        project_name='1d_cnn_tuning'
    )

    early_stop_tuner = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print("\n--- Starting Hyperparameter Search ---")
    tuner.search(X_train_sub, y_train_sub_cat, validation_split=0.2, epochs=25,
                 callbacks=[early_stop_tuner], verbose=1)

    # Train the best model on the full training set
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    print("\n--- Training Best Model ---")
    early_stop_train = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    history = best_model.fit(X_train_cnn, y_train_cat, epochs=50, validation_split=0.2,
                             callbacks=[early_stop_train], verbose=1)
    
    # Evaluation
    print("\n--- Evaluating Model ---")
    preds_prob = best_model.predict(X_test_cnn)
    preds_enc = np.argmax(preds_prob, axis=1)
    preds_labels = le.inverse_transform(preds_enc)

    # Save artifacts
    pd.DataFrame({'True_Label': y_test_labels, 'Predicted_Label': preds_labels}).to_csv(
        os.path.join(file_output_dir, 'predictions.csv'), index=False)
    
    report = classification_report(y_test_labels, preds_labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(os.path.join(file_output_dir, 'classification_report.csv'))
    
    plot_training_history(history, os.path.join(file_output_dir, 'training_history.png'))
    
    best_model.save(os.path.join(file_output_dir, "best_model.h5"))
    joblib.dump(le, os.path.join(file_output_dir, "label_encoder.pkl"))
    joblib.dump(transformer, os.path.join(file_output_dir, "power_transformer.pkl"))

    print(f"\n[SUCCESS] Finished analysis for {base_name}. Artifacts saved in '{file_output_dir}'.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Finds all .h5 files and processes them."""
    parser = argparse.ArgumentParser(description="1D CNN Tuner and Trainer for EEG Biometrics")
    parser.add_argument('--output-dir', type=str, required=True, help='Parent directory to save all analysis results.')
    args = parser.parse_args()

    # Find HDF5 files in the current working directory
    h5_files = glob.glob('*.h5')
    if not h5_files:
        print("No .h5 files found in the current directory. Exiting.")
        return

    print(f"Found {len(h5_files)} HDF5 files to process.")
    for h5_file in h5_files:
        try:
            process_file(h5_file, args.output_dir)
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred while processing {os.path.basename(h5_file)}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()