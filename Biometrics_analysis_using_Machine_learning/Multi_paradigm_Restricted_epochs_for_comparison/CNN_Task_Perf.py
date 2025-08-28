# -*- coding: utf-8 -*-
"""
Performs a complete 1D-CNN based biometric identification experiment.

This script automates a three-stage workflow:
1.  TRAINING: Trains multiple 1D-CNN models. Each model is trained on a
    different random sample of 40 epochs per subject from the training set ('Run_1').
    The trained model, transformer, and label encoder for each run are saved.

2.  PREDICTION: For each trained model, it generates predictions and probability
    scores on the entire test set ('Run_2') and saves them to temporary CSV files.

3.  ANALYSIS: Aggregates the prediction results from all runs to produce a
    single, final report detailing the mean task-wise accuracy and Equal Error
    Rate (EER), along with a summary plot.
"""
import os
import logging
from typing import Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPooling1D, ReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Helper Functions (used across stages) ---

def load_features_from_hdf5(
    filename: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads feature data, labels, runs, and epoch numbers from the HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, "r") as h5f:
        for class_key in h5f:
            subject_code = class_key.split("_", 1)[-1]
            for run_key in h5f[class_key]:
                for epoch_key in h5f[class_key][run_key]:
                    feats = h5f[class_key][run_key][epoch_key]["features"][()]
                    ep_idx = int(epoch_key.split("_")[-1]) if "_" in epoch_key else 0
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)


def build_1d_cnn_model(
    input_shape: Tuple[int, int], num_classes: int
) -> Sequential:
    """Defines and compiles the 1D-CNN architecture."""
    model = Sequential(
        [
            Conv1D(filters=64, kernel_size=3, input_shape=input_shape),
            BatchNormalization(),
            ReLU(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(filters=128, kernel_size=3),
            BatchNormalization(),
            ReLU(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates the Equal Error Rate from true labels and scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_index])


# --- Stage 1: Training Function ---

def run_training_for_iteration(
    run_number: int,
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    artifact_dir: str,
    n_epochs_to_sample: int,
):
    """Trains one model on a random sample and saves the necessary artifacts."""
    logging.info(f"--- [TRAINING] Starting Run #{run_number} ---")

    # Create training sample for this run using a unique random state
    df = pd.DataFrame({"subject": y, "run": runs, "original_index": np.arange(len(y))})
    df_run1 = df[df["run"] == "Run_1"]
    train_sampled_df = df_run1.groupby("subject", group_keys=False).apply(
        lambda grp: grp.sample(
            n=min(len(grp), n_epochs_to_sample), random_state=run_number
        )
    )
    train_idx = train_sampled_df["original_index"].values
    X_train, y_train = X[train_idx], y[train_idx]

    # Preprocessing
    X_train_flat = X_train.reshape(len(X_train), -1)
    transformer = PowerTransformer(method="yeo-johnson").fit(X_train_flat)
    X_train_tf = transformer.transform(X_train_flat)

    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)
    y_train_cat = to_categorical(y_train_enc)

    # Reshape for CNN
    num_features = X_train_tf.shape[1]
    X_train_cnn = X_train_tf.reshape(len(X_train_tf), num_features, 1)
    num_classes = len(le.classes_)

    # Build and Train Model
    model = build_1d_cnn_model(input_shape=(num_features, 1), num_classes=num_classes)
    model.fit(X_train_cnn, y_train_cat, epochs=50, batch_size=32, verbose=0)

    # Save Artifacts for this run
    model.save(os.path.join(artifact_dir, f"model_run_{run_number}.h5"))
    joblib.dump(transformer, os.path.join(artifact_dir, f"transformer_run_{run_number}.pkl"))
    joblib.dump(le, os.path.join(artifact_dir, f"encoder_run_{run_number}.pkl"))
    logging.info(f"--- [TRAINING] Finished Run #{run_number} ---")


# --- Stage 2: Prediction Function ---

def run_prediction_for_iteration(
    run_number: int,
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    epoch_nums: np.ndarray,
    artifact_dir: str,
    temp_results_dir: str,
):
    """Loads a trained model and saves its predictions on the test set."""
    logging.info(f"--- [PREDICTION] Starting Run #{run_number} ---")

    # Load artifacts for this run
    model = tf.keras.models.load_model(os.path.join(artifact_dir, f"model_run_{run_number}.h5"))
    transformer = joblib.load(os.path.join(artifact_dir, f"transformer_run_{run_number}.pkl"))
    le = joblib.load(os.path.join(artifact_dir, f"encoder_run_{run_number}.pkl"))

    # Prepare test data
    test_idx = np.where(runs == "Run_2")[0]
    X_test, y_test, epochs_test = X[test_idx], y[test_idx], epoch_nums[test_idx]

    X_test_flat = X_test.reshape(len(X_test), -1)
    X_test_tf = transformer.transform(X_test_flat)
    X_test_cnn = X_test_tf.reshape(len(X_test_tf), X_test_tf.shape[1], 1)

    # Predict probabilities and derive labels
    preds_prob = model.predict(X_test_cnn)
    preds_enc = np.argmax(preds_prob, axis=1)
    preds = le.inverse_transform(preds_enc)

    # Create a detailed results DataFrame with scores for each class
    results_df = pd.DataFrame(
        {"True_Label": y_test, "Predicted_Label": preds, "Epoch": epochs_test}
    )
    for i, class_label in enumerate(le.classes_):
        results_df[f"score_{class_label}"] = preds_prob[:, i]

    # Save to a temporary CSV for the analysis stage
    results_df.to_csv(
        os.path.join(temp_results_dir, f"run_{run_number}_preds.csv"), index=False
    )
    logging.info(f"--- [PREDICTION] Finished Run #{run_number} ---")


# --- Stage 3: Analysis Function ---

def analyze_all_runs(
    n_iterations: int,
    temp_results_dir: str,
    mapping_csv_path: str,
    output_dir: str,
):
    """Analyzes all temporary results and generates the final aggregated report."""
    logging.info("--- [ANALYSIS] Starting Final Analysis ---")

    df_map = pd.read_csv(mapping_csv_path)
    df_map["Epoch"] = df_map["New_Epoch"].str.extract(r"epoch_(\d+)").astype(int)
    df_map["Subject"] = df_map["Subject"].str.split("_", expand=True)[1]
    df_map_test = df_map[df_map["Run"] == "Run_2"]

    task_accuracies, task_eers = {}, {}

    for i in range(n_iterations):
        results_file = os.path.join(temp_results_dir, f"run_{i}_preds.csv")
        df_res = pd.read_csv(results_file)

        # Ensure consistent string data types for robust merging and comparison
        df_res["True_Label"] = df_res["True_Label"].astype(str)
        df_res["Predicted_Label"] = df_res["Predicted_Label"].astype(str)
        df_map_test["Subject"] = df_map_test["Subject"].astype(str)

        df_res.rename(columns={"True_Label": "Subject"}, inplace=True)
        df_merged = pd.merge(df_res, df_map_test, on=["Subject", "Epoch"], how="left")

        for task, group in df_merged.groupby("Source_File"):
            if pd.isna(task):
                continue

            # Calculate accuracy for the current task
            acc = accuracy_score(group["Subject"], group["Predicted_Label"])
            task_accuracies.setdefault(task, []).append(acc)

            # Calculate EER for the current task
            group_true_labels = group["Subject"].values
            subjects_in_task = np.unique(group_true_labels)
            subject_eers = []
            for subj in subjects_in_task:
                score_col = f"score_{subj}"
                if score_col in group.columns:
                    y_true_ovr = (group_true_labels == subj).astype(int)
                    # EER requires both positive and negative samples
                    if np.unique(y_true_ovr).size > 1:
                        subject_eers.append(
                            calculate_eer(y_true_ovr, group[score_col].values)
                        )
            
            if subject_eers:
                task_eers.setdefault(task, []).append(np.mean(subject_eers))

    # Aggregate results across all runs and save the final report
    agg_results = []
    for task in sorted(task_accuracies.keys()):
        agg_results.append(
            {
                "Task": task,
                "Mean_Accuracy": np.mean(task_accuracies.get(task, [np.nan])),
                "Std_Accuracy": np.std(task_accuracies.get(task, [np.nan])),
                "Mean_EER (%)": np.mean(task_eers.get(task, [np.nan])) * 100,
                "Std_EER (%)": np.std(task_eers.get(task, [np.nan])) * 100,
            }
        )
    
    df_perf = pd.DataFrame(agg_results).sort_values(by="Mean_Accuracy", ascending=False)
    output_path = os.path.join(output_dir, "task_performance_1dcnn_averaged.csv")
    df_perf.to_csv(output_path, index=False)

    logging.info(f"✅ Final aggregated report saved to '{output_path}'")
    print("\n--- Final Task-wise Performance Report ---")
    print(df_perf.to_string(index=False))

    # Plotting the final results
    plt.figure(figsize=(12, 8))
    plt.bar(
        df_perf["Task"],
        df_perf["Mean_Accuracy"],
        yerr=df_perf["Std_Accuracy"],
        capsize=5,
        color="teal",
    )
    plt.title(f"1D-CNN Mean Task Accuracy ({n_iterations} runs)")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Task")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "task_performance_1dcnn_averaged.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"✅ Final performance plot saved to '{plot_path}'")


# --- Main Orchestrator ---
def main():
    """Orchestrates the entire Training -> Prediction -> Analysis workflow."""

    # --- Configuration ---
    # These are the main parameters you might want to change.
    N_ITERATIONS = 10  # Number of models to train and average over.
    N_EPOCHS_TO_SAMPLE = 40  # Number of epochs to sample per subject for training.

    HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
    MAPPING_CSV = "epoch_mapping.csv"

    # A single parent directory for all outputs of this script.
    OUTPUT_DIR = "results/cnn_analysis"
    ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")
    TEMP_RESULTS_DIR = os.path.join(OUTPUT_DIR, "temp_predictions")
    FINAL_REPORT_DIR = os.path.join(OUTPUT_DIR, "final_report")

    # Create necessary directories
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)
    os.makedirs(FINAL_REPORT_DIR, exist_ok=True)

    # --- Load Data Once ---
    logging.info(f"Loading data from {HDF5_FILE}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(HDF5_FILE)
    logging.info("Data loaded successfully.")

    # --- STAGE 1: TRAINING LOOP ---
    logging.info(f"--- STARTING TRAINING STAGE FOR {N_ITERATIONS} RUNS ---")
    for i in range(N_ITERATIONS):
        run_training_for_iteration(i, X, y, runs, ARTIFACT_DIR, N_EPOCHS_TO_SAMPLE)

    # --- STAGE 2: PREDICTION LOOP ---
    logging.info(f"--- STARTING PREDICTION STAGE FOR {N_ITERATIONS} RUNS ---")
    for i in range(N_ITERATIONS):
        run_prediction_for_iteration(
            i, X, y, runs, epoch_nums, ARTIFACT_DIR, TEMP_RESULTS_DIR
        )

    # --- STAGE 3: FINAL ANALYSIS ---
    logging.info("--- STARTING FINAL ANALYSIS STAGE ---")
    analyze_all_runs(N_ITERATIONS, TEMP_RESULTS_DIR, MAPPING_CSV, FINAL_REPORT_DIR)

    logging.info("--- FULL EXPERIMENT COMPLETE ---")


if __name__ == "__main__":
    main()