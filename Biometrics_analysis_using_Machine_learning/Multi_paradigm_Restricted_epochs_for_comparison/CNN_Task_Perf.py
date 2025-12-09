# -*- coding: utf-8 -*-
"""
Performs a 1D-CNN biometric identification experiment with RESTRICTED EPOCHS.

UPDATED for Uniform Analysis:
- Implements Open-Set style Dev/Test split for threshold selection per task.
- Reports EER, FRR @ 1% FAR, and FRR @ 0.1% FAR with Mean/Std across iterations.
- Generates DET Curves for each task.
"""
import os
import logging
import random
from typing import Tuple, List, Dict, Any

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, MaxPooling1D, ReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Helper Functions ---

def load_features_from_hdf5(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def build_1d_cnn_model(input_shape: Tuple[int, int], num_classes: int) -> Sequential:
    """Defines the 1D-CNN architecture."""
    model = Sequential([
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
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_det_curve(fpr, fnr, eer, save_path, title):
    """Plots the Detection Error Tradeoff (DET) curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, linewidth=2, label=f"EER ≈ {eer*100:.2f}%")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- Biometric Verification Logic (Uniform Protocol) ---

def generate_biometric_trials(y_test, scores_matrix, subject_list, impostor_samples=5):
    """Generates genuine and impostor score/label pairs."""
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        
        # Skip if subject not in model (rare case in restricted subsets, but safe to handle)
        if true_subj not in subject_list:
            continue
            
        true_idx = subject_to_idx[true_subj]
        
        # Genuine
        scores.append(scores_matrix[i, true_idx])
        labels.append(1)
        
        # Impostor
        all_impostors = [s for s in subject_list if s != true_subj]
        n_samples = min(impostor_samples, len(all_impostors))
        sampled_impostors = random.sample(all_impostors, n_samples)
        
        for imp_subj in sampled_impostors:
            imp_idx = subject_to_idx[imp_subj]
            scores.append(scores_matrix[i, imp_idx])
            labels.append(0)

    return np.array(scores), np.array(labels)


def biometric_analysis_dev_test(y_test, scores_matrix, all_subjects, dev_subjects_count=10):
    """Splits task data into Dev (Thresholding) and Test (Evaluation)."""
    test_subjects = np.unique(y_test)
    
    current_dev_count = dev_subjects_count
    if len(test_subjects) <= current_dev_count:
        current_dev_count = len(test_subjects) // 2
    
    test_subjects_list = list(test_subjects)
    random.shuffle(test_subjects_list)
    
    dev_subjects = test_subjects_list[:current_dev_count]
    final_test_subjects = test_subjects_list[current_dev_count:]
    
    dev_mask = np.isin(y_test, dev_subjects)
    test_mask = np.isin(y_test, final_test_subjects)
    
    if not np.any(dev_mask) or not np.any(test_mask):
        return None 

    # --- Dev Phase ---
    y_dev = y_test[dev_mask]
    scores_dev = scores_matrix[dev_mask]
    dev_scores_flat, dev_labels_flat = generate_biometric_trials(y_dev, scores_dev, all_subjects)
    
    fpr, tpr, thresholds = roc_curve(dev_labels_flat, dev_scores_flat, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    fpr_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
    thresh_far_1_pc = fpr_interp(0.01)
    thresh_far_01_pc = fpr_interp(0.001)

    # --- Test Phase ---
    y_final = y_test[test_mask]
    scores_final = scores_matrix[test_mask]
    test_scores_flat, test_labels_flat = generate_biometric_trials(y_final, scores_final, all_subjects)
    
    fpr_test, tpr_test, _ = roc_curve(test_labels_flat, test_scores_flat, pos_label=1)
    fnr_test = 1 - tpr_test
    
    def get_frr(s, l, t):
        preds = (s >= t).astype(int)
        gen = (l == 1)
        fn = np.sum(preds[gen] == 0)
        tp = np.sum(preds[gen] == 1)
        return fn / (fn + tp) if (fn + tp) > 0 else 0
        
    return {
        "eer": eer_rate,
        "frr_1pc": get_frr(test_scores_flat, test_labels_flat, thresh_far_1_pc),
        "frr_01pc": get_frr(test_scores_flat, test_labels_flat, thresh_far_01_pc),
        "fpr_curve": fpr_test,
        "fnr_curve": fnr_test
    }


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

    # Sample Training Data (Run 1)
    df = pd.DataFrame({"subject": y, "run": runs, "original_index": np.arange(len(y))})
    df_run1 = df[df["run"] == "Run_1"]
    
    train_sampled_df = df_run1.groupby("subject", group_keys=False).apply(
        lambda grp: grp.sample(n=min(len(grp), n_epochs_to_sample), random_state=run_number)
    )
    train_idx = train_sampled_df["original_index"].values
    X_train, y_train = X[train_idx], y[train_idx]

    # Preprocessing (Fit on Train Only - No Leakage)
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

    # Train
    model = build_1d_cnn_model(input_shape=(num_features, 1), num_classes=num_classes)
    model.fit(X_train_cnn, y_train_cat, epochs=50, batch_size=32, verbose=0)

    # Save Artifacts
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

    model = tf.keras.models.load_model(os.path.join(artifact_dir, f"model_run_{run_number}.h5"))
    transformer = joblib.load(os.path.join(artifact_dir, f"transformer_run_{run_number}.pkl"))
    le = joblib.load(os.path.join(artifact_dir, f"encoder_run_{run_number}.pkl"))

    # Prepare Test Data
    test_idx = np.where(runs == "Run_2")[0]
    X_test, y_test, epochs_test = X[test_idx], y[test_idx], epoch_nums[test_idx]

    X_test_flat = X_test.reshape(len(X_test), -1)
    X_test_tf = transformer.transform(X_test_flat)
    X_test_cnn = X_test_tf.reshape(len(X_test_tf), X_test_tf.shape[1], 1)

    # Predict
    preds_prob = model.predict(X_test_cnn)
    preds_enc = np.argmax(preds_prob, axis=1)
    preds = le.inverse_transform(preds_enc)

    # Save detailed results
    results_df = pd.DataFrame(
        {"True_Label": y_test, "Predicted_Label": preds, "Epoch": epochs_test}
    )
    for i, class_label in enumerate(le.classes_):
        results_df[f"score_{class_label}"] = preds_prob[:, i]

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
    """Analyzes all temporary results using Dev/Test split per task."""
    logging.info("--- [ANALYSIS] Starting Final Analysis ---")

    df_map = pd.read_csv(mapping_csv_path)
    df_map["Epoch"] = df_map["New_Epoch"].str.extract(r"epoch_(\d+)").astype(int)
    df_map["Subject"] = df_map["Subject"].str.split("_", expand=True)[1]
    df_map_test = df_map[df_map["Run"] == "Run_2"]

    # Metrics storage
    task_metrics = {}

    for i in range(n_iterations):
        results_file = os.path.join(temp_results_dir, f"run_{i}_preds.csv")
        df_res = pd.read_csv(results_file)

        # Standardize types
        df_res["True_Label"] = df_res["True_Label"].astype(str)
        df_res["Predicted_Label"] = df_res["Predicted_Label"].astype(str)
        df_map_test["Subject"] = df_map_test["Subject"].astype(str)

        df_res.rename(columns={"True_Label": "Subject"}, inplace=True)
        df_merged = pd.merge(df_res, df_map_test, on=["Subject", "Epoch"], how="left")

        # Identify score columns
        score_cols = [c for c in df_res.columns if c.startswith("score_")]
        subjects_in_model = [c.replace("score_", "") for c in score_cols]

        for task, group in df_merged.groupby("Source_File"):
            if pd.isna(task): continue

            # 1. Accuracy
            acc = accuracy_score(group["Subject"], group["Predicted_Label"])
            
            # 2. Biometric Verification
            task_y = group["Subject"].values
            task_scores = group[score_cols].values
            
            bio_res = biometric_analysis_dev_test(task_y, task_scores, subjects_in_model)
            
            if task not in task_metrics:
                task_metrics[task] = {"acc": [], "eer": [], "frr_1": [], "frr_01": [], "det_data": None}
            
            task_metrics[task]["acc"].append(acc)
            
            if bio_res:
                task_metrics[task]["eer"].append(bio_res["eer"])
                task_metrics[task]["frr_1"].append(bio_res["frr_1pc"])
                task_metrics[task]["frr_01"].append(bio_res["frr_01pc"])
                if task_metrics[task]["det_data"] is None:
                    task_metrics[task]["det_data"] = bio_res

    # Aggregation & Reporting
    agg_results = []
    det_plot_dir = os.path.join(output_dir, "det_curves")
    os.makedirs(det_plot_dir, exist_ok=True)

    for task in sorted(task_metrics.keys()):
        m = task_metrics[task]
        row = {
            "Task": task,
            "Mean_Accuracy": np.mean(m["acc"]),
            "Std_Accuracy": np.std(m["acc"]),
            "Mean_EER (%)": np.mean(m["eer"]) * 100 if m["eer"] else np.nan,
            "Std_EER (%)": np.std(m["eer"]) * 100 if m["eer"] else np.nan,
            "Mean_FRR@1%": np.mean(m["frr_1"]) * 100 if m["frr_1"] else np.nan,
            "Mean_FRR@0.1%": np.mean(m["frr_01"]) * 100 if m["frr_01"] else np.nan
        }
        agg_results.append(row)
        
        # Plot DET
        if m["det_data"]:
            det = m["det_data"]
            plot_det_curve(
                det["fpr_curve"], det["fnr_curve"], det["eer"],
                os.path.join(det_plot_dir, f"DET_{task}.png"),
                title=f"DET Curve (1D-CNN): {task}"
            )

    df_perf = pd.DataFrame(agg_results).sort_values(by="Mean_Accuracy", ascending=False)
    output_path = os.path.join(output_dir, "task_performance_1dcnn_averaged.csv")
    df_perf.to_csv(output_path, index=False)

    logging.info(f"✅ Final aggregated report saved to '{output_path}'")
    print("\n--- Final Task-wise Performance Report ---")
    print(df_perf.to_string(index=False))

    # Plot Accuracy
    plt.figure(figsize=(12, 8))
    plt.bar(df_perf["Task"], df_perf["Mean_Accuracy"], yerr=df_perf["Std_Accuracy"], capsize=5, color="teal")
    plt.title(f"1D-CNN Mean Task Accuracy ({n_iterations} runs)")
    plt.ylabel("Mean Accuracy"); plt.xlabel("Task")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_performance_1dcnn_averaged.png"))
    plt.close()


# --- Main Orchestrator ---
def main():
    N_ITERATIONS = 10
    N_EPOCHS_TO_SAMPLE = 40

    HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
    MAPPING_CSV = "epoch_mapping.csv"

    OUTPUT_DIR = "results/cnn_restricted_analysis"
    ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "artifacts")
    TEMP_RESULTS_DIR = os.path.join(OUTPUT_DIR, "temp_predictions")
    FINAL_REPORT_DIR = os.path.join(OUTPUT_DIR, "final_report")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)
    os.makedirs(FINAL_REPORT_DIR, exist_ok=True)

    if not os.path.exists(HDF5_FILE):
        raise FileNotFoundError(f"{HDF5_FILE} not found!")

    logging.info(f"Loading data from {HDF5_FILE}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(HDF5_FILE)
    logging.info("Data loaded successfully.")

    # STAGE 1: TRAINING
    for i in range(N_ITERATIONS):
        run_training_for_iteration(i, X, y, runs, ARTIFACT_DIR, N_EPOCHS_TO_SAMPLE)

    # STAGE 2: PREDICTION
    for i in range(N_ITERATIONS):
        run_prediction_for_iteration(i, X, y, runs, epoch_nums, ARTIFACT_DIR, TEMP_RESULTS_DIR)

    # STAGE 3: ANALYSIS
    analyze_all_runs(N_ITERATIONS, TEMP_RESULTS_DIR, MAPPING_CSV, FINAL_REPORT_DIR)

    logging.info("--- FULL EXPERIMENT COMPLETE ---")


if __name__ == "__main__":
    main()