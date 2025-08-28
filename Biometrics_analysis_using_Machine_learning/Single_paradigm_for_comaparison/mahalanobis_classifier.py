#!/usr/bin/env python3
"""
Performs EEG-based subject identification using Mahalanobis distance.

This script processes one or more HDF5 files containing EEG features. For each
file, it splits the data into training (Run_1) and testing (Run_2) sets. It
builds a template (mean feature vector) and a covariance matrix for each subject
from the training data. It then classifies test samples by finding the subject
template with the minimum Mahalanobis distance.

The script outputs performance metrics (Accuracy, F1, EER), prediction CSVs,
and a summary file for all processed HDF5 files.
"""
import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.linalg import inv, pinv, cond
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, roc_curve, auc)
from sklearn.preprocessing import PowerTransformer, label_binarize

# Set matplotlib backend to Agg to prevent plots from showing interactively
plt.switch_backend('Agg')


def get_indices_for_n_epochs(y_data, n_epochs):
    """Selects indices to ensure a fixed number of epochs per subject."""
    subjects = np.unique(y_data)
    final_indices = []
    for subj in subjects:
        subj_indices = np.where(y_data == subj)[0]
        if len(subj_indices) < n_epochs:
            print(f"[WARNING] Subject {subj} has only {len(subj_indices)} samples, "
                  f"less than the required {n_epochs}. Using all available.")
            final_indices.extend(subj_indices)
        else:
            final_indices.extend(subj_indices[:n_epochs])
    return np.array(sorted(final_indices))

# =============================================================================
# DATA LOADING
# =============================================================================
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            for run_key in h5f[class_key].keys():
                for epoch_key, epoch_group in h5f[class_key][run_key].items():
                    X.append(epoch_group['features'][()])
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1])
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_confusion_matrix(cm, subject_list, output_path):
    """Saves a confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=subject_list, yticklabels=subject_list)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_far_frr(thresholds, far_array, frr_array, eer, output_path):
    """Saves FAR/FRR curves."""
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_array, label="FAR")
    plt.plot(thresholds, frr_array, label="FRR")
    tau_eer = thresholds[np.argmin(np.abs(np.array(far_array) - np.array(frr_array)))]
    plt.axvline(x=tau_eer, color='gray', linestyle='--', label=f"EER Threshold ({tau_eer:.4f})")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title(f"FAR and FRR Curves (EER = {eer*100:.2f}%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =============================================================================
# CORE ANALYSIS
# =============================================================================
def build_subject_covariances_flat(X_data, y_data, alpha=1e-6):
    """Builds templates and inverse covariance matrices for each subject."""
    subjects = np.unique(y_data)
    templates = {}
    cov_dict = {}
    for subj in subjects:
        subj_data = X_data[y_data == subj]
        template = np.mean(subj_data, axis=0)
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        try:
            inv_cov = inv(cov_matrix)
        except np.linalg.LinAlgError:
            print(f"[WARNING] Covariance matrix for subject {subj} is singular. Using pseudo-inverse.")
            inv_cov = pinv(cov_matrix)
        templates[subj] = template
        cov_dict[subj] = inv_cov
    return templates, cov_dict

def predict_mahalanobis_flat(sample, templates, cov_dict):
    """Predicts the subject for a sample using Mahalanobis distance."""
    best_subj, min_dist = None, float('inf')
    for subj, tmpl in templates.items():
        inv_cov = cov_dict[subj]
        dist = mahalanobis(sample, tmpl, inv_cov)
        if dist < min_dist:
            min_dist = dist
            best_subj = subj
    return best_subj

def compute_biometric_metrics(genuine_distances, imposter_distances, n_thresholds=1000):
    """Computes Equal Error Rate (EER) and FAR/FRR curves."""
    all_distances = np.concatenate([genuine_distances, imposter_distances])
    min_d, max_d = all_distances.min(), all_distances.max()
    thresholds = np.linspace(min_d, max_d, n_thresholds)
    far_array, frr_array = [], []
    for tau in thresholds:
        frr_array.append(np.mean(np.array(genuine_distances) > tau))
        far_array.append(np.mean(np.array(imposter_distances) < tau))

    far_array = np.array(far_array)
    frr_array = np.array(frr_array)
    idx_eer = np.argmin(np.abs(far_array - frr_array))
    eer = (far_array[idx_eer] + frr_array[idx_eer]) / 2.0
    return eer, thresholds, far_array, frr_array

def across_run_analysis(X, y, runs, epochs_per_subject=40, alpha=1e-4):
    """Main analysis pipeline splitting data by run."""
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("[ERROR] Insufficient data for Run_1/Run_2 split. Skipping analysis.")
        return None

    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test_full, y_test_full = X[test_idx], y[test_idx]

    train_keep_idx = get_indices_for_n_epochs(y_train_full, epochs_per_subject)
    X_train, y_train = X_train_full[train_keep_idx], y_train_full[train_keep_idx]

    test_keep_idx = get_indices_for_n_epochs(y_test_full, epochs_per_subject)
    X_test, y_test = X_test_full[test_keep_idx], y_test_full[test_keep_idx]

    # Ensure enough samples for covariance calculation
    for subj in np.unique(y_train):
        if np.sum(y_train == subj) < 2:
            print(f"[ERROR] Subject {subj} has fewer than 2 training samples after selection. Cannot compute covariance.")
            return None

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    templates, cov_dict = build_subject_covariances_flat(X_train_flat, y_train, alpha=alpha)

    predictions, genuine_distances, imposter_distances = [], [], []
    for i, sample in enumerate(X_test_flat):
        true_label = y_test[i]
        pred_label = predict_mahalanobis_flat(sample, templates, cov_dict)
        predictions.append(pred_label)

        # Calculate genuine and imposter distances for EER
        if true_label in templates:
            genuine_distances.append(mahalanobis(sample, templates[true_label], cov_dict[true_label]))
        for subj, tmpl in templates.items():
            if subj != true_label:
                imposter_distances.append(mahalanobis(sample, tmpl, cov_dict[subj]))

    subjects = sorted(templates.keys())
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=subjects)

    eer, thresholds, far, frr = (0, [], [], [])
    if genuine_distances and imposter_distances:
        eer, thresholds, far, frr = compute_biometric_metrics(genuine_distances, imposter_distances)

    return {"accuracy": accuracy, "f1": f1, "confusion_matrix": cm, "eer": eer,
            "thresholds": thresholds, "far_array": far, "frr_array": frr,
            "true_labels": y_test, "predictions": predictions, "subjects": subjects}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_analysis_on_file(h5_filepath, output_dir, epochs_per_subject):
    """Runs the entire analysis pipeline for a single HDF5 file."""
    base_name = os.path.splitext(os.path.basename(h5_filepath))[0]
    print(f"\n--- Processing file: {base_name} ---")

    try:
        X, y, runs, _ = load_features_from_hdf5(h5_filepath)
        if len(X) == 0:
            print(f"[ERROR] No data loaded from {base_name}. Skipping.")
            return None
    except Exception as e:
        print(f"[ERROR] Could not load data from {base_name}. Reason: {e}")
        return None

    # Preprocessing
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_transformed = transformer.fit_transform(X_flat)
    X = X_transformed.reshape(X.shape)

    # Analysis
    res = across_run_analysis(X, y, runs, epochs_per_subject=epochs_per_subject, alpha=1e-4)
    if res is None:
        print(f"[INFO] Analysis for {base_name} was skipped due to data issues.")
        return None

    # Reporting and Saving
    print(f"  Accuracy: {res['accuracy']:.4f}")
    print(f"  F1 Score (Weighted): {res['f1']:.4f}")
    print(f"  Equal Error Rate (EER): {res['eer']*100:.2f}%")

    # Create a dedicated subdirectory for this file's results
    file_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    # Save plots
    plot_confusion_matrix(res['confusion_matrix'], res['subjects'],
                          os.path.join(file_output_dir, "confusion_matrix.png"))
    if res['thresholds']:
        plot_far_frr(res['thresholds'], res['far_array'], res['frr_array'], res['eer'],
                     os.path.join(file_output_dir, "far_frr_curves.png"))

    # Save predictions CSV
    df_out = pd.DataFrame({"True_Label": res['true_labels'], "Predicted_Label": res['predictions']})
    df_out.to_csv(os.path.join(file_output_dir, "predictions.csv"), index=False)
    
    print(f"[SUCCESS] Results for {base_name} saved to '{file_output_dir}'")

    return {"filename": base_name, "accuracy": res['accuracy'], "f1_score": res['f1'], "eer": res['eer']}


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Mahalanobis Distance Classifier for EEG Biometrics")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the HDF5 data files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save analysis results.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to use per subject.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    file_pattern = os.path.join(args.data_dir, '*.h5')
    h5_files = glob.glob(file_pattern)

    if not h5_files:
        print(f"No .h5 files found in '{args.data_dir}'. Please check the path.")
        return

    print(f"Found {len(h5_files)} HDF5 files to process.")
    all_summaries = []
    for h5_file in h5_files:
        summary_data = run_analysis_on_file(h5_file, args.output_dir, args.epochs)
        if summary_data:
            all_summaries.append(summary_data)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_filepath = os.path.join(args.output_dir, '_summary_of_all_runs.csv')
        summary_df.to_csv(summary_filepath, index=False)
        print(f"\n{'='*50}\nAll processing complete.\nSummary report saved to: {summary_filepath}")
        print(summary_df)


if __name__ == "__main__":
    main()