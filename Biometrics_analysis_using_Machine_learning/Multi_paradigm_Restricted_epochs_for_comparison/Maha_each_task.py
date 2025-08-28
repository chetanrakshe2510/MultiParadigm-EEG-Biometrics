# -*- coding: utf-8 -*-
"""
Performs a biometric identification experiment using the Mahalanobis Distance.

This script automates a multi-iteration workflow to provide a robust evaluation
of a Mahalanobis Distance-based classifier. The key steps are:

1.  **Data Loading & Preprocessing**: Loads feature data from an HDF5 file
    and applies a PowerTransformer to normalize the feature distributions.

2.  **Iterative Analysis**: For a specified number of iterations:
    a. A random sample of training data (40 epochs per subject) is selected.
    b. Subject-specific templates (mean vectors) and inverse covariance
       matrices are calculated from the training sample.
    c. Predictions are made on the test set ('Run_2') by finding the subject
       template with the minimum Mahalanobis Distance for each test sample.
    d. Task-specific and overall metrics (Accuracy, EER) are computed for the run.

3.  **Aggregation & Reporting**: The metrics from all iterations are averaged to
    produce a final, stable performance report. The script generates:
    - A CSV file with mean/std of task-wise Accuracy and EER.
    - Plots from the final iteration (e.g., FAR/FRR curves, distance histograms).
    - An aggregated confusion matrix averaged across all runs.
"""
import os
import logging
from typing import Tuple, List, Dict, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import inv, pinv
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import PowerTransformer

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Data Loading ---
def load_features_from_hdf5(
    filename: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads features, labels, runs, and epoch numbers from the HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, "r") as h5f:
        for class_key in h5f.keys():
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


# --- Plotting Utilities ---
def plot_confusion_matrix(
    cm: np.ndarray, subject_list: List[str], save_path: str, title: str
):
    """Generates and saves a heatmap of the confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=subject_list,
        yticklabels=subject_list,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved confusion matrix to {save_path}")


# --- Core Biometric & Analysis Functions ---
def build_subject_models(
    X_data: np.ndarray, y_data: np.ndarray, alpha: float = 1e-6
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Builds templates (mean vectors) and inverse covariance matrices for each subject."""
    subjects = np.unique(y_data)
    templates, inv_cov_dict = {}, {}
    for subj in subjects:
        subj_data = X_data[y_data == subj]
        if subj_data.shape[0] < 2:
            logging.warning(f"Skipping subject {subj} due to insufficient samples (<2).")
            continue
        
        templates[subj] = np.mean(subj_data, axis=0)
        # Regularize covariance matrix to ensure it's invertible
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        try:
            inv_cov_dict[subj] = inv(cov_matrix)
        except np.linalg.LinAlgError:
            logging.warning(f"Covariance matrix for {subj} is singular; using pseudo-inverse.")
            inv_cov_dict[subj] = pinv(cov_matrix)
            
    return templates, inv_cov_dict


def predict_mahalanobis(
    sample: np.ndarray, templates: Dict[str, np.ndarray], inv_cov_dict: Dict[str, np.ndarray]
) -> str:
    """Predicts the subject for a sample based on minimum Mahalanobis distance."""
    best_subj, min_dist = "Unknown", float("inf")
    for subj, tmpl in templates.items():
        dist = mahalanobis(sample, tmpl, inv_cov_dict[subj])
        if dist < min_dist:
            min_dist, best_subj = dist, subj
    return best_subj


def compute_biometric_metrics(
    genuine_distances: List[float], imposter_distances: List[float]
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes EER, FAR, and FRR from genuine and imposter distance scores."""
    if not genuine_distances or not imposter_distances:
        return 0.5, np.array([0]), np.array([0]), np.array([0]) # Default EER if no scores

    genuine, imposter = np.array(genuine_distances), np.array(imposter_distances)
    min_d = min(genuine.min(), imposter.min())
    max_d = max(genuine.max(), imposter.max())
    thresholds = np.linspace(min_d, max_d, num=1000)
    
    far = np.array([np.mean(imposter < t) for t in thresholds])
    frr = np.array([np.mean(genuine > t) for t in thresholds])
    
    eer_idx = np.argmin(np.abs(far - frr))
    eer = (far[eer_idx] + frr[eer_idx]) / 2.0
    
    return eer, thresholds, far, frr


def perform_single_run_analysis(
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    epoch_nums: np.ndarray,
    mapping_csv: str,
    n_epochs_to_sample: int,
    random_state: int,
) -> Dict[str, Any]:
    """Performs a full analysis for one random sample of training data."""
    # 1. Create train/test split based on 'Run' and sample training data
    df = pd.DataFrame({"subject": y, "run": runs, "original_index": np.arange(len(y))})
    df_run1 = df[df["run"] == "Run_1"]
    train_sampled_df = df_run1.groupby("subject", group_keys=False).apply(
        lambda grp: grp.sample(n=min(len(grp), n_epochs_to_sample), random_state=random_state)
    )
    train_idx = train_sampled_df["original_index"].values
    test_idx = df[df["run"] == "Run_2"]["original_index"].values

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test, epoch_nums_test = X[test_idx], y[test_idx], epoch_nums[test_idx]

    # 2. Build subject models from the training sample
    templates, inv_cov_dict = build_subject_models(X_train, y_train)
    subjects = sorted(templates.keys())

    # 3. Make predictions and collect genuine/imposter scores
    predictions, all_genuine, all_imposter = [], [], []
    for i, sample in enumerate(X_test):
        true_label = y_test[i]
        pred_label = predict_mahalanobis(sample, templates, inv_cov_dict)
        predictions.append(pred_label)

        # Collect scores only if the true subject model exists
        if true_label in templates:
            all_genuine.append(mahalanobis(sample, templates[true_label], inv_cov_dict[true_label]))
            for subj, tmpl in templates.items():
                if subj != true_label:
                    all_imposter.append(mahalanobis(sample, tmpl, inv_cov_dict[subj]))

    # 4. Calculate per-task metrics
    df_res = pd.DataFrame({"Subject": y_test, "Epoch": epoch_nums_test, "Prediction": predictions})
    df_map = pd.read_csv(mapping_csv)
    df_map["Epoch"] = df_map["New_Epoch"].str.extract(r"epoch_(\d+)").astype(int)
    df_map["Subject"] = df_map["Subject"].str.split("_", expand=True)[1]
    df_merged = pd.merge(df_res, df_map[df_map["Run"] == "Run_2"], on=["Subject", "Epoch"], how="left")

    task_metrics = {}
    for task, group in df_merged.groupby("Source_File"):
        if pd.isna(task): continue
        
        task_genuine, task_imposter = [], []
        task_y_true = group["Subject"].values
        task_indices_in_test = group.index.values

        for i, row_idx in enumerate(task_indices_in_test):
            sample, true_label = X_test[row_idx], task_y_true[i]
            if true_label in templates:
                task_genuine.append(mahalanobis(sample, templates[true_label], inv_cov_dict[true_label]))
                for subj, tmpl in templates.items():
                    if subj != true_label:
                        task_imposter.append(mahalanobis(sample, tmpl, inv_cov_dict[subj]))
        
        task_eer, _, _, _ = compute_biometric_metrics(task_genuine, task_imposter)
        task_acc = accuracy_score(task_y_true, group["Prediction"].values)
        task_metrics[task] = {"accuracy": task_acc, "eer": task_eer}

    # 5. Calculate overall metrics for the run
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=subjects)
    eer, thresholds, far, frr = compute_biometric_metrics(all_genuine, all_imposter)

    return {
        "accuracy": accuracy, "f1": f1, "confusion_matrix": cm, "eer": eer,
        "subjects": subjects, "task_metrics": task_metrics,
        # Pass through last run's detailed results for plotting
        "last_run_details": {
            "genuine_distances": all_genuine, "imposter_distances": all_imposter,
            "thresholds": thresholds, "far_array": far, "frr_array": frr,
        }
    }


# --- Main Orchestrator ---
def main():
    """Main function to orchestrate the entire analysis workflow."""
    
    # --- Configuration ---
    # Key parameters for the experiment.
    N_ITERATIONS = 10
    N_EPOCHS_TO_SAMPLE = 40
    
    HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
    MAPPING_CSV = "epoch_mapping.csv"
    
    # All outputs will be saved in this single directory.
    OUTPUT_DIR = "results/mahalanobis_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Load and Preprocess Data ---
    logging.info(f"Loading data from {HDF5_FILE}...")
    X_orig, y, runs, epoch_nums = load_features_from_hdf5(HDF5_FILE)
    
    # Apply PowerTransformer once to the entire flattened dataset
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X_orig.reshape(X_orig.shape[0], -1)
    X_transformed = transformer.fit_transform(X_flat)
    logging.info("Data loaded and transformed successfully.")

    # --- 2. Run Multiple Iterations ---
    all_accuracies, all_f1s, all_cms, all_eers = [], [], [], []
    task_results = {}
    last_run_details = None

    for i in range(N_ITERATIONS):
        logging.info(f"--- Starting Iteration {i+1}/{N_ITERATIONS} ---")
        results = perform_single_run_analysis(
            X_transformed, y, runs, epoch_nums, MAPPING_CSV,
            N_EPOCHS_TO_SAMPLE, random_state=i
        )
        
        all_accuracies.append(results['accuracy'])
        all_f1s.append(results['f1'])
        all_cms.append(results['confusion_matrix'])
        all_eers.append(results['eer'])
        
        for task, metrics in results['task_metrics'].items():
            task_results.setdefault(task, {'accuracies': [], 'eers': []})
            task_results[task]['accuracies'].append(metrics['accuracy'])
            task_results[task]['eers'].append(metrics['eer'])

        if i == N_ITERATIONS - 1:
            last_run_details = results
    
    # --- 3. Aggregate, Report, and Plot Results ---
    logging.info("Aggregating results and generating final reports...")
    
    # Aggregate and save task-wise performance
    agg_task_perf = []
    for task, data in task_results.items():
        agg_task_perf.append({
            'Task': task,
            'Mean_Accuracy': np.mean(data['accuracies']),
            'Std_Accuracy': np.std(data['accuracies']),
            'Mean_EER (%)': np.mean(data['eers']) * 100,
            'Std_EER (%)': np.std(data['eers']) * 100
        })
    df_perf = pd.DataFrame(agg_task_perf).sort_values(by='Mean_Accuracy', ascending=False)
    
    # Save overall summary
    summary = {
        "Mean Accuracy": np.mean(all_accuracies), "Std Accuracy": np.std(all_accuracies),
        "Mean F1 Score": np.mean(all_f1s), "Std F1 Score": np.std(all_f1s),
        "Mean Overall EER (%)": np.mean(all_eers) * 100, "Std Overall EER (%)": np.std(all_eers) * 100
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "overall_performance_summary.csv"), index=False)

    print("\n--- Overall Performance Summary ---")
    print(df_summary.to_string(index=False))
    
    # Save task performance
    df_perf.to_csv(os.path.join(OUTPUT_DIR, "task_performance_averaged.csv"), index=False)
    print("\n--- Average Task Performance ---")
    print(df_perf.to_string(index=False))

    # Plot mean confusion matrix
    mean_cm = np.mean(all_cms, axis=0)
    plot_confusion_matrix(
        mean_cm,
        last_run_details['subjects'],
        save_path=os.path.join(OUTPUT_DIR, "mean_confusion_matrix.png"),
        title=f"Mean Confusion Matrix ({N_ITERATIONS} Runs)"
    )
    
    # Plot last run's FAR/FRR curves
    details = last_run_details['last_run_details']
    plt.figure(figsize=(8, 6))
    plt.plot(details['thresholds'], details['far_array'], label="FAR")
    plt.plot(details['thresholds'], details['frr_array'], label="FRR")
    plt.xlabel("Decision Threshold"); plt.ylabel("Error Rate")
    plt.title(f"FAR & FRR Curves (Last Run) - EER = {last_run_details['eer']*100:.2f}%")
    plt.legend(); plt.grid(True, linestyle='--'); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "far_frr_last_run.png")); plt.close()
    logging.info("Saved FAR/FRR plot for the last run.")

    logging.info(f"✅ Analysis complete. All outputs saved to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()