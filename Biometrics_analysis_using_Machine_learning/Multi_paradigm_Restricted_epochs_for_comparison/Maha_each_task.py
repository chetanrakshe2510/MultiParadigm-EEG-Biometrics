# -*- coding: utf-8 -*-
"""
Evaluates Mahalanobis Distance classifier with RESTRICTED EPOCHS and TASK-SPECIFIC analysis.

UPDATED for Uniform Analysis:
- Fixes Data Leakage (Fit Scaler on Train Only).
- Implements Open-Set style Dev/Test split for threshold selection per task.
- Reports EER, FRR @ 1% FAR, and FRR @ 0.1% FAR with Mean/Std across iterations.
- Generates DET Curves for each task.
"""
import os
import logging
import random
from typing import List, Dict, Any, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import inv, pinv
from scipy.spatial.distance import mahalanobis
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PowerTransformer

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Data Loading & Plotting Utilities ---
def load_features_from_hdf5(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def plot_confusion_matrix(cm: np.ndarray, subject_list: List[str], save_path: str, title: str):
    """Generates and saves a heatmap of the confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=subject_list, yticklabels=subject_list,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    """
    Generates genuine and impostor score/label pairs.
    For Mahalanobis, input scores must already be NEGATIVE distances (higher = better).
    """
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        true_idx = subject_to_idx[true_subj]
        
        # Genuine
        scores.append(scores_matrix[i, true_idx])
        labels.append(1)
        
        # Impostor (Random sampling)
        all_impostors = [s for s in subject_list if s != true_subj]
        n_samples = min(impostor_samples, len(all_impostors))
        sampled_impostors = random.sample(all_impostors, n_samples)
        
        for imp_subj in sampled_impostors:
            imp_idx = subject_to_idx[imp_subj]
            scores.append(scores_matrix[i, imp_idx])
            labels.append(0)

    return np.array(scores), np.array(labels)


def biometric_analysis_dev_test(y_test, scores_matrix, all_subjects, dev_subjects_count=10):
    """
    Splits task data into Dev (Thresholding) and Test (Evaluation).
    Returns EER, FRR@1%, FRR@0.1%, and curve data.
    """
    test_subjects = np.unique(y_test)
    
    current_dev_count = dev_subjects_count
    if len(test_subjects) <= current_dev_count:
        current_dev_count = len(test_subjects) // 2
    
    # Random Split
    test_subjects_list = list(test_subjects)
    random.shuffle(test_subjects_list)
    
    dev_subjects = test_subjects_list[:current_dev_count]
    final_test_subjects = test_subjects_list[current_dev_count:]
    
    dev_mask = np.isin(y_test, dev_subjects)
    test_mask = np.isin(y_test, final_test_subjects)
    
    if not np.any(dev_mask) or not np.any(test_mask):
        return None 

    # --- Dev Phase (Threshold Selection) ---
    y_dev = y_test[dev_mask]
    scores_dev = scores_matrix[dev_mask]
    dev_scores_flat, dev_labels_flat = generate_biometric_trials(y_dev, scores_dev, all_subjects)
    
    fpr, tpr, thresholds = roc_curve(dev_labels_flat, dev_scores_flat, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    # Interpolate for FAR thresholds
    fpr_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
    thresh_far_1_pc = fpr_interp(0.01)
    thresh_far_01_pc = fpr_interp(0.001)

    # --- Test Phase (Evaluation) ---
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


# --- Core Mahalanobis Logic ---
def build_subject_models(X_data, y_data, alpha=1e-6):
    """Builds templates and inverse covariance matrices."""
    subjects = np.unique(y_data)
    templates, inv_cov_dict = {}, {}
    for subj in subjects:
        subj_data = X_data[y_data == subj]
        # Regularization
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        
        try:
            inv_c = inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_c = pinv(cov_matrix)
            
        templates[subj] = np.mean(subj_data, axis=0)
        inv_cov_dict[subj] = inv_c
    return templates, inv_cov_dict

def predict_mahalanobis_scores(sample, templates, inv_cov_dict, sorted_subjects):
    """
    Returns (predicted_label, scores_vector).
    Score = NEGATIVE Mahalanobis Distance (Higher = Better).
    """
    scores = {}
    for subj in sorted_subjects:
        dist = mahalanobis(sample, templates[subj], inv_cov_dict[subj])
        scores[subj] = -dist  # Convert distance to similarity score
        
    best_subj = max(scores, key=scores.get)
    score_vector = [scores[s] for s in sorted_subjects]
    return best_subj, score_vector


# --- Analysis Functions ---
def run_single_iteration(
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    epoch_nums: np.ndarray,
    n_epochs_to_sample: int,
    random_state: int,
    alpha: float = 1e-4
) -> Dict[str, Any]:
    """Trains (Run 1 subset) and Evaluates (Run 2) one iteration."""
    
    # 1. Sample Training Data (Run 1)
    df = pd.DataFrame({"subject": y, "run": runs, "original_index": np.arange(len(y))})
    df_run1 = df[df["run"] == "Run_1"]
    
    train_sampled_df = df_run1.groupby("subject", group_keys=False).apply(
        lambda grp: grp.sample(n=min(len(grp), n_epochs_to_sample), random_state=random_state)
    )
    train_idx = train_sampled_df["original_index"].values
    test_idx = df[df["run"] == "Run_2"]["original_index"].values

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test, epoch_nums_test = X[test_idx], y[test_idx], epoch_nums[test_idx]

    # 2. Preprocess (Fit on Train, Transform Test) - NO LEAKAGE
    transformer = PowerTransformer(method="yeo-johnson")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    
    transformer.fit(X_train_flat) # Fit only on train
    X_train_tf = transformer.transform(X_train_flat)
    X_test_tf = transformer.transform(X_test_flat)

    # 3. Train Models
    templates, inv_cov_dict = build_subject_models(X_train_tf, y_train, alpha)
    subjects_sorted = sorted(templates.keys())

    # 4. Predict & Score
    predictions, scores_matrix = [], []
    for sample in X_test_tf:
        pred, scores = predict_mahalanobis_scores(sample, templates, inv_cov_dict, subjects_sorted)
        predictions.append(pred)
        scores_matrix.append(scores)
    
    scores_matrix = np.array(scores_matrix)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=subjects_sorted)

    return {
        "accuracy": accuracy, "f1": f1, "confusion_matrix": cm,
        "subjects": subjects_sorted, "predictions": predictions,
        "true_labels": y_test, "epoch_nums_test": epoch_nums_test,
        "scores_matrix": scores_matrix, "fitted_classes": subjects_sorted
    }


def analyze_task_performance(
    all_results: Dict[str, Any], mapping_csv: str, output_dir: str
):
    """Aggregates results and calculates Task-Specific Dev/Test Metrics."""
    df_map = pd.read_csv(mapping_csv)
    df_map["epoch"] = df_map["New_Epoch"].str.extract(r"epoch_(\d+)").astype(int)
    df_map["subject_code"] = df_map["Subject"].str.split("_", expand=True)[1]
    df_map_test = df_map[df_map["Run"] == "Run_2"]

    task_metrics = {} 
    n_iterations = len(all_results["all_predictions"])

    for i in range(n_iterations):
        # Build DataFrame for this iteration
        df_res = pd.DataFrame({
            "subject_code": all_results["all_true_labels"][i],
            "epoch": all_results["all_epoch_nums_test"][i],
            "prediction": all_results["all_predictions"][i],
        })
        
        # Add probability scores
        fitted_classes = all_results["all_fitted_classes"][i]
        score_df = pd.DataFrame(all_results["all_scores"][i], columns=fitted_classes)
        df_res = pd.concat([df_res.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
        
        # Merge with Task Info
        df_merged = pd.merge(df_res, df_map_test, on=["subject_code", "epoch"], how="left")
        
        # Iterate Tasks
        for task, group in df_merged.groupby("Source_File"):
            if pd.isna(task): continue
            
            # 1. Accuracy
            acc = accuracy_score(group["subject_code"], group["prediction"])
            
            # 2. Biometric Verification (Dev/Test Split)
            task_y = group["subject_code"].values
            task_scores = group[fitted_classes].values # Select columns for all subjects
            
            bio_res = biometric_analysis_dev_test(task_y, task_scores, fitted_classes)
            
            if task not in task_metrics:
                task_metrics[task] = {"acc": [], "eer": [], "frr_1": [], "frr_01": [], "det_data": None}
            
            task_metrics[task]["acc"].append(acc)
            
            if bio_res:
                task_metrics[task]["eer"].append(bio_res["eer"])
                task_metrics[task]["frr_1"].append(bio_res["frr_1pc"])
                task_metrics[task]["frr_01"].append(bio_res["frr_01pc"])
                # Save DET data from first valid iteration
                if task_metrics[task]["det_data"] is None:
                    task_metrics[task]["det_data"] = bio_res

    # --- Aggregation & Reporting ---
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
                title=f"DET Curve (MDTM): {task}"
            )
            
    df_perf = pd.DataFrame(agg_results).sort_values(by="Mean_Accuracy", ascending=False)
    df_perf.to_csv(os.path.join(output_dir, "task_performance_metrics.csv"), index=False)
    
    # Bar Plot for Accuracy
    plt.figure(figsize=(12, 8))
    plt.bar(df_perf["Task"], df_perf["Mean_Accuracy"], yerr=df_perf["Std_Accuracy"], capsize=5)
    plt.xlabel("Task"); plt.ylabel("Mean Accuracy")
    plt.title("Mean Task-Specific Accuracy (MDTM)")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_accuracy_plot.png")); plt.close()
    
    logging.info(f"Saved aggregated metrics and plots to {output_dir}")


# --- Main Orchestrator ---
def main():
    # --- Configuration ---
    N_ITERATIONS = 10
    N_EPOCHS_TO_SAMPLE = 40
    
    HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
    MAPPING_CSV = "epoch_mapping.csv"
    
    OUTPUT_DIR = "results/mahalanobis_restricted_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(HDF5_FILE):
        raise FileNotFoundError(f"{HDF5_FILE} not found!")

    logging.info(f"Loading data from {HDF5_FILE}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(HDF5_FILE)
    logging.info("Data loaded.")

    all_results = {
        "all_predictions": [], "all_true_labels": [], "all_epoch_nums_test": [],
        "all_scores": [], "all_fitted_classes": [], "all_accuracies": [],
        "all_f1_scores": [], "all_cms": []
    }
    
    for i in range(N_ITERATIONS):
        logging.info(f"Iteration {i+1}/{N_ITERATIONS}...")
        results = run_single_iteration(
            X, y, runs, epoch_nums, N_EPOCHS_TO_SAMPLE, random_state=i, alpha=1e-4
        )
        all_results["all_accuracies"].append(results["accuracy"])
        all_results["all_f1_scores"].append(results["f1"])
        all_results["all_cms"].append(results["confusion_matrix"])
        all_results["all_predictions"].append(results["predictions"])
        all_results["all_true_labels"].append(results["true_labels"])
        all_results["all_epoch_nums_test"].append(results["epoch_nums_test"])
        all_results["all_scores"].append(results["scores_matrix"])
        all_results["all_fitted_classes"].append(results["fitted_classes"])
    
    # Analyze Task Performance
    analyze_task_performance(all_results, MAPPING_CSV, OUTPUT_DIR)
    
    # Plot Mean CM
    mean_cm = np.mean(all_results["all_cms"], axis=0)
    plot_confusion_matrix(
        mean_cm, results["subjects"],
        save_path=os.path.join(OUTPUT_DIR, "mean_confusion_matrix.png"),
        title="Mean Confusion Matrix (MDTM)"
    )
    
    # Overall Summary
    summary = {
        "Classifier": "MDTM",
        "Mean Accuracy": np.mean(all_results["all_accuracies"]),
        "Std Accuracy": np.std(all_results["all_accuracies"]),
        "Mean F1": np.mean(all_results["all_f1_scores"]),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "overall_summary.csv"), index=False)
    
    logging.info(f"Analysis complete. Results in '{OUTPUT_DIR}'.")
    print("\n--- Overall Summary ---")
    print(pd.DataFrame([summary]))

if __name__ == "__main__":
    main()