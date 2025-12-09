# -*- coding: utf-8 -*-
"""
Evaluates and compares classical ML classifiers for biometric identification
with RESTRICTED EPOCHS (e.g., 40 epochs) and TASK-SPECIFIC analysis.

UPDATED for Uniform Analysis:
- Implements Open-Set style Dev/Test split for threshold selection per task.
- Reports EER, FRR @ 1% FAR, and FRR @ 0.1% FAR with Mean/Std across iterations.
- Generates DET Curves for each task.
"""
import os
import logging
import random
from typing import List, Dict, Any, Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

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
    """Generates genuine and impostor scores."""
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        true_idx = subject_to_idx[true_subj]
        
        # Genuine
        scores.append(scores_matrix[i, true_idx])
        labels.append(1)
        
        # Impostor (Random sampling to keep balanced)
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
    
    # Handle cases with very few subjects (e.g. specialized tasks)
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

    # --- Dev Phase ---
    y_dev = y_test[dev_mask]
    scores_dev = scores_matrix[dev_mask]
    dev_scores_flat, dev_labels_flat = generate_biometric_trials(y_dev, scores_dev, all_subjects)
    
    fpr, tpr, thresholds = roc_curve(dev_labels_flat, dev_scores_flat, pos_label=1)
    fnr = 1 - tpr
    
    # EER Threshold
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    # FAR Thresholds
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


# --- Core Analysis Functions ---
def run_single_iteration(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    epoch_nums: np.ndarray,
    n_epochs_to_sample: int,
    random_state: int,
) -> Dict[str, Any]:
    """Trains and evaluates a classifier on one random sample of data."""
    
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

    # 3. Train
    classifier.fit(X_train_tf, y_train)
    
    # 4. Predict
    preds = classifier.predict(X_test_tf)
    probs = classifier.predict_proba(X_test_tf)
    
    subjects_unique = np.unique(y)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=subjects_unique)

    return {
        "accuracy": accuracy, "f1": f1, "confusion_matrix": cm,
        "subjects": subjects_unique, "predictions": preds,
        "true_labels": y_test, "epoch_nums_test": epoch_nums_test,
        "scores_matrix": probs, "fitted_classifier_classes": classifier.classes_,
    }


def analyze_task_performance(
    all_results: Dict[str, Any], mapping_csv: str, output_dir: str, classifier_name: str
):
    """Aggregates results and calculates Task-Specific Dev/Test Metrics."""
    df_map = pd.read_csv(mapping_csv)
    df_map["epoch"] = df_map["New_Epoch"].str.extract(r"epoch_(\d+)").astype(int)
    df_map["subject_code"] = df_map["Subject"].str.split("_", expand=True)[1]
    df_map_test = df_map[df_map["Run"] == "Run_2"]

    # Storage for metrics across iterations
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
        
        # Ensure we have all subjects for correct probability mapping
        all_subs = fitted_classes

        # Iterate Tasks
        for task, group in df_merged.groupby("Source_File"):
            if pd.isna(task): continue
            
            # 1. Accuracy
            acc = accuracy_score(group["subject_code"], group["prediction"])
            
            # 2. Biometric Verification (Dev/Test Split)
            # Reconstruct score matrix for this task group
            task_y = group["subject_code"].values
            
            # Map columns to match `all_subs` order
            task_scores = group[all_subs].values
            
            bio_res = biometric_analysis_dev_test(task_y, task_scores, all_subs)
            
            if task not in task_metrics:
                task_metrics[task] = {"acc": [], "eer": [], "frr_1": [], "frr_01": [], "det_data": None}
            
            task_metrics[task]["acc"].append(acc)
            
            if bio_res:
                task_metrics[task]["eer"].append(bio_res["eer"])
                task_metrics[task]["frr_1"].append(bio_res["frr_1pc"])
                task_metrics[task]["frr_01"].append(bio_res["frr_01pc"])
                
                # Save DET data from the first valid iteration for plotting
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
        
        # Plot DET for this task (using data from first run)
        if m["det_data"]:
            det = m["det_data"]
            plot_det_curve(
                det["fpr_curve"], det["fnr_curve"], det["eer"],
                os.path.join(det_plot_dir, f"DET_{task}.png"),
                title=f"DET Curve: {task}\n({classifier_name})"
            )
            
    df_perf = pd.DataFrame(agg_results).sort_values(by="Mean_Accuracy", ascending=False)
    df_perf.to_csv(os.path.join(output_dir, "task_performance_metrics.csv"), index=False)
    logging.info(f"Saved task performance metrics to {output_dir}")

    # Bar Plot for Accuracy
    plt.figure(figsize=(12, 8))
    plt.bar(df_perf["Task"], df_perf["Mean_Accuracy"], yerr=df_perf["Std_Accuracy"], capsize=5)
    plt.xlabel("Task"); plt.ylabel("Mean Accuracy")
    plt.title(f"Mean Task-Specific Accuracy: {classifier_name}")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_accuracy_plot.png")); plt.close()


# --- Main Orchestrator ---
def main():
    # --- Configuration ---
    N_ITERATIONS = 10
    N_EPOCHS_TO_SAMPLE = 40
    
    HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
    MAPPING_CSV = "epoch_mapping.csv"
    
    OUTPUT_DIR = "results/ml_restricted_epochs_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classifiers = {
        "LogisticRegression": LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_features='sqrt', criterion='gini'),
        "SVM": SVC(C=0.1, kernel='linear', gamma='scale', probability=True)
    }

    if not os.path.exists(HDF5_FILE):
        raise FileNotFoundError(f"{HDF5_FILE} not found!")

    logging.info(f"Loading data from {HDF5_FILE}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(HDF5_FILE)
    logging.info("Data loaded successfully.")

    comparison_summary = []
    
    for name, model in classifiers.items():
        logging.info(f"--- Running analysis for {name} ---")
        clf_output_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(clf_output_dir, exist_ok=True)
        
        all_results = {
            "all_predictions": [], "all_true_labels": [], "all_epoch_nums_test": [],
            "all_scores": [], "all_fitted_classes": [], "all_accuracies": [],
            "all_f1_scores": [], "all_cms": []
        }
        
        for i in range(N_ITERATIONS):
            logging.info(f"  Iteration {i+1}/{N_ITERATIONS}...")
            results = run_single_iteration(
                model, X, y, runs, epoch_nums, N_EPOCHS_TO_SAMPLE, random_state=i
            )
            all_results["all_accuracies"].append(results["accuracy"])
            all_results["all_f1_scores"].append(results["f1"])
            all_results["all_cms"].append(results["confusion_matrix"])
            all_results["all_predictions"].append(results["predictions"])
            all_results["all_true_labels"].append(results["true_labels"])
            all_results["all_epoch_nums_test"].append(results["epoch_nums_test"])
            all_results["all_scores"].append(results["scores_matrix"])
            all_results["all_fitted_classes"].append(results["fitted_classifier_classes"])
        
        # Analyze and Save
        analyze_task_performance(all_results, MAPPING_CSV, clf_output_dir, name)
        
        # Mean CM
        mean_cm = np.mean(all_results["all_cms"], axis=0)
        plot_confusion_matrix(
            mean_cm, results["subjects"],
            save_path=os.path.join(clf_output_dir, "mean_confusion_matrix.png"),
            title=f"{name} Mean Confusion Matrix"
        )
        
        comparison_summary.append({
            "Classifier": name,
            "Mean Accuracy": np.mean(all_results["all_accuracies"]),
            "Std Accuracy": np.std(all_results["all_accuracies"]),
        })

    df_comp = pd.DataFrame(comparison_summary)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, "overall_summary.csv"), index=False)
    
    logging.info(f"Analysis complete. Results in '{OUTPUT_DIR}'.")
    print(df_comp)

if __name__ == "__main__":
    main()