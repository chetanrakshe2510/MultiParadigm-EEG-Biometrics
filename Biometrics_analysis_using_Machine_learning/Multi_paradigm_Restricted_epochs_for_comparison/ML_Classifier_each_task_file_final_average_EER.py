# -*- coding: utf-8 -*-
"""
Evaluates and compares classical ML classifiers for biometric identification.

This script performs a comprehensive, multi-iteration experiment to assess the
performance of several standard machine learning models (Logistic Regression,
Random Forest, SVM) on a task-specific basis.

The workflow is as follows:
1.  **Data Loading**: Loads feature data from an HDF5 file.
2.  **Iterative Training & Evaluation**: For a specified number of iterations:
    a. A random sample of training data (e.g., 40 epochs per subject) is
       selected from 'Run_1' to ensure robust evaluation.
    b. Each classifier is trained on the sampled data after applying a
       PowerTransformer for feature normalization.
    c. The trained model makes predictions on the test set ('Run_2').
    d. Results (predictions, scores, labels) are collected for each iteration.
3.  **Task-Specific Analysis**: The results from all iterations are aggregated.
    For each unique task defined in the mapping file, the script calculates the
    mean and standard deviation of both Accuracy and Equal Error Rate (EER).
4.  **Reporting**: The script generates a main output directory containing:
    - A summary CSV comparing the overall performance of the classifiers.
    - Sub-directories for each classifier with detailed task-performance CSVs,
      mean confusion matrix plots, and task-accuracy bar charts.
"""
import os
import logging
from typing import List, Dict, Any, Tuple

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Data Loading & Plotting Utilities ---
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


# --- Core Analysis Functions ---
def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculates the Equal Error Rate (EER) from true labels and scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_index])


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
    # 1. Create train/test split based on 'Run' and sample training data
    df = pd.DataFrame({"subject": y, "run": runs, "original_index": np.arange(len(y))})
    df_run1 = df[df["run"] == "Run_1"]
    train_sampled_df = df_run1.groupby("subject", group_keys=False).apply(
        lambda grp: grp.sample(n=min(len(grp), n_epochs_to_sample), random_state=random_state)
    )
    train_idx = train_sampled_df["original_index"].values
    test_idx = df[df["run"] == "Run_2"]["original_index"].values

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for training or testing.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test, epoch_nums_test = X[test_idx], y[test_idx], epoch_nums[test_idx]

    # 2. Preprocess data
    transformer = PowerTransformer(method="yeo-johnson")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    X_train_tf = transformer.fit_transform(X_train_flat)
    X_test_tf = transformer.transform(X_test_flat)

    # 3. Train classifier and get predictions/scores
    classifier.fit(X_train_tf, y_train)
    preds = classifier.predict(X_test_tf)
    probs = classifier.predict_proba(X_test_tf)
    
    # 4. Calculate overall metrics
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
    """Aggregates results from all runs to compute task-wise performance."""
    df_map = pd.read_csv(mapping_csv)
    df_map["epoch"] = df_map["New_Epoch"].str.extract(r"epoch_(\d+)").astype(int)
    df_map["subject_code"] = df_map["Subject"].str.split("_", expand=True)[1]
    df_map_test = df_map[df_map["Run"] == "Run_2"]

    task_accuracies, task_eers = {}, {}
    n_iterations = len(all_results["all_predictions"])

    for i in range(n_iterations):
        # Create a DataFrame with results for the current iteration
        df_res = pd.DataFrame({
            "subject_code": all_results["all_true_labels"][i],
            "epoch": all_results["all_epoch_nums_test"][i],
            "prediction": all_results["all_predictions"][i],
        })
        
        # Add probability scores, ensuring columns match the classifier's output
        fitted_classes = all_results["all_fitted_classes"][i]
        score_df = pd.DataFrame(all_results["all_scores"][i], columns=fitted_classes)
        df_res = pd.concat([df_res.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
        
        # Merge with task mapping data
        df_merged = pd.merge(df_res, df_map_test, on=["subject_code", "epoch"], how="left")

        for task, group in df_merged.groupby("Source_File"):
            if pd.isna(task): continue
            
            # 1. Calculate Accuracy for the task
            acc = accuracy_score(group["subject_code"], group["prediction"])
            task_accuracies.setdefault(task, []).append(acc)

            # 2. Calculate EER for the task
            group_true_labels = group["subject_code"].values
            subjects_in_task = np.unique(group_true_labels)
            subject_eers = []
            
            for subj in subjects_in_task:
                if subj not in fitted_classes: continue # Skip if subject wasn't in training
                
                y_true_ovr = (group_true_labels == subj).astype(int)
                y_score_ovr = group[subj].values
                
                if np.unique(y_true_ovr).size > 1: # Need both positive and negative samples
                    subject_eers.append(calculate_eer(y_true_ovr, y_score_ovr))
            
            if subject_eers:
                task_eers.setdefault(task, []).append(np.mean(subject_eers))
                
    # Aggregate and save the final report
    agg_results = []
    for task in sorted(task_accuracies.keys()):
        agg_results.append({
            "Task": task,
            "Mean_Accuracy": np.mean(task_accuracies.get(task, [np.nan])),
            "Std_Accuracy": np.std(task_accuracies.get(task, [np.nan])),
            "Mean_EER (%)": np.mean(task_eers.get(task, [np.nan])) * 100,
            "Std_EER (%)": np.std(task_eers.get(task, [np.nan])) * 100,
        })
    
    df_perf = pd.DataFrame(agg_results).sort_values(by="Mean_Accuracy", ascending=False)
    df_perf.to_csv(os.path.join(output_dir, "task_performance_averaged.csv"), index=False)
    logging.info(f"Saved task performance for {classifier_name} to {output_dir}")

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.bar(df_perf["Task"], df_perf["Mean_Accuracy"], yerr=df_perf["Std_Accuracy"], capsize=5)
    plt.xlabel("Task"); plt.ylabel("Mean Accuracy")
    plt.title(f"Mean Task-Specific Accuracy for {classifier_name} ({n_iterations} runs)")
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_performance_plot.png")); plt.close()


# --- Main Orchestrator ---
def main():
    """Main function to orchestrate the entire analysis workflow."""
    
    # --- Configuration ---
    N_ITERATIONS = 10
    N_EPOCHS_TO_SAMPLE = 40
    
    HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
    MAPPING_CSV = "epoch_mapping.csv"
    
    # All outputs will be saved in this single parent directory
    OUTPUT_DIR = "results/ml_classifiers_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Define Classifiers ---
    classifiers = {
        "LogisticRegression": LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_features='sqrt', criterion='gini'),
        "SVM": SVC(C=0.1, kernel='linear', gamma='scale', probability=True)
    }

    # --- Load Data Once ---
    logging.info(f"Loading data from {HDF5_FILE}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(HDF5_FILE)
    logging.info("Data loaded successfully.")

    comparison_summary = []
    for name, model in classifiers.items():
        logging.info(f"--- Running analysis for {name} ---")
        
        # Create a dedicated output directory for this classifier
        clf_output_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(clf_output_dir, exist_ok=True)
        
        # --- Run Multiple Iterations for the Classifier ---
        all_results = {
            "all_predictions": [], "all_true_labels": [], "all_epoch_nums_test": [],
            "all_scores": [], "all_fitted_classes": [], "all_accuracies": [],
            "all_f1_scores": [], "all_cms": []
        }
        
        for i in range(N_ITERATIONS):
            logging.info(f"Processing iteration {i+1}/{N_ITERATIONS} for {name}...")
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
        
        # --- Analyze and Report for the Classifier ---
        analyze_task_performance(all_results, MAPPING_CSV, clf_output_dir, name)
        
        # Plot mean confusion matrix
        mean_cm = np.mean(all_results["all_cms"], axis=0)
        plot_confusion_matrix(
            mean_cm, results["subjects"],
            save_path=os.path.join(clf_output_dir, "mean_confusion_matrix.png"),
            title=f"{name} Mean Confusion Matrix ({N_ITERATIONS} runs)"
        )
        
        # Append to overall comparison summary
        comparison_summary.append({
            "Classifier": name,
            "Mean Accuracy": np.mean(all_results["all_accuracies"]),
            "Std Accuracy": np.std(all_results["all_accuracies"]),
            "Mean F1": np.mean(all_results["all_f1_scores"]),
            "Std F1": np.std(all_results["all_f1_scores"]),
        })

    # --- Save Final Comparison Summary ---
    df_comp = pd.DataFrame(comparison_summary)
    summary_path = os.path.join(OUTPUT_DIR, "classifier_comparison_summary.csv")
    df_comp.to_csv(summary_path, index=False)
    
    logging.info(f"✅ Analysis complete. All outputs saved to '{OUTPUT_DIR}'.")
    print("\n--- Final Performance Summary ---")
    print(df_comp.to_string(index=False))

if __name__ == "__main__":
    main()