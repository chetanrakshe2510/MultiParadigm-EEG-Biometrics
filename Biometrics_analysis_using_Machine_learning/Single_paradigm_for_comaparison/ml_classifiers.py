#!/usr/bin/env python3
"""
Trains, tunes, and compares standard ML classifiers for EEG-based identification.

This script processes a single HDF5 file containing EEG features. It splits the
data by experimental run (Run_1 for training, Run_2 for testing) and applies a
PowerTransformer for feature normalization.

It then trains and evaluates three classifiers:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Hyperparameter tuning is performed for each model using GridSearchCV. The script
saves detailed performance metrics, prediction CSVs, plots (confusion matrix,
ROC curve), and the trained models for each classifier.
"""
import os
import json
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, auc, classification_report)
from sklearn.preprocessing import PowerTransformer, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.switch_backend('Agg')

# =============================================================================
# DATA LOADING & PREPROCESSING
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

def filter_epochs_per_subject(X, y, runs, epoch_nums, max_epochs=40, random_state=42):
    """Randomly samples up to `max_epochs` per subject per run for balanced analysis."""
    logging.info(f"Randomly sampling up to {max_epochs} epochs per subject per run.")
    np.random.seed(random_state)
    final_indices = []
    for subject in np.unique(y):
        for run in np.unique(runs):
            indices = np.where((y == subject) & (runs == run))[0]
            if len(indices) > 0:
                num_to_sample = min(max_epochs, len(indices))
                selected_indices = np.random.choice(indices, size=num_to_sample, replace=False)
                final_indices.extend(selected_indices)
    final_indices = sorted(final_indices)
    return X[final_indices], y[final_indices], runs[final_indices], epoch_nums[final_indices]

# =============================================================================
# PLOTTING & REPORTING
# =============================================================================
def plot_confusion_matrix(cm, subject_list, save_path, title):
    """Saves a confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=subject_list, yticklabels=subject_list)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_aggregated_roc(scores_matrix, true_labels, subjects, save_path, title):
    """Saves a micro-averaged ROC curve."""
    y_true_bin = label_binarize(true_labels, classes=subjects)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Micro-averaged ROC (AUC = {roc_auc:.3f})", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# CORE ANALYSIS
# =============================================================================
def across_run_analysis_ml(classifier, X, y, runs, classifier_name, param_grid):
    """Performs across-run training, tuning, and evaluation for an ML model."""
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1/Run_2 split.")

    X_train_flat = X[train_idx].reshape(len(train_idx), -1)
    y_train = y[train_idx]
    X_test_flat = X[test_idx].reshape(len(test_idx), -1)
    y_test = y[test_idx]
    
    # Hyperparameter tuning
    logging.info(f"Performing GridSearchCV for {classifier_name}...")
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_flat, y_train)
    best_model = grid_search.best_estimator_
    logging.info(f"Best hyperparameters for {classifier_name}: {grid_search.best_params_}")

    # Evaluation
    predictions = best_model.predict(X_test_flat)
    subjects_unique = np.unique(np.concatenate((y_train, y_test)))

    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_test_flat)
        # Align probabilities with the full set of unique subjects
        scores_matrix = np.zeros((len(y_test), len(subjects_unique)))
        class_map = {cls: i for i, cls in enumerate(best_model.classes_)}
        for i, cls in enumerate(subjects_unique):
            if cls in class_map:
                scores_matrix[:, i] = probs[:, class_map[cls]]
    else: # For models like SVM without direct predict_proba
        scores_matrix = label_binarize(predictions, classes=subjects_unique)

    return {
        "classifier_name": classifier_name,
        "best_params": grid_search.best_params_,
        "fitted_model": best_model,
        "predictions": predictions,
        "scores_matrix": scores_matrix,
        "true_labels": y_test,
        "subjects": subjects_unique
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="ML Classifier Comparison for EEG Biometrics")
    parser.add_argument('--input-file', type=str, required=True, help='Path to the HDF5 data file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save analysis results.')
    parser.add_argument('--epochs', type=int, default=40, help='Max number of epochs to randomly sample per subject.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # --- 1. Load and Preprocess Data ---
    logging.info(f"Loading data from {args.input_file}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(args.input_file)
    X, y, runs, epoch_nums = filter_epochs_per_subject(X, y, runs, epoch_nums, max_epochs=args.epochs)
    
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    X = X_tf.reshape(X.shape)
    
    model_save_dir = os.path.join(args.output_dir, f"{base_name}_models")
    os.makedirs(model_save_dir, exist_ok=True)
    joblib.dump(transformer, os.path.join(model_save_dir, "power_transformer.pkl"))

    # --- 2. Define Classifiers and Parameter Grids ---
    classifiers = {
        "LogisticRegression": (LogisticRegression(max_iter=2000, multi_class='ovr', random_state=42),
                               {'C': [0.1, 1, 10], 'solver': ['liblinear']}),
        "RandomForest": (RandomForestClassifier(random_state=42),
                         {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}),
        "SVM": (SVC(probability=True, random_state=42),
                {'C': [1, 10], 'gamma': ['scale'], 'kernel': ['rbf']})
    }

    # --- 3. Run Analysis for Each Classifier ---
    all_results = []
    for name, (model, params) in classifiers.items():
        logging.info(f"\n{'='*20} Analyzing {name} {'='*20}")
        try:
            result = across_run_analysis_ml(model, X, y, runs, name, params)
            
            # Generate reports and save artifacts
            report = classification_report(result['true_labels'], result['predictions'],
                                           labels=result['subjects'], output_dict=True, zero_division=0)
            df_report = pd.DataFrame(report).transpose()
            df_report.to_csv(os.path.join(args.output_dir, f"{base_name}_{name}_report.csv"))

            cm = confusion_matrix(result['true_labels'], result['predictions'], labels=result['subjects'])
            plot_confusion_matrix(cm, result['subjects'],
                                  os.path.join(args.output_dir, f"{base_name}_{name}_confusion_matrix.png"),
                                  f"{name} Confusion Matrix")

            plot_aggregated_roc(result['scores_matrix'], result['true_labels'], result['subjects'],
                                os.path.join(args.output_dir, f"{base_name}_{name}_roc_curve.png"),
                                f"{name} Aggregated ROC Curve")
            
            joblib.dump(result['fitted_model'], os.path.join(model_save_dir, f"{name}_model.pkl"))

            all_results.append({
                "Classifier": name,
                "Accuracy": report['accuracy'],
                "F1_Score_Weighted": report['weighted avg']['f1-score'],
                "Best_Params": json.dumps(result['best_params'])
            })
            logging.info(f"Finished analysis for {name}.")

        except Exception as e:
            logging.error(f"Failed to process {name}. Error: {e}", exc_info=True)

    # --- 4. Save Final Comparison ---
    if all_results:
        df_comparison = pd.DataFrame(all_results)
        df_comparison.to_csv(os.path.join(args.output_dir, f"{base_name}_comparison_summary.csv"), index=False)
        logging.info(f"\nOverall comparison saved. Results are in '{args.output_dir}'")
        print(df_comparison.to_string())

if __name__ == "__main__":
    main()