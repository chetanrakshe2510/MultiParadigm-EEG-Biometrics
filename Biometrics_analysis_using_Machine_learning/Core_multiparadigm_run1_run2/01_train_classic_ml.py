import os
import json
import argparse # NEW: For command-line arguments
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn assns
import logging
from datetime import datetime

from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# 1. Data Loading (No changes)
###############################################################################
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            class_group = h5f[class_key]
            for run_key in class_group.keys():
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():
                    epoch_group = run_group[epoch_key]
                    feats = epoch_group['features'][()]
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

###############################################################################
# 2. Plotting Functions (No changes)
###############################################################################
def inspect_data_variance(X, y, save_path="data_variance.png"):
    subjects = np.unique(y)
    var_stats = {subj: np.mean(np.std(X[y == subj].reshape(X[y == subj].shape[0], -1), axis=0)) for subj in subjects}
    logging.info("=== Data Variance Inspection ===")
    for subj, mean_std in var_stats.items():
        logging.info(f"Subject {subj}: mean STD = {mean_std:.4f}")
    
    plt.figure(figsize=(8, 4))
    plt.bar(var_stats.keys(), var_stats.values())
    plt.xlabel("Subject")
    plt.ylabel("Mean STD of Features")
    plt.title("Data Variance per Subject")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, subject_list, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subject_list, yticklabels=subject_list)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_aggregated_roc(scores_matrix, true_labels, subjects, save_path, title="Aggregated ROC Curve"):
    y_true_bin = label_binarize(true_labels, classes=subjects)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Micro-averaged ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_epoch_accuracy_heatmap(true_labels, predictions, epoch_nums_test, save_path, csv_name):
    df_accuracy = pd.DataFrame({
        'subject': true_labels,
        'epoch': epoch_nums_test,
        'correct': (true_labels == np.array(predictions)).astype(int)
    })
    heatmap_data = df_accuracy.pivot_table(index='subject', columns='epoch', values='correct', aggfunc='mean')
    heatmap_data = heatmap_data.fillna(np.nan)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, linewidths=.5, linecolor='gray', vmin=0, vmax=1)
    plt.xlabel("Epoch Number")
    plt.ylabel("Subject")
    plt.title("Epoch-wise Classification Accuracy (Mean)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    heatmap_data.to_csv(csv_name)
    logging.info(f"Saved epoch-wise classification matrix to {csv_name}")

###############################################################################
# 3. Evaluation & Analysis Functions (Mostly no changes)
###############################################################################
def evaluate_predictions(y_true, predictions, subjects):
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions, labels=subjects)
    return accuracy, f1, cm

def across_run_analysis_ml(classifier, param_grid, X, y, runs, epoch_nums, classifier_name="Classifier", cv=3):
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_flat, y_train)
    
    best_classifier = grid_search.best_estimator_
    logging.info(f"Best hyperparameters for {classifier_name}: {grid_search.best_params_}")
    
    predictions = best_classifier.predict(X_test_flat)
    
    subjects_unique = np.unique(y) # Use all potential subjects for consistent label ordering
    
    if hasattr(best_classifier, "predict_proba"):
        scores_matrix = best_classifier.predict_proba(X_test_flat)
    elif hasattr(best_classifier, "decision_function"):
        decision_vals = best_classifier.decision_function(X_test_flat)
        # Ensure correct shape for multi-class
        if decision_vals.ndim == 1:
             scores_matrix = np.exp(decision_vals) / (1 + np.exp(decision_vals)) # Sigmoid for binary
             scores_matrix = np.vstack([1-scores_matrix, scores_matrix]).T
        else:
            scores_matrix = decision_vals
    else: # Fallback for classifiers without probability scores
        scores_matrix = label_binarize(predictions, classes=subjects_unique)
        
    accuracy, f1, cm = evaluate_predictions(y_test, predictions, subjects_unique)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "true_labels": y_test,
        "epoch_nums_test": epoch_nums_test,
        "scores_matrix": scores_matrix,
        "subjects": subjects_unique,
        "fitted_classifier": best_classifier
    }

###############################################################################
# 4. Utility Functions (No changes)
###############################################################################
def save_artifact(artifact, filepath):
    try:
        joblib.dump(artifact, filepath)
        logging.info(f"Saved artifact to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save artifact at {filepath}: {e}")

def save_metadata(model_dir, metadata):
    metadata_filepath = os.path.join(model_dir, "metadata.json")
    try:
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Saved metadata to {metadata_filepath}")
    except Exception as e:
        logging.error(f"Failed to save metadata: {e}")

def create_versioned_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    return versioned_dir

###############################################################################
# 5. Main Script Logic
###############################################################################
def main(args):
    # Create a single versioned directory for this entire run
    versioned_run_dir = create_versioned_dir(args.output_dir)
    plots_folder = os.path.join(versioned_run_dir, "ml_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    logging.info(f"Loading data from {args.data_file}")
    X, y, runs, epoch_nums = load_features_from_hdf5(args.data_file)
    
    # NOTE: Reshaping seems to have a typo in the original (32x32 -> 32x28). Correcting.
    # Assuming the features are (n, 32, 32) and reshaping back to that after transformation.
    original_shape = X.shape
    X_flat = X.reshape(original_shape[0], -1)
    # X = X_flat.reshape(original_shape) # Corrected reshaping

    inspect_data_variance(X, y, save_path=os.path.join(plots_folder, "data_variance.png"))
    
    comparison_results = []
    
    # NEW: Central model configuration
    MODEL_CONFIG = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200], 'max_depth': [None, 10, 20]
            }
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {
                'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']
            }
        }
    }

    # REFACTORED: Loop through models instead of repeating code
    for model_name, config in MODEL_CONFIG.items():
        logging.info(f"=== Across-Run Analysis: {model_name} with Tuning ===")
        results = across_run_analysis_ml(
            classifier=config["estimator"],
            param_grid=config["params"],
            X=X, y=y, runs=runs, epoch_nums=epoch_nums,
            classifier_name=model_name,
            cv=3
        )
        
        # Store results for final comparison
        comparison_results.append({
            "Classifier": model_name,
            "Accuracy": results["accuracy"],
            "F1_Score": results["f1"]
        })

        # Save model and plots
        save_artifact(results["fitted_classifier"], os.path.join(versioned_run_dir, f"{model_name.lower()}_model.pkl"))
        
        plot_confusion_matrix(results["confusion_matrix"], results["subjects"],
                              save_path=os.path.join(plots_folder, f"cm_{model_name}.png"),
                              title=f"{model_name} Confusion Matrix")
        
        plot_aggregated_roc(results["scores_matrix"], results["true_labels"], results["subjects"],
                            save_path=os.path.join(plots_folder, f"roc_{model_name}.png"),
                            title=f"{model_name} ROC Curve")

        plot_epoch_accuracy_heatmap(results["true_labels"], results["predictions"], results["epoch_nums_test"],
                                    save_path=os.path.join(plots_folder, f"epoch_heatmap_{model_name}.png"),
                                    csv_name=os.path.join(plots_folder, f"epoch_accuracy_{model_name}.csv"))

    # Save overall comparison CSV
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv(os.path.join(versioned_run_dir, "ml_classifiers_comparison.csv"), index=False)
    logging.info(f"Comparison metrics saved to {os.path.join(versioned_run_dir, 'ml_classifiers_comparison.csv')}")

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data_file,
        "output_directory": versioned_run_dir,
        "hyperparameters": {name: conf["params"] for name, conf in MODEL_CONFIG.items()},
        "data_shape": {"X": list(X.shape), "y": list(y.shape)}
    }
    save_metadata(versioned_run_dir, metadata)
    logging.info("="*50)
    logging.info(f"Experiment finished. All outputs saved to: {versioned_run_dir}")
    logging.info("="*50)


if __name__ == "__main__":
    # NEW: Add argument parser for better usability
    parser = argparse.ArgumentParser(description="Run ML classification experiments on EEG data.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the HDF5 data file.")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save models and plots.")
    
    args = parser.parse_args()
    main(args)