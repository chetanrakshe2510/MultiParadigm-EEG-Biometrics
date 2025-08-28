import os
import json
import argparse
import logging
from datetime import datetime
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from numpy.linalg import inv, pinv, cond
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, roc_curve, auc)
from sklearn.preprocessing import PowerTransformer, label_binarize

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
            for run_key in h5f[class_key].keys():
                for epoch_key in h5f[class_key][run_key].keys():
                    feats = h5f[class_key][run_key][epoch_key]['features'][()]
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    epoch_nums.append(int(epoch_key.split('_')[-1]))
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

###############################################################################
# 2. Plotting Functions (Modified to save plots)
###############################################################################
def plot_data_variance(X, y, save_path):
    """Inspects and plots the variance of features for each subject."""
    subjects = np.unique(y)
    var_stats = {subj: np.mean(np.std(X[y == subj].reshape(X[y == subj].shape[0], -1), axis=0)) for subj in subjects}
    
    plt.figure(figsize=(10, 6))
    plt.bar(var_stats.keys(), var_stats.values())
    plt.xlabel("Subject")
    plt.ylabel("Mean Standard Deviation of Features")
    plt.title("Data Variance per Subject")
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Data variance plot saved to {save_path}")

def plot_confusion_matrix(cm, subject_list, save_path):
    """Saves a heatmap of the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subject_list, yticklabels=subject_list)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {save_path}")

def plot_distance_histograms(genuine_distances, imposter_distances, save_path):
    """Saves a histogram comparing genuine and imposter distances."""
    plt.figure(figsize=(10, 6))
    sns.histplot(genuine_distances, bins=50, color='blue', label='Genuine Distances', kde=True)
    sns.histplot(imposter_distances, bins=50, color='red', label='Imposter Distances', kde=True)
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Genuine and Imposter Distances")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Distance histograms saved to {save_path}")

def plot_far_frr(thresholds, far_array, frr_array, eer, save_path):
    """Saves a plot of FAR and FRR curves to find the EER."""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far_array, label="False Acceptance Rate (FAR)")
    plt.plot(thresholds, frr_array, label="False Rejection Rate (FRR)")
    
    eer_threshold_idx = np.argmin(np.abs(far_array - frr_array))
    eer_threshold = thresholds[eer_threshold_idx]
    
    plt.axvline(x=eer_threshold, color='gray', linestyle='--', label=f"EER Threshold ≈ {eer_threshold:.2f}")
    plt.plot(eer_threshold, eer, 'ro', markersize=8, label=f'EER = {eer*100:.2f}%')
    
    plt.xlabel("Decision Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR and FRR Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"FAR/FRR plot saved to {save_path}")

def plot_aggregated_roc(scores_matrix, true_labels, subjects, save_path):
    """Saves a micro-averaged ROC curve for the classifier."""
    y_true_bin = label_binarize(true_labels, classes=subjects)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"Micro-averaged ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aggregated Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Aggregated ROC plot saved to {save_path}")

def plot_epoch_accuracy_heatmap(true_labels, predictions, epoch_nums_test, save_path, csv_path):
    """Saves a heatmap of classification accuracy per epoch and subject."""
    df = pd.DataFrame({
        'subject': true_labels,
        'epoch': epoch_nums_test,
        'correct': (np.array(true_labels) == np.array(predictions)).astype(int)
    })
    heatmap_data = df.pivot_table(index='subject', columns='epoch', values='correct', aggfunc='mean')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, linewidths=.5, linecolor='gray', fmt='.2f')
    plt.xlabel("Epoch Number")
    plt.ylabel("Subject")
    plt.title("Mean Epoch-wise Classification Accuracy per Subject")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    heatmap_data.to_csv(csv_path)
    logging.info(f"Epoch accuracy heatmap saved to {save_path} and data to {csv_path}")

###############################################################################
# 3. Core Logic (No major changes)
###############################################################################
def build_subject_templates(X_flat, y_data, alpha):
    """Builds mean templates and regularized inverse covariance matrices for each subject."""
    subjects = np.unique(y_data)
    templates, cov_dict = {}, {}
    for subj in subjects:
        subj_data = X_flat[y_data == subj]
        template = np.mean(subj_data, axis=0)
        
        # Regularized covariance
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        
        try:
            inv_cov = inv(cov_matrix)
        except np.linalg.LinAlgError:
            logging.warning(f"Covariance matrix for subject {subj} is singular. Using pseudo-inverse.")
            inv_cov = pinv(cov_matrix)
            
        templates[subj] = template
        cov_dict[subj] = inv_cov
    return templates, cov_dict

def predict_mahalanobis(sample, templates, cov_dict):
    """Predicts the subject for a sample based on minimum Mahalanobis distance."""
    distances = {subj: mahalanobis(sample, tmpl, cov_dict[subj]) for subj, tmpl in templates.items()}
    best_subj = min(distances, key=distances.get)
    # Scores are negative distances (higher is better)
    scores = {subj: -dist for subj, dist in distances.items()}
    return best_subj, scores

def compute_biometric_metrics(genuine_distances, imposter_distances, n_thresholds=1000):
    """Calculates FAR, FRR, and EER from distance scores."""
    all_distances = np.concatenate([genuine_distances, imposter_distances])
    thresholds = np.linspace(all_distances.min(), all_distances.max(), n_thresholds)
    
    far_array = [np.mean(np.array(imposter_distances) < t) for t in thresholds]
    frr_array = [np.mean(np.array(genuine_distances) > t) for t in thresholds]
    
    eer_idx = np.argmin(np.abs(np.array(far_array) - np.array(frr_array)))
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2.0
    
    return eer, thresholds, far_array, frr_array

###############################################################################
# 4. Main Analysis and Reporting Functions
###############################################################################
def across_run_analysis(X, y, runs, epoch_nums, alpha):
    """Performs the main across-run training and testing analysis."""
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    if not train_idx.size or not test_idx.size:
        raise ValueError("Insufficient data for Run_1 or Run_2.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    templates, cov_dict = build_subject_templates(X_train, y_train, alpha)
    
    predictions, scores_list = [], []
    genuine_distances, imposter_distances = [], []
    subjects_sorted = sorted(templates.keys())

    for i, sample in enumerate(X_test):
        true_label = y_test[i]
        pred_label, scores = predict_mahalanobis(sample, templates, cov_dict)
        
        predictions.append(pred_label)
        scores_list.append([scores[s] for s in subjects_sorted])
        
        # Collect distances for EER calculation
        genuine_distances.append(mahalanobis(sample, templates[true_label], cov_dict[true_label]))
        for subj, tmpl in templates.items():
            if subj != true_label:
                imposter_distances.append(mahalanobis(sample, tmpl, cov_dict[subj]))

    eer, thresholds, far, frr = compute_biometric_metrics(genuine_distances, imposter_distances)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=subjects_sorted),
        "eer": eer,
        "genuine_distances": genuine_distances,
        "imposter_distances": imposter_distances,
        "predictions": predictions,
        "true_labels": y_test,
        "thresholds": thresholds,
        "far_array": far,
        "frr_array": frr,
        "epoch_nums_test": epoch_nums[test_idx],
        "scores_matrix": np.array(scores_list),
        "subjects": subjects_sorted
    }

def compute_per_subject_metrics(true_labels, predictions, subjects):
    """Computes binary classification metrics for each subject vs. all others."""
    rows = []
    for subj in subjects:
        y_true_binary = (np.array(true_labels) == subj)
        y_pred_binary = (np.array(predictions) == subj)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        rows.append({
            "Subject": subj,
            "Precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "Recall (Sensitivity)": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "F1-Score": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    return pd.DataFrame(rows)

###############################################################################
# 5. Main Script Execution
###############################################################################
def create_versioned_dir(base_dir):
    """Creates a timestamped directory for a single run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main(args):
    # --- Setup Directories ---
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    run_dir = create_versioned_dir(args.output_dir)
    plots_dir = os.path.join(run_dir, "plots")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    logging.info(f"Outputs for this run will be saved in: {run_dir}")

    # --- Load and Preprocess Data ---
    X, y, runs, epoch_nums = load_features_from_hdf5(args.data_file)
    plot_data_variance(X, y, save_path=os.path.join(plots_dir, "data_variance.png"))
    
    # Flatten features and apply PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_transformed = transformer.fit_transform(X_flat)
    joblib.dump(transformer, os.path.join(run_dir, "power_transformer.pkl"))
    logging.info("PowerTransformer fitted and saved.")
    
    # --- Run Analysis ---
    logging.info(f"Starting across-run analysis with alpha = {args.alpha}...")
    results = across_run_analysis(X_transformed, y, runs, epoch_nums, alpha=args.alpha)
    
    logging.info(f"Analysis Complete. Accuracy: {results['accuracy']:.4f}, EER: {results['eer']*100:.2f}%")

    # --- Save Plots ---
    plot_confusion_matrix(results['confusion_matrix'], results['subjects'], os.path.join(plots_dir, "confusion_matrix.png"))
    plot_distance_histograms(results['genuine_distances'], results['imposter_distances'], os.path.join(plots_dir, "distance_histograms.png"))
    plot_far_frr(results['thresholds'], results['far_array'], results['frr_array'], results['eer'], os.path.join(plots_dir, "far_frr_curves.png"))
    plot_aggregated_roc(results['scores_matrix'], results['true_labels'], results['subjects'], os.path.join(plots_dir, "aggregated_roc.png"))
    plot_epoch_accuracy_heatmap(results['true_labels'], results['predictions'], results['epoch_nums_test'],
                                save_path=os.path.join(plots_dir, "epoch_accuracy_heatmap.png"),
                                csv_path=os.path.join(reports_dir, "epoch_wise_accuracy.csv"))

    # --- Save Reports ---
    df_per_subject = compute_per_subject_metrics(results['true_labels'], results['predictions'], results['subjects'])
    df_per_subject.to_csv(os.path.join(reports_dir, "per_subject_metrics.csv"), index=False)
    logging.info(f"Per-subject metrics saved to {reports_dir}")

    df_predictions = pd.DataFrame({"True_Label": results['true_labels'], "Predicted_Label": results['predictions']})
    df_predictions.to_csv(os.path.join(reports_dir, "predictions.csv"), index=False)
    logging.info(f"Predictions saved to {reports_dir}")

    # --- Save Metadata ---
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data_file,
        "regularization_alpha": args.alpha,
        "key_metrics": {
            "accuracy": results['accuracy'],
            "f1_score_weighted": results['f1'],
            "equal_error_rate": results['eer']
        }
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info("="*50)
    logging.info(f"Experiment finished. All outputs saved to: {run_dir}")
    logging.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mahalanobis distance classification on EEG data.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the HDF5 data file.")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save models, plots, and reports.")
    parser.add_argument('--alpha', type=float, default=1e-4, help="Regularization parameter for the covariance matrix.")
    
    args = parser.parse_args()
    main(args)
