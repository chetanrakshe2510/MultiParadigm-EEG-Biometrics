# File: Maha_task_each_task_Multiparadigm.py

"""
Performs biometric identification using Mahalanobis distance on EEG features.

This script implements a biometric system where subject identification is performed
by calculating the Mahalanobis distance between test samples and subject templates.
The script splits data into training ('Run_1') and testing ('Run_2'), builds
covariance matrices and mean templates for each subject from the training data,
and then evaluates the system on the test data.

It calculates standard classification metrics (Accuracy, F1) as well as biometric
metrics (EER, FAR, FRR). It also provides a per-task performance breakdown. All
results, plots, and metrics are saved to a specified output directory.

Usage:
    python Maha_task_each_task_Multiparadigm.py \\
        --features all_subjects_merged_new_full_epochs.h5 \\
        --mapping epoch_mapping.csv \\
        --output-dir results/biometric
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import inv, pinv, cond
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, roc_curve, auc)
from sklearn.preprocessing import PowerTransformer, label_binarize

# --- Data Loading ---
def load_features_from_hdf5(filename):
    """Loads features, labels, runs, and epoch numbers from an HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f:
            subject_code = class_key.split('_', 1)[-1]
            for run_key in h5f[class_key]:
                for epoch_key in h5f[class_key][run_key]:
                    feats = h5f[class_key][run_key][epoch_key]['features'][()]
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

# --- Core Biometric Functions ---
def build_subject_templates(X_train, y_train, alpha=1e-6):
    """Builds mean templates and inverse covariance matrices for each subject."""
    subjects = np.unique(y_train)
    templates, inv_cov_dict = {}, {}
    for subj in subjects:
        subj_data = X_train[y_train == subj]
        template = np.mean(subj_data, axis=0)
        
        # Regularized covariance
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        
        try:
            inv_cov = inv(cov_matrix)
        except np.linalg.LinAlgError:
            print(f"[WARNING] Covariance matrix for subject {subj} is singular. Using pseudo-inverse.")
            inv_cov = pinv(cov_matrix)
            
        templates[subj] = template
        inv_cov_dict[subj] = inv_cov
    return templates, inv_cov_dict

def predict_mahalanobis(sample, templates, inv_cov_dict):
    """Predicts subject identity based on minimum Mahalanobis distance."""
    scores = {subj: -mahalanobis(sample, tmpl, inv_cov_dict[subj]) for subj, tmpl in templates.items()}
    best_subj = max(scores, key=scores.get)
    return best_subj, scores

def compute_biometric_metrics(genuine_distances, imposter_distances, n_thresholds=1000):
    """Computes FAR, FRR, and EER from genuine and imposter distance scores."""
    all_distances = np.concatenate([genuine_distances, imposter_distances])
    thresholds = np.linspace(all_distances.min(), all_distances.max(), n_thresholds)
    
    far_array, frr_array = [], []
    for tau in thresholds:
        frr = np.mean(np.array(genuine_distances) > tau)
        far = np.mean(np.array(imposter_distances) < tau)
        frr_array.append(frr)
        far_array.append(far)
        
    far_array, frr_array = np.array(far_array), np.array(frr_array)
    idx_eer = np.argmin(np.abs(far_array - frr_array))
    eer = (far_array[idx_eer] + frr_array[idx_eer]) / 2.0
    return eer, thresholds, far_array, frr_array

# --- Analysis Pipeline ---
def across_run_analysis(X, y, runs, epoch_nums, alpha=1e-4):
    """Trains on Run_1 and tests on Run_2, returning a dictionary of results."""
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2. Check run labels.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]

    # Preprocessing
    transformer = PowerTransformer(method='yeo-johnson')
    X_train_flat = transformer.fit_transform(X_train.reshape(len(X_train), -1))
    X_test_flat = transformer.transform(X_test.reshape(len(X_test), -1))

    templates, inv_cov_dict = build_subject_templates(X_train_flat, y_train, alpha)
    
    predictions, scores_matrix = [], []
    genuine_distances, imposter_distances = [], []
    subjects = sorted(templates.keys())

    for i, sample in enumerate(X_test_flat):
        true_label = y_test[i]
        pred, scores = predict_mahalanobis(sample, templates, inv_cov_dict)
        predictions.append(pred)
        scores_matrix.append([scores[s] for s in subjects])
        
        # Collect distances for EER calculation
        genuine_distances.append(mahalanobis(sample, templates[true_label], inv_cov_dict[true_label]))
        for subj, tmpl in templates.items():
            if subj != true_label:
                imposter_distances.append(mahalanobis(sample, tmpl, inv_cov_dict[subj]))
    
    eer, thresholds, far, frr = compute_biometric_metrics(genuine_distances, imposter_distances)
    
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average='weighted', zero_division=0),
        "eer": eer,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=subjects),
        "predictions": predictions,
        "true_labels": y_test,
        "epoch_nums_test": epoch_nums_test,
        "scores_matrix": np.array(scores_matrix),
        "subjects": subjects,
        "genuine_distances": genuine_distances,
        "imposter_distances": imposter_distances,
        "thresholds": thresholds,
        "far_array": far,
        "frr_array": frr,
    }

def compute_task_performance(results, mapping_csv, output_dir, run_label="Run_2"):
    """Computes and saves per-task performance metrics and visualizations."""
    df_res = pd.DataFrame({
        'Subject': results['true_labels'],
        'Epoch': results['epoch_nums_test'],
        'Prediction': results['predictions']
    })

    df_map = pd.read_csv(mapping_csv)
    df_map['Epoch'] = df_map['New_Epoch'].str.extract(r'epoch_(\d+)').astype(int)
    df_map['Subject'] = df_map['Subject'].str.split('_', expand=True)[1]
    df_map = df_map[df_map['Run'] == run_label]

    df_merged = pd.merge(df_res, df_map[['Subject', 'Epoch', 'Source_File']], on=['Subject', 'Epoch'], how='left')

    records = []
    for source, grp in df_merged.groupby('Source_File'):
        if not grp.empty:
            acc = accuracy_score(grp['Subject'], grp['Prediction'])
            f1 = f1_score(grp['Subject'], grp['Prediction'], average='weighted', zero_division=0)
            records.append({'Source_File': source, 'Accuracy': acc, 'F1-Score': f1})
    
    df_perf = pd.DataFrame(records).sort_values('Accuracy', ascending=False)
    df_perf.to_csv(os.path.join(output_dir, 'task_performance_summary.csv'), index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df_perf['Source_File'], df_perf['Accuracy'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.xlabel('Task (Source File)')
    plt.title(f'Biometric Task-Level Accuracy ({run_label})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_performance_plot.png'))
    plt.close()

# --- Plotting Functions ---
def save_plots(results, output_dir):
    """Generates and saves all standard evaluation plots."""
    subjects = results['subjects']
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=subjects, yticklabels=subjects)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Distance Histograms
    plt.figure(figsize=(8, 6))
    plt.hist(results['genuine_distances'], bins=50, alpha=0.7, label='Genuine', density=True)
    plt.hist(results['imposter_distances'], bins=50, alpha=0.7, label='Imposter', density=True)
    plt.title('Distribution of Distances')
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_histogram.png'))
    plt.close()

    # FAR/FRR Curve
    plt.figure(figsize=(8, 6))
    plt.plot(results['thresholds'], results['far_array'], label='FAR')
    plt.plot(results['thresholds'], results['frr_array'], label='FRR')
    eer_threshold = results['thresholds'][np.argmin(np.abs(results['far_array'] - results['frr_array']))]
    plt.axvline(x=eer_threshold, color='r', linestyle='--', label=f'EER = {results["eer"]:.2%}')
    plt.title('FAR and FRR Curves')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'far_frr_curve.png'))
    plt.close()

def main(args):
    """Main execution pipeline."""
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading data...")
    X, y, runs, epoch_nums = load_features_from_hdf5(args.features)

    print("[INFO] Starting across-run analysis...")
    results = across_run_analysis(X, y, runs, epoch_nums, alpha=args.alpha)

    print("\n--- Overall Performance ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Weighted): {results['f1']:.4f}")
    print(f"Equal Error Rate (EER): {results['eer']:.2%}")
    
    # Save overall metrics
    pd.DataFrame([{
        'Accuracy': results['accuracy'], 'F1_Score': results['f1'], 'EER': results['eer']
    }]).to_csv(os.path.join(args.output_dir, 'overall_metrics.csv'), index=False)

    # Save predictions
    pd.DataFrame({
        'True_Label': results['true_labels'], 'Predicted_Label': results['predictions']
    }).to_csv(os.path.join(args.output_dir, 'predictions_vs_true.csv'), index=False)
    
    print("\n[INFO] Saving plots...")
    save_plots(results, args.output_dir)

    print("[INFO] Computing per-task performance breakdown...")
    compute_task_performance(results, args.mapping, args.output_dir)
    print(f"\n[INFO] All results saved in '{args.output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run biometric analysis using Mahalanobis distance.")
    parser.add_argument('--features', type=str, required=True, help='Path to the HDF5 file with extracted features.')
    parser.add_argument('--mapping', type=str, required=True, help='Path to the epoch-to-task mapping CSV file.')
    parser.add_argument('--output-dir', type=str, default='results/biometric', help='Directory to save all output files.')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Regularization parameter for covariance matrix.')

    args = parser.parse_args()
    main(args)