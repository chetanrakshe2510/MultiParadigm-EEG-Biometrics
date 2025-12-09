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
import random
from scipy.interpolate import interp1d
from numpy.linalg import inv, pinv, cond
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, roc_curve, auc)
from sklearn.preprocessing import PowerTransformer, label_binarize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# 1. Data Loading
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
# 2. Biometric Verification Helpers (Uniform Analysis)
###############################################################################
def generate_biometric_trials(y_test, scores_matrix, subject_list, impostor_samples=5):
    """
    Generates genuine and impostor score/label pairs for verification analysis.
    For MDTM, Score = Negative Mahalanobis Distance (Higher is better/closer).
    """
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        true_idx = subject_to_idx[true_subj]
        
        # Genuine Trial
        genuine_score = scores_matrix[i, true_idx]
        scores.append(genuine_score)
        labels.append(1) # Genuine
        
        # Impostor Trials (Random Sampling)
        all_impostors = [s for s in subject_list if s != true_subj]
        n_samples = min(impostor_samples, len(all_impostors))
        sampled_impostors = random.sample(all_impostors, n_samples)
        
        for imp_subj in sampled_impostors:
            imp_idx = subject_to_idx[imp_subj]
            impostor_score = scores_matrix[i, imp_idx]
            scores.append(impostor_score)
            labels.append(0) # Impostor

    return np.array(scores), np.array(labels)

def biometric_analysis_dev_test(y_test, scores_matrix, all_subjects, dev_subjects_count=10):
    """
    Splits the test set (Session 2) subjects into Dev (thresholding) and Test (reporting).
    """
    test_subjects = np.unique(y_test)
    if len(test_subjects) <= dev_subjects_count:
        dev_subjects_count = len(test_subjects) // 2 
    
    test_subjects_list = list(test_subjects)
    random.shuffle(test_subjects_list)
    
    dev_subjects = test_subjects_list[:dev_subjects_count]
    final_test_subjects = test_subjects_list[dev_subjects_count:]
    
    dev_mask = np.isin(y_test, dev_subjects)
    test_mask = np.isin(y_test, final_test_subjects)
    
    if not np.any(dev_mask) or not np.any(test_mask):
        return {}

    y_dev = y_test[dev_mask]
    scores_dev = scores_matrix[dev_mask]
    
    y_final_test = y_test[test_mask]
    scores_final_test = scores_matrix[test_mask]

    # --- Dev Set (Threshold Selection) ---
    dev_scores, dev_labels = generate_biometric_trials(y_dev, scores_dev, all_subjects)
    fpr, tpr, thresholds = roc_curve(dev_labels, dev_scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    # Interpolate thresholds for fixed FARs
    fpr_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
    thresh_far_1_pc = fpr_interp(0.01)
    thresh_far_01_pc = fpr_interp(0.001)

    # --- Test Set (Reporting) ---
    test_scores, test_labels = generate_biometric_trials(y_final_test, scores_final_test, all_subjects)
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_scores, pos_label=1)
    fnr_test = 1 - tpr_test
    auroc_test = auc(fpr_test, tpr_test)
    
    def get_frr_at_threshold(scores, labels, threshold):
        preds = (scores >= threshold).astype(int)
        gen_mask = (labels == 1)
        fn = np.sum(preds[gen_mask] == 0)
        tp = np.sum(preds[gen_mask] == 1)
        return fn / (fn + tp) if (fn + tp) > 0 else 0
    
    frr_1pc = get_frr_at_threshold(test_scores, test_labels, thresh_far_1_pc)
    frr_01pc = get_frr_at_threshold(test_scores, test_labels, thresh_far_01_pc)
    
    return {
        "auroc": auroc_test,
        "fpr": fpr_test,
        "fnr": fnr_test,
        "frr_at_far1pc": frr_1pc,
        "frr_at_far01pc": frr_01pc,
        "eer_rate_dev": eer_rate,
        "test_scores": test_scores,
        "test_labels": test_labels
    }

def compute_biometric_ci(y_test, scores_matrix, all_subjects, n_bootstraps=200):
    """Computes confidence intervals using subject-level bootstrap."""
    logging.info(f"Starting Bootstrapping ({n_bootstraps} iterations)...")
    test_subjects_all = np.unique(y_test)
    
    bootstrapped_metrics = {
        'auroc': [],
        'eer': [],
        'frr_at_far1pc': [],
        'frr_at_far01pc': []
    }
    
    subj_indices_map = {s: np.where(y_test == s)[0] for s in test_subjects_all}

    for b in range(n_bootstraps):
        if b % 50 == 0 and b > 0: logging.info(f"Bootstrap {b}/{n_bootstraps}")
        
        sampled_subjects = np.random.choice(test_subjects_all, size=len(test_subjects_all), replace=True)
        
        # Collect indices for the sampled subjects
        indices = []
        for s in sampled_subjects:
            indices.extend(subj_indices_map[s])
        
        if not indices: continue
        indices = np.array(indices)
        
        # Run analysis on this bootstrap sample
        res = biometric_analysis_dev_test(y_test[indices], scores_matrix[indices], all_subjects)
        
        if res:
            bootstrapped_metrics['auroc'].append(res['auroc'])
            bootstrapped_metrics['eer'].append(res['eer_rate_dev'])
            bootstrapped_metrics['frr_at_far1pc'].append(res['frr_at_far1pc'])
            bootstrapped_metrics['frr_at_far01pc'].append(res['frr_at_far01pc'])
            
    cis = {}
    for metric, values in bootstrapped_metrics.items():
        if values:
            cis[metric] = np.percentile(values, [2.5, 97.5])
        else:
            cis[metric] = [0, 0]
            
    return cis, bootstrapped_metrics

###############################################################################
# 3. Plotting Functions
###############################################################################
def plot_data_variance(X, y, save_path):
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

def plot_confusion_matrix(cm, subject_list, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subject_list, yticklabels=subject_list)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_det_curve(fpr_test, fnr_test, eer, save_path, title="DET Curve"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, fnr_test, linewidth=2, label=f"Test Set (EER ≈ {eer*100:.2f}%)")
    eer_idx = np.argmin(np.abs(fpr_test - fnr_test))
    plt.plot(fpr_test[eer_idx], fnr_test[eer_idx], 'ro', markersize=8, label='EER Point')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.0001, 1.0])
    plt.ylim([0.0001, 1.0])
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"DET curve saved to {save_path}")

def plot_epoch_accuracy_heatmap(true_labels, predictions, epoch_nums_test, save_path, csv_path):
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

###############################################################################
# 4. Core Logic
###############################################################################
def build_subject_templates(X_flat, y_data, alpha):
    subjects = np.unique(y_data)
    templates, cov_dict = {}, {}
    for subj in subjects:
        subj_data = X_flat[y_data == subj]
        template = np.mean(subj_data, axis=0)
        
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
    """Predicts subject and returns scores (Negative Distance) for all subjects."""
    distances = {subj: mahalanobis(sample, tmpl, cov_dict[subj]) for subj, tmpl in templates.items()}
    best_subj = min(distances, key=distances.get)
    # Convert distances to scores (Higher score = Better match)
    # Using negative distance preserves the ranking order
    scores = {subj: -dist for subj, dist in distances.items()}
    return best_subj, scores

###############################################################################
# 5. Main Analysis Function
###############################################################################
def across_run_analysis(X, y, runs, epoch_nums, alpha):
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    if not train_idx.size or not test_idx.size:
        raise ValueError("Insufficient data for Run_1 or Run_2.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    templates, cov_dict = build_subject_templates(X_train, y_train, alpha)
    
    predictions, scores_list = [], []
    subjects_sorted = sorted(templates.keys())

    # --- Prediction Loop ---
    for i, sample in enumerate(X_test):
        pred_label, scores = predict_mahalanobis(sample, templates, cov_dict)
        predictions.append(pred_label)
        # Create a consistent score vector for this sample
        scores_list.append([scores[s] for s in subjects_sorted])

    scores_matrix = np.array(scores_list)

    # --- NEW: Biometric Verification Analysis (Bootstrapping) ---
    cis, metrics = compute_biometric_ci(y_test, scores_matrix, subjects_sorted, n_bootstraps=200)
    
    # Get point estimates for plotting
    point_est = biometric_analysis_dev_test(y_test, scores_matrix, subjects_sorted)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=subjects_sorted),
        "predictions": predictions,
        "true_labels": y_test,
        "epoch_nums_test": epoch_nums[test_idx],
        "scores_matrix": scores_matrix,
        "subjects": subjects_sorted,
        "cis": cis,            # Confidence Intervals
        "point_est": point_est # ROC/DET curve data
    }

def compute_per_subject_metrics(true_labels, predictions, subjects):
    rows = []
    for subj in subjects:
        y_true_binary = (np.array(true_labels) == subj)
        y_pred_binary = (np.array(predictions) == subj)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        rows.append({
            "Subject": subj,
            "Precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "Recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "F1-Score": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    return pd.DataFrame(rows)

###############################################################################
# 6. Main Execution
###############################################################################
def create_versioned_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_mdtm_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main(args):
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    run_dir = create_versioned_dir(args.output_dir)
    plots_dir = os.path.join(run_dir, "plots")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    logging.info(f"Outputs will be saved in: {run_dir}")

    # --- Load Data ---
    X, y, runs, epoch_nums = load_features_from_hdf5(args.data_file)
    plot_data_variance(X, y, save_path=os.path.join(plots_dir, "data_variance.png"))
    
    # --- FIX: DATA LEAKAGE PREVENTION ---
    logging.info("Splitting Run 1 (Train) and Run 2 (Test) BEFORE scaler fitting...")
    train_mask = runs == "Run_1"
    
    X_flat = X.reshape(X.shape[0], -1)
    
    transformer = PowerTransformer(method='yeo-johnson')
    
    # Fit ONLY on Run 1
    transformer.fit(X_flat[train_mask])
    
    # Transform ALL data
    X_transformed = transformer.transform(X_flat)
    
    joblib.dump(transformer, os.path.join(run_dir, "power_transformer.pkl"))
    logging.info("PowerTransformer fitted on Run 1 and applied to all.")

    # --- Run Analysis ---
    logging.info(f"Starting analysis with alpha = {args.alpha}...")
    results = across_run_analysis(X_transformed, y, runs, epoch_nums, alpha=args.alpha)
    
    # Format CI Strings for Report
    def fmt_ci(key, scale=1.0):
        mean = results['point_est'].get(key, 0) * scale
        ci_low = results['cis'].get(key, [0,0])[0] * scale
        ci_high = results['cis'].get(key, [0,0])[1] * scale
        return f"{mean:.2f} ({ci_low:.2f}-{ci_high:.2f})"

    logging.info(f"Accuracy: {results['accuracy']:.4f}")
    logging.info(f"EER (Dev): {fmt_ci('eer_rate_dev', 100)}%")

    # --- Save Plots ---
    plot_confusion_matrix(results['confusion_matrix'], results['subjects'], os.path.join(plots_dir, "confusion_matrix.png"))
    
    # DET Curve
    if results['point_est']:
        plot_det_curve(results['point_est']['fpr'], results['point_est']['fnr'], 
                       results['point_est']['eer_rate_dev'], 
                       os.path.join(plots_dir, "det_curve_mdtm.png"),
                       title="DET Curve (MDTM)")

    plot_epoch_accuracy_heatmap(results['true_labels'], results['predictions'], results['epoch_nums_test'],
                                save_path=os.path.join(plots_dir, "epoch_accuracy_heatmap.png"),
                                csv_path=os.path.join(reports_dir, "epoch_wise_accuracy.csv"))

    # --- Save CSV Reports ---
    df_per_subject = compute_per_subject_metrics(results['true_labels'], results['predictions'], results['subjects'])
    df_per_subject.to_csv(os.path.join(reports_dir, "per_subject_metrics.csv"), index=False)

    df_predictions = pd.DataFrame({"True_Label": results['true_labels'], "Predicted_Label": results['predictions']})
    df_predictions.to_csv(os.path.join(reports_dir, "predictions.csv"), index=False)

    # Save Summary CSV
    summary_data = {
        "Model": "MDTM",
        "Accuracy": results['accuracy'],
        "F1_Score": results['f1'],
        "EER_Dev_CI": fmt_ci('eer_rate_dev', 100) + "%",
        "FRR@FAR1%_CI": fmt_ci('frr_at_far1pc', 100) + "%",
        "FRR@FAR0.1%_CI": fmt_ci('frr_at_far01pc', 100) + "%"
    }
    pd.DataFrame([summary_data]).to_csv(os.path.join(run_dir, "mdtm_summary_metrics.csv"), index=False)

    # --- Save Metadata ---
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data_file,
        "alpha": args.alpha,
        "summary": summary_data
    }
    with open(os.path.join(run_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info("="*50)
    logging.info(f"Analysis Finished. Results saved to: {run_dir}")
    logging.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mahalanobis distance classification on EEG data.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the HDF5 data file.")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save models, plots, and reports.")
    parser.add_argument('--alpha', type=float, default=1e-4, help="Regularization parameter for the covariance matrix.")
    
    args = parser.parse_args()
    main(args)