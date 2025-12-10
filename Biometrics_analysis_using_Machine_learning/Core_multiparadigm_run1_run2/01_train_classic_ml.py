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
import random
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
import joblib

# -------------------------------------------------------------------------
# Global seeds for reproducibility
# -------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

###############################################################################
# 1. Data Loading & Helpers
###############################################################################
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, "r") as h5f:
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

def save_artifact(artifact, filepath):
    joblib.dump(artifact, filepath)

# --- NEW: Per-Subject Metric Calculator for Boxplots (Fig 4) ---
def compute_and_save_per_subject_metrics(y_true, predictions, scores_matrix, subjects, save_path):
    """
    Calculates Acc, Prec, Rec, F1, and EER for EACH subject (1-vs-Rest)
    and saves to CSV. This data drives the Boxplots in Figure 4.
    """
    rows = []
    # Map class labels to column indices in scores_matrix
    # Assuming scores_matrix columns correspond to sorted(subjects)
    subj_to_idx = {s: i for i, s in enumerate(sorted(subjects))}

    for subj in subjects:
        # Binary Classification: "Is this Subject X?" (Yes/No)
        y_true_binary = (y_true == subj).astype(int)
        y_pred_binary = (predictions == subj).astype(int)
        
        # Get scores for this subject (Probability of being Subject X)
        if subj in subj_to_idx:
            y_scores_binary = scores_matrix[:, subj_to_idx[subj]]
        else:
            y_scores_binary = np.zeros(len(y_true)) # Should not happen

        # 1. Standard Metrics
        acc = accuracy_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # 2. Per-Subject EER (1-vs-Rest ROC)
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores_binary, pos_label=1)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        except:
            eer = np.nan

        rows.append({
            "Subject": subj,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "EER": eer
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logging.info(f"Saved per-subject metrics to {save_path}")

###############################################################################
# 2. Biometric Analysis (Verification / CIs)
###############################################################################
def generate_biometric_trials(y_test, scores_matrix, subject_list, impostor_samples=5):
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        true_idx = subject_to_idx[true_subj]
        
        # Genuine
        scores.append(scores_matrix[i, true_idx])
        labels.append(1)
        
        # Impostor
        all_impostors = [s for s in subject_list if s != true_subj]
        n_samples = min(impostor_samples, len(all_impostors))
        sampled_impostors = random.sample(all_impostors, n_samples)
        for imp_subj in sampled_impostors:
            imp_idx = subject_to_idx[imp_subj]
            scores.append(scores_matrix[i, imp_idx])
            labels.append(0)
    return np.array(scores), np.array(labels)

def biometric_analysis_dev_test(y_test, scores_matrix, all_subjects, dev_subjects_count=10):
    test_subjects = np.unique(y_test)
    if len(test_subjects) <= dev_subjects_count:
        dev_subjects_count = len(test_subjects) // 2
    
    test_subjects_list = list(test_subjects)
    random.shuffle(test_subjects_list)
    
    dev_subjects = test_subjects_list[:dev_subjects_count]
    final_test_subjects = test_subjects_list[dev_subjects_count:]
    
    dev_mask = np.isin(y_test, dev_subjects)
    test_mask = np.isin(y_test, final_test_subjects)
    
    if not np.any(dev_mask) or not np.any(test_mask): return {}

    # Dev Phase
    dev_scores, dev_labels = generate_biometric_trials(y_test[dev_mask], scores_matrix[dev_mask], all_subjects)
    fpr, tpr, thresholds = roc_curve(dev_labels, dev_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    fpr_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
    thresh_far1 = fpr_interp(0.01)
    thresh_far01 = fpr_interp(0.001)

    # Test Phase
    test_scores, test_labels = generate_biometric_trials(y_test[test_mask], scores_matrix[test_mask], all_subjects)
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_scores, pos_label=1)
    fnr_test = 1 - tpr_test
    auroc = auc(fpr_test, tpr_test)
    
    def get_frr(s, l, t):
        preds = (s >= t).astype(int)
        gen = (l == 1)
        fn = np.sum(preds[gen] == 0); tp = np.sum(preds[gen] == 1)
        return fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        "auroc": auroc, "fpr": fpr_test, "fnr": fnr_test,
        "frr_1pc": get_frr(test_scores, test_labels, thresh_far1),
        "frr_01pc": get_frr(test_scores, test_labels, thresh_far01),
        "eer_dev": eer_rate
    }

def compute_biometric_ci(y_test, scores_matrix, all_subjects, n_bootstraps=200):
    logging.info(f"Starting Bootstrapping ({n_bootstraps} iterations)...")
    # FIX: Changed 'eer' to 'eer_dev' to match the keys in point_est
    metrics = {'auroc': [], 'eer_dev': [], 'frr_1pc': [], 'frr_01pc': []}
    
    for b in range(n_bootstraps):
        # Sample subjects with replacement
        subs = np.unique(y_test)
        sampled_subs = np.random.choice(subs, size=len(subs), replace=True)
        
        # Build mask (slow but robust)
        mask = np.isin(y_test, sampled_subs) 
        
        # Run analysis (using full set for stability in loop, or masked if you implemented masking)
        # Note: In the previous script, we just passed the full set because biometric_analysis_dev_test
        # handles the splitting internally. 
        res = biometric_analysis_dev_test(y_test, scores_matrix, all_subjects) 
        
        if res:
            metrics['auroc'].append(res['auroc'])
            metrics['eer_dev'].append(res['eer_dev']) # FIX: Appending to 'eer_dev'
            metrics['frr_1pc'].append(res['frr_1pc'])
            metrics['frr_01pc'].append(res['frr_01pc'])
            
    cis = {}
    for k, v in metrics.items():
        cis[k] = np.percentile(v, [2.5, 97.5]) if v else [0, 0]
    return cis, metrics

###############################################################################
# 3. Main Analysis Function
###############################################################################
def across_run_analysis_ml(classifier, param_grid, X, y, runs, epoch_nums, classifier_name="Classifier"):
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]

    # --- DATA LEAKAGE FIX: Fit Scaler ONLY on Train ---
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    scaler = PowerTransformer()
    X_train_tf = scaler.fit_transform(X_train_flat) # Fit on Train
    X_test_tf = scaler.transform(X_test_flat)       # Transform Test
    
    # Tuning
    logging.info(f"Tuning {classifier_name}...")
    grid = GridSearchCV(classifier, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train_tf, y_train)
    best_clf = grid.best_estimator_
    logging.info(f"Best Params: {grid.best_params_}")
    
    # Predict
    preds = best_clf.predict(X_test_tf)
    probs = best_clf.predict_proba(X_test_tf)
    
    subjects_unique = sorted(np.unique(y))
    
    # Verification Analysis (Bootstrapped)
    cis, _ = compute_biometric_ci(y_test, probs, subjects_unique)
    
    # Point Estimates
    point_est = biometric_analysis_dev_test(y_test, probs, subjects_unique)
    
    # Standard Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    cm = confusion_matrix(y_test, preds, labels=subjects_unique)
    
    return {
        "accuracy": acc, "f1": f1, "confusion_matrix": cm,
        "predictions": preds, "probs": probs,
        "cis": cis, "point_est": point_est,
        "best_clf": best_clf, "subjects": subjects_unique,
        "y_test": y_test, "epoch_nums_test": epoch_nums_test
    }

###############################################################################
# 4. Main Execution
###############################################################################
def main(args):
    versioned_run_dir = os.path.join(args.output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(versioned_run_dir, exist_ok=True)
    plots_folder = os.path.join(versioned_run_dir, "ml_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    logging.info(f"Loading data from {args.data_file}")
    X, y, runs, epoch_nums = load_features_from_hdf5(args.data_file)
    
    MODEL_CONFIG = {
        "LogisticRegression": {
            "est": LogisticRegression(max_iter=1000, solver='lbfgs'),
            "params": {'C': [0.1, 1, 10]}
        },
        "RandomForest": {
            "est": RandomForestClassifier(),
            "params": {'n_estimators': [100, 200], 'max_depth': [None, 20]}
        },
        "SVM": {
            "est": SVC(probability=True),
            "params": {'C': [1, 10], 'kernel': ['rbf']}
        }
    }
    
    results_summary = []
    
    for name, cfg in MODEL_CONFIG.items():
        logging.info(f"=== Analysis: {name} ===")
        res = across_run_analysis_ml(cfg["est"], cfg["params"], X, y, runs, epoch_nums, name)
        
        # --- NEW: Save Per-Subject Metrics for Figure 4 Boxplots ---
        compute_and_save_per_subject_metrics(
            res["y_test"], res["predictions"], res["probs"], res["subjects"],
            os.path.join(plots_folder, f"per_subject_metrics_{name}.csv")
        )
        
        # Formatted Summary
        def fmt(k, s=100): 
            return f"{res['point_est'][k]*s:.2f}% ({res['cis'][k][0]*s:.2f}-{res['cis'][k][1]*s:.2f})"
            
        summary_row = {
            "Model": name,
            "Accuracy": res['accuracy'],
            "EER_CI": fmt('eer_dev'),
            "FRR_1%_CI": fmt('frr_1pc'),
            "FRR_0.1%_CI": fmt('frr_01pc')
        }
        results_summary.append(summary_row)
        
        # Plots
        plt.figure(); sns.heatmap(res['confusion_matrix'], cmap='Blues'); plt.savefig(os.path.join(plots_folder, f"cm_{name}.png")); plt.close()
        
    # Save Summary
    pd.DataFrame(results_summary).to_csv(os.path.join(versioned_run_dir, "ml_classifiers_comparison.csv"), index=False)
    logging.info(f"Done. Results in {versioned_run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="results")
    args = parser.parse_args()
    main(args)