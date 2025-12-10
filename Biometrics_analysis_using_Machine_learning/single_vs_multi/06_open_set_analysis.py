# -*- coding: utf-8 -*-
"""
Priority 3: Open-Set Biometric Analysis (Held-Out Subjects).

This script evaluates the system's ability to reject "Unknown" (Non-Enrolled) impostors.
Protocol:
1. 5-Fold Cross-Validation:
   - Total Subjects: 25
   - In each fold: 20 Enrolled (Known), 5 Non-Enrolled (Unknown/Impostors).
2. Training:
   - Train Logistic Regression ONLY on Enrolled subjects (Run 1).
3. Evaluation (Run 2):
   - Split Enrolled subjects into Dev (10) and Eval (10).
   - Dev Set: Used to calculate thresholds (EER, FAR=1%, etc.).
   - Test Set includes:
     a) Genuine: Eval subjects claiming themselves.
     b) Closed-Set Impostors: Eval subjects claiming other Eval subjects.
     c) Open-Set Impostors: Non-Enrolled subjects claiming Eval subjects.
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import KFold

# --- Configuration ---
HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
MAPPING_CSV = "epoch_mapping.csv"
OUTPUT_DIR = "results/open_set_analysis"
N_FOLDS = 5
DEV_SPLIT_RATIO = 0.5  # 50% of Enrolled used for Dev (Thresholding), 50% for Eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- 1. Data Loading ---
def load_data(hdf5_path):
    if not os.path.exists(hdf5_path): raise FileNotFoundError(f"{hdf5_path} not found")
    
    X, y, runs = [], [], []
    with h5py.File(hdf5_path, 'r') as f:
        for subj in f.keys():
            subj_code = subj.split('_')[1]
            for run in f[subj].keys():
                for epoch_key in f[subj][run].keys():
                    feats = f[subj][run][epoch_key]['features'][()]
                    X.append(feats)
                    y.append(subj_code)
                    runs.append(run)
    return np.array(X), np.array(y), np.array(runs)

# --- 2. Helper: Generate Scores ---
def get_verification_scores(y_true, scores_matrix, enrolled_classes, dataset_type="closed_set", non_enrolled_data=None):
    """
    Generates genuine and impostor scores.
    
    Args:
        y_true: True labels of the test data.
        scores_matrix: Probability matrix from classifier (N_samples x N_enrolled).
        enrolled_classes: List of enrolled subject IDs (matching columns of scores_matrix).
        dataset_type: "closed_set" (enrolled subjects) or "open_set" (impostors).
    """
    gen_scores = []
    imp_scores = []
    
    subj_to_idx = {s: i for i, s in enumerate(enrolled_classes)}
    
    if dataset_type == "closed_set":
        # Standard verification within enrolled users
        for i, true_s in enumerate(y_true):
            if true_s not in subj_to_idx: continue
            
            # Genuine: Claiming self
            true_idx = subj_to_idx[true_s]
            gen_scores.append(scores_matrix[i, true_idx])
            
            # Impostor: Claiming random other ENROLLED user
            others = [s for s in enrolled_classes if s != true_s]
            if others:
                imp_claim = random.choice(others)
                imp_idx = subj_to_idx[imp_claim]
                imp_scores.append(scores_matrix[i, imp_idx])
                
    elif dataset_type == "open_set":
        # Open-set impostors: Unknown subjects claiming Enrolled identities
        # We simulate them claiming random enrolled identities
        for i in range(len(scores_matrix)):
            # Claim random enrolled identity
            claim_s = random.choice(enrolled_classes)
            claim_idx = subj_to_idx[claim_s]
            imp_scores.append(scores_matrix[i, claim_idx])
            
    return gen_scores, imp_scores

# --- 3. Fold Logic ---
def run_open_set_folds(X, y, runs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_subjects = np.unique(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (enrolled_idx, non_enrolled_idx) in enumerate(kf.split(unique_subjects)):
        enrolled_subjs = unique_subjects[enrolled_idx]
        non_enrolled_subjs = unique_subjects[non_enrolled_idx] # The "Unknowns"
        
        logging.info(f"--- Fold {fold_idx+1}/{N_FOLDS} ---")
        logging.info(f"Enrolled ({len(enrolled_subjs)}): {enrolled_subjs[:5]}...")
        logging.info(f"Open-Set Impostors ({len(non_enrolled_subjs)}): {non_enrolled_subjs}")
        
        # --- A. Prepare Training Data (Enrolled Run 1) ---
        train_mask = np.isin(y, enrolled_subjs) & (runs == 'Run_1')
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # Preprocess (Fit on Enrolled Train only)
        X_train_flat = X_train.reshape(len(X_train), -1)
        transformer = PowerTransformer().fit(X_train_flat)
        X_train_tf = transformer.transform(X_train_flat)
        
        # Train Classifier
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_tf, y_train)
        
        # --- B. Prepare Test Data (Run 2) ---
        # 1. Enrolled Run 2 (For Dev and Eval Genuine/Closed-Imp)
        enrolled_test_mask = np.isin(y, enrolled_subjs) & (runs == 'Run_2')
        X_test_enrolled = transformer.transform(X[enrolled_test_mask].reshape(np.sum(enrolled_test_mask), -1))
        y_test_enrolled = y[enrolled_test_mask]
        probs_enrolled = clf.predict_proba(X_test_enrolled)
        
        # 2. Non-Enrolled Run 2 (Strictly Open-Set Impostors)
        # Note: We can use Run 1 & 2 for open-set since they were never seen, but sticking to Run 2 is cleaner.
        open_set_mask = np.isin(y, non_enrolled_subjs) & (runs == 'Run_2')
        X_test_open = transformer.transform(X[open_set_mask].reshape(np.sum(open_set_mask), -1))
        probs_open = clf.predict_proba(X_test_open)
        
        # --- C. Dev/Test Split within Enrolled ---
        # Split enrolled subjects into Dev (Thresholding) and Eval (Reporting)
        random.shuffle(enrolled_subjs)
        n_dev = int(len(enrolled_subjs) * DEV_SPLIT_RATIO)
        dev_subjs = enrolled_subjs[:n_dev]
        eval_subjs = enrolled_subjs[n_dev:]
        
        # 1. Dev Phase: Calculate Thresholds
        dev_mask = np.isin(y_test_enrolled, dev_subjs)
        dev_gen, dev_imp = get_verification_scores(
            y_test_enrolled[dev_mask], probs_enrolled[dev_mask], clf.classes_, "closed_set"
        )
        
        labels_dev = [1]*len(dev_gen) + [0]*len(dev_imp)
        scores_dev = dev_gen + dev_imp
        fpr_dev, tpr_dev, thresh_dev = roc_curve(labels_dev, scores_dev, pos_label=1)
        fnr_dev = 1 - tpr_dev
        
        # Find EER Threshold
        eer_idx = np.nanargmin(np.abs(fpr_dev - fnr_dev))
        thresh_eer = thresh_dev[eer_idx]
        
        # Find FAR=1% and FAR=0.1% Thresholds
        interp_func = interp1d(fpr_dev, thresh_dev, bounds_error=False, fill_value="extrapolate")
        thresh_far1 = interp_func(0.01)
        thresh_far01 = interp_func(0.001)
        
        # --- D. Eval Phase (Open-Set Evaluation) ---
        # Eval Set = (Eval Enrolled Genuine) + (Eval Enrolled Closed-Imp) + (Open-Set Impostors)
        
        # 1. Eval Enrolled (Genuine + Closed Impostors)
        eval_mask = np.isin(y_test_enrolled, eval_subjs)
        eval_gen, eval_imp_closed = get_verification_scores(
            y_test_enrolled[eval_mask], probs_enrolled[eval_mask], clf.classes_, "closed_set"
        )
        
        # 2. Open-Set Impostors (Unknowns claiming Eval identities)
        # Note: We filter open-set claims to only target `eval_subjs` for consistency, 
        # or we just let them claim any enrolled. Let's strictly check claims against valid enrolled classes.
        _, eval_imp_open = get_verification_scores(
            None, probs_open, clf.classes_, "open_set"
        )
        
        # Combine Impostors
        all_eval_imp = eval_imp_closed + eval_imp_open
        
        # Calculate Metrics on Combined Eval Set
        metrics = {}
        for name, thresh in [("EER", thresh_eer), ("FAR_1%", thresh_far1), ("FAR_0.1%", thresh_far01)]:
            frr = np.mean(np.array(eval_gen) < thresh)
            far_combined = np.mean(np.array(all_eval_imp) >= thresh)
            far_open_only = np.mean(np.array(eval_imp_open) >= thresh)
            
            metrics[f"FRR_at_{name}"] = frr
            metrics[f"FAR_Combined_at_{name}"] = far_combined
            metrics[f"FAR_OpenOnly_at_{name}"] = far_open_only
            
        # Calculate AUROC (Combined)
        y_eval = [1]*len(eval_gen) + [0]*len(all_eval_imp)
        s_eval = eval_gen + all_eval_imp
        fpr_final, tpr_final, _ = roc_curve(y_eval, s_eval, pos_label=1)
        metrics["AUROC_Combined"] = auc(fpr_final, tpr_final)
        
        fold_results.append(metrics)
        
    # --- 4. Aggregation ---
    df_res = pd.DataFrame(fold_results)
    summary = df_res.agg(['mean', 'std']).T
    
    print("\n=== Open-Set Analysis Results (5-Fold) ===")
    print(summary)
    
    df_res.to_csv(os.path.join(OUTPUT_DIR, "fold_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
    
    # Plot Distribution (Last Fold Example)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(eval_gen, fill=True, label='Genuine (Enrolled)', color='green')
    sns.kdeplot(eval_imp_closed, fill=True, label='Impostor (Closed-Set)', color='orange', linestyle='--')
    sns.kdeplot(eval_imp_open, fill=True, label='Impostor (Open-Set)', color='red')
    plt.title("Score Distributions: Genuine vs. Closed/Open Impostors")
    plt.xlabel("Verification Score (Probability)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "open_set_distributions.png"))
    logging.info(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    X_data, y_data, runs_data = load_data(HDF5_FILE)
    run_open_set_folds(X_data, y_data, runs_data)