# -*- coding: utf-8 -*-
"""
Priority 3: Open-Set Biometric Analysis (Held-Out Subjects) - Journal Version.

Updates:
1. Generates DET Curves (Genuine vs. Open-Set Impostors).
2. Saves raw plotting data (CSV) for Origin/Excel.
3. Prints LaTeX table code for direct insertion into papers.
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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold

# --- Configuration ---
HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
OUTPUT_DIR = "results/open_set_analysis_journal"
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
def get_verification_scores(y_true, scores_matrix, enrolled_classes, dataset_type="closed_set"):
    """
    Generates genuine and impostor scores.
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
        for i in range(len(scores_matrix)):
            # Claim random enrolled identity
            claim_s = random.choice(enrolled_classes)
            claim_idx = subj_to_idx[claim_s]
            imp_scores.append(scores_matrix[i, claim_idx])
            
    return gen_scores, imp_scores

# --- 3. New: Journal Artifacts (DET Curves & Data) ---
def generate_journal_artifacts(gen_scores, closed_imp_scores, open_imp_scores):
    """
    Generates DET Curves and saves source data for journal publication.
    """
    print("\n--- Generating Journal Artifacts (DET Curves) ---")
    
    # Helper to compute DET coordinates
    def compute_det(genuine, impostor):
        y_true = [1] * len(genuine) + [0] * len(impostor)
        y_scores = list(genuine) + list(impostor)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        return fpr, fnr
        
    # 1. Compute Curves
    # Scenario A: Baseline (Closed-Set)
    fpr_closed, fnr_closed = compute_det(gen_scores, closed_imp_scores)
    
    # Scenario B: Open-Set (The specific analysis)
    fpr_open, fnr_open = compute_det(gen_scores, open_imp_scores)

    # 2. Save Data for Paper (CSV)
    det_data_closed = pd.DataFrame({'FAR': fpr_closed, 'FRR': fnr_closed})
    det_data_open = pd.DataFrame({'FAR': fpr_open, 'FRR': fnr_open})
    
    det_data_closed.to_csv(os.path.join(OUTPUT_DIR, "det_data_closed_set.csv"), index=False)
    det_data_open.to_csv(os.path.join(OUTPUT_DIR, "det_data_open_set.csv"), index=False)
    print(f"Saved plotting data to {OUTPUT_DIR}/det_data_*.csv")

    # 3. Plotting the DET Curve
    plt.figure(figsize=(8, 8))
    
    # Plot Closed-Set (Baseline)
    plt.plot(fpr_closed, fnr_closed, label='Baseline: Closed-Set Impostors', 
             color='blue', linestyle='--', linewidth=2)
    
    # Plot Open-Set (Target)
    plt.plot(fpr_open, fnr_open, label='Analysis: Open-Set Impostors', 
             color='red', linewidth=2.5)

    # Aesthetics
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Acceptance Rate (FAR) [%]')
    plt.ylabel('False Rejection Rate (FRR) [%]')
    plt.title('DET Curve: Vulnerability to Open-Set Impostors')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend(loc="upper right")
    
    # Axis formatting
    ticks = [0.001, 0.01, 0.1, 1.0]
    tick_labels = ["0.1%", "1%", "10%", "100%"]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)
    plt.xlim([0.0001, 1.0])
    plt.ylim([0.0001, 1.0])
    
    plt.savefig(os.path.join(OUTPUT_DIR, "journal_det_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved DET Curve plot.")

def print_latex_table(summary_df):
    """
    Prints a journal-ready table comparing Combined vs Open-Only performance.
    """
    rows_of_interest = [
        'FAR_Combined_at_EER', 'FAR_OpenOnly_at_EER',
        'FAR_Combined_at_FAR_1%', 'FAR_OpenOnly_at_FAR_1%'
    ]
    
    print("\n=== Journal Table LaTeX Code ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Metric & Mean & Std Dev \\\\")
    print("\\hline")
    
    for row in rows_of_interest:
        if row in summary_df.index:
            mean_val = summary_df.loc[row, 'mean']
            std_val = summary_df.loc[row, 'std']
            row_clean = row.replace('_', ' ').replace('Combined', 'Comb.').replace('OpenOnly', 'Open')
            print(f"{row_clean} & {mean_val*100:.2f}\\% & $\\pm${std_val*100:.2f} \\\\")
            
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Impact of Open-Set Impostors on Verification Performance}")
    print("\\end{table}")

# --- 4. Main Fold Logic ---
def run_open_set_folds(X, y, runs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_subjects = np.unique(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    # --- GLOBAL STORAGE FOR DET PLOT ---
    global_gen_scores = []
    global_closed_imp_scores = []
    global_open_imp_scores = []
    
    for fold_idx, (enrolled_idx, non_enrolled_idx) in enumerate(kf.split(unique_subjects)):
        enrolled_subjs = unique_subjects[enrolled_idx]
        non_enrolled_subjs = unique_subjects[non_enrolled_idx] 
        
        logging.info(f"--- Fold {fold_idx+1}/{N_FOLDS} ---")
        
        # --- A. Prepare Training Data (Run 1) ---
        train_mask = np.isin(y, enrolled_subjs) & (runs == 'Run_1')
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        X_train_flat = X_train.reshape(len(X_train), -1)
        transformer = PowerTransformer().fit(X_train_flat)
        X_train_tf = transformer.transform(X_train_flat)
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_tf, y_train)
        
        # --- B. Prepare Test Data (Run 2) ---
        # 1. Enrolled Run 2
        enrolled_test_mask = np.isin(y, enrolled_subjs) & (runs == 'Run_2')
        X_test_enrolled = transformer.transform(X[enrolled_test_mask].reshape(np.sum(enrolled_test_mask), -1))
        y_test_enrolled = y[enrolled_test_mask]
        probs_enrolled = clf.predict_proba(X_test_enrolled)
        
        # 2. Open-Set Run 2
        open_set_mask = np.isin(y, non_enrolled_subjs) & (runs == 'Run_2')
        X_test_open = transformer.transform(X[open_set_mask].reshape(np.sum(open_set_mask), -1))
        probs_open = clf.predict_proba(X_test_open)
        
        # --- C. Dev/Test Split ---
        random.shuffle(enrolled_subjs)
        n_dev = int(len(enrolled_subjs) * DEV_SPLIT_RATIO)
        dev_subjs = enrolled_subjs[:n_dev]
        eval_subjs = enrolled_subjs[n_dev:]
        
        # Dev Phase (Thresholding)
        dev_mask = np.isin(y_test_enrolled, dev_subjs)
        dev_gen, dev_imp = get_verification_scores(
            y_test_enrolled[dev_mask], probs_enrolled[dev_mask], clf.classes_, "closed_set"
        )
        
        labels_dev = [1]*len(dev_gen) + [0]*len(dev_imp)
        scores_dev = dev_gen + dev_imp
        fpr_dev, tpr_dev, thresh_dev = roc_curve(labels_dev, scores_dev, pos_label=1)
        fnr_dev = 1 - tpr_dev
        
        eer_idx = np.nanargmin(np.abs(fpr_dev - fnr_dev))
        thresh_eer = thresh_dev[eer_idx]
        
        interp_func = interp1d(fpr_dev, thresh_dev, bounds_error=False, fill_value="extrapolate")
        thresh_far1 = interp_func(0.01)
        thresh_far01 = interp_func(0.001)
        
        # --- D. Eval Phase ---
        eval_mask = np.isin(y_test_enrolled, eval_subjs)
        eval_gen, eval_imp_closed = get_verification_scores(
            y_test_enrolled[eval_mask], probs_enrolled[eval_mask], clf.classes_, "closed_set"
        )
        
        _, eval_imp_open = get_verification_scores(
            None, probs_open, clf.classes_, "open_set"
        )
        
        # --- ACCUMULATE SCORES FOR JOURNAL DET PLOT ---
        global_gen_scores.extend(eval_gen)
        global_closed_imp_scores.extend(eval_imp_closed)
        global_open_imp_scores.extend(eval_imp_open)
        
        # Calculate Metrics
        all_eval_imp = eval_imp_closed + eval_imp_open
        metrics = {}
        for name, thresh in [("EER", thresh_eer), ("FAR_1%", thresh_far1), ("FAR_0.1%", thresh_far01)]:
            frr = np.mean(np.array(eval_gen) < thresh)
            far_combined = np.mean(np.array(all_eval_imp) >= thresh)
            far_open_only = np.mean(np.array(eval_imp_open) >= thresh)
            
            metrics[f"FRR_at_{name}"] = frr
            metrics[f"FAR_Combined_at_{name}"] = far_combined
            metrics[f"FAR_OpenOnly_at_{name}"] = far_open_only
            
        y_eval = [1]*len(eval_gen) + [0]*len(all_eval_imp)
        s_eval = eval_gen + all_eval_imp
        fpr_final, tpr_final, _ = roc_curve(y_eval, s_eval, pos_label=1)
        metrics["AUROC_Combined"] = auc(fpr_final, tpr_final)
        
        fold_results.append(metrics)
        
    # --- 5. Aggregation & Artifact Generation ---
    df_res = pd.DataFrame(fold_results)
    summary = df_res.agg(['mean', 'std']).T
    
    print("\n=== Open-Set Analysis Results (5-Fold) ===")
    print(summary)
    
    df_res.to_csv(os.path.join(OUTPUT_DIR, "fold_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
    
    # CALL NEW ARTIFACT GENERATORS
    generate_journal_artifacts(global_gen_scores, global_closed_imp_scores, global_open_imp_scores)
    print_latex_table(summary)
    
    logging.info(f"All results and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Ensure HDF5 file exists or replace with your path
    if os.path.exists(HDF5_FILE):
        X_data, y_data, runs_data = load_data(HDF5_FILE)
        run_open_set_folds(X_data, y_data, runs_data)
    else:
        print(f"File {HDF5_FILE} not found. Please update the path.")