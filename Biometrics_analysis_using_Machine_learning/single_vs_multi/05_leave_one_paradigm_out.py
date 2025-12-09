# -*- coding: utf-8 -*-
"""
Priority 2.3: Leave-One-Paradigm-Out (LOPO) Generalization Analysis.

Research Question:
"Can the system recognize a subject in a NEW task they were never trained on?"

Methodology:
1. Iterate through each task T (Held-Out Target).
2. Train Model on Run 1 data from ALL OTHER tasks (Task set - {T}).
3. Test Model on Run 2 data from Task T.
4. If performance >> chance, it proves 'Task-Independent Identity Representation'.
"""

import os
import logging
import random
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import PowerTransformer, LabelEncoder

# --- Configuration ---
# We use all available epochs for the training tasks to maximize generalization power
HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
MAPPING_CSV = "epoch_mapping.csv"
OUTPUT_DIR = "results/leave_one_paradigm_out"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- 1. Data Loading ---
def load_data_with_metadata(hdf5_path, mapping_path):
    """Loads features and merges with task metadata."""
    if not os.path.exists(hdf5_path): raise FileNotFoundError(f"{hdf5_path} missing")
    
    # Load Features
    X, y, runs, epochs = [], [], [], []
    with h5py.File(hdf5_path, 'r') as f:
        for subj in f.keys():
            subj_code = subj.split('_')[1]
            for run in f[subj].keys():
                for epoch_key in f[subj][run].keys():
                    feats = f[subj][run][epoch_key]['features'][()]
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    X.append(feats)
                    y.append(subj_code)
                    runs.append(run)
                    epochs.append(ep_idx)
    
    df = pd.DataFrame({'subject': y, 'run': runs, 'epoch': epochs, 'orig_idx': range(len(y))})
    
    # Load Mapping
    df_map = pd.read_csv(mapping_path)
    df_map['subject'] = df_map['Subject'].str.split('_').str[1]
    df_map['run'] = df_map['Run']
    df_map['epoch'] = df_map['New_Epoch'].str.extract(r'epoch_(\d+)').astype(int)
    
    df_merged = pd.merge(df, df_map[['subject', 'run', 'epoch', 'Source_File']], 
                         on=['subject', 'run', 'epoch'], how='left')
    
    # Simplify task names
    df_merged['task'] = df_merged['Source_File'].astype(str).apply(
        lambda x: x.split('.')[0].replace('_raw', '').replace('subject_', '') if pd.notna(x) else "Unknown"
    )
    
    return np.array(X), df_merged

# --- 2. Biometric Verification Helpers ---
def biometric_analysis_dev_test(y_true, scores_matrix, subjects, dev_ratio=0.5):
    """
    Performs Dev/Test split on the *subjects* to find robust EER/FRR.
    Note: For LOPO, we test on a specific task. We split that task's subjects.
    """
    unique_subjs = np.unique(y_true)
    n_dev = int(len(unique_subjs) * dev_ratio)
    if n_dev < 2: n_dev = len(unique_subjs) // 2 # Fallback
    
    # Random split of subjects
    random.shuffle(unique_subjs)
    dev_subjs = unique_subjs[:n_dev]
    test_subjs = unique_subjs[n_dev:]
    
    # Helper to generate scores
    def get_scores(target_subjs):
        mask = np.isin(y_true, target_subjs)
        y_subset = y_true[mask]
        s_subset = scores_matrix[mask]
        
        gen_scores, imp_scores = [], []
        subj_map = {s: i for i, s in enumerate(subjects)}
        
        for i, true_s in enumerate(y_subset):
            if true_s not in subj_map: continue
            
            # Genuine
            gen_scores.append(s_subset[i, subj_map[true_s]])
            
            # Impostor (Random One)
            others = [s for s in subjects if s != true_s]
            if others:
                imp_s = random.choice(others)
                imp_scores.append(s_subset[i, subj_map[imp_s]])
                
        return gen_scores, imp_scores

    # --- Dev Phase (Thresholding) ---
    dev_gen, dev_imp = get_scores(dev_subjs)
    if not dev_gen: return None
    
    labels = [1]*len(dev_gen) + [0]*len(dev_imp)
    scores = dev_gen + dev_imp
    fpr, tpr, thresh = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_thresh = thresh[eer_idx]
    
    # --- Test Phase (Evaluation) ---
    test_gen, test_imp = get_scores(test_subjs)
    if not test_gen: return None
    
    # Calculate Metrics on Test Set using Dev Threshold
    test_gen = np.array(test_gen)
    test_imp = np.array(test_imp)
    
    frr = np.mean(test_gen < eer_thresh) # False Rejection
    far = np.mean(test_imp >= eer_thresh) # False Acceptance
    eer = (frr + far) / 2
    
    return eer

# --- 3. LOPO Engine ---
def run_lopo(X_all, df_meta):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    valid_tasks = [t for t in df_meta['task'].unique() if "Unknown" not in t and "baseline" not in t.lower()]
    logging.info(f"Tasks for LOPO: {valid_tasks}")
    
    results = []
    
    for target_task in valid_tasks:
        logging.info(f"--- Holding Out: {target_task} ---")
        
        # 1. Define Train/Test Sets
        # Train: Run 1 of ALL tasks EXCEPT target
        train_mask = (df_meta['run'] == 'Run_1') & (df_meta['task'] != target_task)
        # Test: Run 2 of TARGET task
        test_mask = (df_meta['run'] == 'Run_2') & (df_meta['task'] == target_task)
        
        X_train = X_all[train_mask]
        y_train = df_meta[train_mask]['subject'].values
        
        X_test = X_all[test_mask]
        y_test = df_meta[test_mask]['subject'].values
        
        if len(X_test) < 10:
            logging.warning(f"Skipping {target_task}: Not enough test data.")
            continue

        # 2. Preprocess
        transformer = PowerTransformer().fit(X_train.reshape(len(X_train), -1))
        X_train_tf = transformer.transform(X_train.reshape(len(X_train), -1))
        X_test_tf = transformer.transform(X_test.reshape(len(X_test), -1))
        
        # 3. Train Classifier
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_tf, y_train)
        
        # 4. Predict
        probs = clf.predict_proba(X_test_tf)
        preds = clf.predict(X_test_tf)
        
        # 5. Metrics
        # A. Basic Accuracy
        acc = accuracy_score(y_test, preds)
        
        # B. Biometric EER (using Dev/Test split on the held-out task subjects)
        # We run this 5 times to get a stable estimate since subject split is random
        eers = []
        for _ in range(5):
            e = biometric_analysis_dev_test(y_test, probs, clf.classes_)
            if e is not None: eers.append(e)
        
        mean_eer = np.mean(eers) if eers else np.nan
        
        logging.info(f"Task: {target_task} | Acc: {acc:.4f} | EER: {mean_eer:.4f}")
        
        results.append({
            "Held_Out_Task": target_task,
            "Train_Tasks_Count": len(valid_tasks) - 1,
            "Accuracy": acc,
            "EER": mean_eer
        })

    # --- 6. Save & Plot ---
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "lopo_results.csv"), index=False)
    
    print("\n=== Leave-One-Paradigm-Out Results ===")
    print(df_res.sort_values('EER'))
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Held_Out_Task', y='EER', color='#3498db')
    plt.title("Generalization: EER on Unseen Tasks")
    plt.ylabel("Equal Error Rate (Lower is Better)")
    plt.xlabel("Held-Out Task (Test Set)")
    plt.xticks(rotation=45)
    plt.axhline(0.5, color='r', linestyle='--', label='Random Guess')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lopo_eer_barplot.png"))

if __name__ == "__main__":
    X_data, df_metadata = load_data_with_metadata(HDF5_FILE, MAPPING_CSV)
    run_lopo(X_data, df_metadata)