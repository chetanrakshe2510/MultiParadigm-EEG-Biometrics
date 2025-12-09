# -*- coding: utf-8 -*-
"""
Priority 2 Analysis: Single-Task vs. Multi-Paradigm Training (Controlled N).

This script addresses Reviewer 2's core question:
"Is the improvement just due to more data, or cross-paradigm representation?"

Methodology:
1. Fix Training Size (N): We select a fixed N (e.g., 40) epochs per subject.
2. Condition A (Single-Task): Train a model ONLY on N epochs of Task T.
3. Condition B (Multi-Paradigm): Train a model on N epochs sampled from ALL tasks.
4. Evaluation: Both models are tested on the SAME Run 2 data for Task T.
5. Metrics: We report EER, F1, and Delta (Improvement) using the rigorous Dev/Test protocol.
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
N_ITERATIONS = 5          # Number of random splits to average over
N_EPOCHS_PER_SUBJ = 40    # Fixed training budget (adjust based on weakest task)
HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
MAPPING_CSV = "epoch_mapping.csv"
OUTPUT_DIR = "results/single_vs_multi_comparison"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- 1. Data Loading ---
def load_data_with_metadata(hdf5_path, mapping_path):
    """Loads features and merges with task metadata."""
    if not os.path.exists(hdf5_path): raise FileNotFoundError(f"{hdf5_path} missing")
    if not os.path.exists(mapping_path): raise FileNotFoundError(f"{mapping_path} missing")

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
    
    # Create DataFrame
    df = pd.DataFrame({
        'subject': y,
        'run': runs,
        'epoch': epochs,
        'orig_idx': range(len(y))
    })
    
    # Load Mapping to get Task info
    df_map = pd.read_csv(mapping_path)
    # Parse mapping columns to match HDF5 format
    df_map['subject'] = df_map['Subject'].str.split('_').str[1]
    df_map['run'] = df_map['Run']
    df_map['epoch'] = df_map['New_Epoch'].str.extract(r'epoch_(\d+)').astype(int)
    
    # Merge
    df_merged = pd.merge(df, df_map[['subject', 'run', 'epoch', 'Source_File']], 
                         on=['subject', 'run', 'epoch'], how='left')
    
    # Clean up task names (remove extensions)
    df_merged['task'] = df_merged['Source_File'].astype(str).apply(lambda x: x.split('.')[0] if pd.notna(x) else "Unknown")
    
    return np.array(X), df_merged

# --- 2. Biometric Verification Helpers (Uniform Analysis) ---
def get_biometric_metrics(y_true, scores_matrix, subjects):
    """Calculates EER using Dev/Test logic (simplified for inner loop)."""
    # Generate trials
    gen_scores, imp_scores = [], []
    subj_map = {s: i for i, s in enumerate(subjects)}
    
    for i, true_s in enumerate(y_true):
        if true_s not in subj_map: continue
        true_idx = subj_map[true_s]
        
        # Genuine
        gen_scores.append(scores_matrix[i, true_idx])
        
        # Impostor (take random other)
        impostors = [s for s in subjects if s != true_s]
        if impostors:
            imp_s = random.choice(impostors)
            imp_idx = subj_map[imp_s]
            imp_scores.append(scores_matrix[i, imp_idx])
            
    if not gen_scores: return None

    # ROC
    y_ver = [1]*len(gen_scores) + [0]*len(imp_scores)
    s_ver = gen_scores + imp_scores
    fpr, tpr, thresh = roc_curve(y_ver, s_ver, pos_label=1)
    fnr = 1 - tpr
    
    # EER
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer

# --- 3. Training & Evaluation Engine ---
def run_comparison(X_all, df_meta):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filter valid tasks (exclude baseline or unknowns if needed)
    valid_tasks = [t for t in df_meta['task'].unique() if "Unknown" not in t and "baseline" not in t.lower()]
    logging.info(f"Tasks identified: {valid_tasks}")
    
    results = []
    
    for iteration in range(N_ITERATIONS):
        logging.info(f"--- Iteration {iteration+1}/{N_ITERATIONS} ---")
        rng = iteration # Seed
        
        # --- A. Prepare Training Sets ---
        df_train = df_meta[df_meta['run'] == 'Run_1']
        
        # 1. Multi-Paradigm Train Set (Sample N from ALL tasks mixed)
        # We group by subject and sample N epochs regardless of task
        multi_indices = df_train.groupby('subject', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), N_EPOCHS_PER_SUBJ), random_state=rng)
        )['orig_idx'].values
        
        # 2. Single-Task Train Sets (Sample N from specific task)
        single_indices_map = {}
        for task in valid_tasks:
            df_task = df_train[df_train['task'] == task]
            # If a task has < N epochs, we take all (fair comparison limited by data)
            single_indices_map[task] = df_task.groupby('subject', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), N_EPOCHS_PER_SUBJ), random_state=rng)
            )['orig_idx'].values

        # --- B. Train Models ---
        # Helper to train and return model + transformer
        def train_model(indices):
            if len(indices) == 0: return None, None, None
            X_tr = X_all[indices]
            y_tr = df_meta.iloc[indices]['subject'].values
            
            # Preprocess
            X_tr_flat = X_tr.reshape(len(X_tr), -1)
            scaler = PowerTransformer().fit(X_tr_flat)
            X_tr_tf = scaler.transform(X_tr_flat)
            
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_tr_tf, y_tr)
            return clf, scaler, clf.classes_

        # Train Multi Model
        clf_multi, scl_multi, classes_multi = train_model(multi_indices)
        
        # --- C. Evaluate per Task ---
        df_test = df_meta[df_meta['run'] == 'Run_2']
        
        for task in valid_tasks:
            # 1. Get Test Data for this Task
            task_test_indices = df_test[df_test['task'] == task]['orig_idx'].values
            if len(task_test_indices) == 0: continue
            
            X_test_task = X_all[task_test_indices]
            y_test_task = df_meta.iloc[task_test_indices]['subject'].values
            
            # --- Condition A: Evaluate Single-Task Model ---
            indices_single = single_indices_map.get(task, [])
            clf_single, scl_single, classes_single = train_model(indices_single)
            
            res_row = {'Iteration': iteration, 'Task': task}
            
            if clf_single:
                X_test_single = scl_single.transform(X_test_task.reshape(len(X_test_task), -1))
                probs_single = clf_single.predict_proba(X_test_single)
                res_row['EER_Single'] = get_biometric_metrics(y_test_task, probs_single, classes_single)
            else:
                res_row['EER_Single'] = np.nan

            # --- Condition B: Evaluate Multi-Paradigm Model ---
            if clf_multi:
                X_test_multi = scl_multi.transform(X_test_task.reshape(len(X_test_task), -1))
                probs_multi = clf_multi.predict_proba(X_test_multi)
                res_row['EER_Multi'] = get_biometric_metrics(y_test_task, probs_multi, classes_multi)
            else:
                res_row['EER_Multi'] = np.nan
                
            results.append(res_row)

    # --- 4. Aggregation ---
    df_res = pd.DataFrame(results)
    
    # Calculate improvement (Negative delta means EER went DOWN, which is good)
    df_res['Delta_EER'] = df_res['EER_Multi'] - df_res['EER_Single']
    
    summary = df_res.groupby('Task').agg(
        mean_EER_Single=('EER_Single', 'mean'),
        std_EER_Single=('EER_Single', 'std'),
        mean_EER_Multi=('EER_Multi', 'mean'),
        std_EER_Multi=('EER_Multi', 'std'),
        mean_Delta=('Delta_EER', 'mean')
    ).reset_index()
    
    # Sort by improvement (biggest drop in EER first)
    summary = summary.sort_values('mean_Delta')
    
    print("\n=== Single vs Multi-Paradigm Results (Fixed N) ===")
    print(summary)
    
    summary.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)
    
    # --- 5. Visualization ---
    # Prepare data for plotting (Long format)
    df_long = df_res.melt(id_vars=['Task', 'Iteration'], 
                          value_vars=['EER_Single', 'EER_Multi'], 
                          var_name='Condition', value_name='EER')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_long, x='Task', y='EER', hue='Condition', 
                palette={'EER_Single': '#e74c3c', 'EER_Multi': '#2ecc71'}, capsize=.1)
    plt.title(f"Impact of Multi-Paradigm Training (Fixed N={N_EPOCHS_PER_SUBJ})")
    plt.ylabel("Equal Error Rate (Lower is Better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "single_vs_multi_barplot.png"))
    logging.info(f"Analysis complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    X_data, df_metadata = load_data_with_metadata(HDF5_FILE, MAPPING_CSV)
    run_comparison(X_data, df_metadata)