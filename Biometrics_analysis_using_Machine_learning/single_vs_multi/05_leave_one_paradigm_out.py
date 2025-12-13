# -*- coding: utf-8 -*-
"""
Priority 2.3: Leave-One-Paradigm-Out (LOPO) Generalization Analysis - Journal Version.

Methodology:
1. Iterate through each task T (Held-Out Target).
2. Train Model on Run 1 data from ALL OTHER tasks.
3. Test Model on Run 2 data from Task T.
4. Generates:
   - Combined ROC Plot (All tasks).
   - LaTeX Table (Acc, EER, AUC).
   - CSV Data for Origin Pro.
"""

import os
import logging
import random
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import PowerTransformer

# --- Configuration ---
HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
MAPPING_CSV = "epoch_mapping.csv"
OUTPUT_DIR = "results/leave_one_paradigm_out_journal"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- 1. Data Loading ---
def load_data_with_metadata(hdf5_path, mapping_path):
    if not os.path.exists(hdf5_path): raise FileNotFoundError(f"{hdf5_path} missing")
    
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
    
    df_map = pd.read_csv(mapping_path)
    df_map['subject'] = df_map['Subject'].str.split('_').str[1]
    df_map['run'] = df_map['Run']
    df_map['epoch'] = df_map['New_Epoch'].str.extract(r'epoch_(\d+)').astype(int)
    
    df_merged = pd.merge(df, df_map[['subject', 'run', 'epoch', 'Source_File']], 
                         on=['subject', 'run', 'epoch'], how='left')
    
    df_merged['task'] = df_merged['Source_File'].astype(str).apply(
        lambda x: x.split('.')[0].replace('_raw', '').replace('subject_', '') if pd.notna(x) else "Unknown"
    )
    
    return np.array(X), df_merged

# --- 2. Biometric Helpers ---
def get_verification_scores_full(y_true, scores_matrix, enrolled_classes):
    """
    Generates genuine and impostor scores for the ENTIRE set (for ROC plotting).
    Does NOT split into dev/test.
    """
    gen_scores = []
    imp_scores = []
    subj_map = {s: i for i, s in enumerate(enrolled_classes)}
    
    for i, true_s in enumerate(y_true):
        if true_s not in subj_map: continue
        
        # Genuine
        gen_scores.append(scores_matrix[i, subj_map[true_s]])
        
        # Impostor (Random One)
        others = [s for s in enrolled_classes if s != true_s]
        if others:
            imp_s = random.choice(others)
            imp_scores.append(scores_matrix[i, subj_map[imp_s]])
            
    return gen_scores, imp_scores

def calculate_eer_rigorous(y_true, scores_matrix, enrolled_classes, n_splits=5):
    """
    Calculates EER using repeated random Dev/Test splits (Rigorous for Table).
    """
    eers = []
    unique_subjs = np.unique(y_true)
    
    for _ in range(n_splits):
        # Split Subjects
        np.random.shuffle(unique_subjs)
        n_dev = max(2, int(len(unique_subjs) * 0.5))
        dev_subjs = unique_subjs[:n_dev]
        test_subjs = unique_subjs[n_dev:]
        
        # Helper inner function
        def get_subset_scores(target_subjs):
            mask = np.isin(y_true, target_subjs)
            if not np.any(mask): return [], []
            return get_verification_scores_full(y_true[mask], scores_matrix[mask], enrolled_classes)

        # Dev (Thresholding)
        dev_gen, dev_imp = get_subset_scores(dev_subjs)
        if not dev_gen: continue
        
        labels_dev = [1]*len(dev_gen) + [0]*len(dev_imp)
        fpr, tpr, thresh = roc_curve(labels_dev, dev_gen + dev_imp, pos_label=1)
        eer_thresh = thresh[np.nanargmin(np.abs(fpr - (1-tpr)))]
        
        # Test (Evaluation)
        test_gen, test_imp = get_subset_scores(test_subjs)
        if not test_gen: continue
        
        frr = np.mean(np.array(test_gen) < eer_thresh)
        far = np.mean(np.array(test_imp) >= eer_thresh)
        eers.append((frr + far) / 2)
        
    return np.mean(eers) if eers else np.nan

# --- 3. Artifact Generators ---
def generate_artifacts(results_list, roc_data_list):
    print("\n--- Generating Journal Artifacts ---")
    
    # A. LaTeX Table
    print("\n=== Journal Table LaTeX Code ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Held-Out Task & Accuracy & EER & AUC \\\\")
    print("\\hline")
    
    mean_acc, mean_eer, mean_auc = [], [], []
    
    for res in results_list:
        print(f"{res['Held_Out_Task']} & {res['Accuracy']*100:.1f}\\% & {res['EER']*100:.2f}\\% & {res['AUC']:.3f} \\\\")
        mean_acc.append(res['Accuracy'])
        mean_eer.append(res['EER'])
        mean_auc.append(res['AUC'])
        
    print("\\hline")
    print(f"\\textbf{{Average}} & \\textbf{{{np.mean(mean_acc)*100:.1f}\\%}} & \\textbf{{{np.mean(mean_eer)*100:.2f}\\%}} & \\textbf{{{np.mean(mean_auc):.3f}}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Generalization Performance (Leave-One-Paradigm-Out)}")
    print("\\end{table}")

    # B. Plotting ROCs
    plt.figure(figsize=(9, 7))
    
    # Save Data for Origin
    origin_rows = []
    
    for task_name, fpr, tpr, auc_val in roc_data_list:
        plt.plot(fpr, tpr, lw=2, label=f'{task_name} (AUC = {auc_val:.2f})')
        
        # Prepare Origin Data (Stacking columns)
        # We'll save a "Long" format csv which is easy to filter in Origin
        for f, t in zip(fpr, tpr):
            origin_rows.append({'Task': task_name, 'FPR': f, 'TPR': t})

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Generalization ROC: Performance on Unseen Tasks')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "journal_lopo_roc_combined.png"), dpi=300)
    print(f"Saved Plot to {OUTPUT_DIR}")
    
    # C. Save Origin CSV
    df_origin = pd.DataFrame(origin_rows)
    df_origin.to_csv(os.path.join(OUTPUT_DIR, "lopo_roc_data_for_origin.csv"), index=False)
    print("Saved Origin Data CSV.")

# --- 4. Main Engine ---
def run_lopo_journal(X_all, df_meta):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    valid_tasks = [t for t in df_meta['task'].unique() if "Unknown" not in t and "baseline" not in t.lower()]
    logging.info(f"Tasks for LOPO: {valid_tasks}")
    
    results = []
    roc_data_list = []
    
    for target_task in valid_tasks:
        logging.info(f"--- Processing: {target_task} ---")
        
        # Split Data
        train_mask = (df_meta['run'] == 'Run_1') & (df_meta['task'] != target_task)
        test_mask = (df_meta['run'] == 'Run_2') & (df_meta['task'] == target_task)
        
        X_train = X_all[train_mask]
        y_train = df_meta[train_mask]['subject'].values
        X_test = X_all[test_mask]
        y_test = df_meta[test_mask]['subject'].values
        
        if len(X_test) < 10: continue

        # Train
        transformer = PowerTransformer().fit(X_train.reshape(len(X_train), -1))
        X_train_tf = transformer.transform(X_train.reshape(len(X_train), -1))
        X_test_tf = transformer.transform(X_test.reshape(len(X_test), -1))
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_tf, y_train)
        
        # Predict
        probs = clf.predict_proba(X_test_tf)
        preds = clf.predict(X_test_tf)
        
        # --- Metrics ---
        # 1. Accuracy
        acc = accuracy_score(y_test, preds)
        
        # 2. EER (Rigorous)
        eer = calculate_eer_rigorous(y_test, probs, clf.classes_)
        
        # 3. AUC & ROC (Full Set for Plotting)
        gen, imp = get_verification_scores_full(y_test, probs, clf.classes_)
        y_roc = [1]*len(gen) + [0]*len(imp)
        s_roc = gen + imp
        fpr, tpr, _ = roc_curve(y_roc, s_roc, pos_label=1)
        auc_val = auc(fpr, tpr)
        
        # Store
        results.append({
            "Held_Out_Task": target_task,
            "Accuracy": acc,
            "EER": eer,
            "AUC": auc_val
        })
        roc_data_list.append((target_task, fpr, tpr, auc_val))
        
    # Generate Outputs
    generate_artifacts(results, roc_data_list)
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "lopo_final_results.csv"), index=False)

if __name__ == "__main__":
    if os.path.exists(HDF5_FILE):
        X_data, df_metadata = load_data_with_metadata(HDF5_FILE, MAPPING_CSV)
        run_lopo_journal(X_data, df_metadata)
    else:
        print("Data file not found.")