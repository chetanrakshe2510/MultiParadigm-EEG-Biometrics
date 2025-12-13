# -*- coding: utf-8 -*-
"""
Priority 3 (Final): Open-Set Sensitivity Analysis + DET Curves
+ Closed-Set Baseline
+ FRR @ 1% FAR (Security Operating Point)

NO changes to training, scoring, or DET logic.
Only metric extraction and CSV summary added.
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------- CONFIGURATION ----------------

HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
OUTPUT_DIR = "results/open_set_sensitivity_det"

N_ITERATIONS = 5
DEV_SPLIT_RATIO = 0.5
OPEN_SET_RATIOS = [0.1, 0.3, 0.5]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ---------------- HELPERS ----------------

def compute_det_curve(genuine, impostor):
    y_true = [1] * len(genuine) + [0] * len(impostor)
    y_scores = list(genuine) + list(impostor)
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr

def compute_eer(fpr, fnr):
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx], fnr[idx]

def frr_at_far(fpr, fnr, target_far=0.01):
    idx = np.nanargmin(np.abs(fpr - target_far))
    return fnr[idx]

# ---------------- DATA LOADING ----------------

def load_data(hdf5_path):
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

# ---------------- SCORING ----------------

def get_verification_scores(y_true, scores_matrix, enrolled_classes, mode):
    gen, imp = [], []
    subj_to_idx = {s: i for i, s in enumerate(enrolled_classes)}

    if mode == "closed_set":
        for i, true_s in enumerate(y_true):
            gen.append(scores_matrix[i, subj_to_idx[true_s]])
            other = random.choice([s for s in enrolled_classes if s != true_s])
            imp.append(scores_matrix[i, subj_to_idx[other]])

    elif mode == "open_set":
        for i in range(len(scores_matrix)):
            claim = random.choice(enrolled_classes)
            imp.append(scores_matrix[i, subj_to_idx[claim]])

    return gen, imp

# ---------------- MAIN ANALYSIS ----------------

def run_sensitivity_analysis(X, y, runs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_subjects = np.unique(y)
    summary_rows = []

    ratio_scores = {r: {'gen': [], 'imp_closed': [], 'imp_open': []}
                    for r in OPEN_SET_RATIOS}

    for ratio in OPEN_SET_RATIOS:
        n_held_out = max(1, int(len(unique_subjects) * ratio))
        logging.info(f"Ratio {ratio}: Held-out subjects = {n_held_out}")

        for _ in range(N_ITERATIONS):
            shuffled = np.random.permutation(unique_subjects)
            non_enrolled = shuffled[:n_held_out]
            enrolled = shuffled[n_held_out:]

            train_mask = np.isin(y, enrolled) & (runs == 'Run_1')
            test_enrolled_mask = np.isin(y, enrolled) & (runs == 'Run_2')
            test_open_mask = np.isin(y, non_enrolled) & (runs == 'Run_2')

            X_train = X[train_mask].reshape(np.sum(train_mask), -1)
            y_train = y[train_mask]

            pipe = make_pipeline(
                StandardScaler(),
                PowerTransformer(),
                LogisticRegression(max_iter=1000, n_jobs=-1)
            )
            pipe.fit(X_train, y_train)

            probs_enrolled = pipe.predict_proba(
                X[test_enrolled_mask].reshape(np.sum(test_enrolled_mask), -1)
            )
            probs_open = pipe.predict_proba(
                X[test_open_mask].reshape(np.sum(test_open_mask), -1)
            )

            random.shuffle(enrolled)
            eval_subjs = enrolled[int(len(enrolled) * DEV_SPLIT_RATIO):]
            eval_mask = np.isin(y[test_enrolled_mask], eval_subjs)

            gen, imp_closed = get_verification_scores(
                y[test_enrolled_mask][eval_mask],
                probs_enrolled[eval_mask],
                pipe.classes_,
                "closed_set"
            )

            _, imp_open = get_verification_scores(
                None,
                probs_open,
                pipe.classes_,
                "open_set"
            )

            ratio_scores[ratio]['gen'] += gen
            ratio_scores[ratio]['imp_closed'] += imp_closed
            ratio_scores[ratio]['imp_open'] += imp_open

        # -------- METRICS --------

        data = ratio_scores[ratio]

        fpr_c, fnr_c = compute_det_curve(data['gen'], data['imp_closed'])
        fpr_o, fnr_o = compute_det_curve(data['gen'], data['imp_open'])
        fpr_comb, fnr_comb = compute_det_curve(
            data['gen'], data['imp_closed'] + data['imp_open']
        )

        eer_far_c, eer_frr_c = compute_eer(fpr_c, fnr_c)
        eer_far_o, _ = compute_eer(fpr_o, fnr_o)
        eer_far_comb, eer_frr_comb = compute_eer(fpr_comb, fnr_comb)

        frr_1pct_closed = frr_at_far(fpr_c, fnr_c, 0.01)
        frr_1pct_open = frr_at_far(fpr_o, fnr_o, 0.01)
        frr_1pct_comb = frr_at_far(fpr_comb, fnr_comb, 0.01)

        summary_rows.append({
            "Ratio": ratio,
            "ClosedSet_FAR_EER (%)": eer_far_c * 100,
            "OpenSet_FAR_EER (%)": eer_far_o * 100,
            "Combined_FAR_EER (%)": eer_far_comb * 100,
            "FRR_EER (%)": eer_frr_comb * 100,
            "FRR_1pctFAR_Closed (%)": frr_1pct_closed * 100,
            "FRR_1pctFAR_Open (%)": frr_1pct_open * 100,
            "FRR_1pctFAR_Combined (%)": frr_1pct_comb * 100,
            "AUROC_Combined": auc(fpr_comb, 1 - fnr_comb)
        })

    df_summary = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUTPUT_DIR, "sensitivity_summary_with_closed_and_security.csv")
    df_summary.to_csv(out_csv, index=False)

    print(f"\nSaved summary CSV -> {out_csv}")
    print(df_summary)

# ---------------- RUN ----------------

if __name__ == "__main__":
    if os.path.exists(HDF5_FILE):
        X, y, runs = load_data(HDF5_FILE)
        run_sensitivity_analysis(X, y, runs)
    else:
        print("HDF5 file not found.")
