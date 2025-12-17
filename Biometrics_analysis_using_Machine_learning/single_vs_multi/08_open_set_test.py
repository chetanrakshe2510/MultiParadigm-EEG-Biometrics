# -*- coding: utf-8 -*-
"""
FINAL: Open-Set Sensitivity Analysis with DET Curves + EER Markers

Outputs:
1) sensitivity_metrics_final.csv   -> Table (EER, FRR@1%, FRR@0.1%, AUROC)
2) det_curve_open_set.png          -> Publication DET figure (with EER markers)
3) det_curve_data.xlsx             -> DET curve data for Origin
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
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import make_pipeline

# ---------------- CONFIG ----------------

HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
OUTPUT_DIR = "results/open_set_sensitivity_new2"

OPEN_SET_RATIOS = [0.1, 0.3, 0.5]
N_ITERATIONS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ---------------- METRIC HELPERS ----------------

def compute_det_curve(genuine, impostor):
    y_true = [1] * len(genuine) + [0] * len(impostor)
    y_scores = list(genuine) + list(impostor)
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr

def compute_eer(fpr, fnr):
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx], fnr[idx]

def frr_at_far(fpr, fnr, target_far):
    idx = np.nanargmin(np.abs(fpr - target_far))
    return fnr[idx]

# ---------------- PLOTTING ----------------

def plot_det_curves(det_data, eer_data, save_path):
    plt.figure(figsize=(7, 6))

    for label, (fpr, fnr) in det_data.items():
        plt.plot(fpr * 100, fnr * 100, linewidth=2, label=label)

        # --- EER marker ---
        if label in eer_data:
            eer_far, eer_frr = eer_data[label]
            plt.scatter(
                eer_far * 100,
                eer_frr * 100,
                s=60,
                edgecolors="black",
                zorder=5
            )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FAR (%)")
    plt.ylabel("FRR (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_det_to_excel(det_data, save_path):
    rows = []
    for label, (fpr, fnr) in det_data.items():
        for far, frr in zip(fpr, fnr):
            rows.append({
                "Curve": label,
                "FAR (%)": far * 100,
                "FRR (%)": frr * 100
            })
    pd.DataFrame(rows).to_excel(save_path, index=False)

# ---------------- DATA LOADING ----------------

def load_data(hdf5_path):
    X, y, runs = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        for subj in f.keys():
            subj_code = subj.split("_")[1]
            for run in f[subj].keys():
                for ep in f[subj][run].keys():
                    X.append(f[subj][run][ep]["features"][()])
                    y.append(subj_code)
                    runs.append(run)
    return np.array(X), np.array(y), np.array(runs)

# ---------------- SCORE EXTRACTION ----------------

def get_verification_scores(y_true, scores, classes, mode):
    gen, imp = [], []
    cls_map = {c: i for i, c in enumerate(classes)}

    if mode == "closed_set":
        for i, true_s in enumerate(y_true):
            gen.append(scores[i, cls_map[true_s]])
            other = random.choice([c for c in classes if c != true_s])
            imp.append(scores[i, cls_map[other]])

    elif mode == "open_set":
        for i in range(len(scores)):
            claim = random.choice(classes)
            imp.append(scores[i, cls_map[claim]])

    return gen, imp

# ---------------- MAIN ANALYSIS ----------------

def run_sensitivity_analysis(X, y, runs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subjects = np.unique(y)
    det_curves = {}
    eer_points = []
    summary = []

    for ratio in OPEN_SET_RATIOS:
        n_unknown = max(1, int(len(subjects) * ratio))
        logging.info(f"Open-set ratio {ratio}, Unknowns = {n_unknown}")

        all_gen, all_imp_closed, all_imp_open = [], [], []

        for _ in range(N_ITERATIONS):
            shuffled = np.random.permutation(subjects)
            unknown = shuffled[:n_unknown]
            enrolled = shuffled[n_unknown:]

            tr = np.isin(y, enrolled) & (runs == "Run_1")
            te_en = np.isin(y, enrolled) & (runs == "Run_2")
            te_op = np.isin(y, unknown) & (runs == "Run_2")

            pipe = make_pipeline(
                StandardScaler(),
                PowerTransformer(),
                LogisticRegression(max_iter=1000, n_jobs=-1)
            )
            pipe.fit(X[tr].reshape(np.sum(tr), -1), y[tr])

            pe = pipe.predict_proba(X[te_en].reshape(np.sum(te_en), -1))
            po = pipe.predict_proba(X[te_op].reshape(np.sum(te_op), -1))

            g, ic = get_verification_scores(y[te_en], pe, pipe.classes_, "closed_set")
            _, io = get_verification_scores(None, po, pipe.classes_, "open_set")

            all_gen.extend(g)
            all_imp_closed.extend(ic)
            all_imp_open.extend(io)

        impostors = all_imp_closed + all_imp_open

        fpr_c, fnr_c = compute_det_curve(all_gen, all_imp_closed)
        fpr_o, fnr_o = compute_det_curve(all_gen, impostors)

        if ratio == OPEN_SET_RATIOS[0]:
            det_curves["Closed-Set"] = (fpr_c, fnr_c)
            eer_c = compute_eer(fpr_c, fnr_c)
            eer_points.append(("Closed-Set", eer_c))

        det_curves[f"Open Set (Ratio {ratio})"] = (fpr_o, fnr_o)
        eer_o = compute_eer(fpr_o, fnr_o)
        eer_points.append((f"Open Set (Ratio {ratio})", eer_o))

        summary.append({
            "Ratio": ratio,
            "N_Unknowns": n_unknown,
            "EER (%)": np.mean(eer_o) * 100,
            "FRR @ 1% FAR (%)": frr_at_far(fpr_o, fnr_o, 0.01) * 100,
            "FRR @ 0.1% FAR (%)": frr_at_far(fpr_o, fnr_o, 0.001) * 100,
            "AUROC": auc(fpr_o, 1 - fnr_o)
        })

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_metrics_final.csv"), index=False)

    eer_dict = {k: v for k, v in eer_points}

    plot_det_curves(
        det_curves,
        eer_dict,
        os.path.join(OUTPUT_DIR, "det_curve_open_set.png")
    )

    save_det_to_excel(
        det_curves,
        os.path.join(OUTPUT_DIR, "det_curve_data.xlsx")
    )

    print("\nFINAL RESULTS:")
    print(df.to_string(index=False))

# ---------------- RUN ----------------

if __name__ == "__main__":
    if os.path.exists(HDF5_FILE):
        X, y, runs = load_data(HDF5_FILE)
        run_sensitivity_analysis(X, y, runs)
    else:
        print(f"HDF5 file not found: {HDF5_FILE}")
