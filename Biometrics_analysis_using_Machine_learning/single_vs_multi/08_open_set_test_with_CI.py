# -*- coding: utf-8 -*-
"""
FINAL++: Open-Set Sensitivity Analysis
- DET curves (log–log)
- EER markers
- Confidence bands (bootstrap)
- Excel exports for Origin:
    1) det_curve_data.xlsx
    2) eer_points.xlsx
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
from sklearn.utils import resample

# ---------------- CONFIG ----------------

HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
OUTPUT_DIR = "results/open_set_sensitivity_det_CI"

OPEN_SET_RATIOS = [0.1, 0.3, 0.5]
N_ITERATIONS = 5
N_BOOTSTRAPS = 200        # for confidence bands
CI_ALPHA = 0.95           # 95% CI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ---------------- METRIC HELPERS ----------------

def compute_det_curve(genuine, impostor):
    y_true = np.array([1]*len(genuine) + [0]*len(impostor))
    y_scores = np.array(list(genuine) + list(impostor))
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr

def compute_eer(fpr, fnr):
    idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[idx], fnr[idx]

def frr_at_far(fpr, fnr, target_far):
    idx = np.nanargmin(np.abs(fpr - target_far))
    return fnr[idx]

# ---------------- BOOTSTRAP CI ----------------

def bootstrap_det_ci(genuine, impostor, base_fpr):
    """
    Returns lower and upper FRR confidence bounds
    interpolated at base_fpr locations
    """
    fnr_samples = []

    for _ in range(N_BOOTSTRAPS):
        g_bs = resample(genuine, replace=True)
        i_bs = resample(impostor, replace=True)
        fpr, fnr = compute_det_curve(g_bs, i_bs)
        fnr_interp = np.interp(base_fpr, fpr, fnr)
        fnr_samples.append(fnr_interp)

    fnr_samples = np.array(fnr_samples)
    low = np.percentile(fnr_samples, (1-CI_ALPHA)/2*100, axis=0)
    high = np.percentile(fnr_samples, (1+CI_ALPHA)/2*100, axis=0)
    return low, high

# ---------------- PLOTTING ----------------

def plot_det_curves(det_data, eer_data, ci_data, save_path):
    plt.figure(figsize=(7, 6))

    for label, (fpr, fnr) in det_data.items():
        plt.plot(fpr*100, fnr*100, linewidth=2, label=label)

        # Confidence band
        if label in ci_data:
            lo, hi = ci_data[label]
            plt.fill_between(
                fpr*100, lo*100, hi*100,
                alpha=0.2
            )

        # EER marker
        if label in eer_data:
            eer_far, eer_frr = eer_data[label]
            plt.scatter(
                eer_far*100, eer_frr*100,
                s=60, edgecolors="black", zorder=5
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

# ---------------- EXPORT HELPERS ----------------

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

def save_eer_to_excel(eer_dict, save_path):
    rows = []
    for label, (far, frr) in eer_dict.items():
        rows.append({
            "Curve": label,
            "EER FAR (%)": far * 100,
            "EER FRR (%)": frr * 100,
            "EER (%)": np.mean([far, frr]) * 100
        })
    pd.DataFrame(rows).to_excel(save_path, index=False)

# ---------------- DATA LOADING ----------------

def load_data(hdf5_path):
    X, y, runs = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        for subj in f.keys():
            sid = subj.split("_")[1]
            for run in f[subj].keys():
                for ep in f[subj][run].keys():
                    X.append(f[subj][run][ep]["features"][()])
                    y.append(sid)
                    runs.append(run)
    return np.array(X), np.array(y), np.array(runs)

# ---------------- SCORE EXTRACTION ----------------

def get_verification_scores(y_true, scores, classes, mode):
    gen, imp = [], []
    cmap = {c:i for i,c in enumerate(classes)}

    if mode == "closed_set":
        for i, s in enumerate(y_true):
            gen.append(scores[i, cmap[s]])
            other = random.choice([c for c in classes if c != s])
            imp.append(scores[i, cmap[other]])

    elif mode == "open_set":
        for i in range(len(scores)):
            claim = random.choice(classes)
            imp.append(scores[i, cmap[claim]])

    return gen, imp

# ---------------- MAIN ----------------

def run_sensitivity_analysis(X, y, runs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subjects = np.unique(y)
    det_curves, ci_curves, eer_points = {}, {}, {}
    summary = []

    for ratio in OPEN_SET_RATIOS:
        n_unknown = max(1, int(len(subjects) * ratio))
        logging.info(f"Open-set ratio {ratio}, Unknowns={n_unknown}")

        gen_all, imp_closed_all, imp_open_all = [], [], []

        for _ in range(N_ITERATIONS):
            perm = np.random.permutation(subjects)
            unknown, enrolled = perm[:n_unknown], perm[n_unknown:]

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

            gen_all.extend(g)
            imp_closed_all.extend(ic)
            imp_open_all.extend(io)

        impostors = imp_closed_all + imp_open_all

        fpr_o, fnr_o = compute_det_curve(gen_all, impostors)
        eer_far, eer_frr = compute_eer(fpr_o, fnr_o)

        label = f"Open Set (Ratio {ratio})"
        det_curves[label] = (fpr_o, fnr_o)
        eer_points[label] = (eer_far, eer_frr)

        lo, hi = bootstrap_det_ci(gen_all, impostors, fpr_o)
        ci_curves[label] = (lo, hi)

        summary.append({
            "Ratio": ratio,
            "N_Unknowns": n_unknown,
            "EER (%)": np.mean([eer_far, eer_frr]) * 100,
            "FRR @ 1% FAR (%)": frr_at_far(fpr_o, fnr_o, 0.01) * 100,
            "FRR @ 0.1% FAR (%)": frr_at_far(fpr_o, fnr_o, 0.001) * 100,
            "AUROC": auc(fpr_o, 1 - fnr_o)
        })

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_metrics_final.csv"), index=False)

    plot_det_curves(
        det_curves,
        eer_points,
        ci_curves,
        os.path.join(OUTPUT_DIR, "det_curve_open_set.png")
    )

    save_det_to_excel(det_curves,
                      os.path.join(OUTPUT_DIR, "det_curve_data.xlsx"))

    save_eer_to_excel(eer_points,
                      os.path.join(OUTPUT_DIR, "eer_points.xlsx"))

    print("\nFINAL RESULTS:")
    print(df.to_string(index=False))

# ---------------- RUN ----------------

if __name__ == "__main__":
    if os.path.exists(HDF5_FILE):
        X, y, runs = load_data(HDF5_FILE)
        run_sensitivity_analysis(X, y, runs)
    else:
        print(f"HDF5 file not found: {HDF5_FILE}")
