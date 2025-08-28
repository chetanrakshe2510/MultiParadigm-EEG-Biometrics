import os
import numpy as np
import h5py
import pandas as pd
from numpy.linalg import inv, pinv
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

###############################################################################
# 1. Data Loading (Unchanged)
###############################################################################
def load_features_from_hdf5(filename):
    """
    Loads EEG features from an HDF5 file in (n_samples, n_feat, n_chan) format.
    """
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            for run_key in h5f[class_key].keys():
                for epoch_key in h5f[class_key][run_key].keys():
                    feats = h5f[class_key][run_key][epoch_key]['features'][()]
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)


###############################################################################
# 2. Building Covariances & Templates (Unchanged)
###############################################################################
def build_subject_covariances_flat(X_data, y_data, alpha=1e-6):
    """
    For each subject, compute mean template and (regularized) covariance matrix + its inverse.
    """
    templates = {}
    cov_dict = {}
    for subj in np.unique(y_data):
        subj_data = X_data[y_data == subj]
        template = subj_data.mean(axis=0)
        emp_cov = np.cov(subj_data, rowvar=False)
        cov_matrix = emp_cov + alpha * np.eye(emp_cov.shape[1])
        try:
            inv_cov = inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = pinv(cov_matrix)
        templates[subj] = template
        cov_dict[subj] = inv_cov
    return templates, cov_dict


###############################################################################
# 3. Mahalanobis Prediction (Unchanged)
###############################################################################
def predict_mahalanobis_flat(sample, templates, cov_dict):
    """
    Compute Mahalanobis distance to each subject template, pick the smallest.
    """
    best_subj, best_dist = None, float('inf')
    for subj, tmpl in templates.items():
        inv_cov = cov_dict[subj]
        d = mahalanobis(sample, tmpl, inv_cov)
        if d < best_dist:
            best_dist = d
            best_subj = subj
    return best_subj


###############################################################################
# 4. Across-Run Analysis (Unchanged)
###############################################################################
def across_run_analysis(X, y, runs, epoch_nums, alpha=1e-4):
    """
    Train on Run_1, test on Run_2, using Mahalanobis classifier.
    Returns dict with predictions, true labels, and metrics.
    """
    # split indices
    train_idx = np.where(runs == "Run_1")[0]
    test_idx  = np.where(runs == "Run_2")[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Need both Run_1 and Run_2 data.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # build templates & inverse covariances
    templates, cov_dict = build_subject_covariances_flat(X_train, y_train, alpha=alpha)

    # predict
    preds = np.array([predict_mahalanobis_flat(s, templates, cov_dict)
                      for s in X_test])

    # compute metrics
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro")
    cm  = confusion_matrix(y_test, preds)

    return {
        "true_labels":   y_test,
        "predictions":   preds,
        "accuracy":      acc,
        "f1_macro":      f1,
        "confusion_mat": cm
    }


###############################################################################
# 5. Main Script (Fully Updated)
###############################################################################
def main():
    # --- 1. Path and Directory Setup (UPDATED) ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script's location
    DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'all_subjects_merged_new_full_epochs.h5')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'mahalanobis_results')

    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # --- End Path Setup ---
    
    # 5.1 Load raw X (n_samples, dim1, dim2)
    X_raw, y, runs, epoch_nums = load_features_from_hdf5(DATA_FILE)
    n_samples, dim1, dim2 = X_raw.shape
    print(f"[INFO] Loaded data: {n_samples} samples, dims=({dim1}, {dim2})")

    # 5.2 Infer which axis is features vs. channels
    if dim1 > dim2:
        feat_axis, chan_axis = 1, 2
    elif dim2 > dim1:
        feat_axis, chan_axis = 2, 1
    else:
        feat_axis, chan_axis = 1, 2
        print("[WARNING] equal dims; assuming axis 1=features, axis 2=channels")

    if (feat_axis, chan_axis) == (2, 1):
        X_raw = X_raw.transpose(0, 2, 1)

    n_feat = X_raw.shape[1]
    n_chan = X_raw.shape[2]
    print(f"[INFO] Interpreting as {n_feat} feature‐types × {n_chan} channels")

    # 5.3 Flatten and normalize
    X_flat = X_raw.reshape(n_samples, n_feat * n_chan)
    transformer = PowerTransformer(method="yeo-johnson")
    X = transformer.fit_transform(X_flat)

    # 5.4 Define your feature‐type ranges
    feature_ranges = {
        "Set_1_Time":                 list(range(0, 5)),
        "Set_2_Frequency":            list(range(5, 10)),
        "Set_3_TimeFrequency":        list(range(14, 28)),
        "Set_4_NonLinear":            list(range(10, 14)),
        "Set_5_Time+Frequency":       list(range(0, 10)),
        "Set_6_Time+TimeFrequency":   list(range(0, 5))  + list(range(14, 28)),
        "Set_7_Frequency+TimeFrequency": list(range(5, 10)) + list(range(14, 28)),
        "Set_8_Time+Frequency+NonLinear": list(range(0, 14)),
        "Set_9_Time+TimeFrequency+NonLinear": list(range(0, 5)) + list(range(10, 14)) + list(range(14, 28)),
        "Set_10_Time+Frequency+NonLinear (No TimeFreq)": list(range(0, 14)),
        "Set_11_All":                 list(range(0, n_feat))
    }

    # 5.5 Build actual flat‐vector indices for each set
    feature_combinations = {}
    for set_name, f_idxs in feature_ranges.items():
        indices = [f * n_chan + c for f in f_idxs for c in range(n_chan)]
        feature_combinations[set_name] = indices

    # 5.6 Loop through subsets and run analysis
    alpha = 1e-4
    for set_name, indices in feature_combinations.items():
        print(f"\n--- Running Mahalanobis on {set_name} ({len(indices)} features) ---")
        X_sub = X[:, indices]
        res = across_run_analysis(X_sub, y, runs, epoch_nums, alpha=alpha)

        # save detailed results (UPDATED to use RESULTS_DIR)
        df_out = pd.DataFrame({
            "True_Label":      res["true_labels"],
            "Predicted_Label": res["predictions"]
        })
        out_fname = f"results_Mahalanobis_{set_name}.csv"
        out_path  = os.path.join(RESULTS_DIR, out_fname)
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Saved labels to {out_path}")

        # also log metrics
        print(f"[METRICS] accuracy = {res['accuracy']:.4f}, "
              f"f1_macro = {res['f1_macro']:.4f}")
        print(f"[METRICS] confusion matrix:\n{res['confusion_mat']}")

if __name__ == "__main__":
    main()