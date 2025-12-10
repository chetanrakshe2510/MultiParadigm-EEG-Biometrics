import os
import numpy as np
import h5py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from scipy.interpolate import interp1d
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import keras_tuner as kt

# -------------------------------------------------------------------------
# Global seeds for reproducibility
# -------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

###############################################################################
# 1. Data Loading
###############################################################################
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
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

###############################################################################
# 2. Biometric Verification Helpers (Uniform Analysis)
###############################################################################
def generate_biometric_trials(y_test, scores_matrix, subject_list, impostor_samples=5):
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        true_idx = subject_to_idx[true_subj]
        
        # Genuine Trial
        scores.append(scores_matrix[i, true_idx])
        labels.append(1) 
        
        # Impostor Trials
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

    # Dev Set (Thresholding)
    dev_scores, dev_labels = generate_biometric_trials(y_test[dev_mask], scores_matrix[dev_mask], all_subjects)
    fpr, tpr, thresholds = roc_curve(dev_labels, dev_scores, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    fpr_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
    thresh_far_1_pc = fpr_interp(0.01)
    thresh_far_01_pc = fpr_interp(0.001)

    # Test Set (Reporting)
    test_scores, test_labels = generate_biometric_trials(y_test[test_mask], scores_matrix[test_mask], all_subjects)
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_scores, pos_label=1)
    fnr_test = 1 - tpr_test
    auroc_test = auc(fpr_test, tpr_test)
    
    def get_frr_at_thresh(s, l, t):
        preds = (s >= t).astype(int)
        gen_mask = (l == 1)
        fn = np.sum(preds[gen_mask] == 0)
        tp = np.sum(preds[gen_mask] == 1)
        return fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        "auroc": auroc_test,
        "fpr": fpr_test,
        "fnr": fnr_test,
        "frr_at_far1pc": get_frr_at_thresh(test_scores, test_labels, thresh_far_1_pc),
        "frr_at_far01pc": get_frr_at_thresh(test_scores, test_labels, thresh_far_01_pc),
        "eer_rate_dev": eer_rate
    }

def compute_biometric_ci(y_test, scores_matrix, all_subjects, n_bootstraps=200):
    print(f"Starting Bootstrapping ({n_bootstraps} iterations)...")
    test_subjects_all = np.unique(y_test)
    metrics = {'auroc': [], 'eer': [], 'frr_at_far1pc': [], 'frr_at_far01pc': []}
    
    subj_indices_map = {s: np.where(y_test == s)[0] for s in test_subjects_all}

    for b in range(n_bootstraps):
        sampled_subjects = np.random.choice(test_subjects_all, size=len(test_subjects_all), replace=True)
        
        indices = []
        for s in sampled_subjects:
            indices.extend(subj_indices_map[s])
        
        if not indices: continue
        indices = np.array(indices)
        
        res = biometric_analysis_dev_test(y_test[indices], scores_matrix[indices], all_subjects)
        if res:
            metrics['auroc'].append(res['auroc'])
            metrics['eer'].append(res['eer_rate_dev'])
            metrics['frr_at_far1pc'].append(res['frr_at_far1pc'])
            metrics['frr_at_far01pc'].append(res['frr_at_far01pc'])
            
    cis = {}
    for m, vals in metrics.items():
        cis[m] = np.percentile(vals, [2.5, 97.5]) if vals else [0, 0]
    return cis, metrics

# --- NEW: Function to save Per-Subject Metrics for Boxplots (Fig 4) ---
def compute_and_save_per_subject_metrics(y_true, predictions, scores_matrix, subjects, save_path):
    rows = []
    subj_to_idx = {s: i for i, s in enumerate(sorted(subjects))}

    for subj in subjects:
        y_true_binary = (y_true == subj).astype(int)
        y_pred_binary = (predictions == subj).astype(int)
        
        # Binary Scores for this subject (One-vs-Rest)
        if subj in subj_to_idx:
            y_scores_binary = scores_matrix[:, subj_to_idx[subj]]
        else:
            y_scores_binary = np.zeros(len(y_true))

        acc = accuracy_score(y_true_binary, y_pred_binary)
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
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
        
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[INFO] Saved per-subject metrics to {save_path}")

###############################################################################
# 3. Hypermodel
###############################################################################
def build_tunable_1d_cnn_model(hp):
    filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16, default=32)
    kernel_size_1 = hp.Choice('kernel_size_1', values=[3, 5, 7], default=5)
    filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32, default=64)
    kernel_size_2 = hp.Choice('kernel_size_2', values=[3, 5], default=3)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64, default=128)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)

    global input_shape, num_classes
    model = models.Sequential([
        layers.Conv1D(filters=filters_1, kernel_size=kernel_size_1, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=filters_2, kernel_size=kernel_size_2, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

###############################################################################
# 4. Main Script
###############################################################################
def main():
    filename = "all_subjects_merged_new_full_epochs.h5"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")

    X, y, runs, epoch_nums = load_features_from_hdf5(filename)

    # === FIX: DATA LEAKAGE PREVENTION ===
    train_idx = np.where(runs == "Run_1")[0]
    test_idx  = np.where(runs == "Run_2")[0]
    
    if len(train_idx)==0 or len(test_idx)==0:
        raise ValueError("Insufficient data for Run_1 or Run_2.")

    # 2. Reshape for Transformer
    X_flat = X.reshape(X.shape[0], -1)
    
    # 3. Fit Transformer ONLY on Training Data
    transformer = PowerTransformer(method='yeo-johnson')
    X_train_flat = X_flat[train_idx]
    transformer.fit(X_train_flat)
    
    # 4. Transform both sets
    X_train_flat = transformer.transform(X_flat[train_idx])
    X_test_flat = transformer.transform(X_flat[test_idx])
    
    flattened_length = X_train_flat.shape[1]
    X_train_cnn = X_train_flat.reshape(X_train_flat.shape[0], flattened_length, 1)
    X_test_cnn = X_test_flat.reshape(X_test_flat.shape[0], flattened_length, 1)

    y_train = y[train_idx]
    y_test  = y[test_idx]

    joblib.dump(transformer, "power_transformer_cnn.pkl")

    # === Label Encoding ===
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    global num_classes, input_shape
    num_classes = len(np.unique(y_train_enc))
    input_shape = (flattened_length, 1)
    
    y_train_cat = utils.to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat = utils.to_categorical(y_test_enc, num_classes=num_classes)

    # === Tuning Split ===
    X_train_sub, X_val, y_train_sub_cat, y_val_cat = train_test_split(
        X_train_cnn, y_train_cat, test_size=0.1, stratify=y_train_enc, random_state=42
    )

    tuner = kt.RandomSearch(
        build_tunable_1d_cnn_model,
        objective='val_accuracy',
        max_trials=10,
        directory='kt_1d_cnn_tuning',
        project_name='1d_cnn_v2'
    )
    
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    tuner.search(X_train_sub, y_train_sub_cat, validation_data=(X_val, y_val_cat),
                 epochs=20, callbacks=[early_stop], verbose=1)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    
    history = best_model.fit(X_train_sub, y_train_sub_cat, epochs=20, 
                             validation_data=(X_val, y_val_cat), callbacks=[early_stop], verbose=1)

    # === Basic Evaluation ===
    test_loss, test_acc = best_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    preds_prob = best_model.predict(X_test_cnn)
    preds_enc = np.argmax(preds_prob, axis=1)
    preds = le.inverse_transform(preds_enc)
    
    # === NEW: BIOMETRIC VERIFICATION ANALYSIS ===
    print("\n--- Running Biometric Verification Analysis (Bootstrapped) ---")
    cis, metrics = compute_biometric_ci(y_test, preds_prob, le.classes_, n_bootstraps=200)
    
    # Get point estimates (single run)
    point_est = biometric_analysis_dev_test(y_test, preds_prob, le.classes_)
    
    # === PLOTS ===
    # 1. DET Curve
    plt.figure(figsize=(8, 6))
    plt.plot(point_est['fpr'], point_est['fnr'], label=f"CNN (EER={point_est['eer_rate_dev']*100:.2f}%)")
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('False Acceptance Rate (FAR)'); plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Curve (1D CNN)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('det_curve_cnn.png')
    plt.close()
    
    # === REPORTING ===
    results = {
        "Test Accuracy": f"{test_acc:.4f}",
        "EER (Dev)": f"{point_est['eer_rate_dev']*100:.2f}% ({cis['eer'][0]*100:.2f}-{cis['eer'][1]*100:.2f})",
        "FRR @ 1% FAR": f"{point_est['frr_at_far1pc']*100:.2f}% ({cis['frr_at_far1pc'][0]*100:.2f}-{cis['frr_at_far1pc'][1]*100:.2f})",
        "FRR @ 0.1% FAR": f"{point_est['frr_at_far01pc']*100:.2f}% ({cis['frr_at_far01pc'][0]*100:.2f}-{cis['frr_at_far01pc'][1]*100:.2f})"
    }
    
    print("\n=== FINAL RESULTS (Uniform Analysis) ===")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    # Save Results
    pd.DataFrame([results]).to_csv("cnn_biometric_results.csv", index=False)
    
    # === NEW: Save Per-Subject Metrics for Figure 4 Boxplots ===
    # Use the sorted classes from LabelEncoder to ensure column mapping is correct
    compute_and_save_per_subject_metrics(
        y_test, preds, preds_prob, le.classes_,
        "per_subject_metrics_cnn.csv"
    )
    
    best_model.save("best_1d_cnn_model.h5")

if __name__ == "__main__":
    main()