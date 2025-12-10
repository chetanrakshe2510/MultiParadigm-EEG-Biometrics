# -*- coding: utf-8 -*-
"""
Priority 4: Feature Importance & Interpretability Analysis.

Research Question:
"Which features (Channel, Domain, Band) contribute most to biometric identification?"

Methodology:
1. Train a 'Main' Logistic Regression model on Session 1 (Enrolled subjects).
2. Extract the Weight Matrix W (Shape: Classes x Features).
3. Compute Feature Importance: Average absolute weight across all classes.
   Imp_j = (1/C) * sum(|W_cj|)
4. Map indices j back to (Channel, Domain, Feature Name).
5. Aggregate and visualize importance by Domain and Channel.
"""

import os
import logging
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer

# --- Configuration ---
HDF5_FILE = "all_subjects_merged_new_full_epochs.h5"
OUTPUT_DIR = "results/feature_importance"
N_MFCC = 13  # Must match the extraction config

# Updated Channel Names provided by user
CHANNEL_NAMES = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5',
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6',
    'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2'
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- 1. Helper: Reconstruct Feature Names ---
def get_feature_names_order(n_mfcc=13):
    """
    Reconstructs the exact list of features per channel 
    based on features.py logic.
    """
    # Order must match features.py dictionary insertion order
    names = [
        "skewness", "kurtosis", "zero_crossing_rate",  # Stats
        "hjorth_mobility", "hjorth_complexity",        # Hjorth
        "delta_relative_power", "beta_relative_power", # Freq
        "spectral_entropy",
        "theta_alpha_ratio", "beta_theta_ratio",       # Ratios
        "permutation_entropy", "hurst_exponent",       # Non-linear
        "katz_fd", "higuchi_fd"
    ]
    # MFCCs
    for i in range(n_mfcc):
        names.append(f"mfcc_{i+1}")
    
    names.append("imf_1_entropy") # EMD
    
    return names

def map_feature_to_domain(feat_name):
    """Maps a feature name to a high-level domain."""
    if any(x in feat_name for x in ["skewness", "kurtosis", "zero_crossing", "hjorth"]):
        return "Time"
    elif "mfcc" in feat_name:
        return "Time-Frequency"
    elif any(x in feat_name for x in ["power", "ratio", "spectral_entropy"]):
        return "Frequency"
    elif any(x in feat_name for x in ["permutation", "hurst", "fd", "imf"]):
        return "Non-Linear"
    return "Other"

# --- 2. Data Loading ---
def load_data(hdf5_path):
    if not os.path.exists(hdf5_path): raise FileNotFoundError(f"{hdf5_path} missing")
    
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

# --- 3. Main Analysis ---
def run_feature_importance_analysis(X, y, runs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- A. Train Model ---
    logging.info("Training Main LR Model (Run 1)...")
    train_mask = (runs == 'Run_1')
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Flatten: (N_samples, N_channels, N_feats) -> (N_samples, N_flat)
    n_samples, n_channels, n_feats_per_ch = X_train.shape
    
    # Check if channel count matches provided names
    if n_channels != len(CHANNEL_NAMES):
        logging.warning(f"Data has {n_channels} channels, but {len(CHANNEL_NAMES)} names provided!")
        logging.warning("Using generic indices for channels instead.")
        ch_labels = [f"Ch{i+1}" for i in range(n_channels)]
    else:
        ch_labels = CHANNEL_NAMES

    X_train_flat = X_train.reshape(n_samples, -1)
    
    # Preprocess
    scaler = PowerTransformer()
    X_train_tf = scaler.fit_transform(X_train_flat)
    
    # Train
    clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, n_jobs=-1)
    clf.fit(X_train_tf, y_train)
    
    # --- B. Compute Importance ---
    # Weight Matrix W: (n_classes, n_features)
    W = clf.coef_ 
    
    # Importance = Mean Absolute Weight across classes
    # If binary, W is (1, n_features), imp is just abs(W)
    if W.ndim == 1: W = W.reshape(1, -1)
    
    feature_importance_flat = np.mean(np.abs(W), axis=0) # Shape: (n_flat,)
    
    # Reshape back to (Channels, Features)
    feature_importance_matrix = feature_importance_flat.reshape(n_channels, n_feats_per_ch)
    
    # --- C. Mapping & Aggregation ---
    feat_names = get_feature_names_order(N_MFCC)
    
    # Validate dimensions
    if len(feat_names) != n_feats_per_ch:
        logging.warning(f"Feature count mismatch! Expected {len(feat_names)}, got {n_feats_per_ch}. Check N_MFCC.")
        # Fallback: create dummy names
        feat_names = [f"feat_{i}" for i in range(n_feats_per_ch)]

    # 1. Aggregate by Channel
    channel_importance = np.mean(feature_importance_matrix, axis=1) # Mean imp per channel
    
    df_channel = pd.DataFrame({'Channel': ch_labels, 'Importance': channel_importance})
    df_channel = df_channel.sort_values('Importance', ascending=False)
    
    # 2. Aggregate by Domain
    domain_map = {name: map_feature_to_domain(name) for name in feat_names}
    
    # We need to average importance for each feature name across ALL channels first
    feat_avg_importance = np.mean(feature_importance_matrix, axis=0) # Shape: (n_feats_per_ch,)
    
    df_feat = pd.DataFrame({
        'Feature': feat_names,
        'Importance': feat_avg_importance,
        'Domain': [domain_map[n] for n in feat_names]
    })
    
    df_domain = df_feat.groupby('Domain')['Importance'].mean().reset_index().sort_values('Importance', ascending=False)
    
    # --- D. Saving & Plotting ---
    
    # Save CSVs
    df_channel.to_csv(os.path.join(OUTPUT_DIR, "importance_by_channel.csv"), index=False)
    df_feat.to_csv(os.path.join(OUTPUT_DIR, "importance_by_feature.csv"), index=False)
    df_domain.to_csv(os.path.join(OUTPUT_DIR, "importance_by_domain.csv"), index=False)
    
    print("\n=== Top 5 Channels ===")
    print(df_channel.head(5))
    print("\n=== Domain Importance ===")
    print(df_domain)

    # Plot 1: Domain Importance
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_domain, x='Domain', y='Importance', palette='viridis')
    plt.title("Mean Feature Importance by Domain")
    plt.ylabel("Mean Absolute Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_domain_importance.png"))
    
    # Plot 2: Channel Importance (Top 15)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_channel.head(15), x='Channel', y='Importance', color='#3498db')
    plt.title("Top 15 Channels by Importance")
    plt.xticks(rotation=45)
    plt.ylabel("Mean Absolute Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_channel_importance.png"))
    
    # Plot 3: Detailed Feature Importance (Top 20)
    plt.figure(figsize=(12, 8))
    df_feat_sorted = df_feat.sort_values('Importance', ascending=False).head(20)
    sns.barplot(data=df_feat_sorted, y='Feature', x='Importance', hue='Domain', dodge=False)
    plt.title("Top 20 Specific Features (Averaged across Channels)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_feature_detail_importance.png"))
    
    logging.info(f"Analysis complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    X_data, y_data, runs_data = load_data(HDF5_FILE)
    run_feature_importance_analysis(X_data, y_data, runs_data)