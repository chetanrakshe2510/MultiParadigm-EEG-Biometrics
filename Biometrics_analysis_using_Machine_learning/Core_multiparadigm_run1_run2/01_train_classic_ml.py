import os
import json
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from datetime import datetime
import random # NEW: For random sampling in verification

from scipy.interpolate import interp1d # NEW: For threshold interpolation
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# 2. Plotting Functions
###############################################################################
def inspect_data_variance(X, y, save_path="data_variance.png"):
    subjects = np.unique(y)
    var_stats = {subj: np.mean(np.std(X[y == subj].reshape(X[y == subj].shape[0], -1), axis=0)) for subj in subjects}
    logging.info("=== Data Variance Inspection ===")
    for subj, mean_std in var_stats.items():
        logging.info(f"Subject {subj}: mean STD = {mean_std:.4f}")
    
    plt.figure(figsize=(8, 4))
    plt.bar(var_stats.keys(), var_stats.values())
    plt.xlabel("Subject")
    plt.ylabel("Mean STD of Features")
    plt.title("Data Variance per Subject")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, subject_list, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subject_list, yticklabels=subject_list)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_aggregated_roc(scores_matrix, true_labels, subjects, save_path, title="Aggregated ROC Curve"):
    y_true_bin = label_binarize(true_labels, classes=subjects)
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), scores_matrix.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Micro-averaged ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_epoch_accuracy_heatmap(true_labels, predictions, epoch_nums_test, save_path, csv_name):
    df_accuracy = pd.DataFrame({
        'subject': true_labels,
        'epoch': epoch_nums_test,
        'correct': (true_labels == np.array(predictions)).astype(int)
    })
    heatmap_data = df_accuracy.pivot_table(index='subject', columns='epoch', values='correct', aggfunc='mean')
    heatmap_data = heatmap_data.fillna(np.nan)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, linewidths=.5, linecolor='gray', vmin=0, vmax=1)
    plt.xlabel("Epoch Number")
    plt.ylabel("Subject")
    plt.title("Epoch-wise Classification Accuracy (Mean)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    heatmap_data.to_csv(csv_name)
    logging.info(f"Saved epoch-wise classification matrix to {csv_name}")

def plot_det_curve(fpr_test, fnr_test, eer, save_path, title="DET Curve"):
    """Plots the Detection Error Tradeoff (DET) curve (FNR vs FPR)."""
    plt.figure(figsize=(8, 6))
    
    # Plot FNR vs FPR
    plt.plot(fpr_test, fnr_test, linewidth=2, label=f"Test Set (EER ≈ {eer*100:.2f}%)")
    
    # Locate EER point
    eer_idx = np.argmin(np.abs(fpr_test - fnr_test))
    plt.plot(fpr_test[eer_idx], fnr_test[eer_idx], 'ro', markersize=8, label='EER Point')
    
    plt.xscale('log') # Log scale is standard for DET curves
    plt.yscale('log')
    
    # Set limits to avoid log(0) issues, adjust based on your data range
    plt.xlim([0.0001, 1.0])
    plt.ylim([0.0001, 1.0])
    
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

###############################################################################
# 3. Biometric Verification Logic (NEW)
###############################################################################

def generate_biometric_trials(y_test, scores_matrix, subject_list, impostor_samples=5):
    """
    Generates genuine and impostor score/label pairs for verification analysis.
    Score = p(y=s | x), where s is the claimed identity.
    """
    scores, labels = [], []
    subject_to_idx = {subj: i for i, subj in enumerate(subject_list)}
    
    for i in range(len(y_test)):
        true_subj = y_test[i]
        true_idx = subject_to_idx[true_subj]
        
        # 1. Genuine Trial (Claim = True Subject)
        genuine_score = scores_matrix[i, true_idx]
        scores.append(genuine_score)
        labels.append(1) # Genuine
        
        # 2. Impostor Trials (Claim = Random Impostor Subject)
        all_impostors = [s for s in subject_list if s != true_subj]
        n_samples = min(impostor_samples, len(all_impostors))
        
        # Randomly sample impostor claimed identities
        sampled_impostors = random.sample(all_impostors, n_samples)
        
        for imp_subj in sampled_impostors:
            imp_idx = subject_to_idx[imp_subj]
            impostor_score = scores_matrix[i, imp_idx]
            scores.append(impostor_score)
            labels.append(0) # Impostor

    return np.array(scores), np.array(labels)

def biometric_analysis_dev_test(y_test, scores_matrix, all_subjects, dev_subjects_count=10):
    """
    Splits the test set (Session 2) subjects into Dev (thresholding) and Test (reporting).
    """
    test_subjects = np.unique(y_test)
    # Ensure we don't try to take more dev subjects than exist
    if len(test_subjects) <= dev_subjects_count:
        dev_subjects_count = len(test_subjects) // 2 
    
    # Shuffle subjects to create random split
    # Note: For reproducibility in main runs, seeds are handled externally or by luck. 
    # In bootstrapping, we want randomness.
    test_subjects_list = list(test_subjects)
    random.shuffle(test_subjects_list)
    
    dev_subjects = test_subjects_list[:dev_subjects_count]
    final_test_subjects = test_subjects_list[dev_subjects_count:]
    
    # --- 1. Split Data based on Subject IDs ---
    dev_mask = np.isin(y_test, dev_subjects)
    test_mask = np.isin(y_test, final_test_subjects)
    
    # Handle edge case where split results in empty set
    if not np.any(dev_mask) or not np.any(test_mask):
        logging.warning("Dev/Test split resulted in empty set. Returning NaNs.")
        return {}

    y_dev = y_test[dev_mask]
    scores_dev = scores_matrix[dev_mask]
    
    y_final_test = y_test[test_mask]
    scores_final_test = scores_matrix[test_mask]

    # --- 2. Generate Trials for Dev Set (Threshold Selection) ---
    dev_scores, dev_labels = generate_biometric_trials(y_dev, scores_dev, all_subjects)
    fpr, tpr, thresholds = roc_curve(dev_labels, dev_scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find EER threshold
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    eer_rate = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    # Interpolate to find thresholds for specific target FARs
    # We use 'extrapolate' cautiously; usually 0.01 and 0.001 are within range if N is large enough
    fpr_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value="extrapolate")
    
    thresh_far_1_pc = fpr_interp(0.01)
    thresh_far_01_pc = fpr_interp(0.001)

    # --- 3. Apply Thresholds to Final Test Set ---
    test_scores, test_labels = generate_biometric_trials(y_final_test, scores_final_test, all_subjects)
    
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_scores, pos_label=1)
    fnr_test = 1 - tpr_test
    auroc_test = auc(fpr_test, tpr_test)
    
    # Calculate FRR and FAR at operating points
    def get_frr_far_at_threshold(scores, labels, threshold):
        predictions = (scores >= threshold).astype(int)
        
        genuine_mask = labels == 1
        impostor_mask = labels == 0
        
        # FRR (False Rejection Rate) -> Genuine rejected
        fn = np.sum(predictions[genuine_mask] == 0)
        tp = np.sum(predictions[genuine_mask] == 1)
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # FAR (False Acceptance Rate) -> Impostor accepted
        fp = np.sum(predictions[impostor_mask] == 1)
        tn = np.sum(predictions[impostor_mask] == 0)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return frr, far
    
    frr_1pc, far_1pc = get_frr_far_at_threshold(test_scores, test_labels, thresh_far_1_pc)
    frr_01pc, far_01pc = get_frr_far_at_threshold(test_scores, test_labels, thresh_far_01_pc)
    
    return {
        "auroc": auroc_test,
        "fpr": fpr_test,
        "fnr": fnr_test,
        "frr_at_far1pc": frr_1pc,
        "far_at_far1pc": far_1pc,
        "frr_at_far01pc": frr_01pc,
        "far_at_far01pc": far_01pc,
        "eer_rate_dev": eer_rate,
        "eer_threshold_dev": eer_threshold,
        "test_scores": test_scores,
        "test_labels": test_labels,
    }

def compute_biometric_ci(y_test, scores_matrix, all_subjects, n_bootstraps=200, dev_subjects_count=10):
    """Computes confidence intervals using subject-level bootstrap."""
    logging.info(f"Starting Bootstrapping ({n_bootstraps} iterations)...")
    test_subjects_all = np.unique(y_test)
    
    bootstrapped_metrics = {
        'auroc': [],
        'eer': [],
        'frr_at_far1pc': [],
        'frr_at_far01pc': []
    }
    
    for b in range(n_bootstraps):
        if b % 50 == 0: logging.info(f"Bootstrap iteration {b}/{n_bootstraps}")
        
        # 1. Sample subjects with replacement
        sampled_subjects = np.random.choice(test_subjects_all, size=len(test_subjects_all), replace=True)
        
        # 2. Split this specific bootstrap sample into Dev and Test
        # (We shuffle the *names* we just picked)
        random.shuffle(sampled_subjects)
        dev_subjects = sampled_subjects[:dev_subjects_count]
        final_test_subjects = sampled_subjects[dev_subjects_count:]
        
        if len(dev_subjects) < 2 or len(final_test_subjects) < 2:
            continue

        # 3. Construct the dataset for this bootstrap iteration
        # Since we sampled with replacement, a subject might appear multiple times.
        # We need to collect epochs for *each instance* of the subject.
        y_sample_list = []
        scores_sample_list = []
        
        # Pre-calculate indices for speed
        subj_indices_map = {s: np.where(y_test == s)[0] for s in test_subjects_all}

        for s in sampled_subjects:
            indices = subj_indices_map[s]
            y_sample_list.append(y_test[indices])
            scores_sample_list.append(scores_matrix[indices])
            
        if not y_sample_list: continue
            
        y_sample = np.concatenate(y_sample_list)
        scores_sample = np.concatenate(scores_sample_list)
        
        # --- Run Dev/Test Logic on this sample ---
        # Note: We must pass the *original* all_subjects list for trial generation consistent with columns
        res = biometric_analysis_dev_test(y_sample, scores_sample, all_subjects, dev_subjects_count)
        
        if not res: continue

        bootstrapped_metrics['auroc'].append(res['auroc'])
        bootstrapped_metrics['eer'].append(res['eer_rate_dev'])
        bootstrapped_metrics['frr_at_far1pc'].append(res['frr_at_far1pc'])
        bootstrapped_metrics['frr_at_far01pc'].append(res['frr_at_far01pc'])
        
    # Calculate CIs (2.5 and 97.5 percentiles)
    cis = {}
    for metric, values in bootstrapped_metrics.items():
        if len(values) > 0:
            cis[metric] = np.percentile(values, [2.5, 97.5])
        else:
            cis[metric] = [0, 0]
            
    return cis, bootstrapped_metrics

###############################################################################
# 4. Evaluation & Analysis Functions
###############################################################################
def evaluate_predictions(y_true, predictions, subjects):
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions, labels=subjects)
    return accuracy, f1, cm

def across_run_analysis_ml(classifier, param_grid, X, y, runs, epoch_nums, classifier_name="Classifier", cv=3):
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Hyperparameter tuning
    logging.info(f"Tuning {classifier_name}...")
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_flat, y_train)
    
    best_classifier = grid_search.best_estimator_
    logging.info(f"Best hyperparameters for {classifier_name}: {grid_search.best_params_}")
    
    predictions = best_classifier.predict(X_test_flat)
    
    subjects_unique = np.unique(y) # Use all potential subjects for consistent label ordering
    
    if hasattr(best_classifier, "predict_proba"):
        scores_matrix = best_classifier.predict_proba(X_test_flat)
    elif hasattr(best_classifier, "decision_function"):
        decision_vals = best_classifier.decision_function(X_test_flat)
        # Ensure correct shape for multi-class
        if decision_vals.ndim == 1:
             scores_matrix = np.exp(decision_vals) / (1 + np.exp(decision_vals)) # Sigmoid for binary
             scores_matrix = np.vstack([1-scores_matrix, scores_matrix]).T
        else:
            # For SVM, decision_function usually gives distances. 
            # We can use sigmoid to approximate probabilities if predict_proba=False, 
            # but we set probability=True in main config.
            scores_matrix = decision_vals 
    else: 
        scores_matrix = label_binarize(predictions, classes=subjects_unique)
        
    # --- NEW: Primary Biometric Verification (Single Pass for Point Estimates) ---
    logging.info("Running initial Dev/Test split for point estimates...")
    biometric_results = biometric_analysis_dev_test(
        y_test, scores_matrix, subjects_unique
    )
    
    # Simple classification for reference
    accuracy, f1, cm = evaluate_predictions(y_test, predictions, subjects_unique)

    # Combine everything
    full_results = {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "true_labels": y_test,
        "epoch_nums_test": epoch_nums_test,
        "scores_matrix": scores_matrix,
        "subjects": subjects_unique,
        "fitted_classifier": best_classifier
    }
    
    # Add biometric results if successful
    if biometric_results:
        full_results.update(biometric_results)
        
    return full_results

###############################################################################
# 5. Utility Functions
###############################################################################
def save_artifact(artifact, filepath):
    try:
        joblib.dump(artifact, filepath)
        logging.info(f"Saved artifact to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save artifact at {filepath}: {e}")

def save_metadata(model_dir, metadata):
    metadata_filepath = os.path.join(model_dir, "metadata.json")
    try:
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Saved metadata to {metadata_filepath}")
    except Exception as e:
        logging.error(f"Failed to save metadata: {e}")

def create_versioned_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    return versioned_dir

###############################################################################
# 6. Main Script Logic
###############################################################################
def main(args):
    # Create a single versioned directory for this entire run
    versioned_run_dir = create_versioned_dir(args.output_dir)
    plots_folder = os.path.join(versioned_run_dir, "ml_plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    logging.info(f"Loading data from {args.data_file}")
    X, y, runs, epoch_nums = load_features_from_hdf5(args.data_file)
    
    original_shape = X.shape
    X_flat = X.reshape(original_shape[0], -1)

    inspect_data_variance(X, y, save_path=os.path.join(plots_folder, "data_variance.png"))
    
    comparison_results = []
    
    MODEL_CONFIG = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200], 'max_depth': [None, 10, 20]
            }
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {
                'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']
            }
        }
    }

    for model_name, config in MODEL_CONFIG.items():
        logging.info(f"=== Across-Run Analysis: {model_name} with Tuning ===")
        results = across_run_analysis_ml(
            classifier=config["estimator"],
            param_grid=config["params"],
            X=X, y=y, runs=runs, epoch_nums=epoch_nums,
            classifier_name=model_name,
            cv=3
        )
        
        # --- NEW: Compute 95% CIs via Bootstrapping ---
        cis, raw_metrics = compute_biometric_ci(
            results["true_labels"], 
            results["scores_matrix"], 
            results["subjects"],
            n_bootstraps=200  # Adjust as needed (e.g., 1000 for final paper)
        )
        
        # Format results string with Mean and CI
        def fmt_ci(metric_key, val_key, scale=1.0):
            mean_val = results.get(val_key, 0) * scale
            ci_low = cis[metric_key][0] * scale
            ci_high = cis[metric_key][1] * scale
            return f"{mean_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"

        # Store results for final comparison
        row = {
            "Classifier": model_name,
            "Accuracy": results["accuracy"],
            "F1_Score": results["f1"],
            "AUROC_CI": fmt_ci('auroc', 'auroc', 1.0),
            "EER_Dev_CI": fmt_ci('eer', 'eer_rate_dev', 100.0) + "%",
            "FRR@FAR1%_CI": fmt_ci('frr_at_far1pc', 'frr_at_far1pc', 100.0) + "%",
            "FRR@FAR0.1%_CI": fmt_ci('frr_at_far01pc', 'frr_at_far01pc', 100.0) + "%"
        }
        comparison_results.append(row)
        
        # Save raw metrics for later custom plotting if needed
        save_artifact(raw_metrics, os.path.join(versioned_run_dir, f"ci_metrics_{model_name.lower()}.pkl"))

        # Save model and standard plots
        save_artifact(results["fitted_classifier"], os.path.join(versioned_run_dir, f"{model_name.lower()}_model.pkl"))
        
        plot_confusion_matrix(results["confusion_matrix"], results["subjects"],
                              save_path=os.path.join(plots_folder, f"cm_{model_name}.png"),
                              title=f"{model_name} Confusion Matrix")
        
        plot_aggregated_roc(results["scores_matrix"], results["true_labels"], results["subjects"],
                            save_path=os.path.join(plots_folder, f"roc_{model_name}.png"),
                            title=f"{model_name} ROC Curve")

        plot_epoch_accuracy_heatmap(results["true_labels"], results["predictions"], results["epoch_nums_test"],
                                    save_path=os.path.join(plots_folder, f"epoch_heatmap_{model_name}.png"),
                                    csv_name=os.path.join(plots_folder, f"epoch_accuracy_{model_name}.csv"))
                                    
        # --- NEW: Plot DET Curve ---
        if 'fpr' in results and 'fnr' in results:
             plot_det_curve(results['fpr'], results['fnr'], results.get('eer_rate_dev', 0),
                            save_path=os.path.join(plots_folder, f"det_{model_name}.png"),
                            title=f"{model_name} DET Curve (Log-Log)")

    # Save overall comparison CSV
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv(os.path.join(versioned_run_dir, "ml_classifiers_comparison.csv"), index=False)
    logging.info(f"Comparison metrics saved to {os.path.join(versioned_run_dir, 'ml_classifiers_comparison.csv')}")
    print(df_comparison)

    # Save experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_file": args.data_file,
        "output_directory": versioned_run_dir,
        "hyperparameters": {name: conf["params"] for name, conf in MODEL_CONFIG.items()},
        "data_shape": {"X": list(X.shape), "y": list(y.shape)}
    }
    save_metadata(versioned_run_dir, metadata)
    logging.info("="*50)
    logging.info(f"Experiment finished. All outputs saved to: {versioned_run_dir}")
    logging.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML classification experiments on EEG data.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the HDF5 data file.")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save models and plots.")
    
    args = parser.parse_args()
    main(args)