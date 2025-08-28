import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import PowerTransformer, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import argparse # Added for command-line arguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.switch_backend('Agg')

###############################################################################
# 1. Data Loading and Filtering
###############################################################################
def load_features_from_hdf5(filename):
    """Loads EEG features from an HDF5 file."""
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

def filter_epochs_per_subject(X, y, runs, epoch_nums, max_epochs=40, random_state=42):
    """Filters dataset by RANDOMLY SAMPLING up to `max_epochs` per subject per run."""
    np.random.seed(random_state)
    final_indices = []
    unique_subjects, unique_runs = np.unique(y), np.unique(runs)
    for subject in unique_subjects:
        for run in unique_runs:
            indices = np.where((y == subject) & (runs == run))[0]
            if len(indices) > 0:
                num_to_sample = min(max_epochs, len(indices))
                selected_indices = np.random.choice(indices, size=num_to_sample, replace=False)
                final_indices.extend(selected_indices)
    final_indices = sorted(final_indices)
    logging.info(f"Original size: {len(X)}. Filtered size (max {max_epochs} epochs/subject/run): {len(final_indices)}.")
    return X[final_indices], y[final_indices], runs[final_indices], epoch_nums[final_indices]

###############################################################################
# 2. Plotting & Evaluation
###############################################################################
def plot_confusion_matrix(cm, subjects, save_path, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=subjects, yticklabels=subjects)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_mean_eer(y_true, scores_matrix, subjects):
    """Calculates the Mean Equal Error Rate (EER) for a multi-class problem."""
    y_true_bin = label_binarize(y_true, classes=subjects)
    eer_per_class = []
    for i in range(len(subjects)):
        y_true_class, y_scores_class = y_true_bin[:, i], scores_matrix[:, i]
        if len(np.unique(y_true_class)) < 2:
            logging.warning(f"Skipping EER for subject '{subjects[i]}' due to single-class data.")
            continue
        fpr, tpr, _ = roc_curve(y_true_class, y_scores_class)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - fnr))
        eer_per_class.append((fpr[eer_index] + fnr[eer_index]) / 2.0)
    return np.mean(eer_per_class) * 100 if eer_per_class else np.nan

###############################################################################
# 3. Core Analysis Logic
###############################################################################
def across_run_analysis_ml(classifier, X_flat, y, runs, param_grid=None, cv=3):
    train_idx, test_idx = np.where(runs == "Run_1")[0], np.where(runs == "Run_2")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1/Run_2 split.")

    X_train, y_train = X_flat[train_idx], y[train_idx]
    X_test, y_test = X_flat[test_idx], y[test_idx]

    if param_grid:
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        classifier = grid_search.best_estimator_
        logging.info(f"Best params: {grid_search.best_params_}")

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    
    subjects_unique = np.unique(np.concatenate((y_train, y_test)))
    probs = classifier.predict_proba(X_test) if hasattr(classifier, "predict_proba") else label_binarize(predictions, classes=subjects_unique)

    scores_matrix = np.zeros((len(y_test), len(subjects_unique)))
    class_map = {cls: i for i, cls in enumerate(classifier.classes_)}
    for i, cls in enumerate(subjects_unique):
        if cls in class_map:
            scores_matrix[:, i] = probs[:, class_map[cls]]
    
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, average='weighted', zero_division=0),
        "mean_eer_percent": calculate_mean_eer(y_test, scores_matrix, subjects_unique),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=subjects_unique),
        "subjects": subjects_unique,
        "fitted_classifier": classifier
    }

###############################################################################
# 4. Main Script Execution
###############################################################################
def main(args):
    """Main function to run the ML classifier comparison."""
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(args.output_dir, f"run_{base_name}_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    logging.info(f"Created output directory: {versioned_dir}")

    X, y, runs, epoch_nums = load_features_from_hdf5(args.input_file)
    X, y, runs, epoch_nums = filter_epochs_per_subject(X, y, runs, epoch_nums, max_epochs=args.epochs)
    
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = transformer.fit_transform(X.reshape(X.shape[0], -1))
    joblib.dump(transformer, os.path.join(versioned_dir, "power_transformer.pkl"))

    classifiers = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), 
                               {'C': [0.1, 1, 10], 'solver': ['liblinear']}),
        "RandomForest": (RandomForestClassifier(random_state=42),
                         {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}),
        "SVM": (SVC(probability=True, random_state=42),
                {'C': [1, 10], 'gamma': ['scale'], 'kernel': ['rbf']})
    }

    comparison_results = []
    for name, (model, params) in classifiers.items():
        logging.info(f"=== Training and Evaluating: {name} ===")
        results = across_run_analysis_ml(model, X_flat, y, runs, param_grid=params)
        
        comparison_results.append({
            "Classifier": name,
            "Accuracy": results["accuracy"],
            "F1_Score": results["f1"],
            "Mean_EER_Percent": results["mean_eer_percent"]
        })
        
        joblib.dump(results["fitted_classifier"], os.path.join(versioned_dir, f"{name}_model.pkl"))
        plot_confusion_matrix(results["confusion_matrix"], results["subjects"],
                              save_path=os.path.join(versioned_dir, f"cm_{name}.png"),
                              title=f"{name} Confusion Matrix")
        
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv(os.path.join(versioned_dir, "model_comparison.csv"), index=False)
    logging.info(f"Comparison metrics saved:\n{df_comparison}")

    metadata = {
        "timestamp": timestamp,
        "input_file": args.input_file,
        "sampling_strategy": f"Randomly sampled max {args.epochs} epochs per subject/run.",
        "best_hyperparameters": {name: res["fitted_classifier"].get_params() for name, res in zip(classifiers.keys(), [results]*len(classifiers))},
    }
    with open(os.path.join(versioned_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and compare classical ML classifiers on a single HDF5 file.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input HDF5 data file.')
    parser.add_argument('--output-dir', type=str, default='../results/classic_ml', help='Base directory to save versioned results.')
    parser.add_argument('--epochs', type=int, default=40, help='Max number of epochs to randomly sample per subject per run.')
    args = parser.parse_args()
    main(args)