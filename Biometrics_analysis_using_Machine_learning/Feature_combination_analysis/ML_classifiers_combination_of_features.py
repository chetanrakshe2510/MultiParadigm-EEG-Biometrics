import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
# 1. Data Loading Function
###############################################################################
def load_features_from_hdf5(filename):
    """
    Loads EEG features from an HDF5 file in 32x32 format.
    Returns:
      X: (n_samples, 32, 32)
      y: subject labels
      runs: run labels
      epoch_nums: epoch indices
    """
    X, y, runs, epoch_nums = [], [], [], []
    with h5py.File(filename, 'r') as h5f:
        for class_key in h5f.keys():
            subject_code = class_key.split('_', 1)[-1]
            class_group = h5f[class_key]
            for run_key in class_group.keys():
                run_group = class_group[run_key]
                for epoch_key in run_group.keys():
                    epoch_group = run_group[epoch_key]
                    feats = epoch_group['features'][()]  # shape (32,32)
                    X.append(feats)
                    y.append(subject_code)
                    runs.append(run_key)
                    ep_idx = int(epoch_key.split('_')[-1]) if '_' in epoch_key else 0
                    epoch_nums.append(ep_idx)
    return np.array(X), np.array(y), np.array(runs), np.array(epoch_nums)

###############################################################################
# 2. Evaluation Function
###############################################################################
def evaluate_predictions(y_true, predictions):
    """Compute evaluation metrics: accuracy, precision, and f1 score."""
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    return accuracy, precision, f1

###############################################################################
# 3. Across-Run Analysis Function (Without Hyperparameter Tuning)
###############################################################################
def across_run_analysis_ml(classifier, X, y, runs, epoch_nums, classifier_name="Classifier", debug=False):
    """
    Performs across-run analysis for a scikit-learn classifier.
    Trains on Run_1 and tests on Run_2.
    Returns a dictionary with true labels, predictions, evaluation metrics, and the fitted classifier.
    """
    # Select indices for training and testing
    train_idx = np.where(runs == "Run_1")[0]
    test_idx = np.where(runs == "Run_2")[0]
    
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient data for Run_1 or Run_2. Check your run labels.")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    epoch_nums_test = epoch_nums[test_idx]
    
    # Flatten the 2D features into vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Train and predict
    classifier.fit(X_train_flat, y_train)
    predictions = classifier.predict(X_test_flat)
    
    # Calculate metrics
    accuracy, precision, f1 = evaluate_predictions(y_test, predictions)
    
    results = {
        "classifier_name": classifier_name,
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "predictions": predictions,
        "true_labels": y_test,
        "epoch_nums_test": epoch_nums_test,
        "fitted_classifier": classifier
    }
    
    if debug:
        logging.info(f"[DEBUG] {classifier_name}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, F1 = {f1:.4f}")
    
    return results

###############################################################################
# 4. Artifact and Metadata Saving Functions (Optional)
###############################################################################
def save_artifact(artifact, filepath, save_method="joblib"):
    try:
        if save_method == "joblib":
            joblib.dump(artifact, filepath)
        else:
            raise ValueError("Unsupported save method specified.")
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
    """Creates a timestamped sub-directory within a given base directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    return versioned_dir

###############################################################################
# 5. Main Script: Loop Over Feature Combinations and Classifiers
###############################################################################
def main():
    # Setup directories for saving results
    # Get the absolute path of the directory where the script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script's location for better portability
    DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'all_subjects_merged_new_full_epochs.h5')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'ml_results')
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'results', 'models')

    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    versioned_model_dir = create_versioned_dir(base_dir=MODELS_DIR)
    logging.info(f"Results will be saved in: {RESULTS_DIR}")
    logging.info(f"Models and metadata will be saved in: {versioned_model_dir}")


    # Load data from the HDF5 file (adjust file name/path as needed)
    logging.info(f"Loading data from {DATA_FILE}...")
    X, y, runs, epoch_nums = load_features_from_hdf5(DATA_FILE)
    
    # Preprocess data using PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    X_flat = X.reshape(X.shape[0], -1)
    X_tf = transformer.fit_transform(X_flat)
    # Reshape to (n_samples, 32, 28)
    X = X_tf.reshape(X.shape[0], 32, 28)
    
    # Save the transformer for future use
    transformer_path = os.path.join(versioned_model_dir, "power_transformer.pkl")
    save_artifact(transformer, transformer_path)
    
    # Define feature combination sets (indices along the last axis of X)
    feature_combinations = {
        "Set_1_Time": [0, 1, 2, 3, 4],
        "Set_2_Frequency": [5, 6, 7, 8, 9],
        "Set_3_TimeFrequency": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        "Set_4_NonLinear": [10, 11, 12, 13],
        "Set_5_Time+Frequency": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Set_6_Time+TimeFrequency": [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        "Set_7_Frequency+TimeFrequency": [5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        "Set_8_Time+Frequency+NonLinear": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "Set_9_Time+TimeFrequency+NonLinear": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        "Set_10_Time+Frequency+NonLinear (No TimeFreq)": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "Set_11_All": list(range(0, 28))
    }
    
    # Instantiate classifiers with the given hyperparameters
    classifiers = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            class_weight=None,
            dual=False,
            fit_intercept=True,
            intercept_scaling=1,
            l1_ratio=None,
            max_iter=1000,
            multi_class='deprecated',
            n_jobs=None,
            penalty='l2',
            random_state=None,
            solver='lbfgs',
            tol=0.0001,
            verbose=0,
            warm_start=False
        ),
        "RandomForest": RandomForestClassifier(
            bootstrap=True,
            ccp_alpha=0.0,
            class_weight=None,
            criterion='gini',
            max_depth=None,
            max_features='sqrt',
            max_leaf_nodes=None,
            max_samples=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            monotonic_cst=None,
            n_estimators=200,
            n_jobs=None,
            oob_score=False,
            random_state=None,
            verbose=0,
            warm_start=False
        ),
        "SVM": SVC(
            C=0.1,
            break_ties=False,
            cache_size=200,
            class_weight=None,
            coef0=0.0,
            decision_function_shape='ovr',
            degree=3,
            gamma='scale',
            kernel='linear',
            max_iter=-1,
            probability=True,
            random_state=None,
            shrinking=True,
            tol=0.001,
            verbose=False
        )
    }
    
    # Prepare a list to gather summary results for all experiments
    summary_results = []
    
    for set_name, indices in feature_combinations.items():
        logging.info(f"Processing Feature Combination: {set_name}")
        X_subset = X[:, :, indices]
        
        for clf_name, clf in classifiers.items():
            logging.info(f"Running {clf_name} on {set_name}...")
            
            results = across_run_analysis_ml(
                classifier=clf, X=X_subset, y=y, runs=runs,
                epoch_nums=epoch_nums, classifier_name=clf_name, debug=True
            )
            
            # Save detailed CSV using the new results path
            df_results = pd.DataFrame({
                "True_Label": results["true_labels"],
                "Predicted_Label": results["predictions"]
            })
            csv_filename = f"results_{clf_name}_{set_name}.csv"
            csv_filepath = os.path.join(RESULTS_DIR, csv_filename)
            df_results.to_csv(csv_filepath, index=False)
            
            summary_results.append({
                "Feature_Set": set_name, "Classifier": clf_name,
                "Accuracy": results["accuracy"], "Precision": results["precision"], "F1_Score": results["f1"]
            })
    
    # --- 5. Save Summary and Metadata ---
    df_summary = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(RESULTS_DIR, "comparison_metrics_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    logging.info(f"Saved summary comparison CSV to {summary_csv_path}")
    
    # Optionally, save experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_shape": {
            "X": list(X.shape),
            "y": list(y.shape)
        },
        "feature_combinations": feature_combinations,
        "model_parameters": {
            "LogisticRegression": {
                "C": 0.1,
                "class_weight": None,
                "dual": False,
                "fit_intercept": True,
                "intercept_scaling": 1,
                "l1_ratio": None,
                "max_iter": 1000,
                "multi_class": "deprecated",
                "n_jobs": None,
                "penalty": "l2",
                "random_state": None,
                "solver": "lbfgs",
                "tol": 0.0001,
                "verbose": 0,
                "warm_start": False
            },
            "RandomForest": {
                "bootstrap": True,
                "ccp_alpha": 0.0,
                "class_weight": None,
                "criterion": "gini",
                "max_depth": None,
                "max_features": "sqrt",
                "max_leaf_nodes": None,
                "max_samples": None,
                "min_impurity_decrease": 0.0,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "min_weight_fraction_leaf": 0.0,
                "monotonic_cst": None,
                "n_estimators": 200,
                "n_jobs": None,
                "oob_score": False,
                "random_state": None,
                "verbose": 0,
                "warm_start": False
            },
            "SVM": {
                "C": 0.1,
                "break_ties": False,
                "cache_size": 200,
                "class_weight": None,
                "coef0": 0.0,
                "decision_function_shape": "ovr",
                "degree": 3,
                "gamma": "scale",
                "kernel": "linear",
                "max_iter": -1,
                "probability": True,
                "random_state": None,
                "shrinking": True,
                "tol": 0.001,
                "verbose": False
            }
        }
    }
    metadata = { "timestamp": datetime.now().isoformat(), "data_shape": list(X.shape), # ... etc.
    }
    save_metadata(versioned_model_dir, metadata)

if __name__ == "__main__":
    main()
