# EEG-based Biometric Analysis Suite

This repository contains a suite of Python scripts for performing biometric identification using EEG feature data stored in HDF5 files. Three different classification approaches are implemented:

1.  **Mahalanobis Distance Classifier:** A classical statistical approach for biometric verification and identification.
2.  **1D Convolutional Neural Network (CNN):** A deep learning model with hyperparameter tuning to find an optimal architecture.
3.  **Classical Machine Learning Classifiers:** A comparison of standard models including Logistic Regression, Random Forest, and Support Vector Machines (SVM).

---

## 📋 Requirements

The scripts are written in Python 3. To run them, you'll need to install the necessary libraries. It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

---

## 📦 Project Structure

A recommended directory structure for your project would be:

```
eeg-biometrics/
├── data/
│   ├── dataset1.h5
│   └── dataset2.h5
├── results/
│   ├── mahalanobis/
│   ├── cnn_tuning/
│   └── classic_ml/
├── src/
│   ├── 1_mahalanobis_classifier.py
│   ├── 2_cnn_hyper_tuning.py
│   ├── 3_classic_ml_classifiers.py
├── requirements.txt
└── README.md
```

-   **`data/`**: Place your input `.h5` files here.
-   **`results/`**: All output artifacts (CSVs, plots, models) will be saved here in corresponding subdirectories.
-   **`src/`**: Contains the three executable Python scripts.

---

## 🚀 Usage

All scripts are run from the command line and accept arguments to specify input and output directories.

### 1. Mahalanobis Classifier

This script processes all `.h5` files in a directory, performs a run-to-run analysis using Mahalanobis distance, and saves summary and epoch-wise results.

**Command:**
```bash
python src/1_mahalanobis_classifier.py --data-dir <path_to_data> --output-dir <path_to_results> [--epochs 40] [--save-plots]
```

**Arguments:**
-   `--data-dir`: Directory containing your input `.h5` files. (Default: `../data`)
-   `--output-dir`: Directory where results will be saved. (Default: `../results/mahalanobis`)
-   `--epochs`: The maximum number of epochs to use per subject. (Default: `40`)
-   `--save-plots`: If included, saves analysis plots (e.g., confusion matrix, ROC).

**Example:**
```bash
python src/1_mahalanobis_classifier.py --data-dir ../data --output-dir ../results/mahalanobis --save-plots
```

### 2. 1D CNN with Hyperparameter Tuning

This script iterates through each `.h5` file, uses Keras Tuner to find optimal hyperparameters for a 1D CNN, trains the best model, and saves the results.

**Command:**
```bash
python src/2_cnn_hyper_tuning.py --data-dir <path_to_data> --output-dir <path_to_results>
```

**Arguments:**
-   `--data-dir`: Directory containing your input `.h5` files. (Default: `../data`)
-   `--output-dir`: Directory where results and tuning artifacts will be saved. (Default: `../results/cnn_tuning`)

**Example:**
```bash
python src/2_cnn_hyper_tuning.py --data-dir ../data --output-dir ../results/cnn_tuning
```

### 3. Classical Machine Learning Classifiers

This script processes a single `.h5` file, applies three different ML models (Logistic Regression, Random Forest, SVM) with hyperparameter tuning, and saves a detailed comparison and all model artifacts.

**Command:**
```bash
python src/3_classic_ml_classifiers.py --input-file <path_to_file.h5> --output-dir <path_to_results> [--epochs 40]
```

**Arguments:**
-   `--input-file`: Path to the single `.h5` file to be processed.
-   `--output-dir`: Directory where versioned model runs and plots will be saved. (Default: `../results/classic_ml`)
-   `--epochs`: Maximum number of epochs to randomly sample per subject. (Default: `40`)

**Example:**
```bash
python src/3_classic_ml_classifiers.py --input-file ../data/raw_all_subject_eeg_features_S5_hierarchical.h5 --output-dir ../results/classic_ml
```
---

## 📝 Best Practices Note

For a production-level GitHub repository, you would also typically include:
-   A `.gitignore` file to exclude `venv/`, `__pycache__/`, and other temporary files from version control.
-   Unit tests to ensure code reliability.