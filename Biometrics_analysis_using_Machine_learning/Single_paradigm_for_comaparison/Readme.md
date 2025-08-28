# EEG-Based Biometric Identification Analysis

This repository contains a suite of Python scripts for analyzing EEG data for biometric identification. It explores and compares three different classification approaches: a traditional statistical method (Mahalanobis Distance), standard machine learning classifiers (Logistic Regression, Random Forest, SVM), and a deep learning model (1D Convolutional Neural Network).

## Repository Structure

* `README.md`: You are here!
* `mahalanobis_classifier.py`: Implements a subject identification pipeline using Mahalanobis distance. It establishes a template and covariance matrix for each subject from a training run and classifies test samples based on the minimum distance.
* `ml_classifiers.py`: Trains, tunes, and compares three standard machine learning models (Logistic Regression, Random Forest, SVM) for the same identification task. It performs hyperparameter tuning using `GridSearchCV`.
* `cnn_classifier.py`: Implements a 1D Convolutional Neural Network (CNN) for EEG-based identification. It uses Keras Tuner to find the best hyperparameters before training and evaluating the final model.

## Setup

### Prerequisites
* Python 3.8+
* Your EEG data, preprocessed and stored in HDF5 (`.h5`) files. The scripts expect a specific hierarchical structure within each HDF5 file: `subject_code/run_key/epoch_key/features`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a file named `requirements.txt` with the content below and run `pip install -r requirements.txt`.

    ```text
    # requirements.txt
    numpy
    h5py
    matplotlib
    pandas
    seaborn
    scikit-learn
    tensorflow
    keras-tuner
    joblib
    ```

## Usage

Place your HDF5 data files in a directory (e.g., `data/`). The scripts are run from the command line, and you can specify input and output paths.

### 1. Mahalanobis Distance Classifier

This script processes all `.h5` files in a given directory and saves the results in an output folder.

**Command:**
```bash
python mahalanobis_classifier.py --data-dir ./data --output-dir ./results/mahalanobis --epochs 40
```
* `--data-dir`: Directory containing your `.h5` files.
* `--output-dir`: Directory where results (CSVs, plots) will be saved.
* `--epochs`: The number of epochs to sample per subject for training and testing.

### 2. Standard ML Classifiers

This script processes a single `.h5` file to train and compare Logistic Regression, Random Forest, and SVM models.

**Command:**
```bash
python ml_classifiers.py --input-file ./data/features.h5 --output-dir ./results/ml_models --epochs 40
```
* `--input-file`: Path to a single HDF5 data file.
* `--output-dir`: Directory where results (CSVs, plots, saved models) will be saved.
* `--epochs`: The number of epochs to randomly sample per subject.

### 3. 1D CNN Classifier

This script finds all `.h5` files in the current directory, then tunes, trains, and evaluates a separate CNN model for each file.

**Command:**
```bash
python cnn_classifier.py --output-dir ./results/cnn
```
* `--output-dir`: A parent directory where all outputs for each file (tuning logs, reports, models) will be stored in uniquely named subfolders.

## Expected Output

Each script will generate an output directory containing:
* **CSV files**: Detailed epoch-wise predictions, per-subject performance metrics (Precision, Recall, F1), and overall summary reports.
* **Plots (`.png`)**: Confusion matrices, ROC curves, and other relevant visualizations.
* **Saved Models**: The trained classifiers (`.pkl` for ML models, `.h5` for the CNN) and data transformers for future use.