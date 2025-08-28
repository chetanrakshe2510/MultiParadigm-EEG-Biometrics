EEG-based Subject Identification and Classification
This repository contains a suite of Python scripts for classifying and identifying subjects based on EEG (Electroencephalography) signal features. The project explores three distinct methodologies: traditional machine learning, deep learning with a 1D Convolutional Neural Network (CNN), and a biometric approach using Mahalanobis distance.

All experiments follow a consistent "across-run" analysis protocol, where models are trained on data from Run_1 and evaluated on unseen data from Run_2.

Scripts Overview
This project includes three main scripts, each implementing a different classification approach:

ml_classifiers.py:

Method: Trains and evaluates classic machine learning models.

Models: Logistic Regression, Random Forest, and Support Vector Machine (SVM).

Features: Includes hyperparameter tuning using GridSearchCV to find the best model configuration.

Evaluation: Reports standard classification metrics like Accuracy and F1-Score.

cnn_tuner.py:

Method: Implements a deep learning approach for classification.

Model: A 1D Convolutional Neural Network (CNN).

Features: Uses Keras Tuner to automatically search for the optimal network architecture and hyperparameters (e.g., number of filters, dropout rate, learning rate).

Evaluation: Provides a full suite of deep learning metrics, including training history plots, a classification report, and ROC curves.

mahalanobis_classifier.py:

Method: Implements a biometric identification system based on statistical distance.

Model: Mahalanobis Distance Classifier. This method builds a statistical template (mean and covariance) for each subject.

Features: Designed for identity verification tasks.

Evaluation: In addition to classification accuracy, it calculates key biometric metrics like the Equal Error Rate (EER) from FAR (False Acceptance Rate) and FRR (False Rejection Rate) curves.

Getting Started
Follow these steps to set up your environment and run the experiments.

1. Clone the Repository
git clone <your-repo-url>
cd <your-repo-name>

2. Create and Activate a Virtual Environment (Recommended)
# Create the environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

3. Install Dependencies
All required libraries are listed in the requirements.txt file. Install them with a single command:

pip install -r requirements.txt

How to Run the Experiments
Each script is run from the command line and requires, at a minimum, the path to the HDF5 data file. All generated outputs (models, plots, reports, and metadata) will be saved into a unique, timestamped sub-directory inside the specified --output_dir.

1. Running the Machine Learning Classifiers
python ml_classifiers.py --data_file your_data.h5 --output_dir results/ml

2. Running the 1D CNN Tuner and Trainer
python cnn_tuner.py --data_file your_data.h5 --output_dir results/cnn --max_trials 15 --epochs 30

--max_trials: The number of hyperparameter combinations to test.

--epochs: The number of epochs for each trial in the search.

3. Running the Mahalanobis Distance Classifier
python mahalanobis_classifier.py --data_file your_data.h5 --output_dir results/mahalanobis --alpha 1e-4

--alpha: The regularization parameter for the covariance matrix calculation.

Output Structure
Each script will generate a new directory for its run, for example: results/cnn/run_20250828_104500/. Inside this directory, you will find:

Trained Models: Saved models (.pkl or .h5) and data preprocessors (power_transformer.pkl, label_encoder.pkl).

plots/: A folder containing all visualizations, such as confusion matrices, ROC curves, training history, and FAR/FRR curves.

reports/: A folder with detailed performance metrics saved as CSV files.

metadata.json: A summary of the experiment, including parameters and final performance scores, for easy reproducibility.

Dependencies
TensorFlow / Keras

Keras Tuner

Scikit-learn

Pandas & NumPy

Matplotlib & Seaborn

h5py & Joblib