# EEG-Based Subject Identification Analysis

This repository contains three Python scripts for performing subject identification from EEG data. Each script implements a different classification approach to analyze various feature combinations extracted from the EEG signals.

The three methods explored are:
1.  **`ML_classifiers_combination_of_features.py`**: Compares traditional machine learning models, including Logistic Regression, Random Forest, and Support Vector Machines (SVM).
2.  **`cnn_combination.py`**: Implements a 1D Convolutional Neural Network (CNN) with hyperparameter tuning using Keras Tuner to classify the feature sets.
3.  **`Combination_maha.py`**: Uses a Mahalanobis distance-based classifier, which is effective for template-matching problems.

---

## 📂 Project Structure

The project is organized to keep code, data, and results separate.

```
eeg-analysis-project/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── all_subjects_merged_new_full_epochs.h5  (You must add this file)
│
├── results/
│   ├── cnn_results/
│   ├── mahalanobis_results/
│   └── ml_results/
│
├── ML_classifiers_combination_of_features.py
├── cnn_combination.py
└── Combination_maha.py
```

---

## ⚙️ Setup Instructions

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
Open your terminal and clone this repository:
```bash
git clone [https://github.com/your-username/eeg-analysis-project.git](https://github.com/your-username/eeg-analysis-project.git)
cd eeg-analysis-project
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Add the Data File
You must provide your own data file.
* Create a folder named `data` in the root of the project.
* Place your HDF5 data file, named `all_subjects_merged_new_full_epochs.h5`, inside this `data` folder.

---

## 🚀 How to Run the Analyses

You can run each analysis by executing its corresponding Python script from your terminal. All results, including prediction CSVs and model artifacts, will be automatically saved in the `results/` directory.

### Run Traditional ML Classifiers
```bash
python ML_classifiers_combination_of_features.py
```

### Run the 1D-CNN Analysis
```bash
python cnn_combination.py
```

### Run the Mahalanobis Distance Analysis
```bash
python Combination_maha.py
```