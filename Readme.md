
# Multi-Paradigm Fusion for Task-Independent EEG-based Biometric Authentication

This repository contains the official source code and analysis pipelines for the research paper, **"Multi-Paradigm Fusion for Task-Independent EEG-based Biometric Authentication"**. Our work introduces a robust, task-independent biometric system by fusing EEG data from multiple cognitive and motor paradigms.

## 📜 Abstract

EEG‐based biometric systems offer a promising and secure alternative to traditional authentication methods. However, current systems often lack cross-paradigm robustness and are limited by the non-stationary characteristics of EEG signals across sessions. This study proposes a unified multi-paradigm pipeline for biometric authentication that integrates six distinct EEG tasks. By training models on a rich, aggregated dataset and performing strict cross-session validation, we demonstrate that combining diverse EEG paradigms with a comprehensive multi-domain feature set enables precise, flexible, and task-independent biometric identification, paving the way for real-world EEG-based authentication systems.

-----

## 🔬 Research Workflow

Our research pipeline is organized into a series of distinct stages, from raw data processing to final biometric analysis. Each stage corresponds to a directory in this repository.

1.  **Manual Segmentation (`/raw_data_manual_Segmentation`)**: Initial manual segmentation of raw EEG data using a custom GUI to define task boundaries.
2.  **Automated Task Processing (`/Each_task_processing`)**: Automated scripts to segment the raw BrainVision data files based on predefined markers for each of the six experimental tasks.
3.  **Feature Extraction (`/Feature_extraction`)**: A comprehensive pipeline to extract 25 features from four distinct domains (Time, Frequency, Time-Frequency, and Non-Linear) for each EEG epoch.
4.  **Biometric Analysis (`/Biometrics_analysis_using_Machine_learning`)**: The core of the project, where various machine learning and deep learning models are trained and evaluated for subject identification. This includes our primary multi-paradigm analysis as well as several ablation studies.

-----

## 📂 Repository Structure

The repository is organized into modules, each handling a specific part of the workflow.

```
.
├── 📄 EEG_biometric_Final_Updated_28_08_25.docx  # The research manuscript
├── 📁 Biometrics_analysis_using_Machine_learning/
│   ├── 📁 Core_multiparadigm_run1_run2/          # Main multi-pardigm cross-session analysis (Fig 4)
│   ├── 📁 Feature_combination_analysis/        # Ablation study on feature domains (Fig 5)
│   ├── 📁 Multi_pardigm_All_epochs_Task_performance_ablation_study/ # Task performance analysis 
│   ├── 📁 Single_paradigm_for_comaparison/       # Comparison of unified vs. single-task models (Fig 3)
│   └── 📁 Multi_paradigm_Restricted_epochs_for_comparison/ # Analysis with limited epochs per task (Fig 3)
├── 📁 Each_task_processing/                        # Scripts for automated task-based segmentation
├── 📁 Feature_extraction/                          # Scripts for feature extraction pipeline
├── 📁 raw_data_manual_Segmentation/                # GUI for manual data segmentation
└── 📁 Complimentary_scripts/                      # Utility scripts for merging and inspecting results
```

-----

## ⚙️ Setup and Installation

To run the experiments in this repository, follow these setup instructions.

Option A: Using Anaconda (Recommended)
This is the easiest way to ensure all dependencies are installed correctly.

Create the Conda Environment:
This command will create a new environment named eeg_biometrics using the provided environment.yml file.

Bash

conda env create -f environment.yml
Activate the Environment:
Before running any scripts, you must activate the environment.

Bash

conda activate eeg_biometrics
Option B: Using pip and a Virtual Environment
If you are not using Anaconda, you can set up a virtual environment and install the packages using pip.

Create and Activate a Virtual Environment:

Bash

# Create the environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
Install Dependencies:
Each analysis folder contains a requirements.txt file. You will need to install the dependencies for the specific analysis you wish to run.

Bash

# Example for the core analysis
cd Biometrics_analysis_using_Machine_learning/Core_multiparadigm_run1_run2/
pip install -r requirements.txt


## 🚀 How to Run the Analyses

This project includes several key analyses that correspond to the main figures and findings in the paper. For detailed instructions, **please refer to the `Readme.md` file inside each respective directory.**

### Unified Multi-Paradigm Model

  * **Directory**: `/Biometrics_analysis_using_Machine_learning/Core_multiparadigm_run1_run2/`
  * **Description**: This is the main experiment. It trains five classifiers (MDTM, LR, SVM, RF, 1D-CNN) on a unified dataset from Session 1 and evaluates them on Session 2. 

### Task Performance and Comparison

  * **Directory**: `/Biometrics_analysis_using_Machine_learning/Multi_pardigm_All_epochs_Task_performance_ablation_study/`
  * **Description**: This set of scripts evaluates the performance of the unified model on each of the individual tasks and compares it against models trained on single paradigms. 

### Feature Domain Contributions

  * **Directory**: `/Biometrics_analysis_using_Machine_learning/Feature_combination_analysis/`
  * **Description**: This analysis evaluates the performance of the classifiers using different combinations of feature domains (Time, Frequency, TF, NL) to determine their respective contributions. 


-----



```

## 📧 Contact : chetantanajirakshe.rs.bme22@itbhu.ac.in

For any questions or collaborations, please contact Chetan Rakshe at the Computational Neuroscience and Biology Lab, IIT (BHU), Varanasi.