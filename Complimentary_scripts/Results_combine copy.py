# -*- coding: utf-8 -*-
"""
This script automates the analysis of machine learning results from a structured
directory of experiments. It recursively scans a root folder for result files
matching the pattern 'ml_results_*.csv', calculates detailed performance
metrics for each, and aggregates them into summary files for each classifier.

The script calculates two main sets of metrics:
1.  Multiclass Performance: Includes Accuracy, Macro/Weighted Precision, Recall,
    F1-Score, and the newly added Average EER (%).
2.  Error Rates per Class: Includes False Accept Rate (FAR), False Reject Rate
    (FRR), and Equal Error Rate (EER), with EER reported as a percentage.

The final outputs are separate files for each classifier. The multiclass
performance is saved as a CSV, while the detailed FAR/FRR/EER metrics are
saved in an Excel file with each condition on a separate sheet. For example:
-   'LogisticRegression_multiclass_performance.csv'
-   'LogisticRegression_far_frr_eer_performance.xlsx'
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import re
import sys
import numpy as np

# Note: You may need to install the library required for writing Excel files:
# pip install openpyxl

def analyze_all_results(root_directory):
    """
    Scans a directory, processes all ML result files, and saves combined metrics.

    Args:
        root_directory (str): The absolute or relative path to the root folder
                              containing the result subdirectories.
    """
    # Lists to hold the results from all files
    all_multiclass_metrics = []
    all_far_frr_eer_metrics = []

    print(f"--- Starting Analysis in Root Directory: '{os.path.abspath(root_directory)}' ---")

    # Check if the root directory exists
    if not os.path.isdir(root_directory):
        print(f"Error: The specified root directory does not exist: '{root_directory}'")
        sys.exit(1) # Exit the script if the path is invalid

    # Walk through the directory structure
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            # Process only the files that match the pattern 'ml_results_*.csv'
            if filename.startswith('ml_results_') and filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                
                # Extract 'condition' from the folder name and 'classifier' from the file name
                condition = os.path.basename(dirpath)
                classifier_match = re.search(r'ml_results_(.*)\.csv', filename)
                
                if not classifier_match:
                    print(f"Warning: Skipping file with unexpected name format: {filename}")
                    continue
                classifier = classifier_match.group(1)

                print(f"Processing: Condition='{condition}', Classifier='{classifier}'")
                
                try:
                    # --- Core Performance Calculation Logic ---
                    data = pd.read_csv(file_path)
                    
                    # Ensure required columns exist
                    if 'True_Label' not in data.columns or 'Predicted_Label' not in data.columns:
                        print(f"Warning: Skipping file '{file_path}' due to missing 'True_Label' or 'Predicted_Label' columns.")
                        continue

                    y_true = data['True_Label']
                    y_pred = data['Predicted_Label']

                    # 1. Calculate FAR, FRR, and EER per class (One-vs-All)
                    unique_classes = sorted(y_true.unique())
                    class_eers = [] # To store EER for averaging
                    for cls in unique_classes:
                        binary_true = (y_true == cls).astype(int)
                        binary_pred = (y_pred == cls).astype(int)
                        
                        try:
                            tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
                        except ValueError:
                            print(f"  - Skipping FAR/FRR for class {cls} (likely no true samples).")
                            continue

                        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                        eer_percent = ((far + frr) / 2.0) * 100
                        class_eers.append(eer_percent)
                        
                        class_metrics = {
                            'Condition': condition,
                            'Classifier': classifier,
                            'Class': cls,
                            'FAR': far,
                            'FRR': frr,
                            'EER (%)': eer_percent
                        }
                        all_far_frr_eer_metrics.append(class_metrics)
                    
                    # Calculate the average EER across all classes for this file
                    average_eer = np.mean(class_eers) if class_eers else 0.0

                    # 2. Compute Multiclass Performance Metrics
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    
                    multiclass_metrics = {
                        'Condition': condition,
                        'Classifier': classifier,
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'Average EER (%)': average_eer, # Add the new average EER metric
                        'Macro Precision': report['macro avg']['precision'],
                        'Macro Recall': report['macro avg']['recall'],
                        'Macro F1 Score': report['macro avg']['f1-score'],
                        'Weighted Precision': report['weighted avg']['precision'],
                        'Weighted Recall': report['weighted avg']['recall'],
                        'Weighted F1 Score': report['weighted avg']['f1-score'],
                    }
                    all_multiclass_metrics.append(multiclass_metrics)

                except Exception as e:
                    print(f"Error processing file '{file_path}': {e}")
                    continue # Move to the next file

    # --- Combine and Save Final Results Separately for Each Classifier ---
    if all_multiclass_metrics and all_far_frr_eer_metrics:
        # Create master DataFrames from the lists of dictionaries
        combined_multiclass_df = pd.DataFrame(all_multiclass_metrics)
        combined_far_frr_eer_df = pd.DataFrame(all_far_frr_eer_metrics)

        # Get a list of unique classifiers found during processing
        unique_classifiers = combined_multiclass_df['Classifier'].unique()

        print("\n--- ✅ Processing Complete ---")

        # Save separate files for each classifier
        for classifier_name in unique_classifiers:
            # --- Handle Multiclass Performance Data ---
            multiclass_subset_df = combined_multiclass_df[combined_multiclass_df['Classifier'] == classifier_name]
            multiclass_output_path = os.path.join(root_directory, f'{classifier_name}_multiclass_performance.csv')
            multiclass_subset_df.to_csv(multiclass_output_path, index=False)
            print(f"Saved multiclass metrics for '{classifier_name}' to '{multiclass_output_path}'")

            # --- Handle FAR/FRR/EER Performance Data (Save to Excel) ---
            far_frr_eer_subset_df = combined_far_frr_eer_df[combined_far_frr_eer_df['Classifier'] == classifier_name]
            excel_output_path = os.path.join(root_directory, f'{classifier_name}_far_frr_eer_performance.xlsx')
            
            # Create an Excel writer object to save multiple sheets
            with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
                # Get unique conditions for this specific classifier
                unique_conditions = far_frr_eer_subset_df['Condition'].unique()
                
                # Loop through each condition and save it to a separate sheet
                for condition_name in unique_conditions:
                    condition_df = far_frr_eer_subset_df[far_frr_eer_subset_df['Condition'] == condition_name]
                    
                    # Excel sheet names have a 31-character limit, so truncate if necessary
                    sheet_name = condition_name[:31]
                    condition_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"Saved FAR/FRR/EER metrics for '{classifier_name}' to '{excel_output_path}'")

    else:
        print("\n--- ❗ No 'ml_results_*.csv' files were found to process. ---")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # SET THE PATH TO YOUR MAIN PROJECT FOLDER HERE.
    # Use '.' to represent the current directory if you place the script
    # inside your main project folder (e.g., inside 'RUN2_run_1').
    # Or provide the full path, e.g., r"G:\Analysis_3_streamlined\RUN2_run_1"
    
    project_root_directory = '.' 
    
    analyze_all_results(project_root_directory)
