# In 03_feature_extraction/main.py
import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import os

# Import our custom modules and config
import config
import preprocessing
import features

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_feature_extraction_pipeline(root_dir: Path, task_name: str):
    """
    A generic pipeline to extract features for any configured task.
    """
    if task_name not in config.TASK_CONFIGS:
        logger.error(f"Task '{task_name}' not found in config.py. Available tasks: {list(config.TASK_CONFIGS.keys())}")
        return

    task_cfg = config.TASK_CONFIGS[task_name]
    pipe_cfg = config.PIPELINE_CONFIG
    fs = pipe_cfg["sampling_frequency"]
    
    window_samples = int(pipe_cfg["window_size_seconds"] * fs)
    step_size = int(window_samples * (1 - pipe_cfg["overlap_percentage"]))
    
    feature_matrices = []

    for subject_folder in sorted(root_dir.glob("*Subject_*")):
        subject_code = subject_folder.name.split('_')[0]
        logger.info(f"--- Processing Subject: {subject_code} ---")

        for run_folder in (subject_folder / "Run_1", subject_folder / "Run_2"):
            if not run_folder.exists(): continue
            
            run_id = run_folder.name.split("_")[-1]
            preprocessed_base = run_folder / "Preprocessed"
            
            # Dynamically determine search path and pattern from config
            folder_name = task_cfg["folder_name"].format(subject_code=subject_code)
            pattern = task_cfg["file_pattern"].format(subject_code=subject_code)
            
            search_path = preprocessed_base / folder_name
            if folder_name == "search_all": # Special case for baseline
                eeg_files = list(preprocessed_base.rglob(pattern))
            else:
                eeg_files = list(search_path.glob(pattern))

            if not eeg_files:
                logger.warning(f"No files found for task '{task_name}' in {search_path} with pattern '{pattern}'")
                continue

            logger.info(f"Found {len(eeg_files)} file(s) for task '{task_name}' in Run {run_id}")

            for eeg_file in eeg_files:
                try:
                    raw = preprocessing.load_raw_data(eeg_file)
                    raw = preprocessing.preprocess_raw_data(raw, target_fs=fs)
                    eeg_data = raw.get_data()
                    
                    # Trim and apply baseline correction
                    n_samples_total = eeg_data.shape[1]
                    trim_samples = int(fs * 1)
                    if n_samples_total < 2 * trim_samples: continue
                    eeg_data = eeg_data[:, trim_samples:-trim_samples]
                    eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

                    # Create epochs and extract features
                    epoch_counter = 0
                    for start in range(0, eeg_data.shape[1] - window_samples + 1, step_size):
                        epoch_features_list = []
                        for ch_data in eeg_data:
                            windowed_data = ch_data[start : start + window_samples]
                            feat_vector = features.extract_all_features(windowed_data, fs, n_mfcc=13, n_fft=2048) # Pass params directly
                            epoch_features_list.append(feat_vector)
                        
                        feature_matrices.append({
                            'features': np.vstack(epoch_features_list),
                            'class_label': subject_code,
                            'epoch': epoch_counter,
                            'run_id': run_id
                        })
                        epoch_counter += 1
                except Exception as e:
                    logger.error(f"Failed to process file {eeg_file}: {e}")

    # Save the results
    if feature_matrices:
        output_filename = f"features_{task_name}.h5"
        save_features_to_h5(feature_matrices, output_filename)
    else:
        logger.error(f"No features were extracted for task '{task_name}'. No output file was saved.")


def save_features_to_h5(feature_matrices, filename):
    """Saves features to an HDF5 file, organized by subject, run, and epoch."""
    if Path(filename).exists():
        os.remove(filename)
    
    with h5py.File(filename, 'w') as h5f:
        for data in feature_matrices:
            class_label, run_id, epoch_num = data['class_label'], data['run_id'], data['epoch']
            group_path = f'class_{class_label}/Run_{run_id}/epoch_{epoch_num}'
            epoch_group = h5f.create_group(group_path)
            epoch_group.create_dataset('features', data=data['features'])
    logger.info(f"Successfully saved {len(feature_matrices)} epochs to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Feature Extraction Pipeline.")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the subject data.")
    parser.add_argument("--task", type=str, required=True, choices=config.TASK_CONFIGS.keys(), help="The task to extract features for.")
    args = parser.parse_args()

    run_feature_extraction_pipeline(Path(args.root_dir), args.task)