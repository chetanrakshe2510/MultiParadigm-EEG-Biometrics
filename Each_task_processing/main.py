# --- main.py ---
import argparse
from pathlib import Path
import re
import logging
import utils
import tasks

# A dictionary to map task names to their specific settings
TASK_CONFIG = {
    "eyes_open": {
        "file_pattern": "*[eE]yes[_][oO]pen.vhdr",
        "extractor_func": tasks.extract_eyes_open_segments,
    },
    "motor": {
        "file_pattern": "*[Mm]otor.vhdr",
        "extractor_func": tasks.extract_motor_segments,
    },
    "memory": {
        "file_pattern": "*[Mm]emory.vhdr",
        "extractor_func": tasks.extract_memory_segments,
    },
    "ssaep": {
        "file_pattern": "*SSAEP.vhdr",
        "extractor_func": tasks.extract_ssaep_segments,
    },
    "ssvep": {
        "file_pattern": "*SSVEP.vhdr",
        "extractor_func": None, # Placeholder for SSVEP logic
    },
}

def process_single_file(file_path: Path, output_dir: Path, subject_code: str, extractor_func):
    """Main processing workflow for one EEG file."""
    logging.info(f"Processing file: {file_path.name}")
    raw = utils.load_eeg_data(file_path)
    if not raw:
        return

    annotations_df = utils.get_annotations(raw)

    # 1. Extract and save the baseline
    baseline_raw = utils.extract_baseline(raw, annotations_df)
    if baseline_raw:
        base_name = f"{subject_code}_Baseline"
        utils.export_as_fif(baseline_raw, output_dir, base_name)
        utils.export_as_brainvision(baseline_raw, output_dir, base_name)

    # 2. Extract and save task-specific segments
    if extractor_func:
        task_segments = extractor_func(raw, annotations_df)
        if not task_segments:
            logging.warning(f"No task-specific segments were extracted from {file_path.name}.")
            return
            
        for name, segment_raw in task_segments.items():
            base_name = f"{subject_code}_{name}"
            utils.export_as_fif(segment_raw, output_dir, base_name)
            utils.export_as_brainvision(segment_raw, output_dir, base_name)
    else:
        logging.warning("No extractor function provided for this task.")


def batch_processor(root_dir: Path, task: str, clean: bool):
    """Iterates through subjects and runs to process EEG data for a specific task."""
    if task not in TASK_CONFIG:
        logging.error(f"Task '{task}' is not recognized. Available tasks: {list(TASK_CONFIG.keys())}")
        return

    config = TASK_CONFIG[task]
    file_pattern = config["file_pattern"]
    extractor_func = config["extractor_func"]
    
    logging.info(f"--- Starting Batch Processing for Task: '{task.upper()}' ---")
    
    for subject_folder in sorted(root_dir.glob("*Subject_*")):
        match = re.match(r"(\d+)_Subject_", subject_folder.name)
        if not match:
            continue
        subject_code = match.group(1)
        logging.info(f"--- Processing Subject: {subject_code} ---")

        for run_folder in sorted(subject_folder.glob("Run_*")):
            segmented_folder = run_folder / "Segmented"
            task_files = list(segmented_folder.glob(file_pattern))

            if not task_files:
                logging.warning(f"No files matching '{file_pattern}' found in {segmented_folder}")
                continue

            # Define and optionally clean the output directory
            output_dir = run_folder / "Preprocessed" / f"{subject_code}_{task.capitalize()}"
            if clean:
                utils.clean_output_folder(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for eeg_file in task_files:
                process_single_file(eeg_file, output_dir, subject_code, extractor_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A modular toolkit for batch processing EEG experimental tasks."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="The root directory containing the subject data folders."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIG.keys()),
        help="The experimental task to process."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="If set, deletes the task's output folder before running to ensure a fresh start."
    )
    args = parser.parse_args()
    
    root_path = Path(args.root_dir)
    if not root_path.is_dir():
        logging.error(f"Root directory not found: {root_path}")
    else:
        batch_processor(root_dir=root_path, task=args.task, clean=args.clean)