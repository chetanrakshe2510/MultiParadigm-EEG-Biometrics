# --- utils.py ---
import mne
import pandas as pd
from pathlib import Path
import logging
import shutil

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_eeg_data(file_path: Path) -> mne.io.BaseRaw | None:
    """Loads EEG data from .vhdr or .edf files."""
    try:
        if file_path.suffix == '.vhdr':
            return mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        elif file_path.suffix == '.edf':
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        else:
            logging.warning(f"Unsupported file format for {file_path}. Skipping.")
            return None
    except Exception as e:
        logging.error(f"Error loading EEG data from {file_path}: {e}")
        return None

def get_annotations(raw_data: mne.io.BaseRaw) -> pd.DataFrame:
    """Retrieves and cleans annotations from raw EEG data."""
    annotations = raw_data.annotations
    return pd.DataFrame({
        "Description": [desc.strip() for desc in annotations.description],
        "Onset (s)": annotations.onset,
        "Duration (s)": annotations.duration
    })

def export_as_brainvision(raw: mne.io.BaseRaw, output_dir: Path, filename_base: str):
    """Exports an MNE Raw object to BrainVision format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vhdr_path = output_dir / f"{filename_base}_raw.vhdr"
    try:
        raw.export(vhdr_path, fmt='brainvision', overwrite=True, verbose=False)
        logging.info(f"Exported BrainVision file: {vhdr_path.name}")
    except Exception as e:
        logging.error(f"Failed to export BrainVision file for {filename_base}: {e}")

def export_as_fif(raw: mne.io.BaseRaw, output_dir: Path, filename_base: str):
    """Exports an MNE Raw object to FIF format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fif_path = output_dir / f"{filename_base}_raw.fif"
    try:
        raw.save(fif_path, overwrite=True, verbose=False)
        logging.info(f"Exported FIF file: {fif_path.name}")
    except Exception as e:
        logging.error(f"Failed to export FIF file for {filename_base}: {e}")

def extract_baseline(raw: mne.io.BaseRaw, annotations_df: pd.DataFrame) -> mne.io.BaseRaw | None:
    """Extracts the baseline segment defined by two 'Stimulus/S  2' markers."""
    baseline_stimulus = "Stimulus/S  2"
    baseline_indices = annotations_df[annotations_df["Description"] == baseline_stimulus].index.tolist()
    if len(baseline_indices) < 2:
        logging.warning(f"Insufficient markers ({len(baseline_indices)}) for baseline '{baseline_stimulus}'.")
        return None

    start_time = annotations_df.loc[baseline_indices[0], "Onset (s)"]
    end_time = annotations_df.loc[baseline_indices[1], "Onset (s)"]
    logging.info(f"Extracting baseline from {start_time:.2f}s to {end_time:.2f}s.")
    return raw.copy().crop(tmin=start_time, tmax=end_time)

def clean_output_folder(output_dir: Path):
    """Deletes a folder if it exists."""
    if output_dir.exists() and output_dir.is_dir():
        try:
            shutil.rmtree(output_dir)
            logging.info(f"Cleaned previous output folder: {output_dir}")
        except Exception as e:
            logging.error(f"Error deleting folder {output_dir}: {e}")