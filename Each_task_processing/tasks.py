# --- tasks.py ---
import logging
import numpy as np

def extract_eyes_open_segments(raw, annotations_df):
    """Extracts segment between the first two 'S 1' markers."""
    segments = {}
    marker = "Stimulus/S 1"
    marker_indices = annotations_df[annotations_df["Description"] == marker].index.tolist()
    if len(marker_indices) >= 2:
        start_time = annotations_df.loc[marker_indices[0], "Onset (s)"]
        end_time = annotations_df.loc[marker_indices[1], "Onset (s)"]
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)
        segments['EyesOpen_segment'] = segment
        logging.info(f"Extracted 'Eyes Open' segment from {start_time:.2f}s to {end_time:.2f}s.")
    else:
        logging.warning("Not enough 'S 1' markers found for 'Eyes Open' task.")
    return segments

def extract_ssaep_segments(raw, annotations_df):
    """Extracts segments between S10/S11 and S15 markers."""
    segments = {}
    start_indices = np.where(annotations_df["Description"].str.contains("Stimulus/S 10|Stimulus/S 11"))[0]
    end_indices = np.where(annotations_df["Description"] == "Stimulus/S 15")[0]

    for i, start_idx in enumerate(start_indices):
        start_onset = annotations_df.iloc[start_idx]["Onset (s)"]
        marker_type = "S10" if "10" in annotations_df.iloc[start_idx]["Description"] else "S11"
        
        valid_end_indices = end_indices[end_indices > start_idx]
        if valid_end_indices.size > 0:
            end_idx = valid_end_indices[0]
            end_onset = annotations_df.iloc[end_idx]["Onset (s)"]
            segment = raw.copy().crop(tmin=start_onset, tmax=end_onset)
            segments[f'{marker_type}_segment_{i+1}'] = segment
            logging.info(f"Extracted SSAEP segment {i+1} ({marker_type}) from {start_onset:.2f}s to {end_onset:.2f}s.")
    return segments

def extract_memory_segments(raw, annotations_df):
    """Extracts segments between 'S 5' and 'S 14' markers."""
    segments = {}
    start_indices = np.where(annotations_df["Description"].str.contains("Stimulus/S  5", regex=False))[0]
    end_indices = np.where(annotations_df["Description"] == "Stimulus/S 14")[0]
    
    for i, start_idx in enumerate(start_indices):
        start_onset = annotations_df.iloc[start_idx]["Onset (s)"]
        valid_end_indices = end_indices[end_indices > start_idx]
        if valid_end_indices.size > 0:
            end_idx = valid_end_indices[0]
            end_onset = annotations_df.iloc[end_idx]["Onset (s)"]
            segment = raw.copy().crop(tmin=start_onset, tmax=end_onset)
            segments[f'S5_segment_{i+1}'] = segment
            logging.info(f"Extracted Memory segment {i+1} from {start_onset:.2f}s to {end_onset:.2f}s.")
    return segments

def extract_motor_segments(raw, annotations_df):
    """Extracts 7.5s segments starting at S3 and S4 markers."""
    segments = {}
    segment_duration = 7.5
    for stim_label, suffix in [("Stimulus/S  3", "S3"), ("Stimulus/S  4", "S4")]:
        stimulus_markers = annotations_df[annotations_df["Description"] == stim_label]
        for i, row in stimulus_markers.iterrows():
            start = row["Onset (s)"]
            end = start + segment_duration
            if end <= raw.times.max():
                segment = raw.copy().crop(tmin=start, tmax=end)
                segments[f'{suffix}_segment_{i+1}'] = segment
                logging.info(f"Extracted Motor segment {suffix}_{i+1} from {start:.2f}s to {end:.2f}s.")
            else:
                logging.warning(f"Motor segment for {stim_label} skipped; exceeds max time.")
    return segments