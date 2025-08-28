# In 03_feature_extraction/config.py

# This dictionary holds all the settings for the feature extraction pipeline.
# To add a new task, you only need to add a new entry here.

PIPELINE_CONFIG = {
    "sampling_frequency": 500,
    "window_size_seconds": 2,
    "overlap_percentage": 0.5,
}

# Configuration for each specific task
# 'folder_name': The subfolder within 'Preprocessed' where the segments are stored.
# 'file_pattern': The glob pattern to find the specific segment files.
TASK_CONFIGS = {
    "eyes_open": {
        "folder_name": "{subject_code}_EyesOpen",
        "file_pattern": "*.vhdr",
    },
    "eyes_close": {
        "folder_name": "search_all",  # Special case to search recursively
        "file_pattern": "*baseline*.fif",
    },
    "motor_s3": {
        "folder_name": "{subject_code}_Motor",
        "file_pattern": "{subject_code}_Motor_S3_seg_*_raw.vhdr",
    },
    "motor_s4": {
        "folder_name": "{subject_code}_Motor",
        "file_pattern": "{subject_code}_Motor_S4_seg_*_raw.vhdr",
    },
    "memory_s5": {
        "folder_name": "{subject_code}_Memory",
        "file_pattern": "{subject_code}_S5_segment_*_raw.vhdr",
    },
    "ssvep_s6": {
        "folder_name": "{subject_code}_Stimulus",
        "file_pattern": "{subject_code}_S__6_seg_*_raw.vhdr",
    },
    "ssvep_s7": {
        "folder_name": "{subject_code}_Stimulus",
        "file_pattern": "{subject_code}_S__7_seg_*_raw.vhdr",
    },
    "ssvep_s8": {
        "folder_name": "{subject_code}_Stimulus",
        "file_pattern": "{subject_code}_S__8_seg_*_raw.vhdr",
    },
    "ssvep_s9": {
        "folder_name": "{subject_code}_Stimulus",
        "file_pattern": "{subject_code}_S__9_seg_*_raw.vhdr",
    },
    "ssaep_s10": {
        "folder_name": "{subject_code}_SSAEP",
        "file_pattern": "{subject_code}_S10_segment_*_raw.vhdr",
    },
    "ssaep_s11": {
        "folder_name": "{subject_code}_SSAEP",
        "file_pattern": "{subject_code}_S11_segment_*_raw.vhdr",
    },
}