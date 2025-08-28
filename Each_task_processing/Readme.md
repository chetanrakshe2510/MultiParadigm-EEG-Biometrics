# EEG Batch Processing Toolkit

A modular Python toolkit for automatically processing EEG data from various experimental tasks. This tool extracts baseline and task-specific segments from raw BrainVision files and saves them in both `.fif` and BrainVision formats.

## Supported Tasks
* **Eyes Open**: Resting-state data.
* **Motor**: Motor imagery task.
* **Memory**: Memory encoding/retrieval task.
* **SSAEP**: Steady-State Auditory Evoked Potential.
* **SSVEP**: Steady-State Visually Evoked Potential (logic to be added).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/eeg_processing_toolkit.git](https://github.com/your-username/eeg_processing_toolkit.git)
    cd eeg_processing_toolkit
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

This script expects your data to be organized in the following structure:
```
<root_directory>/
├── 101_Subject_Name1/
│   ├── Run_1/
│   │   └── Segmented/
│   │       ├── ..._Motor.vhdr
│   │       └── ..._Memory.vhdr
│   │       └── ...
...
```
## Usage

The main script `main.py` is controlled via command-line arguments.

### Arguments
* `--root-dir`: (Required) The path to the root directory containing your subject data.
* `--task`: (Required) The name of the task you want to process. Choose from: `eyes_open`, `motor`, `memory`, `ssaep`, `ssvep`.
* `--clean`: (Optional) If included, the script will delete any previous output for that task before running.

### Examples

**Process the Motor task for all subjects, cleaning previous results:**
```bash
python main.py --root-dir "/path/to/your/data" --task motor --clean
```

**Process the Memory task for all subjects:**
```bash
python main.py --root-dir "/path/to/your/data" --task memory
```
