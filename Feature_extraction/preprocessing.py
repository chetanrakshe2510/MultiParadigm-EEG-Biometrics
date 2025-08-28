# In 03_feature_extraction/preprocessing.py
import mne
from mne.preprocessing import ICA
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_raw_data(file_path):
    """
    Load raw EEG data from a .fif or .vhdr file using MNE, based on file extension.
    """
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.fif':
        logger.debug("Loading FIF file: %s", file_path)
        return mne.io.read_raw_fif(str(file_path), preload=True)
    elif file_path.suffix.lower() == '.vhdr':
        logger.debug("Loading BrainVision file: %s", file_path)
        return mne.io.read_raw_brainvision(str(file_path), preload=True)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

def preprocess_raw_data(raw, l_freq=0.1, h_freq=70.0, target_fs=500, ica_corr_threshold=0.5):
    """
    Preprocess raw EEG data while ensuring only EEG channels are considered.
    This function includes resampling, filtering, notch filtering, interpolation,
    and artifact removal using ICA.
    """
    current_fs = raw.info['sfreq']
    if current_fs > target_fs + 1:
        logger.info("Resampling from %s Hz to %s Hz...", current_fs, target_fs)
        raw.resample(target_fs, npad="auto")
    else:
        logger.info("No resampling needed. Current sampling frequency: %s Hz.", current_fs)

    try:
        raw.set_montage('easycap-M1', verbose=False)
    except Exception as e:
        logger.warning("Montage setting failed, using standard 10-20 positions: %s", e)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose=False)

    iir_params = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir', iir_params=iir_params, phase='zero', verbose=False)

    if not raw.info['bads']:
        logger.warning("No bad channels marked for interpolation.")
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # Select only EEG channels
    eeg_channels = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks=eeg_channels)
    logger.info("Using %d EEG channels for feature extraction.", len(eeg_channels))

    # Run ICA for artifact removal
    logger.info("Running ICA to remove artifacts...")
    ica = ICA(n_components=20, random_state=97, max_iter=1000)
    ica.fit(raw)

    excluded_components = []
    frontal_channels = ['Fp1', 'Fp2']
    available_channels = set(raw.info['ch_names'])
    frontal_channels = [ch for ch in frontal_channels if ch in available_channels]

    if frontal_channels:
        logger.info("Checking ICA components against frontal channels: %s", frontal_channels)
        frontal_data = raw.copy().pick_channels(frontal_channels).get_data()
        ica_sources = ica.get_sources(raw).get_data()
        for comp_idx in range(ica_sources.shape[0]):
            for ch in range(frontal_data.shape[0]):
                corr_value = np.corrcoef(ica_sources[comp_idx], frontal_data[ch])[0, 1]
                logger.debug("ICA Component %d - %s Correlation: %.4f", comp_idx, frontal_channels[ch], corr_value)
                if abs(corr_value) >= ica_corr_threshold:
                    excluded_components.append(comp_idx)
                    break
        excluded_components = list(set(excluded_components))
        logger.info("Excluding ICA components: %s", excluded_components)
    else:
        logger.warning("Frontal channels (Fp1, Fp2) missing. Skipping ICA artifact removal based on correlation.")

    ica.exclude = excluded_components
    raw = ica.apply(raw)
    logger.info("Preprocessing complete.")
    return raw