# In 03_feature_extraction/features.py
import numpy as np
from scipy.stats import skew, kurtosis
# ... (all other necessary imports)
from librosa.feature import mfcc
from PyEMD import EMD
import logging

logger = logging.getLogger(__name__)

# This is where your feature computation logic lives, unchanged.
def compute_features(data, fs, eps=1e-6):
    features = {}
    # Statistical Features
    features['skewness'] = skew(data)
    features['kurtosis'] = kurtosis(data)
    features['zero_crossing_rate'] = ((data[:-1] * data[1:]) < 0).sum()
    
    # Hjorth Parameters
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    activity = np.var(data)
    mobility = np.sqrt(np.var(diff1) / (activity + eps))
    complexity = (np.sqrt(np.var(diff2) / (np.var(diff1) + eps)) / (mobility + eps))
    features['hjorth_mobility'] = mobility
    features['hjorth_complexity'] = complexity
    
    # Frequency-Domain Features using Welch's method
    nperseg = int(fs * 2) if len(data) >= fs * 2 else len(data)
    freqs, psd = welch(data, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd) + eps
    
    # Delta Relative Power (0.5 - 4 Hz)
    delta_idx = (freqs >= 0.5) & (freqs <= 4)
    delta_power = np.sum(psd[delta_idx])
    features['delta_relative_power'] = delta_power / total_power
    
    # Beta Relative Power (13 - 30 Hz)
    beta_idx = (freqs >= 13) & (freqs <= 30)
    beta_power = np.sum(psd[beta_idx])
    features['beta_relative_power'] = beta_power / total_power
    
    # Theta and Alpha for band ratios
    theta_idx = (freqs >= 4) & (freqs <= 8)
    theta_power = np.sum(psd[theta_idx])
    alpha_idx = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.sum(psd[alpha_idx])
    
    try:
        features['spectral_entropy'] = spectral_entropy(data, sf=fs, method='welch', normalize=True)
    except Exception as e:
        logger.error("Spectral entropy error: %s", e)
        features['spectral_entropy'] = np.nan
    
    features['theta_alpha_ratio'] = theta_power / (alpha_power + eps)
    features['beta_theta_ratio'] = beta_power / (theta_power + eps)
    
    try:
        features['permutation_entropy'] = ordpy.permutation_entropy(data, dx=3)
    except Exception as e:
        logger.error("Permutation entropy error: %s", e)
        features['permutation_entropy'] = np.nan
    
    def hurst_exponent(signal):
        N = len(signal)
        Y = np.cumsum(signal - np.mean(signal))
        R = np.max(Y) - np.min(Y)
        S = np.std(signal)
        return np.log(R / (S + eps) + eps) / np.log(N)
    features['hurst_exponent'] = hurst_exponent(data)
    
    def katz_fd(signal):
        L = np.sum(np.sqrt(1 + np.diff(signal) ** 2))
        d = np.max(np.abs(signal - signal[0]))
        N = len(signal)
        return np.log10(N) / (np.log10(N) + np.log10(d / L + eps))
    features['katz_fd'] = katz_fd(data)
    
    def higuchi_fd(signal, kmax=10):
        L = []
        N = len(signal)
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = np.sum(np.abs(np.diff(signal[m::k])))
                Lmk *= (N - 1) / (len(signal[m::k]) * k)
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        L = np.array(L)
        lnL = np.log(L + eps)
        lnk = np.log(1.0 / np.arange(1, kmax + 1))
        higuchi, _ = np.polyfit(lnk, lnL, 1)
        return higuchi
    features['higuchi_fd'] = higuchi_fd(data)
    
    # MFCCs (n_mfcc coefficients)
    mfccs = mfcc(y=data, sr=fs, n_mfcc=config["n_mfcc"], n_fft=config["n_fft"])
    for i in range(config["n_mfcc"]):
        features[f'mfcc_{i + 1}'] = np.mean(mfccs[i, :])
    
    # EMD-Based Feature: imf_1_entropy
    emd = EMD()
    imfs = emd(data)
    if len(imfs) > 0:
        try:
            features['imf_1_entropy'] = spectral_entropy(imfs[0], sf=fs, method='welch', normalize=True)
        except Exception as e:
            logger.error("IMF entropy error: %s", e)
            features['imf_1_entropy'] = np.nan
    else:
        features['imf_1_entropy'] = np.nan
    
    return features

def get_feature_order(fs=500, n_samples=1000):
    """
    Run compute_features on a dummy signal to determine the feature order.
    """
    dummy_signal = np.random.randn(n_samples)
    feats = compute_features(dummy_signal, fs)
    return list(feats.keys())

def extract_all_features(epoch_data, fs):
    """
    Compute and return features for one epoch of data.
    """
    spectral_feats = compute_features(epoch_data, fs)
    feat_vector = np.array(list(spectral_feats.values()))
    logger.info("Extracted Feature Vector Shape: %s", feat_vector.shape)
    return feat_vector