import numpy as np
def normalize_channels(signal: np.ndarray):
    """
    signal: np.ndarray of shape (C, T)

    returns:
        normalized_signal: np.ndarray of shape (C, T)
    """
    C, T = signal.shape
    normalized = np.zeros_like(signal)

    for c in range(C):
        mean = signal[c].mean()
        std = signal[c].std() + 1e-8
        normalized[c] = (signal[c] - mean) / std

    return normalized

def window_signal(signal: np.ndarray, window_size: int, stride: int):
    """
    signal: np.ndarray of shape (C, T)

    returns:
        windows: np.ndarray of shape (N, C, window_size)
    """
    C, T = signal.shape
    windows = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[:, start:end])

    return np.stack(windows)
