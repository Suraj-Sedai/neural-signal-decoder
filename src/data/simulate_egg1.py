import numpy as np

FREQ_BANDS = {
    0: (8, 12),    # Alpha
    1: (13, 30),   # Beta
    2: (4, 7),     # Theta
    3: (30, 45),   # Gamma
}

def simulate_eeg_sample(
    class_id: int,
    num_channels: int,
    num_timesteps: int,
    sampling_rate: int = 256
):
    t = np.arange(num_timesteps) / sampling_rate
    signal = np.zeros((num_channels, num_timesteps))

    low_f, high_f = FREQ_BANDS[class_id]

    base_freq = np.random.uniform(low_f, high_f)

    for ch in range(num_channels):
        freq = base_freq + np.random.uniform(-0.3, 0.3)

        phase = np.random.uniform(0, 2 * np.pi)

        sinusoid = np.sin(2 * np.pi * freq * t + phase)
        noise = 0.3 * np.random.randn(num_timesteps)

        signal[ch] = sinusoid + noise

    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-6
    signal = (signal - mean) / std

    return signal


import torch
from torch.utils.data import Dataset
from utils.preprocessing import normalize_channels, window_signal

def to_spectral(window, sampling_rate=256, max_freq=60):
    """
    Converts a time-domain signal window into its spectral (frequency-domain) representation
    using the Real Fast Fourier Transform (RFFT).

    Args:
        window (np.ndarray): A 2D numpy array representing a single EEG window,
                             with shape (channels, time_points).
        sampling_rate (int): The sampling rate of the EEG signal in Hz.
        max_freq (int): The maximum frequency to retain in the spectral representation.

    Returns:
        np.ndarray: A 2D numpy array representing the spectral features,
                    with shape (channels, freq_bins), where freq_bins
                    are the frequencies up to max_freq.
    """
    # Compute the magnitude of the Real FFT for each channel along the time axis.
    # np.fft.rfft is optimized for real-valued inputs.
    fft = np.abs(np.fft.rfft(window, axis=1))

    # Calculate the corresponding frequencies for the RFFT output.
    freqs = np.fft.rfftfreq(window.shape[1], d=1 / sampling_rate)

    # Return only the frequency bins up to the specified maximum frequency.
    return fft[:, freqs <= max_freq]


class SimulatedEEGDataset(Dataset):
    """
    A PyTorch Dataset for simulated EEG signals, generating multi-channel
    time-series data and converting them to spectral features for classification.
    """
    def __init__(
        self,
        num_samples_per_class: int,
        num_channels: int,
        total_timesteps: int,
        window_size: int,
        stride: int,
        sampling_rate: int = 256,
        max_freq: int = 60
    ):
        """
        Initializes the dataset by simulating EEG signals for different classes,
        windowing them, and converting them to spectral representations.

        Args:
            num_samples_per_class (int): Number of base signals to simulate for each class.
            num_channels (int): Number of EEG channels per signal.
            total_timesteps (int): Total number of time points in the simulated signal.
            window_size (int): The size of the sliding window for segmentation.
            stride (int): The step size for sliding the window.
            sampling_rate (int): The sampling rate of the simulated EEG signal.
            max_freq (int): The maximum frequency to include in the spectral features.
        """
        self.windows = []  # Stores the spectral representations of the windows
        self.labels = []   # Stores the corresponding class labels

        # Loop through each defined class (0 to 3 for Alpha, Beta, Theta, Gamma)
        for class_id in range(4):
            # Generate a specified number of samples for the current class
            for _ in range(num_samples_per_class):
                # Simulate a single multi-channel EEG signal for the current class
                signal = simulate_eeg_sample(
                    class_id=class_id,
                    num_channels=num_channels,
                    num_timesteps=total_timesteps,
                    sampling_rate=sampling_rate
                )

                # Segment the continuous signal into overlapping windows in the time domain
                windows = window_signal(signal, window_size, stride)

                # Process each generated time window
                for w in windows:
                    # Convert each window from time domain to spectral domain (FFT magnitude)
                    w_spec = to_spectral(
                        w,
                        sampling_rate=sampling_rate,
                        max_freq=max_freq
                    )

                    # Append the spectral window and its corresponding label to the lists
                    self.windows.append(w_spec)
                    self.labels.append(class_id)

        # Convert the collected lists of windows and labels into PyTorch tensors
        # This makes them compatible with PyTorch's DataLoader and model inputs
        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        """
        Returns the total number of samples (windows) in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a sample (spectral window and its label) from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the spectral window (torch.Tensor) and its label (torch.LongTensor).
        """
        return self.windows[idx], self.labels[idx]