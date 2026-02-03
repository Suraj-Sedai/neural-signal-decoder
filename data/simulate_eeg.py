import numpy as np

def simulate_eeg_sample(
    class_id: int,
    num_channels: int,
    num_timesteps: int,
    sampling_rate: int = 256
):
    """
    Returns:
        signal: np.ndarray of shape (num_channels, num_timesteps)
    """

    # Time axis
    t = np.arange(num_timesteps) / sampling_rate

    # Class-dependent frequency bands
    freq_bands = {
        0: (7, 10),    # Left
        1: (11, 14),   # Right
        2: (15, 20),   # Up
        3: (20, 25)    # Down
    }

    low_f, high_f = freq_bands[class_id]

    signal = np.zeros((num_channels, num_timesteps))

    for ch in range(num_channels):
        freq = np.random.uniform(low_f, high_f)
        phase = np.random.uniform(0, 2 * np.pi)

        wave = np.sin(2 * np.pi * freq * t + phase)
        noise = np.random.normal(0, 0.3, size=num_timesteps)

        signal[ch] = wave + noise

    return signal
import torch
from torch.utils.data import Dataset
from utils.preprocessing import normalize_channels, window_signal

class SimulatedEEGDataset(Dataset):
    def __init__(
        self,
        num_samples_per_class: int,
        num_channels: int,
        total_timesteps: int,
        window_size: int,
        stride: int
    ):
        self.windows = []
        self.labels = []

        for class_id in range(4):
            for _ in range(num_samples_per_class):
                signal = simulate_eeg_sample(
                    class_id=class_id,
                    num_channels=num_channels,
                    num_timesteps=total_timesteps
                )

                signal = normalize_channels(signal)
                windows = window_signal(signal, window_size, stride)

                for w in windows:
                    self.windows.append(w)
                    self.labels.append(class_id)

        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]
