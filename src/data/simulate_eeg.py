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
