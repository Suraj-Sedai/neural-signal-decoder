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
