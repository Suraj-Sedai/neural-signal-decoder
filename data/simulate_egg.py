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
