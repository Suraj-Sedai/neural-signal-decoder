import torch
import numpy as np
from torch.utils.data import DataLoader

from data.simulate_eeg import SimulatedEEGDataset
from models.lstm_decoder import LSTMDecoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_channels = 16
    total_timesteps = 1024
    window_size = 256
    stride = 128
    num_samples_per_class = 20

    hidden_size = 128
    num_layers = 2
    num_classes = 4
    batch_size = 32

    dataset = SimulatedEEGDataset(
        num_samples_per_class=num_samples_per_class,
        num_channels=num_channels,
        total_timesteps=total_timesteps,
        window_size=window_size,
        stride=stride
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMDecoder(
        num_channels=num_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    accuracy = (all_preds == all_labels).mean()
    per_class_acc = confusion.diagonal() / confusion.sum(axis=1)

    print(f"Overall Accuracy: {accuracy:.4f}")

    for i, acc in enumerate(per_class_acc):
        print(f"Class {i} Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)


if __name__ == "__main__":
    main()
