import torch
import numpy as np
from torch.utils.data import DataLoader

from data.simulate_eeg import SimulatedEEGDataset
from models.lstm_decoder import LSTMDecoder
from models.transformer_decoder import TransformerDecoder

def evaluate_model(model, loader, device, num_classes):
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

    for true_label, pred_label in zip(all_labels, all_preds):
        confusion[true_label, pred_label] += 1

    accuracy = (all_preds == all_labels).mean()
    per_class_accuracy = confusion.diagonal() / confusion.sum(axis=1)

    return accuracy, per_class_accuracy, confusion


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_channels = 16
    total_timesteps = 1024
    window_size = 256
    stride = 128
    num_samples_per_class = 20

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

    lstm_model = LSTMDecoder(
        num_channels=num_channels,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    transformer_model = TransformerDecoder(
        num_channels=num_channels,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    lstm_acc, lstm_pc, lstm_conf = evaluate_model(
        lstm_model, loader, device, num_classes
    )

    trans_acc, trans_pc, trans_conf = evaluate_model(
        transformer_model, loader, device, num_classes
    )

    print("\n===== LSTM RESULTS =====")
    print(f"Overall Accuracy: {lstm_acc:.4f}")
    for i, acc in enumerate(lstm_pc):
        print(f"Class {i} Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(lstm_conf)

    print("\n===== TRANSFORMER RESULTS =====")
    print(f"Overall Accuracy: {trans_acc:.4f}")
    for i, acc in enumerate(trans_pc):
        print(f"Class {i} Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(trans_conf)


if __name__ == "__main__":
    main()