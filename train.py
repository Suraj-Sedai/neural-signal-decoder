import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.simulate_eeg import SimulatedEEGDataset
from models.lstm_decoder import LSTMDecoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data parameters
    num_channels = 16
    total_timesteps = 1024
    window_size = 256
    stride = 128
    num_samples_per_class = 50

    # Model parameters
    hidden_size = 128
    num_layers = 2
    num_classes = 4

    # Training parameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 10

    dataset = SimulatedEEGDataset(
        num_samples_per_class=num_samples_per_class,
        num_channels=num_channels,
        total_timesteps=total_timesteps,
        window_size=window_size,
        stride=stride
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = LSTMDecoder(
        num_channels=num_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Accuracy: {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
