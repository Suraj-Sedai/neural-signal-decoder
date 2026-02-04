import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 4
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, C, T)
        x = x.permute(0, 2, 1)  # (batch, T, C)

        _, (h_n, _) = self.lstm(x)

        final_hidden = h_n[-1]  # (batch, hidden)
        logits = self.classifier(final_hidden)

        return logits
