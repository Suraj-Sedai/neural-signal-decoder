import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_classes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(num_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, C, T)
        x = x.permute(0, 2, 1)          # (batch, T, C)
        x = self.input_proj(x)          # (batch, T, d_model)

        T = x.size(1)
        x = x + self.pos_embedding[:, :T, :]

        x = self.transformer(x)         # (batch, T, d_model)
        x = x.mean(dim=1)               # temporal pooling

        logits = self.classifier(x)
        return logits
