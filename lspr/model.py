from typing import List

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    layers.append(ResidualBlock1D(in_ch, out_ch, stride=stride))
    for _ in range(1, blocks):
        layers.append(ResidualBlock1D(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


class ResNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(32, 64, blocks=2, stride=2)
        self.layer2 = _make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = _make_layer(128, 128, blocks=2, stride=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).squeeze(-1)
        out = self.fc(out)
        return out


class ResNet1DClassifier(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = ResNet1DEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits
