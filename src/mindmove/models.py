"""Lightweight 1D CNN for a single EEG epoch."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def resolve_device(requested: str) -> torch.device:
    name = str(requested).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class EEGCNN(nn.Module):
    """Conv1d network specified for MindMove.

    Input shape is (batch, channels, time). The linear layer reads the
    second convolution's feature count after global pooling. In the default
    config that count is 64, which is also the EEG channel count, but the
    two numbers are not the same quantity.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 320,
        n_classes: int = 2,
        dropout: float = 0.4,
        conv1_channels: int = 32,
        conv1_kernel: int = 15,
        conv2_channels: int = 64,
        conv2_kernel: int = 7,
        pool_kernel: int = 4,
    ) -> None:
        super().__init__()
        _check_temporal_length(n_times, conv1_kernel, pool_kernel, conv2_kernel)
        self.spec = {
            "n_channels": int(n_channels),
            "n_times": int(n_times),
            "n_classes": int(n_classes),
            "dropout": float(dropout),
            "conv1_channels": int(conv1_channels),
            "conv1_kernel": int(conv1_kernel),
            "conv2_channels": int(conv2_channels),
            "conv2_kernel": int(conv2_kernel),
            "pool_kernel": int(pool_kernel),
        }
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, conv1_channels, kernel_size=conv1_kernel, bias=False),
            nn.BatchNorm1d(conv1_channels),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=pool_kernel),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=conv2_kernel, bias=False),
            nn.BatchNorm1d(conv2_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(conv2_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time), got {tuple(x.shape)}.")
        pooled = self.features(x).squeeze(-1)
        return self.classifier(self.dropout(pooled))


def _check_temporal_length(n_times: int, kernel1: int, pool: int, kernel2: int) -> None:
    length = _conv_length(n_times, kernel1)
    length = _pool_length(length, pool)
    _conv_length(length, kernel2)


def _conv_length(length: int, kernel: int) -> int:
    output = int(length) - int(kernel) + 1
    if output <= 0:
        raise ValueError(f"Convolution kernel {kernel} does not fit a length-{length} signal.")
    return output


def _pool_length(length: int, kernel: int) -> int:
    if int(length) < int(kernel):
        raise ValueError(f"Pooling kernel {kernel} does not fit a length-{length} signal.")
    return (int(length) - int(kernel)) // int(kernel) + 1
