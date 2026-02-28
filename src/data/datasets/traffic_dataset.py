"""Data pipeline utilities for METR-LA traffic forecasting."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficSpeedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for traffic speed forecasting windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 3 or y.ndim != 3:
            msg = "x and y must be rank-3 arrays: [samples, time, nodes]"
            raise ValueError(msg)

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass(frozen=True)
class TrafficDataSplit:
    """Container for train/validation/test arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class StandardScaler:
    """Simple standardization helper."""

    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std) + self.mean


def load_metr_la_h5(h5_path: str | Path) -> np.ndarray:
    """Load METR-LA traffic speeds from h5 and return [time, nodes] ndarray."""

    path = Path(h5_path)

    try:
        frame = pd.read_hdf(path)
        values = frame.to_numpy(dtype=np.float32)
    except Exception:
        import h5py

        with h5py.File(path, "r") as handle:
            first_key = next(iter(handle.keys()))
            values = np.asarray(handle[first_key], dtype=np.float32)

    if values.ndim != 2:
        msg = "METR-LA h5 content must be a 2D array [time, nodes]"
        raise ValueError(msg)
    return values


def load_adjacency_matrix(adj_path: str | Path) -> np.ndarray:
    """Load adjacency matrix from METR-LA adj_mx.pkl."""

    with Path(adj_path).open("rb") as handle:
        content: Any = pickle.load(handle)

    if isinstance(content, tuple) and len(content) == 3:
        _, _, adjacency = content
    elif isinstance(content, list) and len(content) == 3:
        _, _, adjacency = content
    else:
        adjacency = content

    adjacency_array = np.asarray(adjacency, dtype=np.float32)
    if adjacency_array.ndim != 2:
        msg = "adjacency matrix must be rank-2"
        raise ValueError(msg)

    return adjacency_array


def build_standard_scaler(train_values: np.ndarray) -> StandardScaler:
    """Fit a standard scaler from train values only."""

    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    std = std if std > 0 else 1.0
    return StandardScaler(mean=mean, std=std)


def create_sliding_windows(
    values: np.ndarray,
    input_window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create forecasting windows from [time, nodes] sequence."""

    if values.ndim != 2:
        msg = "values must be rank-2 array [time, nodes]"
        raise ValueError(msg)

    if input_window <= 0 or horizon <= 0:
        msg = "input_window and horizon must be positive"
        raise ValueError(msg)

    total_steps = values.shape[0]
    sample_count = total_steps - input_window - horizon + 1
    if sample_count <= 0:
        msg = "Not enough timesteps for the requested input_window and horizon"
        raise ValueError(msg)

    x = np.empty((sample_count, input_window, values.shape[1]), dtype=np.float32)
    y = np.empty((sample_count, horizon, values.shape[1]), dtype=np.float32)

    for i in range(sample_count):
        x[i] = values[i : i + input_window]
        y[i] = values[i + input_window : i + input_window + horizon]

    return x, y


def split_time_series(
    values: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split [time, nodes] values into train/val/test."""

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        msg = "train_ratio and val_ratio must be in [0, 1], with train_ratio > 0"
        raise ValueError(msg)

    if train_ratio + val_ratio >= 1:
        msg = "train_ratio + val_ratio must be < 1"
        raise ValueError(msg)

    steps = values.shape[0]
    train_end = int(steps * train_ratio)
    val_end = train_end + int(steps * val_ratio)

    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    return train_values, val_values, test_values


def prepare_metr_la_pipeline(
    metr_h5_path: str | Path,
    adj_mx_path: str | Path,
    input_window: int,
    horizon: int,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 0,
) -> tuple[TrafficDataSplit, np.ndarray, StandardScaler, dict[str, DataLoader]]:
    """End-to-end METR-LA pipeline with loaders and scaler."""

    values = load_metr_la_h5(metr_h5_path)
    adjacency = load_adjacency_matrix(adj_mx_path)

    train_values, val_values, test_values = split_time_series(
        values=values,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    scaler = build_standard_scaler(train_values)

    train_norm = scaler.transform(train_values)
    val_norm = scaler.transform(val_values)
    test_norm = scaler.transform(test_values)

    x_train, y_train = create_sliding_windows(train_norm, input_window, horizon)
    x_val, y_val = create_sliding_windows(val_norm, input_window, horizon)
    x_test, y_test = create_sliding_windows(test_norm, input_window, horizon)

    split = TrafficDataSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    loaders = {
        "train": DataLoader(
            TrafficSpeedDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            TrafficSpeedDataset(x_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            TrafficSpeedDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return split, adjacency, scaler, loaders
