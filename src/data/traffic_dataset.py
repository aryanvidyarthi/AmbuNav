from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """METR-LA sliding-window dataset for traffic forecasting."""

    def __init__(
        self,
        data_path: str | Path = "data/raw/METR-LA.h5",
        input_length: int = 12,
        output_length: int = 3,
        train_ratio: float = 0.70,
        fit_scaler_on_train: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.input_length = input_length
        self.output_length = output_length
        self.train_ratio = train_ratio
        self.fit_scaler_on_train = fit_scaler_on_train

        if not self.data_path.exists():
            raise FileNotFoundError(f"METR-LA file not found: {self.data_path}")

        values = self._load_h5(self.data_path)
        self.scaler = MinMaxScaler()
        normalized = self._normalize_values(values)

        self.x, self.y = self._create_windows(normalized)

    def _load_h5(self, path: Path) -> np.ndarray:
        frame = pd.read_hdf(path)
        values = frame.to_numpy(dtype=np.float32)
        if values.ndim != 2:
            msg = "Expected METR-LA data shape [time, nodes]"
            raise ValueError(msg)
        return values

    def _create_windows(self, values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        total_steps, num_nodes = values.shape
        sample_count = total_steps - self.input_length - self.output_length + 1
        if sample_count <= 0:
            msg = "Not enough timesteps for requested input/output lengths"
            raise ValueError(msg)

        x = np.empty((sample_count, self.input_length, num_nodes), dtype=np.float32)
        y = np.empty((sample_count, self.output_length, num_nodes), dtype=np.float32)

        for idx in range(sample_count):
            x[idx] = values[idx : idx + self.input_length]
            y[idx] = values[
                idx + self.input_length : idx + self.input_length + self.output_length
            ]

        return torch.from_numpy(x), torch.from_numpy(y)

    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        if not self.fit_scaler_on_train:
            return self.scaler.fit_transform(values)

        total_steps = int(values.shape[0])
        sample_count = total_steps - self.input_length - self.output_length + 1
        if sample_count <= 0:
            msg = "Not enough timesteps for requested input/output lengths"
            raise ValueError(msg)

        train_samples = int(sample_count * self.train_ratio)
        if train_samples <= 0:
            msg = "train_ratio is too small; no training samples available."
            raise ValueError(msg)

        # Last train window uses values up to (start + input + output - 1), so include that span.
        train_value_end = train_samples + self.input_length + self.output_length - 1
        train_values = values[:train_value_end]
        self.scaler.fit(train_values)
        return self.scaler.transform(values)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # X: (input_length, num_nodes), Y: (output_length, num_nodes)
        return self.x[index], self.y[index]


def get_dataloaders(
    batch_size: int = 64,
    data_path: str | Path = "data/raw/METR-LA.h5",
    input_length: int = 12,
    output_length: int = 3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    return_scaler: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]:
    """Return train/val/test dataloaders with 70/15/15 chronological split."""

    dataset = TrafficDataset(
        data_path=data_path,
        input_length=input_length,
        output_length=output_length,
        train_ratio=train_ratio,
        fit_scaler_on_train=True,
    )

    total_samples = len(dataset)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    x_train, y_train = dataset.x[:train_end], dataset.y[:train_end]
    x_val, y_val = dataset.x[train_end:val_end], dataset.y[train_end:val_end]
    x_test, y_test = dataset.x[val_end:], dataset.y[val_end:]

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if return_scaler:
        return train_loader, val_loader, test_loader, dataset.scaler

    return train_loader, val_loader, test_loader
