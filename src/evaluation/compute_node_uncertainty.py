from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from src.data.traffic_dataset import TrafficDataset
    from src.models.graph_wavenet import GraphWaveNet
except ModuleNotFoundError:
    # Allows running as: python src/evaluation/compute_node_uncertainty.py
    from data.traffic_dataset import TrafficDataset
    from models.graph_wavenet import GraphWaveNet


def _inverse_scale(batch_values: np.ndarray, scaler: Any) -> np.ndarray:
    original_shape = batch_values.shape
    restored = scaler.inverse_transform(batch_values.reshape(-1, original_shape[-1]))
    return restored.reshape(original_shape)


def _load_validation_loader(batch_size: int) -> tuple[DataLoader, Any, int, int]:
    dataset = TrafficDataset(
        data_path=PROJECT_ROOT / "data" / "raw" / "METR-LA.h5",
        input_length=12,
        output_length=3,
    )

    total_samples = len(dataset)
    train_end = int(total_samples * 0.70)
    val_end = train_end + int(total_samples * 0.15)

    x_val = dataset.x[train_end:val_end]
    y_val = dataset.y[train_end:val_end]
    loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, dataset.scaler, int(x_val.shape[-1]), int(y_val.shape[1])


def _load_model(device: torch.device, num_nodes: int, out_dim: int) -> GraphWaveNet:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = GraphWaveNet(
        num_nodes=num_nodes,
        in_dim=1,
        out_dim=out_dim,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=64,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_node_uncertainty(batch_size: int = 64) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, scaler, num_nodes, out_dim = _load_validation_loader(batch_size=batch_size)
    model = _load_model(device=device, num_nodes=num_nodes, out_dim=out_dim)

    err_sum = np.zeros(num_nodes, dtype=np.float64)
    err_sq_sum = np.zeros(num_nodes, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch.to(device)).cpu().numpy()  # [B, H, N]
            target = y_batch.numpy()  # [B, H, N]

            preds_inv = _inverse_scale(preds, scaler)
            target_inv = _inverse_scale(target, scaler)
            error = target_inv - preds_inv

            err_sum += error.sum(axis=(0, 1))
            err_sq_sum += np.square(error).sum(axis=(0, 1))
            count += int(error.shape[0] * error.shape[1])

    if count <= 0:
        raise RuntimeError("Validation set is empty. Unable to compute node uncertainty.")

    mean_err = err_sum / count
    variance = np.maximum((err_sq_sum / count) - np.square(mean_err), 0.0)
    std_per_node = np.sqrt(variance).astype(np.float32)
    return std_per_node


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute node-wise uncertainty from validation errors.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="node_uncertainty.npy")
    args = parser.parse_args()

    std_error_per_node = compute_node_uncertainty(batch_size=args.batch_size)
    output_path = Path(args.output)
    np.save(output_path, std_error_per_node)
    print(f"Saved node uncertainty to: {output_path.resolve()}")
    print(f"Shape: {std_error_per_node.shape}, min: {std_error_per_node.min():.6f}, max: {std_error_per_node.max():.6f}")


if __name__ == "__main__":
    main()
