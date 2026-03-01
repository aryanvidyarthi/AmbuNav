from __future__ import annotations

from dataclasses import dataclass
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
    from src.inference import run_inference
    from src.models.graph_wavenet import GraphWaveNet
    from src.routing.dynamic_astar import (
        get_optimized_path,
        get_static_path,
        load_adjacency_matrix,
    )
except ModuleNotFoundError:
    # Allows running as: python src/evaluation/research_analysis.py
    from data.traffic_dataset import TrafficDataset
    from inference import run_inference
    from models.graph_wavenet import GraphWaveNet
    from routing.dynamic_astar import get_optimized_path, get_static_path, load_adjacency_matrix


@dataclass(frozen=True)
class TestDataBundle:
    loader: DataLoader
    scaler: Any
    num_nodes: int


def _load_model(device: torch.device, num_nodes: int, out_dim: int = 3) -> GraphWaveNet:
    checkpoint_path = Path(__file__).resolve().parents[2] / "checkpoints" / "model.pth"
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


def _build_test_data(batch_size: int = 64) -> TestDataBundle:
    dataset = TrafficDataset(
        data_path="data/raw/METR-LA.h5",
        input_length=12,
        output_length=3,
    )

    total_samples = len(dataset)
    train_end = int(total_samples * 0.70)
    val_end = train_end + int(total_samples * 0.15)

    x_test = dataset.x[val_end:]
    y_test = dataset.y[val_end:]

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return TestDataBundle(loader=test_loader, scaler=dataset.scaler, num_nodes=int(x_test.shape[-1]))


def _inverse_scale(batch_values: np.ndarray, scaler: Any) -> np.ndarray:
    # scaler expects [samples, num_nodes], so flatten horizon into sample dimension.
    original_shape = batch_values.shape
    restored = scaler.inverse_transform(batch_values.reshape(-1, original_shape[-1]))
    return restored.reshape(original_shape)


def evaluate_forecast_horizons(batch_size: int = 64) -> dict[str, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = _build_test_data(batch_size=batch_size)
    model = _load_model(device=device, num_nodes=test_data.num_nodes, out_dim=3)

    mae_sum = np.zeros(3, dtype=np.float64)
    rmse_sum = np.zeros(3, dtype=np.float64)
    mape_sum = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)

    with torch.no_grad():
        for x_batch, y_batch in test_data.loader:
            preds = model(x_batch.to(device)).cpu().numpy()  # [B, 3, N]
            target = y_batch.numpy()  # [B, 3, N]

            preds_inv = _inverse_scale(preds, test_data.scaler)
            target_inv = _inverse_scale(target, test_data.scaler)

            abs_err = np.abs(preds_inv - target_inv)
            sq_err = np.square(preds_inv - target_inv)
            denom = np.maximum(np.abs(target_inv), 1e-5)
            ape = abs_err / denom * 100.0

            mae_sum += abs_err.sum(axis=(0, 2))
            rmse_sum += sq_err.sum(axis=(0, 2))
            mape_sum += ape.sum(axis=(0, 2))
            count += np.array([target_inv.shape[0] * target_inv.shape[2]] * 3, dtype=np.float64)

    mae = mae_sum / np.maximum(count, 1.0)
    rmse = np.sqrt(rmse_sum / np.maximum(count, 1.0))
    mape = mape_sum / np.maximum(count, 1.0)

    return {
        "5min": {"MAE": float(mae[0]), "RMSE": float(rmse[0]), "MAPE": float(mape[0])},
        "10min": {"MAE": float(mae[1]), "RMSE": float(rmse[1]), "MAPE": float(mape[1])},
        "15min": {"MAE": float(mae[2]), "RMSE": float(rmse[2]), "MAPE": float(mape[2])},
    }


def _sample_node_pairs(num_nodes: int, n_pairs: int = 100, seed: int = 42) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []

    while len(pairs) < n_pairs:
        start = int(rng.integers(0, num_nodes))
        end = int(rng.integers(0, num_nodes))
        if start != end:
            pairs.append((start, end))

    return pairs


def _summarize_improvements(improvements: list[float]) -> dict[str, float]:
    if not improvements:
        return {
            "mean_improvement": 0.0,
            "median_improvement": 0.0,
            "std_improvement": 0.0,
            "max_improvement": 0.0,
            "min_improvement": 0.0,
            "valid_pairs": 0.0,
        }

    values = np.asarray(improvements, dtype=np.float64)
    return {
        "mean_improvement": float(np.mean(values)),
        "median_improvement": float(np.median(values)),
        "std_improvement": float(np.std(values)),
        "max_improvement": float(np.max(values)),
        "min_improvement": float(np.min(values)),
        "valid_pairs": float(len(values)),
    }


def evaluate_routing_distribution(
    predicted_speeds: np.ndarray,
    n_pairs: int = 100,
    seed: int = 42,
    adj_path: str | Path = "data/raw/adj_METR-LA.pkl",
) -> dict[str, Any]:
    adjacency = load_adjacency_matrix(adj_path)
    pairs = _sample_node_pairs(num_nodes=int(adjacency.shape[0]), n_pairs=n_pairs, seed=seed)

    improvements: list[float] = []
    for start_node, end_node in pairs:
        try:
            _, static_time = get_static_path(start_node, end_node, adj_path=adj_path)
            _, predictive_time = get_optimized_path(
                start_node=start_node,
                end_node=end_node,
                predicted_speeds=predicted_speeds,
                adj_path=adj_path,
            )
        except ValueError:
            continue

        if static_time <= 0:
            continue
        improvement = ((static_time - predictive_time) / static_time) * 100.0
        improvements.append(float(improvement))

    return {
        "improvements": improvements,
        "statistics": _summarize_improvements(improvements),
    }


def evaluate_congestion_stress_test(
    base_predicted_speeds: np.ndarray,
    n_pairs: int = 100,
    seed: int = 42,
    adj_path: str | Path = "data/raw/adj_METR-LA.pkl",
) -> dict[str, Any]:
    scenarios = {
        "baseline": 1.00,
        "congestion_20pct": 0.80,
        "congestion_40pct": 0.60,
    }

    scenario_results: dict[str, Any] = {}
    for scenario_name, speed_scale in scenarios.items():
        adjusted_speeds = np.clip(base_predicted_speeds * speed_scale, 1e-3, None)
        scenario_results[scenario_name] = evaluate_routing_distribution(
            predicted_speeds=adjusted_speeds,
            n_pairs=n_pairs,
            seed=seed,
            adj_path=adj_path,
        )["statistics"]

    return scenario_results


def run_research_analysis(batch_size: int = 64, n_pairs: int = 100, seed: int = 42) -> dict[str, Any]:
    forecast_metrics = evaluate_forecast_horizons(batch_size=batch_size)
    predicted_speeds = np.asarray(run_inference(), dtype=np.float32)

    routing_distribution = evaluate_routing_distribution(
        predicted_speeds=predicted_speeds,
        n_pairs=n_pairs,
        seed=seed,
    )

    congestion_stress = evaluate_congestion_stress_test(
        base_predicted_speeds=predicted_speeds,
        n_pairs=n_pairs,
        seed=seed,
    )

    return {
        "forecast_horizon_metrics": forecast_metrics,
        "routing_improvement_distribution": routing_distribution["statistics"],
        "congestion_stress_test": congestion_stress,
    }


def main() -> dict[str, Any]:
    results = run_research_analysis()

    print("\n=== Forecast Horizon Evaluation ===")
    for horizon, metrics in results["forecast_horizon_metrics"].items():
        print(
            f"{horizon:>5} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"MAPE: {metrics['MAPE']:.2f}%"
        )

    routing_stats = results["routing_improvement_distribution"]
    print("\n=== Routing Improvement Distribution (100 pairs) ===")
    print(f"Mean improvement   : {routing_stats['mean_improvement']:.2f}%")
    print(f"Median improvement : {routing_stats['median_improvement']:.2f}%")
    print(f"Std deviation      : {routing_stats['std_improvement']:.2f}")
    print(f"Max improvement    : {routing_stats['max_improvement']:.2f}%")
    print(f"Min improvement    : {routing_stats['min_improvement']:.2f}%")
    print(f"Valid pairs used   : {int(routing_stats['valid_pairs'])}")

    print("\n=== Congestion Stress Test ===")
    for scenario, stats in results["congestion_stress_test"].items():
        print(
            f"{scenario:>16} | mean: {stats['mean_improvement']:.2f}% | "
            f"median: {stats['median_improvement']:.2f}% | "
            f"std: {stats['std_improvement']:.2f}"
        )

    return results


if __name__ == "__main__":
    main()
