from __future__ import annotations

from pathlib import Path

import numpy as np

from src.routing.dynamic_astar import (
    compute_route_reliability,
    get_optimized_path_detailed,
    get_static_path,
    load_adjacency_matrix,
)


def _summarize_distribution(improvements: list[float], reliabilities: list[float]) -> dict[str, float]:
    if not improvements:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "positive_rate": 0.0,
            "avg_reliability": 0.0,
        }

    values = np.asarray(improvements, dtype=np.float64)
    reliability_values = np.asarray(reliabilities, dtype=np.float64) if reliabilities else np.asarray([], dtype=np.float64)
    positive_rate = float(np.mean(values > 0.0) * 100.0)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "positive_rate": positive_rate,
        "avg_reliability": float(np.mean(reliability_values)) if reliability_values.size else 0.0,
    }


def evaluate_random_od_pairs(
    predicted_speeds: np.ndarray,
    node_uncertainty: np.ndarray | None = None,
    lambda_value: float = 0.3,
    n_pairs: int = 100,
    seed: int = 42,
    adj_path: str | Path = "data/raw/adj_METR-LA.pkl",
    max_attempts: int = 20000,
) -> dict[str, float | list[float]]:
    """
    Evaluate routing improvement on random valid OD pairs.

    Returns:
    {
      "mean": float,
      "median": float,
      "std": float,
      "positive_rate": float,  # percentage in [0, 100]
      "avg_reliability": float,
      "distribution": list[float],  # risk-adjusted improvement distribution
      "physical_distribution": list[float],  # physical ETA improvement distribution
    }
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0.")

    adjacency = load_adjacency_matrix(adj_path)
    num_nodes = int(adjacency.shape[0])
    rng = np.random.default_rng(seed)

    improvements: list[float] = []
    physical_improvements: list[float] = []
    reliabilities: list[float] = []
    attempts = 0

    while len(improvements) < n_pairs and attempts < max_attempts:
        attempts += 1
        start_node = int(rng.integers(0, num_nodes))
        end_node = int(rng.integers(0, num_nodes))

        if start_node == end_node:
            continue

        try:
            _, static_time = get_static_path(start_node=start_node, end_node=end_node, adj_path=adj_path)
            predictive_path, predictive_physical_time, predictive_risk_cost = get_optimized_path_detailed(
                start_node=start_node,
                end_node=end_node,
                predicted_speeds=predicted_speeds,
                node_uncertainty=node_uncertainty,
                lambda_value=lambda_value,
                adj_path=adj_path,
            )
        except ValueError:
            continue

        if static_time <= 0:
            continue

        improvement = ((float(static_time) - float(predictive_risk_cost)) / float(static_time)) * 100.0
        physical_improvement = ((float(static_time) - float(predictive_physical_time)) / float(static_time)) * 100.0
        improvements.append(float(improvement))
        physical_improvements.append(float(physical_improvement))
        if node_uncertainty is not None:
            _, reliability_score, _ = compute_route_reliability(
                predictive_path,
                node_uncertainty,
                adj_path=adj_path,
            )
            reliabilities.append(float(reliability_score))

    if len(improvements) < n_pairs:
        raise RuntimeError(
            f"Unable to collect {n_pairs} valid OD pairs. "
            f"Collected {len(improvements)} after {attempts} attempts."
        )

    summary = _summarize_distribution(improvements, reliabilities)
    physical_summary = _summarize_distribution(physical_improvements, [])
    return {
        "mean": float(summary["mean"]),
        "median": float(summary["median"]),
        "std": float(summary["std"]),
        "positive_rate": float(summary["positive_rate"]),
        "avg_reliability": float(summary["avg_reliability"]),
        "physical_mean": float(physical_summary["mean"]),
        "physical_median": float(physical_summary["median"]),
        "physical_std": float(physical_summary["std"]),
        "physical_positive_rate": float(physical_summary["positive_rate"]),
        "distribution": improvements,
        "physical_distribution": physical_improvements,
    }
