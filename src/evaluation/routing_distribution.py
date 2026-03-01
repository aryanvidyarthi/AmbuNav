from __future__ import annotations

from pathlib import Path

import numpy as np

from src.routing.dynamic_astar import get_optimized_path, get_static_path, load_adjacency_matrix


def _summarize_distribution(improvements: list[float]) -> dict[str, float]:
    if not improvements:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "positive_rate": 0.0,
        }

    values = np.asarray(improvements, dtype=np.float64)
    positive_rate = float(np.mean(values > 0.0) * 100.0)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "positive_rate": positive_rate,
    }


def evaluate_random_od_pairs(
    predicted_speeds: np.ndarray,
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
      "distribution": list[float],  # len == n_pairs
    }
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0.")

    adjacency = load_adjacency_matrix(adj_path)
    num_nodes = int(adjacency.shape[0])
    rng = np.random.default_rng(seed)

    improvements: list[float] = []
    attempts = 0

    while len(improvements) < n_pairs and attempts < max_attempts:
        attempts += 1
        start_node = int(rng.integers(0, num_nodes))
        end_node = int(rng.integers(0, num_nodes))

        if start_node == end_node:
            continue

        try:
            _, static_time = get_static_path(start_node=start_node, end_node=end_node, adj_path=adj_path)
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

        improvement = ((float(static_time) - float(predictive_time)) / float(static_time)) * 100.0
        improvements.append(float(improvement))

    if len(improvements) < n_pairs:
        raise RuntimeError(
            f"Unable to collect {n_pairs} valid OD pairs. "
            f"Collected {len(improvements)} after {attempts} attempts."
        )

    summary = _summarize_distribution(improvements)
    return {
        "mean": float(summary["mean"]),
        "median": float(summary["median"]),
        "std": float(summary["std"]),
        "positive_rate": float(summary["positive_rate"]),
        "distribution": improvements,
    }
