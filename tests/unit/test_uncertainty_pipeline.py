from __future__ import annotations

import numpy as np
import pytest

from src.routing import dynamic_astar


def test_normalize_node_uncertainty_in_unit_interval() -> None:
    values = np.array([4.0, 8.0, 12.0], dtype=np.float32)
    norm = dynamic_astar.normalize_node_uncertainty(values)
    assert norm.shape == (3,)
    assert float(norm.min()) == pytest.approx(0.0)
    assert float(norm.max()) == pytest.approx(1.0)


def test_compute_route_reliability_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(dynamic_astar, "load_adjacency_matrix", lambda _: adjacency)

    path = [0, 1, 2]
    uncertainty = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    avg_unc, reliability, risk = dynamic_astar.compute_route_reliability(path, uncertainty, adj_path="unused.pkl")

    assert 0.0 <= avg_unc <= 1.0
    assert 0.0 <= reliability <= 100.0
    assert risk in {"Low", "Medium", "High"}


def test_risk_adjusted_cost_non_decreasing_with_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    adjacency = np.array(
        [
            [0.0, 1.0, 2.5, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(dynamic_astar, "load_adjacency_matrix", lambda _: adjacency)

    predicted_speeds = np.array([20.0, 15.0, 12.0, 10.0], dtype=np.float32)
    node_uncertainty = np.array([0.05, 0.95, 0.10, 0.20], dtype=np.float32)

    costs: list[float] = []
    for lam in (0.0, 0.2, 0.5, 0.8, 1.0):
        _, cost = dynamic_astar.get_optimized_path(
            start_node=0,
            end_node=3,
            predicted_speeds=predicted_speeds,
            node_uncertainty=node_uncertainty,
            lambda_value=float(lam),
            adj_path="unused.pkl",
        )
        costs.append(float(cost))

    assert all(costs[i] <= costs[i + 1] + 1e-9 for i in range(len(costs) - 1))
