from __future__ import annotations

import numpy as np
import pytest

from src.routing import dynamic_astar


def test_get_static_path_uses_undirected_fallback_for_asymmetric_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(dynamic_astar, "load_adjacency_matrix", lambda _: adjacency)

    path, travel_time = dynamic_astar.get_static_path(
        start_node=2,
        end_node=0,
        adj_path="unused.pkl",
        constant_speed=10.0,
        allow_undirected_fallback=True,
    )

    assert path == [2, 1, 0]
    assert travel_time == pytest.approx(0.2)


def test_get_static_path_raises_when_fallback_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(dynamic_astar, "load_adjacency_matrix", lambda _: adjacency)

    with pytest.raises(ValueError, match="No path found between nodes 2 and 0"):
        dynamic_astar.get_static_path(
            start_node=2,
            end_node=0,
            adj_path="unused.pkl",
            allow_undirected_fallback=False,
        )
