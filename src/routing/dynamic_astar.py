from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np


def load_adjacency_matrix(adj_path: str | Path = "data/raw/adj_METR-LA.pkl") -> np.ndarray:
    """Load adjacency matrix from METR-LA pickle."""
    path = Path(adj_path)
    with path.open("rb") as handle:
        try:
            content = pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            content = pickle.load(handle, encoding="latin1")

    if isinstance(content, (list, tuple)) and len(content) == 3:
        adjacency = np.asarray(content[2], dtype=np.float32)
    else:
        adjacency = np.asarray(content, dtype=np.float32)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        msg = "Adjacency matrix must be square with shape [num_nodes, num_nodes]."
        raise ValueError(msg)

    return adjacency


def build_sensor_graph(adjacency: np.ndarray) -> nx.DiGraph:
    """Create directed graph where edge weight is initial distance."""
    num_nodes = adjacency.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    rows, cols = np.where(adjacency > 0)
    for src, dst in zip(rows.tolist(), cols.tolist()):
        if src == dst:
            continue
        distance = float(adjacency[src, dst])
        graph.add_edge(src, dst, distance=distance, weight=distance)

    return graph


def _astar_with_optional_undirected_fallback(
    graph: nx.DiGraph,
    start_node: int,
    end_node: int,
    weight_fn,
    *,
    allow_undirected_fallback: bool,
) -> tuple[list[int], nx.Graph]:
    def heuristic(_: int, __: int) -> float:
        # No coordinate metadata is available, so we use an admissible zero heuristic.
        return 0.0

    try:
        return nx.astar_path(graph, start_node, end_node, heuristic=heuristic, weight=weight_fn), graph
    except nx.NodeNotFound as exc:
        raise ValueError(str(exc)) from exc
    except nx.NetworkXNoPath:
        if not allow_undirected_fallback:
            raise

    undirected_graph = graph.to_undirected(as_view=False)
    return (
        nx.astar_path(undirected_graph, start_node, end_node, heuristic=heuristic, weight=weight_fn),
        undirected_graph,
    )


def _to_node_speed_vector(
    predicted_speeds: np.ndarray | Iterable[float],
    num_nodes: int,
    min_speed: float = 1e-3,
) -> np.ndarray:
    speeds = np.asarray(predicted_speeds, dtype=np.float32)
    if speeds.size == 0:
        raise ValueError("predicted_speeds is empty.")

    if speeds.ndim == 1:
        node_speeds = speeds
    else:
        if speeds.shape[-1] != num_nodes:
            msg = f"Expected predicted_speeds last dimension = {num_nodes}, got {speeds.shape[-1]}"
            raise ValueError(msg)
        # Reduce batch/time dimensions to one speed estimate per node.
        node_speeds = speeds.reshape(-1, num_nodes).mean(axis=0)

    if node_speeds.shape[0] != num_nodes:
        msg = f"Expected {num_nodes} node speeds, got {node_speeds.shape[0]}"
        raise ValueError(msg)

    return np.clip(node_speeds.astype(np.float32), min_speed, None)


def get_optimized_path(
    start_node: int,
    end_node: int,
    predicted_speeds: np.ndarray | Iterable[float],
    adj_path: str | Path = "data/raw/adj_METR-LA.pkl",
    allow_undirected_fallback: bool = True,
) -> tuple[list[int], float]:
    """
    Run dynamic A* where edge travel time is updated by predicted speeds.

    travel_time = distance / predicted_speed
    """
    adjacency = load_adjacency_matrix(adj_path)
    graph = build_sensor_graph(adjacency)
    node_speeds = _to_node_speed_vector(predicted_speeds, num_nodes=adjacency.shape[0])

    def dynamic_weight(u: int, v: int, edge_data: dict[str, float]) -> float:
        distance = float(edge_data["distance"])
        speed = float((node_speeds[u] + node_speeds[v]) * 0.5)
        return distance / speed

    try:
        path, routing_graph = _astar_with_optional_undirected_fallback(
            graph,
            start_node,
            end_node,
            dynamic_weight,
            allow_undirected_fallback=allow_undirected_fallback,
        )
    except nx.NodeNotFound as exc:
        raise ValueError(str(exc)) from exc
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"No path found between nodes {start_node} and {end_node}.") from exc

    total_travel_time = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total_travel_time += dynamic_weight(u, v, routing_graph[u][v])

    return path, total_travel_time


def get_static_path(
    start_node: int,
    end_node: int,
    adj_path: str | Path = "data/raw/adj_METR-LA.pkl",
    constant_speed: float = 40.0,
    allow_undirected_fallback: bool = True,
) -> tuple[list[int], float]:
    """
    Run static A* with constant speed routing.

    travel_time = distance / constant_speed
    """
    if constant_speed <= 0:
        raise ValueError("constant_speed must be > 0.")

    adjacency = load_adjacency_matrix(adj_path)
    graph = build_sensor_graph(adjacency)

    def static_weight(_: int, __: int, edge_data: dict[str, float]) -> float:
        distance = float(edge_data["distance"])
        return distance / constant_speed

    try:
        path, routing_graph = _astar_with_optional_undirected_fallback(
            graph,
            start_node,
            end_node,
            static_weight,
            allow_undirected_fallback=allow_undirected_fallback,
        )
    except nx.NodeNotFound as exc:
        raise ValueError(str(exc)) from exc
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"No path found between nodes {start_node} and {end_node}.") from exc

    total_travel_time = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total_travel_time += static_weight(u, v, routing_graph[u][v])

    return path, total_travel_time


def compare_static_vs_predictive(
    start_node: int,
    end_node: int,
    predicted_speeds: np.ndarray | Iterable[float],
    adj_path: str | Path = "data/raw/adj_METR-LA.pkl",
    constant_speed: float = 40.0,
) -> tuple[tuple[list[int], float], tuple[list[int], float], float]:
    """
    Compare static and predictive routing and print timing improvement.

    Prints:
    - static_time
    - predictive_time
    - percentage_improvement
    """
    static_path, static_time = get_static_path(
        start_node=start_node,
        end_node=end_node,
        adj_path=adj_path,
        constant_speed=constant_speed,
    )
    predictive_path, predictive_time = get_optimized_path(
        start_node=start_node,
        end_node=end_node,
        predicted_speeds=predicted_speeds,
        adj_path=adj_path,
    )

    if static_time > 0:
        percentage_improvement = ((static_time - predictive_time) / static_time) * 100.0
    else:
        percentage_improvement = 0.0

    print(f"static_time: {static_time:.6f}")
    print(f"predictive_time: {predictive_time:.6f}")
    print(f"percentage_improvement: {percentage_improvement:.2f}%")

    return (static_path, static_time), (predictive_path, predictive_time), percentage_improvement
