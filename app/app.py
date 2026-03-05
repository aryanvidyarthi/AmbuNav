from __future__ import annotations

import sys
from datetime import datetime, timedelta
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PAGES_DIR = Path(__file__).resolve().parent / "pages"
if str(PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PAGES_DIR))

from Analysis import render_analysis_page
from src.deploy.artifacts import ensure_artifacts
from src.inference import run_inference
from src.routing import dynamic_astar as routing_dynamic_astar


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #07090f;
            --bg-secondary: #0d111b;
            --text-main: #f4f7ff;
            --text-muted: #aeb8ca;
            --accent-red: #FF3B3B;
            --accent-red-dark: #c82323;
            --card-bg: rgba(255, 255, 255, 0.08);
            --card-border: rgba(255, 255, 255, 0.16);
            --card-glow: rgba(255, 59, 59, 0.18);
            --green-good: #16c784;
            --red-bad: #ff5a5a;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(255, 59, 59, 0.14), transparent 32%),
                radial-gradient(circle at 85% 20%, rgba(96, 122, 255, 0.08), transparent 35%),
                linear-gradient(140deg, var(--bg-primary) 0%, var(--bg-secondary) 60%, #090c14 100%);
            color: var(--text-main);
        }

        .stApp::before {
            content: "\\1F691";
            position: fixed;
            right: 4vw;
            bottom: 4vh;
            font-size: clamp(140px, 16vw, 220px);
            color: rgba(255, 255, 255, 0.035);
            z-index: 0;
            pointer-events: none;
            user-select: none;
        }

        .main .block-container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .header-wrap {
            text-align: center;
            margin: 0.25rem auto 1.1rem;
        }

        .header-row {
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            justify-content: center;
            flex-wrap: wrap;
        }

        .pulse-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--accent-red);
            box-shadow: 0 0 0 rgba(255, 59, 59, 0.7);
            animation: pulse 1.8s infinite;
            flex: 0 0 auto;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 59, 59, 0.7); }
            70% { box-shadow: 0 0 0 14px rgba(255, 59, 59, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 59, 59, 0); }
        }

        .main-title {
            margin: 0;
            font-size: clamp(1.65rem, 2.5vw, 2.9rem);
            font-weight: 800;
            letter-spacing: 0.4px;
            color: var(--text-main);
        }

        .subtitle {
            margin: 0.35rem 0 0;
            font-size: 1.02rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .clock-pill {
            display: inline-block;
            margin-top: 0.7rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(255, 255, 255, 0.06);
            color: #e9efff;
            font-weight: 700;
            font-size: 0.93rem;
            backdrop-filter: blur(6px);
        }

        .glass-shell {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.08);
            padding: 1rem 1rem 1.15rem;
        }

        .metric-card {
            text-align: center;
            min-height: 150px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.09);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(255, 59, 59, 0.45);
            box-shadow: 0 16px 34px rgba(255, 59, 59, 0.22);
        }

        .metric-label {
            color: #c6d2ea;
            font-size: 0.96rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }

        .metric-value {
            color: var(--text-main);
            font-size: clamp(1.5rem, 2.0vw, 2.3rem);
            font-weight: 800;
            line-height: 1.2;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b101a 0%, #0a0f18 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        section[data-testid="stSidebar"] * {
            color: #eef3ff;
        }

        .stButton > button {
            background: linear-gradient(90deg, var(--accent-red) 0%, var(--accent-red-dark) 100%);
            border: 0;
            color: white;
            font-weight: 800;
            letter-spacing: 0.3px;
            border-radius: 12px;
            padding: 0.62rem 1rem;
            box-shadow: 0 0 18px rgba(255, 59, 59, 0.45);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 0 24px rgba(255, 59, 59, 0.65);
        }

        .viz-title {
            text-align: center;
            margin: 0.5rem 0 0.8rem;
            font-weight: 700;
            font-size: 1.15rem;
            color: #eaf1ff;
        }

        .status-chip {
            display: inline-block;
            margin: 0.3rem auto 0.75rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.06);
            color: #d7e2fb;
            font-size: 0.86rem;
            font-weight: 600;
        }

        .analytics-card {
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.07);
            padding: 0.9rem 1rem;
            min-height: 118px;
        }
        .analytics-title {
            color: #c7d7f7;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }
        .analytics-value {
            color: #f3f8ff;
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.3;
        }
        .badge {
            display: inline-block;
            padding: 0.26rem 0.62rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(title: str, value: str, value_color: str = "#f4f7ff") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value" style="color: {value_color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _get_graph_and_layout(adj_path: str) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    adjacency = routing_dynamic_astar.load_adjacency_matrix(adj_path)
    directed_graph = routing_dynamic_astar.build_sensor_graph(adjacency)
    # Route search can fallback to undirected edges; draw on undirected graph to avoid broken segments.
    graph = directed_graph.to_undirected(as_view=False)
    layout = nx.spring_layout(graph, seed=42, k=0.24)
    return graph, layout


@st.cache_resource
def _load_normalized_uncertainty(uncertainty_path: str) -> np.ndarray:
    if hasattr(routing_dynamic_astar, "load_node_uncertainty") and hasattr(
        routing_dynamic_astar, "normalize_node_uncertainty"
    ):
        node_uncertainty = routing_dynamic_astar.load_node_uncertainty(uncertainty_path)
        return routing_dynamic_astar.normalize_node_uncertainty(node_uncertainty)

    node_uncertainty = np.asarray(np.load(uncertainty_path), dtype=np.float32).reshape(-1)
    unc_min = float(np.min(node_uncertainty))
    unc_max = float(np.max(node_uncertainty))
    if np.isclose(unc_max, unc_min):
        return np.zeros_like(node_uncertainty, dtype=np.float32)
    normalized = (node_uncertainty - unc_min) / (unc_max - unc_min)
    return np.clip(normalized.astype(np.float32), 0.0, 1.0)


def _edge_risk_color(uncertainty_value: float) -> str:
    if uncertainty_value < 0.3:
        return "#22c55e"
    if uncertainty_value < 0.6:
        return "#facc15"
    return "#ef4444"


def _compute_route_reliability(path: list[int], node_uncertainty_norm: np.ndarray) -> tuple[float, float, str]:
    if hasattr(routing_dynamic_astar, "compute_route_reliability"):
        return routing_dynamic_astar.compute_route_reliability(path, node_uncertainty_norm)

    if not path:
        return 1.0, 0.0, "High"
    path_unc = node_uncertainty_norm[np.asarray(path, dtype=np.int64)]
    avg_unc = float(np.mean(path_unc))
    reliability = float(np.clip(100.0 - avg_unc * 100.0, 0.0, 100.0))
    if avg_unc < 0.3:
        return avg_unc, reliability, "Low"
    if avg_unc < 0.6:
        return avg_unc, reliability, "Medium"
    return avg_unc, reliability, "High"


def _compare_static_vs_predictive_compat(
    start_node: int,
    end_node: int,
    predicted_speeds: np.ndarray,
    adj_path: Path,
    node_uncertainty_norm: np.ndarray | None,
    lambda_value: float,
):
    compare_fn = routing_dynamic_astar.compare_static_vs_predictive
    params = inspect.signature(compare_fn).parameters
    kwargs: dict[str, object] = {
        "start_node": int(start_node),
        "end_node": int(end_node),
        "predicted_speeds": predicted_speeds,
        "adj_path": adj_path,
    }
    if "node_uncertainty" in params:
        kwargs["node_uncertainty"] = node_uncertainty_norm
    if "lambda_value" in params:
        kwargs["lambda_value"] = float(lambda_value)
    return compare_fn(**kwargs)


def _plot_paths(
    graph: nx.Graph,
    layout: dict[int, tuple[float, float]],
    static_path: list[int],
    predictive_path: list[int],
    start_node: int,
    end_node: int,
    node_uncertainty_norm: np.ndarray | None = None,
    show_risk_layer: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#090f1a")
    ax.set_facecolor("#090f1a")

    if show_risk_layer and node_uncertainty_norm is not None:
        edge_list = list(graph.edges())
        edge_colors = []
        for u, v in edge_list:
            unc = float((node_uncertainty_norm[u] + node_uncertainty_norm[v]) * 0.5)
            edge_colors.append(_edge_risk_color(unc))
        nx.draw_networkx_edges(
            graph,
            pos=layout,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=1.2,
            alpha=0.55,
            arrows=False,
            ax=ax,
        )
    else:
        nx.draw_networkx_edges(
            graph,
            pos=layout,
            edge_color="#8fa0bf",
            width=0.55,
            alpha=0.16,
            arrows=False,
            ax=ax,
        )
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        node_size=20,
        node_color="#90a7ce",
        alpha=0.50,
        ax=ax,
    )

    static_edges = list(zip(static_path[:-1], static_path[1:]))
    predictive_edges = list(zip(predictive_path[:-1], predictive_path[1:]))

    nx.draw_networkx_edges(
        graph,
        pos=layout,
        edgelist=static_edges,
        edge_color="#FF3B3B",
        width=3.2,
        alpha=0.98,
        arrows=False,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos=layout,
        edgelist=predictive_edges,
        edge_color="#00f5d4",
        width=3.5,
        alpha=0.98,
        arrows=False,
        ax=ax,
    )

    # Requested node colors: start blue, end orange.
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        nodelist=[start_node],
        node_color="#36a3ff",
        node_size=195,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        nodelist=[end_node],
        node_color="#ff9f1a",
        node_size=195,
        ax=ax,
    )

    ax.set_axis_off()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _format_minutes(value_minutes: float | None) -> str:
    if value_minutes is None:
        return "--"
    return f"{value_minutes:.2f} min"


def _format_hhmm(dt_value: datetime | None) -> str:
    if dt_value is None:
        return "--"
    return dt_value.strftime("%H:%M")


def _format_ampm(dt_value: datetime | None) -> str:
    if dt_value is None:
        return "--"
    return dt_value.strftime("%I:%M %p").lstrip("0")


def _format_eta_readable(minutes_value: float | None) -> str:
    if minutes_value is None:
        return "--"
    scaled_minutes = int(round(float(minutes_value) * 60.0))
    if scaled_minutes < 60:
        return f"{scaled_minutes} min"
    hours = scaled_minutes // 60
    mins = scaled_minutes % 60
    return f"{hours} hr {mins} min"


def _node_speed_vector(predicted_speeds: np.ndarray) -> np.ndarray:
    speeds = np.asarray(predicted_speeds, dtype=np.float32)
    return speeds.reshape(-1, speeds.shape[-1]).mean(axis=0)


def _average_route_speed(path: list[int], node_speeds: np.ndarray) -> float | None:
    if not path or len(path) < 2:
        return None
    edge_speeds = [float((node_speeds[u] + node_speeds[v]) * 0.5) for u, v in zip(path[:-1], path[1:])]
    return float(np.mean(edge_speeds)) if edge_speeds else None


def _congestion_level(avg_speed: float | None) -> tuple[str, str]:
    if avg_speed is None:
        return "Unknown", "#9ca7bd"
    if avg_speed > 60:
        return "Low Congestion", "#16c784"
    if avg_speed >= 30:
        return "Moderate Congestion", "#f5c451"
    return "High Congestion", "#ff5a5a"


def _human_time_saved(static_minutes: float | None, predictive_minutes: float | None) -> str:
    if static_minutes is None or predictive_minutes is None:
        return "--"
    delta_seconds = int(round((static_minutes - predictive_minutes) * 60))
    sign = "saved" if delta_seconds >= 0 else "longer"
    delta_seconds = abs(delta_seconds)
    mins = delta_seconds // 60
    secs = delta_seconds % 60
    return f"{mins} min {secs} sec {sign}"


def _top_bottlenecks(path: list[int], node_speeds: np.ndarray, top_k: int = 2) -> list[tuple[int, int, float]]:
    segments: list[tuple[int, int, float]] = []
    for u, v in zip(path[:-1], path[1:]):
        seg_speed = float((node_speeds[u] + node_speeds[v]) * 0.5)
        segments.append((u, v, seg_speed))
    segments.sort(key=lambda item: item[2])
    return segments[:top_k]


def _route_stability(static_path: list[int], predictive_path: list[int]) -> tuple[str, str]:
    static_edges = set(zip(static_path[:-1], static_path[1:]))
    predictive_edges = set(zip(predictive_path[:-1], predictive_path[1:]))
    if not static_edges and not predictive_edges:
        return "⚪ Unknown", "#9ca7bd"
    overlap = len(static_edges & predictive_edges)
    base = max(len(static_edges | predictive_edges), 1)
    overlap_ratio = overlap / base
    if overlap_ratio < 0.5:
        return "⚠ Dynamic route change", "#ffb454"
    return "✅ Stable route", "#16c784"


def _confidence_by_horizon(travel_after_min: int) -> tuple[str, int]:
    if travel_after_min <= 5:
        return "5 min horizon", 90
    if travel_after_min <= 10:
        return "10 min horizon", 80
    return "15 min horizon", 70


def _analytics_card(title: str, value_html: str) -> None:
    st.markdown(
        f"""
        <div class="analytics-card">
            <div class="analytics-title">{title}</div>
            <div class="analytics-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _route_optimizer_page() -> None:
    adj_path = PROJECT_ROOT / "data" / "raw" / "adj_METR-LA.pkl"
    uncertainty_path = PROJECT_ROOT / "node_uncertainty.npy"
    adjacency = routing_dynamic_astar.load_adjacency_matrix(adj_path)
    max_node_idx = int(adjacency.shape[0] - 1)

    with st.sidebar:
        st.markdown("### Route Inputs")
        start_node = st.number_input(
            "Start Node",
            min_value=0,
            max_value=max_node_idx,
            value=0,
            step=1,
        )
        end_node = st.number_input(
            "End Node",
            min_value=0,
            max_value=max_node_idx,
            value=min(10, max_node_idx),
            step=1,
        )
        lambda_value = st.slider(
            "Risk penalty (lambda)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        show_risk_layer = st.checkbox("Show Risk Layer", value=True)
        optimize_clicked = st.button("Optimize Route", width="stretch")

    st.markdown(
        """
        <div class="header-wrap">
            <div class="header-row">
                <div class="pulse-dot"></div>
                <h1 class="main-title">&#128657; AmbuNav - AI Emergency Route Optimizer</h1>
            </div>
            <p class="subtitle">Real-time Predictive Traffic Intelligence for Ambulance Routing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "route_result" not in st.session_state:
        st.session_state.route_result = {
            "predictive_minutes": None,
            "static_path": None,
            "predictive_path": None,
            "node_uncertainty_norm": None,
            "reliability_score": None,
            "predicted_speeds": None,
            "last_inputs": None,
        }

    current_inputs = (int(start_node), int(end_node), round(float(lambda_value), 3))
    has_previous_run = st.session_state.route_result["predicted_speeds"] is not None
    params_changed = st.session_state.route_result["last_inputs"] != current_inputs
    should_recompute = bool(optimize_clicked) or (has_previous_run and params_changed)

    if should_recompute:
        try:
            if optimize_clicked or st.session_state.route_result["predicted_speeds"] is None:
                predicted_speeds = run_inference()
                try:
                    node_uncertainty_norm = _load_normalized_uncertainty(str(uncertainty_path))
                except FileNotFoundError:
                    node_uncertainty_norm = None
                    st.warning(
                        "node_uncertainty.npy not found. "
                        "Run: python src/evaluation/compute_node_uncertainty.py"
                    )
            else:
                predicted_speeds = st.session_state.route_result["predicted_speeds"]
                node_uncertainty_norm = st.session_state.route_result["node_uncertainty_norm"]

            (static_path, _), (predictive_path, predictive_time), _ = _compare_static_vs_predictive_compat(
                start_node=int(start_node),
                end_node=int(end_node),
                predicted_speeds=predicted_speeds,
                node_uncertainty_norm=node_uncertainty_norm,
                lambda_value=float(lambda_value),
                adj_path=adj_path,
            )

            if node_uncertainty_norm is not None:
                _, reliability_score, _ = _compute_route_reliability(predictive_path, node_uncertainty_norm)
            else:
                reliability_score = None

            st.session_state.route_result = {
                "predictive_minutes": max(float(predictive_time) * 60.0, 0.0),
                "static_path": static_path,
                "predictive_path": predictive_path,
                "node_uncertainty_norm": node_uncertainty_norm,
                "reliability_score": reliability_score,
                "predicted_speeds": predicted_speeds,
                "last_inputs": current_inputs,
            }
        except Exception as exc:  # pragma: no cover
            st.error(f"Failed to optimize route: {exc}")

    predictive_minutes = st.session_state.route_result["predictive_minutes"]
    static_path = st.session_state.route_result["static_path"]
    predictive_path = st.session_state.route_result["predictive_path"]
    node_uncertainty_norm = st.session_state.route_result["node_uncertainty_norm"]
    reliability_score = st.session_state.route_result["reliability_score"]

    departure_dt = datetime.now()
    scaled_eta_minutes = (
        float(int(round(float(predictive_minutes) * 60.0)))
        if predictive_minutes is not None
        else None
    )
    arrival_dt = (
        departure_dt + timedelta(minutes=scaled_eta_minutes)
        if scaled_eta_minutes is not None
        else None
    )

    with st.container():
        st.markdown('<div class="glass-shell">', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("\U0001F552 Departure Time", _format_ampm(departure_dt))
        with c2:
            st.metric("\U0001F552 Arrival Time", _format_ampm(arrival_dt))
        with c3:
            st.metric("\u23F3 ETA", _format_eta_readable(predictive_minutes))
        with c4:
            st.metric(
                "\U0001F6E1 Reliability",
                "--" if reliability_score is None else f"{float(reliability_score):.1f}%",
            )

        st.markdown('<div class="viz-title">Route Visualization</div>', unsafe_allow_html=True)

        if static_path is not None and predictive_path is not None:
            graph, layout = _get_graph_and_layout(str(adj_path))
            _plot_paths(
                graph=graph,
                layout=layout,
                static_path=static_path,
                predictive_path=predictive_path,
                start_node=int(start_node),
                end_node=int(end_node),
                node_uncertainty_norm=node_uncertainty_norm,
                show_risk_layer=bool(show_risk_layer),
            )
        else:
            st.info("Configure start/end and click 'Optimize Route' to render the route.")

        st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _ensure_deploy_artifacts() -> dict[str, list[str]]:
    extra_urls: dict[str, str] = {}
    try:
        for key in (
            "AMBUNAV_MODEL_URL",
            "AMBUNAV_METR_H5_URL",
            "AMBUNAV_ADJ_URL",
            "AMBUNAV_UNCERTAINTY_URL",
        ):
            value = st.secrets.get(key, "")
            if value:
                extra_urls[key] = str(value)
    except Exception:
        # No secrets configured; fallback to environment variables.
        pass
    return ensure_artifacts(extra_urls=extra_urls)

def main() -> None:
    st.set_page_config(page_title="AmbuNav - AI Emergency Route Optimizer", layout="wide")
    _inject_styles()
    artifact_status = _ensure_deploy_artifacts()
    if artifact_status["missing_required"]:
        st.error(
            "Missing required artifacts. Set Streamlit secrets or environment variables for: "
            + ", ".join(artifact_status["missing_required"])
        )
        st.info(
            "Required keys: AMBUNAV_MODEL_URL, AMBUNAV_METR_H5_URL, AMBUNAV_ADJ_URL. "
            "Optional: AMBUNAV_UNCERTAINTY_URL."
        )
        return
    if artifact_status["downloaded"]:
        st.caption("Downloaded artifacts: " + ", ".join(artifact_status["downloaded"]))

    with st.sidebar:
        page = st.radio("Navigation", ["Emergency Optimizer", "Forecasting Analysis"])

    if page == "Forecasting Analysis":
        render_analysis_page()
        return

    _route_optimizer_page()


if __name__ == "__main__":
    main()


