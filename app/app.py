from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PAGES_DIR = Path(__file__).resolve().parent / "pages"
if str(PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PAGES_DIR))

from Analysis import render_analysis_page
from src.inference import run_inference
from src.routing.dynamic_astar import (
    build_sensor_graph,
    compare_static_vs_predictive,
    load_adjacency_matrix,
)


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
    adjacency = load_adjacency_matrix(adj_path)
    directed_graph = build_sensor_graph(adjacency)
    # Route search can fallback to undirected edges; draw on undirected graph to avoid broken segments.
    graph = directed_graph.to_undirected(as_view=False)
    layout = nx.spring_layout(graph, seed=42, k=0.24)
    return graph, layout


def _plot_paths(
    graph: nx.Graph,
    layout: dict[int, tuple[float, float]],
    static_path: list[int],
    predictive_path: list[int],
    start_node: int,
    end_node: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#090f1a")
    ax.set_facecolor("#090f1a")

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
        edge_color="#10d876",
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


def _render_live_clock() -> None:
    components.html(
        """
        <div id="ambu-clock" style="text-align:center; color:#e9efff; font-weight:700;
             font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
             font-size:0.96rem; margin-top:0.55rem;">
          Device Time: --:--:--
        </div>
        <script>
          function updateClock() {
            const now = new Date();
            const time = now.toLocaleTimeString([], {hour12: false});
            const node = document.getElementById('ambu-clock');
            if (node) node.textContent = `Device Time: ${time}`;
          }
          updateClock();
          setInterval(updateClock, 1000);
        </script>
        """,
        height=36,
    )


def _format_minutes(value_minutes: float | None) -> str:
    if value_minutes is None:
        return "--"
    return f"{value_minutes:.2f} min"


def _format_hhmm(dt_value: datetime | None) -> str:
    if dt_value is None:
        return "--"
    return dt_value.strftime("%H:%M")


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
    adjacency = load_adjacency_matrix(adj_path)
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
        travel_after_min = st.slider(
            "Travel after (minutes)",
            min_value=0,
            max_value=120,
            value=0,
            step=5,
        )
        current_device_dt = datetime.now()
        scheduled_departure_dt = current_device_dt + timedelta(minutes=int(travel_after_min))
        st.caption(f"Scheduled Departure: {scheduled_departure_dt.strftime('%H:%M')}")
        optimize_clicked = st.button("Optimize Route", width="stretch")

    st.markdown(
        """
        <div class="header-wrap">
            <div class="header-row">
                <div class="pulse-dot"></div>
                <h1 class="main-title">&#128657; AmbuNav - AI Emergency Route Optimizer</h1>
            </div>
            <p class="subtitle">Real-time Predictive Traffic Intelligence for Ambulance Routing</p>
            <span class="clock-pill">Live Device Time</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_live_clock()

    if "route_result" not in st.session_state:
        st.session_state.route_result = {
            "static_minutes": None,
            "predictive_minutes": None,
            "improvement_pct": None,
            "static_path": None,
            "predictive_path": None,
            "predicted_speeds": None,
        }

    if optimize_clicked:
        try:
            predicted_speeds = run_inference()
            (static_path, static_time), (predictive_path, predictive_time), percentage_improvement = (
                compare_static_vs_predictive(
                    start_node=int(start_node),
                    end_node=int(end_node),
                    predicted_speeds=predicted_speeds,
                    adj_path=adj_path,
                )
            )

            # Existing backend time outputs are converted to minutes for UI display.
            st.session_state.route_result = {
                "static_minutes": max(float(static_time) * 60.0, 0.0),
                "predictive_minutes": max(float(predictive_time) * 60.0, 0.0),
                "improvement_pct": float(percentage_improvement),
                "static_path": static_path,
                "predictive_path": predictive_path,
                "predicted_speeds": predicted_speeds,
            }
        except Exception as exc:  # pragma: no cover - UI error path
            st.error(f"Failed to optimize route: {exc}")

    static_minutes = st.session_state.route_result["static_minutes"]
    predictive_minutes = st.session_state.route_result["predictive_minutes"]
    improvement_pct = st.session_state.route_result["improvement_pct"]
    static_path = st.session_state.route_result["static_path"]
    predictive_path = st.session_state.route_result["predictive_path"]
    predicted_speeds = st.session_state.route_result["predicted_speeds"]

    # Time flow:
    # device_time -> scheduled_departure (+travel_after) -> ETA (+predictive travel minutes)
    if predictive_minutes is not None:
        eta_dt = scheduled_departure_dt + timedelta(minutes=float(predictive_minutes))
    else:
        eta_dt = None

    with st.container():
        st.markdown('<div class="glass-shell">', unsafe_allow_html=True)
        chip_eta = _format_hhmm(eta_dt) if eta_dt is not None else "Pending"
        st.markdown(
            f'<div style="text-align:center;"><span class="status-chip">Estimated Arrival Time (ETA): {chip_eta}</span></div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        with cols[0]:
            _metric_card("Device Time", current_device_dt.strftime("%H:%M"))
        with cols[1]:
            _metric_card("Scheduled Departure", _format_hhmm(scheduled_departure_dt))
        with cols[2]:
            _metric_card("Predictive Travel Time", _format_minutes(predictive_minutes))
        with cols[3]:
            _metric_card("Estimated Arrival Time", _format_hhmm(eta_dt))

        if improvement_pct is not None:
            imp_color = "#16c784" if improvement_pct >= 0 else "#ff5a5a"
            st.markdown(
                f'<div style="text-align:center; margin-top:0.6rem; font-weight:700; color:{imp_color};">'
                f'📊 Route Improvement: {improvement_pct:.2f}%'
                "</div>",
                unsafe_allow_html=True,
            )

        if static_path is not None and predictive_path is not None and predicted_speeds is not None:
            node_speeds = _node_speed_vector(np.asarray(predicted_speeds))
            avg_speed = _average_route_speed(predictive_path, node_speeds)
            congestion_text, congestion_color = _congestion_level(avg_speed)
            time_saved_text = _human_time_saved(static_minutes, predictive_minutes)
            bottlenecks = _top_bottlenecks(predictive_path, node_speeds, top_k=2)
            stability_text, stability_color = _route_stability(static_path, predictive_path)
            confidence_horizon, confidence_score = _confidence_by_horizon(travel_after_min)

            st.markdown(
                '<div style="text-align:center; margin:0.8rem 0 0.5rem; font-size:1.08rem; font-weight:800;">'
                "🚑 Advanced Route Intelligence"
                "</div>",
                unsafe_allow_html=True,
            )

            a1, a2, a3, a4 = st.columns(4)
            with a1:
                badge = (
                    f'<span class="badge" style="background:{congestion_color}20; color:{congestion_color}; '
                    f'border:1px solid {congestion_color}66;">{congestion_text}</span>'
                )
                speed_text = "--" if avg_speed is None else f"{avg_speed:.2f}"
                _analytics_card("⚠ Congestion Level", f"{badge}<br/><span style='font-size:0.95rem;'>Avg Speed: {speed_text}</span>")
            with a2:
                _analytics_card("⏱ Time Saved", f"<span style='font-size:1.35rem;'>{time_saved_text}</span>")
            with a3:
                _analytics_card("📊 Route Stability", f"<span style='color:{stability_color};'>{stability_text}</span>")
            with a4:
                _analytics_card("🚑 AI Confidence", f"{confidence_score}%<br/><span style='font-size:0.88rem;'>{confidence_horizon}</span>")
                st.progress(confidence_score / 100.0)

            st.markdown("#### Critical Segments")
            if bottlenecks:
                for u, v, seg_speed in bottlenecks:
                    st.markdown(
                        f"- `Node {u} -> Node {v}` | Predicted speed: `{seg_speed:.2f}`",
                    )
            else:
                st.markdown("- No critical segments identified.")

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
            )
        else:
            st.info("Configure inputs from the sidebar and click 'Optimize Route' to generate emergency route intelligence.")

        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="AmbuNav - AI Emergency Route Optimizer", layout="wide")
    _inject_styles()

    with st.sidebar:
        page = st.radio("Navigation", ["Emergency Optimizer", "Forecasting Analysis"])

    if page == "Forecasting Analysis":
        render_analysis_page()
        return

    _route_optimizer_page()


if __name__ == "__main__":
    main()
