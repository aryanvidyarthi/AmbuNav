from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.traffic_dataset import get_dataloaders
from src.data.traffic_dataset import TrafficDataset
from src.evaluation.routing_distribution import evaluate_random_od_pairs
from src.inference import run_inference
from src.models.graph_wavenet import GraphWaveNet
from src.routing.dynamic_astar import load_node_uncertainty


@st.cache_resource(show_spinner=False)
def _load_model(device_type: str, num_nodes: int) -> GraphWaveNet:
    device = torch.device(device_type)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = GraphWaveNet(
        num_nodes=num_nodes,
        in_dim=1,
        out_dim=3,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=64,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def _get_test_artifacts(batch_size: int = 64) -> tuple[object, object, int]:
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    scaler = TrafficDataset().scaler
    sample_x, _ = next(iter(test_loader))
    num_nodes = int(sample_x.shape[-1])
    return test_loader, scaler, num_nodes


def _inverse_scale(batch_values: np.ndarray, scaler: object) -> np.ndarray:
    original_shape = batch_values.shape
    restored = scaler.inverse_transform(batch_values.reshape(-1, original_shape[-1]))
    return restored.reshape(original_shape)


@st.cache_data(show_spinner=False)
def compute_forecast_metrics(batch_size: int = 64) -> dict[str, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, scaler, num_nodes = _get_test_artifacts(batch_size=batch_size)
    model = _load_model(device_type=device.type, num_nodes=num_nodes)

    mae_sum = np.zeros(3, dtype=np.float64)
    rmse_sum = np.zeros(3, dtype=np.float64)
    mape_sum = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)
    mape_count = np.zeros(3, dtype=np.float64)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.numpy()

            preds = model(x_batch).cpu().numpy()  # [B, 3, N]
            preds_inv = _inverse_scale(preds, scaler)
            target_inv = _inverse_scale(y_batch, scaler)

            abs_err = np.abs(preds_inv - target_inv)
            sq_err = np.square(preds_inv - target_inv)

            mae_sum += abs_err.sum(axis=(0, 2))
            rmse_sum += sq_err.sum(axis=(0, 2))
            count += np.array([target_inv.shape[0] * target_inv.shape[2]] * 3, dtype=np.float64)
            for h in range(3):
                target_h = target_inv[:, h, :]
                err_h = abs_err[:, h, :]
                mask = np.abs(target_h) > 1.0
                if np.any(mask):
                    mape_sum[h] += float((err_h[mask] / np.abs(target_h[mask]) * 100.0).sum())
                    mape_count[h] += float(mask.sum())

    mae = mae_sum / np.maximum(count, 1.0)
    rmse = np.sqrt(rmse_sum / np.maximum(count, 1.0))
    mape = mape_sum / np.maximum(mape_count, 1.0)

    return {
        "5min": {"mae": float(mae[0]), "rmse": float(rmse[0]), "mape": float(mape[0])},
        "10min": {"mae": float(mae[1]), "rmse": float(rmse[1]), "mape": float(mape[1])},
        "15min": {"mae": float(mae[2]), "rmse": float(rmse[2]), "mape": float(mape[2])},
    }


def _reliability_text(mae: float) -> str:
    if mae < 3.0:
        return "High reliability"
    if mae <= 6.0:
        return "Moderate reliability"
    return "Low reliability"


@st.cache_data(show_spinner=False)
def compute_routing_distribution_metrics(
    n_pairs: int = 100,
    seed: int = 42,
    lambda_value: float = 0.3,
) -> dict[str, float | list[float]]:
    predicted_speeds = np.asarray(run_inference(), dtype=np.float32)
    node_uncertainty = load_node_uncertainty(PROJECT_ROOT / "node_uncertainty.npy")
    return evaluate_random_od_pairs(
        predicted_speeds=predicted_speeds,
        node_uncertainty=node_uncertainty,
        lambda_value=lambda_value,
        n_pairs=n_pairs,
        seed=seed,
    )


def _inject_analysis_styles() -> None:
    st.markdown(
        """
        <style>
        .fa-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            color: #f5f7ff;
        }
        .fa-subtitle {
            color: #b7c2da;
            margin-bottom: 1rem;
        }
        .ci-box {
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.08);
            padding: 0.8rem 0.9rem;
            min-height: 160px;
        }
        .ci-title {
            font-weight: 800;
            color: #e9efff;
            margin-bottom: 0.35rem;
        }
        .ci-line {
            color: #f5f7ff;
            font-weight: 700;
            line-height: 1.35;
            font-size: 0.95rem;
        }
        .summary-card {
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.08);
            padding: 0.9rem 0.95rem;
            min-height: 120px;
        }
        .summary-label {
            color: #c8d4ef;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .summary-value {
            color: #f5f7ff;
            font-size: 1.6rem;
            font-weight: 800;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_page() -> None:
    _inject_analysis_styles()
    st.markdown('<div class="fa-title">\U0001F4CA Forecasting Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fa-subtitle">\u23F1 Horizon-wise evaluation of GraphWaveNet on the test split using the existing checkpoint.</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        try:
            metric_dict = compute_forecast_metrics(batch_size=64)
        except Exception as exc:
            st.error(f"Failed to run forecasting analysis: {exc}")
            return

        rows = []
        for horizon in ("5min", "10min", "15min"):
            rows.append(
                {
                    "Horizon": horizon.replace("min", " min"),
                    "MAE": metric_dict[horizon]["mae"],
                    "RMSE": metric_dict[horizon]["rmse"],
                    "MAPE": metric_dict[horizon]["mape"],
                    "Reliability": _reliability_text(metric_dict[horizon]["mae"]),
                }
            )
        results_df = pd.DataFrame(rows)

        st.markdown("### \U0001F691 Confidence Interpretation", unsafe_allow_html=True)
        cards = st.columns(3)
        for col, (_, row) in zip(cards, results_df.iterrows()):
            with col:
                st.markdown(
                    (
                        f'<div class="ci-box">'
                        f'<div class="ci-title">{row["Horizon"]}</div>'
                        f'<div class="ci-line">MAE: {row["MAE"]:.4f}</div>'
                        f'<div class="ci-line">RMSE: {row["RMSE"]:.4f}</div>'
                        f'<div class="ci-line">MAPE: {row["MAPE"]:.2f}%</div>'
                        f'<div class="ci-line">{row["Reliability"]}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        c1, c2 = st.columns([1.15, 1.1])
        with c1:
            st.markdown("### \U0001F4CA Metrics Table")
            table_df = results_df[["Horizon", "MAE", "RMSE", "MAPE"]].copy()
            table_df["MAE"] = table_df["MAE"].map(lambda x: f"{x:.6f}")
            table_df["RMSE"] = table_df["RMSE"].map(lambda x: f"{x:.6f}")
            table_df["MAPE"] = table_df["MAPE"].map(lambda x: f"{x:.2f}%")
            st.table(table_df)

        with c2:
            st.markdown("### \u23F1 MAE by Horizon")
            chart_df = results_df.copy()
            chart_df["HorizonMin"] = [5, 10, 15]
            mae_chart = (
                alt.Chart(chart_df)
                .mark_line(point=True, strokeWidth=3, color="#36a3ff")
                .encode(
                    x=alt.X("HorizonMin:Q", title="Horizon (minutes)", scale=alt.Scale(domain=[5, 15])),
                    y=alt.Y("MAE:Q", title="MAE"),
                    tooltip=[
                        alt.Tooltip("Horizon:N"),
                        alt.Tooltip("MAE:Q", format=".4f"),
                        alt.Tooltip("RMSE:Q", format=".4f"),
                        alt.Tooltip("MAPE:Q", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(mae_chart, width="stretch")

        st.info(
            "Prediction error increases with forecasting horizon, validating short-term forecasting reliability."
        )

        st.markdown("---")
        lambda_used = 0.3
        st.markdown("## \U0001F4C8 Statistical Routing Evaluation (100 OD Pairs)")
        st.caption(f"Uncertainty-aware routing enabled (lambda = {lambda_used:.2f})")

        try:
            routing_eval = compute_routing_distribution_metrics(n_pairs=100, seed=42, lambda_value=lambda_used)
        except Exception as exc:
            st.error(f"Failed to run routing distribution evaluation: {exc}")
            return

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        with r1:
            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">\U0001F4CA Avg Improvement (Risk Cost)</div>'
                    f'<div class="summary-value">{float(routing_eval["mean"]):.2f}%</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">\U0001F4C8 Median Improvement</div>'
                    f'<div class="summary-value">{float(routing_eval["median"]):.2f}%</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">\U0001F691 % Routes Improved</div>'
                    f'<div class="summary-value">{float(routing_eval["positive_rate"]):.1f}%</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">\U0001F4CA Std Deviation</div>'
                    f'<div class="summary-value">{float(routing_eval["std"]):.2f}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with r5:
            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">\U0001F6E1 Avg Reliability</div>'
                    f'<div class="summary-value">{float(routing_eval["avg_reliability"]):.1f}%</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with r6:
            st.markdown(
                (
                    '<div class="summary-card">'
                    '<div class="summary-label">\u23F3 Avg Improvement (Physical ETA)</div>'
                    f'<div class="summary-value">{float(routing_eval["physical_mean"]):.2f}%</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        distribution = [float(x) for x in routing_eval["distribution"]]
        dist_df = pd.DataFrame(
            {
                "Improvement": distribution,
                "RouteIndex": list(range(1, len(distribution) + 1)),
            }
        )

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("### \U0001F4CA Improvement Distribution")
            hist = (
                alt.Chart(dist_df)
                .mark_bar(color="#43c6ff")
                .encode(
                    x=alt.X("Improvement:Q", bin=alt.Bin(maxbins=20), title="% Improvement"),
                    y=alt.Y("count()", title="Frequency"),
                    tooltip=[alt.Tooltip("count()", title="Frequency")],
                )
                .properties(height=300)
            )
            st.altair_chart(hist, width="stretch")

        with p2:
            st.markdown("### \U0001F4C8 Route Index vs Improvement")
            line = (
                alt.Chart(dist_df)
                .mark_line(point=True, color="#36a3ff")
                .encode(
                    x=alt.X("RouteIndex:Q", title="Route Index"),
                    y=alt.Y("Improvement:Q", title="% Improvement"),
                    tooltip=[
                        alt.Tooltip("RouteIndex:Q"),
                        alt.Tooltip("Improvement:Q", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(line, width="stretch")

        if float(routing_eval["positive_rate"]) > 60.0:
            msg = "Uncertainty-aware predictive routing consistently outperforms static routing."
        else:
            msg = "Uncertainty-aware predictive advantage is limited under current dataset conditions."
        st.info(msg)


def main() -> None:
    st.set_page_config(page_title="Forecasting Analysis", layout="wide")
    render_analysis_page()


if __name__ == "__main__":
    main()


