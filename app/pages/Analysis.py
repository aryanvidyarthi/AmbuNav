from __future__ import annotations

import sys
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.traffic_dataset import TrafficDataset
from src.evaluation.routing_distribution import evaluate_random_od_pairs
from src.inference import run_inference
from src.models.graph_wavenet import GraphWaveNet
from src.routing.dynamic_astar import load_node_uncertainty

FORECAST_CACHE_CSV = PROJECT_ROOT / "reports" / "final_eval_smoke" / "forecast_metrics_30min.csv"
WINDOW_ABLATION_CACHE_CSV = PROJECT_ROOT / "reports" / "final_eval_smoke" / "input_window_ablation.csv"
LAMBDA_ABLATION_CACHE_CSV = PROJECT_ROOT / "reports" / "final_eval_smoke" / "routing_lambda_ablation_full.csv"
ROUTING_EVAL_CACHE_JSON = PROJECT_ROOT / "reports" / "final_eval_smoke" / "routing_eval_100_pairs_lambda_0_30.json"


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
def _get_test_artifacts(batch_size: int = 64, output_length: int = 6) -> tuple[object, object, int]:
    dataset = TrafficDataset(
        data_path=PROJECT_ROOT / "data" / "raw" / "METR-LA.h5",
        input_length=12,
        output_length=output_length,
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

    num_nodes = int(x_test.shape[-1])
    return test_loader, dataset.scaler, num_nodes


@st.cache_resource(show_spinner=False)
def _get_validation_artifacts(batch_size: int = 128, input_length: int = 24, output_length: int = 6) -> tuple[object, object, int]:
    dataset = TrafficDataset(
        data_path=PROJECT_ROOT / "data" / "raw" / "METR-LA.h5",
        input_length=input_length,
        output_length=output_length,
    )

    total_samples = len(dataset)
    train_end = int(total_samples * 0.70)
    val_end = train_end + int(total_samples * 0.15)

    x_val = dataset.x[train_end:val_end]
    y_val = dataset.y[train_end:val_end]
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    num_nodes = int(x_val.shape[-1])
    return val_loader, dataset.scaler, num_nodes


def _inverse_scale(batch_values: np.ndarray, scaler: object) -> np.ndarray:
    original_shape = batch_values.shape
    restored = scaler.inverse_transform(batch_values.reshape(-1, original_shape[-1]))
    return restored.reshape(original_shape)


@st.cache_data(show_spinner=False)
def compute_forecast_metrics(batch_size: int = 64) -> dict[str, dict[str, float]]:
    if FORECAST_CACHE_CSV.exists():
        cached_df = pd.read_csv(FORECAST_CACHE_CSV)
        cached: dict[str, dict[str, float]] = {}
        for _, row in cached_df.iterrows():
            horizon_label = str(row["horizon"])
            cached[horizon_label] = {
                "mae": float(row["MAE"]),
                "rmse": float(row["RMSE"]),
                "mape": float(row["MAPE"]),
            }
        if cached:
            return cached

    horizon_steps = 6
    model_steps = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, scaler, num_nodes = _get_test_artifacts(
        batch_size=batch_size,
        output_length=horizon_steps,
    )
    model = _load_model(device_type=device.type, num_nodes=num_nodes)

    mae_sum = np.zeros(horizon_steps, dtype=np.float64)
    rmse_sum = np.zeros(horizon_steps, dtype=np.float64)
    mape_sum = np.zeros(horizon_steps, dtype=np.float64)
    count = np.zeros(horizon_steps, dtype=np.float64)
    mape_count = np.zeros(horizon_steps, dtype=np.float64)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            current_x = x_batch.to(device)
            target = y_batch.numpy()

            pred_chunks: list[np.ndarray] = []
            rollout_steps = horizon_steps // model_steps
            for _ in range(rollout_steps):
                preds_chunk = model(current_x)  # [B, 3, N]
                pred_chunks.append(preds_chunk.cpu().numpy())
                current_x = torch.cat([current_x[:, model_steps:, :], preds_chunk], dim=1)

            preds = np.concatenate(pred_chunks, axis=1)  # [B, 6, N]
            preds_inv = _inverse_scale(preds, scaler)
            target_inv = _inverse_scale(target, scaler)

            abs_err = np.abs(preds_inv - target_inv)
            sq_err = np.square(preds_inv - target_inv)

            mae_sum += abs_err.sum(axis=(0, 2))
            rmse_sum += sq_err.sum(axis=(0, 2))
            count += np.array(
                [target_inv.shape[0] * target_inv.shape[2]] * horizon_steps,
                dtype=np.float64,
            )
            for h in range(horizon_steps):
                target_h = target_inv[:, h, :]
                err_h = abs_err[:, h, :]
                mask = np.abs(target_h) > 1.0
                if np.any(mask):
                    mape_sum[h] += float((err_h[mask] / np.abs(target_h[mask]) * 100.0).sum())
                    mape_count[h] += float(mask.sum())

    mae = mae_sum / np.maximum(count, 1.0)
    rmse = np.sqrt(rmse_sum / np.maximum(count, 1.0))
    mape = mape_sum / np.maximum(mape_count, 1.0)

    results: dict[str, dict[str, float]] = {}
    for i in range(horizon_steps):
        horizon_label = f"{(i + 1) * 5}min"
        results[horizon_label] = {
            "mae": float(mae[i]),
            "rmse": float(rmse[i]),
            "mape": float(mape[i]),
        }

    save_df = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "MAE": vals["mae"],
                "RMSE": vals["rmse"],
                "MAPE": vals["mape"],
            }
            for horizon, vals in results.items()
        ]
    )
    FORECAST_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(FORECAST_CACHE_CSV, index=False)
    return results


@st.cache_data(show_spinner=False)
def compute_input_window_ablation(
    batch_size: int = 128,
    horizon_steps: int = 6,
) -> list[dict[str, float]]:
    if WINDOW_ABLATION_CACHE_CSV.exists():
        cached_df = pd.read_csv(WINDOW_ABLATION_CACHE_CSV)
        return [
            {
                "window_minutes": float(row["window_minutes"]),
                "mape_5min": float(row["mape_5min"]),
                "mape_15min": float(row["mape_15min"]),
                "mape_30min": float(row["mape_30min"]),
                "avg_mape_5_to_30": float(row["avg_mape_5_to_30"]),
                "avg_accuracy_5_to_30": float(row["avg_accuracy_5_to_30"]),
            }
            for _, row in cached_df.iterrows()
        ]

    model_steps = 3
    history_windows = (6, 12, 18, 24)  # 30, 60, 90, 120 min
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loader, scaler, num_nodes = _get_validation_artifacts(
        batch_size=batch_size,
        input_length=max(history_windows),
        output_length=horizon_steps,
    )
    model = _load_model(device_type=device.type, num_nodes=num_nodes)

    results: list[dict[str, float]] = []
    for window_steps in history_windows:
        mape_sum = np.zeros(horizon_steps, dtype=np.float64)
        mape_count = np.zeros(horizon_steps, dtype=np.float64)

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                current_x = x_batch[:, -window_steps:, :].to(device)
                target = y_batch.numpy()

                pred_chunks: list[np.ndarray] = []
                rollout_steps = horizon_steps // model_steps
                for _ in range(rollout_steps):
                    preds_chunk = model(current_x)  # [B, 3, N]
                    pred_chunks.append(preds_chunk.cpu().numpy())
                    current_x = torch.cat([current_x[:, model_steps:, :], preds_chunk], dim=1)

                preds = np.concatenate(pred_chunks, axis=1)  # [B, 6, N]
                preds_inv = _inverse_scale(preds, scaler)
                target_inv = _inverse_scale(target, scaler)
                abs_err = np.abs(preds_inv - target_inv)

                for h in range(horizon_steps):
                    target_h = target_inv[:, h, :]
                    err_h = abs_err[:, h, :]
                    mask = np.abs(target_h) > 1.0
                    if np.any(mask):
                        mape_sum[h] += float((err_h[mask] / np.abs(target_h[mask]) * 100.0).sum())
                        mape_count[h] += float(mask.sum())

        mape = mape_sum / np.maximum(mape_count, 1.0)
        avg_mape = float(np.mean(mape))
        results.append(
            {
                "window_minutes": float(window_steps * 5),
                "mape_5min": float(mape[0]),
                "mape_15min": float(mape[2]),
                "mape_30min": float(mape[5]),
                "avg_mape_5_to_30": avg_mape,
                "avg_accuracy_5_to_30": float(100.0 - avg_mape),
            }
        )

    pd.DataFrame(results).to_csv(WINDOW_ABLATION_CACHE_CSV, index=False)
    return results


def _reliability_text(mae: float) -> str:
    if mae < 3.0:
        return "High reliability"
    if mae <= 6.0:
        return "Moderate reliability"
    return "Low reliability"


def compute_routing_distribution_metrics(
    n_pairs: int = 100,
    seed: int = 42,
    lambda_value: float = 0.3,
) -> dict[str, float | list[float]]:
    if ROUTING_EVAL_CACHE_JSON.exists():
        return json.loads(ROUTING_EVAL_CACHE_JSON.read_text(encoding="utf-8"))

    predicted_speeds = np.asarray(run_inference(), dtype=np.float32)
    node_uncertainty = load_node_uncertainty(PROJECT_ROOT / "node_uncertainty.npy")
    result = evaluate_random_od_pairs(
        predicted_speeds=predicted_speeds,
        node_uncertainty=node_uncertainty,
        lambda_value=lambda_value,
        n_pairs=n_pairs,
        seed=seed,
    )
    ROUTING_EVAL_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    ROUTING_EVAL_CACHE_JSON.write_text(json.dumps(result), encoding="utf-8")
    return result


@st.cache_data(show_spinner=False)
def compute_lambda_ablation_metrics(
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
    n_pairs: int = 100,
    seed: int = 42,
) -> list[dict[str, float]]:
    if LAMBDA_ABLATION_CACHE_CSV.exists():
        cached_df = pd.read_csv(LAMBDA_ABLATION_CACHE_CSV)
        return [
            {
                "lambda": float(row["lambda"]),
                "risk_mean_improvement_pct": float(row["risk_mean_improvement_pct"]),
                "physical_mean_improvement_pct": float(row["physical_mean_improvement_pct"]),
                "avg_reliability_pct": float(row["avg_reliability_pct"]),
                "risk_positive_rate_pct": float(row["risk_positive_rate_pct"]),
            }
            for _, row in cached_df.iterrows()
        ]

    predicted_speeds = np.asarray(run_inference(), dtype=np.float32)
    node_uncertainty = load_node_uncertainty(PROJECT_ROOT / "node_uncertainty.npy")

    rows: list[dict[str, float]] = []
    for lam in lambdas:
        stats = evaluate_random_od_pairs(
            predicted_speeds=predicted_speeds,
            node_uncertainty=node_uncertainty,
            lambda_value=float(lam),
            n_pairs=int(n_pairs),
            seed=int(seed),
        )
        rows.append(
            {
                "lambda": float(lam),
                "risk_mean_improvement_pct": float(stats["mean"]),
                "physical_mean_improvement_pct": float(stats["physical_mean"]),
                "avg_reliability_pct": float(stats["avg_reliability"]),
                "risk_positive_rate_pct": float(stats["positive_rate"]),
            }
        )

    save_df = pd.DataFrame(rows)
    LAMBDA_ABLATION_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(LAMBDA_ABLATION_CACHE_CSV, index=False)
    return rows


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
        horizon_labels = tuple(f"{minute}min" for minute in (5, 10, 15, 20, 25, 30))
        for horizon in horizon_labels:
            rows.append(
                {
                    "Horizon": horizon.replace("min", " min"),
                    "HorizonMin": int(horizon.replace("min", "")),
                    "MAE": metric_dict[horizon]["mae"],
                    "RMSE": metric_dict[horizon]["rmse"],
                    "MAPE": metric_dict[horizon]["mape"],
                    "Reliability": _reliability_text(metric_dict[horizon]["mae"]),
                }
            )
        results_df = pd.DataFrame(rows)

        st.markdown("### \U0001F691 Confidence Interpretation", unsafe_allow_html=True)
        for start in range(0, len(results_df), 3):
            cards = st.columns(3)
            for col, (_, row) in zip(cards, results_df.iloc[start : start + 3].iterrows()):
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
            mae_chart = (
                alt.Chart(chart_df)
                .mark_line(point=True, strokeWidth=3, color="#36a3ff")
                .encode(
                    x=alt.X("HorizonMin:Q", title="Horizon (minutes)", scale=alt.Scale(domain=[5, 30])),
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

        st.info("Prediction error increases with forecasting horizon from 5 to 30 minutes.")

        st.markdown("### \U0001F50E Validation: Input History Window Ablation")
        if WINDOW_ABLATION_CACHE_CSV.exists():
            st.caption(f"Showing cached snapshot from: {WINDOW_ABLATION_CACHE_CSV.name}")
            recompute_ablation = st.button("Recompute Input-Window Validation", key="run_input_window_ablation")
        else:
            st.caption("No cached snapshot yet. Run once to generate it.")
            recompute_ablation = st.button("Run Input-Window Validation", key="run_input_window_ablation")

        if recompute_ablation and WINDOW_ABLATION_CACHE_CSV.exists():
            WINDOW_ABLATION_CACHE_CSV.unlink(missing_ok=True)

        if WINDOW_ABLATION_CACHE_CSV.exists() or recompute_ablation:
            try:
                window_rows = compute_input_window_ablation(batch_size=128, horizon_steps=6)
            except Exception as exc:
                st.error(f"Failed to run input-window ablation: {exc}")
                return

            window_df = pd.DataFrame(window_rows)
            window_df["window_label"] = window_df["window_minutes"].astype(int).astype(str) + " min"
            display_df = window_df[
                [
                    "window_label",
                    "mape_5min",
                    "mape_15min",
                    "mape_30min",
                    "avg_mape_5_to_30",
                    "avg_accuracy_5_to_30",
                ]
            ].copy()
            display_df.columns = [
                "Input Window",
                "MAPE @5m",
                "MAPE @15m",
                "MAPE @30m",
                "Avg MAPE (5-30m)",
                "Avg Accuracy (5-30m)",
            ]
            for col in ["MAPE @5m", "MAPE @15m", "MAPE @30m", "Avg MAPE (5-30m)", "Avg Accuracy (5-30m)"]:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%")

            w1, w2 = st.columns([1.1, 1.1])
            with w1:
                st.table(display_df)

            with w2:
                ablation_chart = (
                    alt.Chart(window_df)
                    .mark_line(point=True, strokeWidth=3, color="#00c2ff")
                    .encode(
                        x=alt.X("window_minutes:Q", title="Input History (minutes)", scale=alt.Scale(domain=[30, 120])),
                        y=alt.Y("avg_mape_5_to_30:Q", title="Avg MAPE (5-30 min)"),
                        tooltip=[
                            alt.Tooltip("window_label:N", title="Window"),
                            alt.Tooltip("avg_mape_5_to_30:Q", format=".2f", title="Avg MAPE"),
                            alt.Tooltip("avg_accuracy_5_to_30:Q", format=".2f", title="Avg Accuracy"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(ablation_chart, width="stretch")

            best_row = min(window_rows, key=lambda r: r["avg_mape_5_to_30"])
            best_window = int(best_row["window_minutes"])
            st.info(
                f"Best validation performance is at {best_window} min input history "
                f"(Avg MAPE: {best_row['avg_mape_5_to_30']:.2f}%, "
                f"Avg Accuracy: {best_row['avg_accuracy_5_to_30']:.2f}%)."
            )
        else:
            st.info("Run once to generate the snapshot.")

        st.markdown("---")
        lambda_used = 0.3
        st.markdown("## \U0001F4C8 Statistical Routing Evaluation (100 OD Pairs)")
        st.caption(f"Uncertainty-aware routing enabled (lambda = {lambda_used:.2f})")
        if ROUTING_EVAL_CACHE_JSON.exists():
            st.caption(f"Showing cached snapshot from: {ROUTING_EVAL_CACHE_JSON.name}")
            run_routing_eval = st.button("Recompute Routing Evaluation", key="run_routing_eval")
        else:
            st.caption("No cached snapshot yet. Run once to generate it.")
            run_routing_eval = st.button("Run Routing Evaluation", key="run_routing_eval")

        if run_routing_eval and ROUTING_EVAL_CACHE_JSON.exists():
            ROUTING_EVAL_CACHE_JSON.unlink(missing_ok=True)

        if ROUTING_EVAL_CACHE_JSON.exists() or run_routing_eval:
            try:
                routing_eval = compute_routing_distribution_metrics(
                    n_pairs=100,
                    seed=42,
                    lambda_value=lambda_used,
                )
            except Exception as exc:
                st.error(f"Failed to run routing distribution evaluation: {exc}")
                return
        else:
            st.info("Run once to generate the snapshot.")
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

        st.markdown("### \U0001F4C9 Lambda Sensitivity (Uncertainty Weight)")
        st.caption("Run once to generate data; cached CSV is reused on reload.")
        run_lambda_ablation = st.button("Run / Refresh Lambda Ablation", key="run_lambda_ablation")
        if run_lambda_ablation or LAMBDA_ABLATION_CACHE_CSV.exists():
            try:
                lambda_rows = compute_lambda_ablation_metrics(
                    lambdas=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
                    n_pairs=100,
                    seed=42,
                )
            except Exception as exc:
                st.error(f"Failed to run lambda ablation: {exc}")
                return

            lambda_df = pd.DataFrame(lambda_rows).sort_values("lambda").reset_index(drop=True)
            table_df = lambda_df.copy()
            table_df["risk_mean_improvement_pct"] = table_df["risk_mean_improvement_pct"].map(lambda x: f"{x:.2f}%")
            table_df["physical_mean_improvement_pct"] = table_df["physical_mean_improvement_pct"].map(lambda x: f"{x:.2f}%")
            table_df["avg_reliability_pct"] = table_df["avg_reliability_pct"].map(lambda x: f"{x:.1f}%")
            table_df["risk_positive_rate_pct"] = table_df["risk_positive_rate_pct"].map(lambda x: f"{x:.1f}%")
            table_df.columns = [
                "Lambda",
                "Risk Mean Improvement",
                "Physical ETA Mean Improvement",
                "Avg Reliability",
                "% Routes Improved",
            ]

            l1, l2 = st.columns([1.05, 1.2])
            with l1:
                st.table(table_df)

            with l2:
                lambda_chart_df = lambda_df.melt(
                    id_vars=["lambda"],
                    value_vars=[
                        "risk_mean_improvement_pct",
                        "physical_mean_improvement_pct",
                        "avg_reliability_pct",
                    ],
                    var_name="Metric",
                    value_name="Value",
                )
                metric_labels = {
                    "risk_mean_improvement_pct": "Risk Mean Improvement",
                    "physical_mean_improvement_pct": "Physical ETA Mean Improvement",
                    "avg_reliability_pct": "Avg Reliability",
                }
                lambda_chart_df["Metric"] = lambda_chart_df["Metric"].map(metric_labels)

                lambda_chart = (
                    alt.Chart(lambda_chart_df)
                    .mark_line(point=True, strokeWidth=3)
                    .encode(
                        x=alt.X("lambda:Q", title="Lambda"),
                        y=alt.Y("Value:Q", title="Percentage"),
                        color=alt.Color("Metric:N", title="Metric"),
                        tooltip=[
                            alt.Tooltip("lambda:Q", format=".2f", title="Lambda"),
                            alt.Tooltip("Metric:N"),
                            alt.Tooltip("Value:Q", format=".2f"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(lambda_chart, width="stretch")

            best_lambda = float(
                lambda_df.loc[lambda_df["risk_mean_improvement_pct"].idxmax(), "lambda"]
            )
            st.info(
                f"Best lambda by mean risk improvement in this run: {best_lambda:.2f}. "
                "Use this as the primary tuning justification in the paper."
            )
        else:
            st.info("Lambda ablation not generated yet. Click the button to compute and cache it.")

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


