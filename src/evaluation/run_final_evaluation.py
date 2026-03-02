from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from src.data.traffic_dataset import get_dataloaders
    from src.evaluation.routing_distribution import evaluate_random_od_pairs
    from src.inference import run_inference
    from src.models.graph_wavenet import GraphWaveNet
    from src.routing.dynamic_astar import load_node_uncertainty
except ModuleNotFoundError:
    from data.traffic_dataset import get_dataloaders
    from evaluation.routing_distribution import evaluate_random_od_pairs
    from inference import run_inference
    from models.graph_wavenet import GraphWaveNet
    from routing.dynamic_astar import load_node_uncertainty


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _load_model(device: torch.device, num_nodes: int) -> GraphWaveNet:
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


def _inverse_scale(batch_values: np.ndarray, scaler: Any) -> np.ndarray:
    original_shape = batch_values.shape
    restored = scaler.inverse_transform(batch_values.reshape(-1, original_shape[-1]))
    return restored.reshape(original_shape)


def evaluate_forecast_metrics(batch_size: int = 64) -> dict[str, dict[str, float]]:
    train_loader, _, test_loader, scaler = get_dataloaders(batch_size=batch_size, return_scaler=True)
    sample_x, _ = next(iter(train_loader))
    num_nodes = int(sample_x.shape[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device=device, num_nodes=num_nodes)

    mae_sum = np.zeros(3, dtype=np.float64)
    rmse_sum = np.zeros(3, dtype=np.float64)
    mape_sum = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)
    mape_count = np.zeros(3, dtype=np.float64)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch.to(device)).cpu().numpy()
            target = y_batch.numpy()

            preds_inv = _inverse_scale(preds, scaler)
            target_inv = _inverse_scale(target, scaler)
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
        "5min": {"MAE": float(mae[0]), "RMSE": float(rmse[0]), "MAPE": float(mape[0])},
        "10min": {"MAE": float(mae[1]), "RMSE": float(rmse[1]), "MAPE": float(mape[1])},
        "15min": {"MAE": float(mae[2]), "RMSE": float(rmse[2]), "MAPE": float(mape[2])},
    }


def run_ablation(
    predicted_speeds: np.ndarray,
    node_uncertainty: np.ndarray,
    lambdas: list[float],
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for lam in lambdas:
        stats = evaluate_random_od_pairs(
            predicted_speeds=predicted_speeds,
            node_uncertainty=node_uncertainty,
            lambda_value=float(lam),
            n_pairs=n_pairs,
            seed=seed,
        )
        rows.append(
            {
                "lambda": float(lam),
                "risk_mean_improvement_pct": float(stats["mean"]),
                "risk_median_improvement_pct": float(stats["median"]),
                "risk_positive_rate_pct": float(stats["positive_rate"]),
                "physical_mean_improvement_pct": float(stats["physical_mean"]),
                "physical_median_improvement_pct": float(stats["physical_median"]),
                "physical_positive_rate_pct": float(stats["physical_positive_rate"]),
                "avg_reliability_pct": float(stats["avg_reliability"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_ablation(df: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["lambda"], df["risk_mean_improvement_pct"], marker="o", label="Risk-adjusted mean improvement")
    ax.plot(df["lambda"], df["physical_mean_improvement_pct"], marker="s", label="Physical ETA mean improvement")
    ax.plot(df["lambda"], df["avg_reliability_pct"], marker="^", label="Avg reliability")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Percentage")
    ax.set_title("Uncertainty-Aware Routing Ablation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final reproducible evaluation and export paper-ready artifacts.")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.3, 0.7, 1.0])
    parser.add_argument("--out-dir", type=str, default="reports/final_eval")
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_metrics = evaluate_forecast_metrics(batch_size=args.batch_size)
    predicted_speeds = np.asarray(run_inference(), dtype=np.float32)
    node_uncertainty = load_node_uncertainty(PROJECT_ROOT / "node_uncertainty.npy")
    ablation_df = run_ablation(
        predicted_speeds=predicted_speeds,
        node_uncertainty=node_uncertainty,
        lambdas=[float(x) for x in args.lambdas],
        n_pairs=int(args.n_pairs),
        seed=int(args.seed),
    )

    forecast_df = pd.DataFrame.from_dict(forecast_metrics, orient="index").reset_index().rename(columns={"index": "horizon"})
    forecast_csv = out_dir / "forecast_metrics.csv"
    ablation_csv = out_dir / "routing_lambda_ablation.csv"
    summary_json = out_dir / "summary.json"
    ablation_png = out_dir / "routing_lambda_ablation.png"

    forecast_df.to_csv(forecast_csv, index=False)
    ablation_df.to_csv(ablation_csv, index=False)
    _plot_ablation(ablation_df, ablation_png)

    summary = {
        "metadata": {
            "seed": int(args.seed),
            "n_pairs": int(args.n_pairs),
            "batch_size": int(args.batch_size),
            "lambdas": [float(x) for x in args.lambdas],
            "git_commit": _get_git_commit(),
        },
        "forecast_metrics": forecast_metrics,
        "best_lambda_by_risk_improvement": float(ablation_df.loc[ablation_df["risk_mean_improvement_pct"].idxmax(), "lambda"]),
        "best_lambda_by_physical_improvement": float(ablation_df.loc[ablation_df["physical_mean_improvement_pct"].idxmax(), "lambda"]),
        "best_lambda_by_reliability": float(ablation_df.loc[ablation_df["avg_reliability_pct"].idxmax(), "lambda"]),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {forecast_csv}")
    print(f"Saved: {ablation_csv}")
    print(f"Saved: {ablation_png}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
