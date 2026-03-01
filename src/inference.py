from __future__ import annotations

from pathlib import Path
import torch

try:
    from src.data.traffic_dataset import TrafficDataset
    from src.models.graph_wavenet import GraphWaveNet
except ModuleNotFoundError:
    # Allows running as: python src/inference.py
    from data.traffic_dataset import TrafficDataset
    from models.graph_wavenet import GraphWaveNet


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "checkpoints" / "model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = TrafficDataset(
        data_path=project_root / "data" / "raw" / "METR-LA.h5",
        input_length=12,
        output_length=3,
    )

    # Latest history window
    latest_x, _ = dataset[-1]
    latest_x = latest_x.unsqueeze(0).to(device)

    num_nodes = latest_x.shape[-1]

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
    with torch.no_grad():
        prediction = model(latest_x)

    prediction_np = prediction.cpu().numpy()

    # Inverse normalization (VERY IMPORTANT)
    prediction_np = dataset.scaler.inverse_transform(
        prediction_np.reshape(-1, prediction_np.shape[-1])
    ).reshape(prediction_np.shape)

    print(f"Prediction shape: {tuple(prediction_np.shape)}")
    return prediction_np


if __name__ == "__main__":
    run_inference()