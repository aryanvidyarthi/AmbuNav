from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from src.data.traffic_dataset import get_dataloaders
    from src.models.graph_wavenet import GraphWaveNet
except ModuleNotFoundError:
    # Allows running as: python src/train.py
    from data.traffic_dataset import get_dataloaders
    from models.graph_wavenet import GraphWaveNet


def train(
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    sample_x, sample_y = next(iter(train_loader))
    num_nodes = sample_x.shape[-1]
    output_length = sample_y.shape[1]

    model = GraphWaveNet(
        num_nodes=num_nodes,
        in_dim=1,
        out_dim=output_length,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=64,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pth"

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * x_batch.size(0)

        avg_train_loss = train_loss_total / len(train_loader.dataset)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss_total += loss.item() * x_batch.size(0)

        avg_val_loss = val_loss_total / len(val_loader.dataset)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    train()
