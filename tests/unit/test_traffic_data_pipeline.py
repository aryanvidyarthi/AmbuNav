import pickle

import numpy as np
import pandas as pd

from src.data.datasets.traffic_dataset import prepare_metr_la_pipeline


def test_prepare_metr_la_pipeline(tmp_path, monkeypatch):
    time_steps = 100
    num_nodes = 5
    speeds = np.arange(time_steps * num_nodes, dtype=np.float32).reshape(time_steps, num_nodes)

    h5_path = tmp_path / "metr-la.h5"
    h5_path.write_text("placeholder")

    monkeypatch.setattr(pd, "read_hdf", lambda _: pd.DataFrame(speeds))

    adj = np.eye(num_nodes, dtype=np.float32)
    adj_path = tmp_path / "adj_mx.pkl"
    with adj_path.open("wb") as handle:
        pickle.dump((list(range(num_nodes)), {i: i for i in range(num_nodes)}, adj), handle)

    split, loaded_adj, scaler, loaders = prepare_metr_la_pipeline(
        metr_h5_path=h5_path,
        adj_mx_path=adj_path,
        input_window=6,
        horizon=3,
        batch_size=4,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    assert loaded_adj.shape == (num_nodes, num_nodes)
    assert np.allclose(loaded_adj, adj)

    # segment lengths: train=60, val=20, test=20
    # windows per split = segment_len - input_window - horizon + 1
    assert split.x_train.shape == (52, 6, num_nodes)
    assert split.y_train.shape == (52, 3, num_nodes)
    assert split.x_val.shape == (12, 6, num_nodes)
    assert split.y_test.shape == (12, 3, num_nodes)

    assert abs(float(np.mean(split.x_train))) < 0.5
    assert scaler.std > 0

    batch_x, batch_y = next(iter(loaders["train"]))
    assert batch_x.shape[1:] == (6, num_nodes)
    assert batch_y.shape[1:] == (3, num_nodes)
