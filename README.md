# Traffic Prediction and Route Optimization System

Research-oriented modular Python project scaffold for:
- **Traffic forecasting** with **Graph WaveNet**
- **Route optimization** with **Dynamic A\***
- **Interactive UI** with **Streamlit**

## Project Structure

```text
AmbuNav/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── model/
│   │   └── graph_wavenet.yaml
│   ├── routing/
│   │   └── dynamic_astar.yaml
│   ├── training/
│   │   └── train.yaml
│   └── ui/
│       └── streamlit.toml
├── docs/
│   ├── architecture/
│   │   └── overview.md
│   └── experiments/
│       └── experiment_log.md
├── notebooks/
│   └── .gitkeep
├── scripts/
│   ├── run_inference.py
│   ├── run_streamlit.sh
│   └── run_training.py
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   └── traffic_dataset.py
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   └── graph_features.py
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   └── load_raw_data.py
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       └── clean_traffic_data.py
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   └── metrics.py
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   └── predict.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── graph_conv.py
│   │   │   └── temporal_conv.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── graph_wavenet.py
│   │   └── training/
│   │       ├── __init__.py
│   │       └── train_graph_wavenet.py
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   └── dynamic_astar.py
│   │   ├── costs/
│   │   │   ├── __init__.py
│   │   │   └── travel_time_cost.py
│   │   └── simulation/
│   │       ├── __init__.py
│   │       └── route_simulator.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── assets/
│   │   │   └── .gitkeep
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── forecast_panel.py
│   │   │   └── map_view.py
│   │   └── pages/
│   │       ├── __init__.py
│   │       ├── dashboard.py
│   │       └── route_planner.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── seed.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── e2e/
│   │   ├── __init__.py
│   │   └── test_streamlit_app.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_forecast_to_route_pipeline.py
│   └── unit/
│       ├── __init__.py
│       ├── test_dynamic_astar.py
│       └── test_graph_wavenet.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. Implement Graph WaveNet blocks in `src/forecasting/models/graph_wavenet.py`
2. Implement time-dependent Dynamic A* in `src/routing/algorithms/dynamic_astar.py`
3. Wire prediction + routing into `src/ui/app.py` Streamlit workflows
4. Add reproducible experiments under `docs/experiments/`
