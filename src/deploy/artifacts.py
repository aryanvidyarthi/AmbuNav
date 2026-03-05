from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_ARTIFACTS = {
    "model": {
        "path": PROJECT_ROOT / "checkpoints" / "model.pth",
        "env_key": "AMBUNAV_MODEL_URL",
    },
    "metr_h5": {
        "path": PROJECT_ROOT / "data" / "raw" / "METR-LA.h5",
        "env_key": "AMBUNAV_METR_H5_URL",
    },
    "adjacency": {
        "path": PROJECT_ROOT / "data" / "raw" / "adj_METR-LA.pkl",
        "env_key": "AMBUNAV_ADJ_URL",
    },
}

OPTIONAL_ARTIFACTS = {
    "node_uncertainty": {
        "path": PROJECT_ROOT / "node_uncertainty.npy",
        "env_key": "AMBUNAV_UNCERTAINTY_URL",
    }
}


def _download_to_path(url: str, destination: Path, timeout_s: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout_s) as response, NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.read())
        tmp_path = Path(tmp.name)
    tmp_path.replace(destination)


def ensure_artifacts(
    *,
    extra_urls: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """
    Ensure required deploy artifacts exist locally.

    If an artifact is missing, it tries to download from:
    1) extra_urls[env_key]
    2) os.environ[env_key]
    """
    extra_urls = extra_urls or {}
    existing: list[str] = []
    downloaded: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []

    def _resolve_url(env_key: str) -> str:
        return str(extra_urls.get(env_key) or os.environ.get(env_key) or "").strip()

    for artifact_name, cfg in REQUIRED_ARTIFACTS.items():
        path = Path(cfg["path"])
        env_key = str(cfg["env_key"])
        if path.exists():
            existing.append(artifact_name)
            continue
        url = _resolve_url(env_key)
        if not url:
            missing_required.append(f"{artifact_name} ({env_key})")
            continue
        _download_to_path(url=url, destination=path)
        downloaded.append(artifact_name)

    for artifact_name, cfg in OPTIONAL_ARTIFACTS.items():
        path = Path(cfg["path"])
        env_key = str(cfg["env_key"])
        if path.exists():
            existing.append(artifact_name)
            continue
        url = _resolve_url(env_key)
        if not url:
            missing_optional.append(f"{artifact_name} ({env_key})")
            continue
        _download_to_path(url=url, destination=path)
        downloaded.append(artifact_name)

    return {
        "existing": existing,
        "downloaded": downloaded,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }
