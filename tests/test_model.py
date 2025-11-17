from pathlib import Path
import json

from ml.config import config
from ml.train import train

def test_training_and_metrics(tmp_path, monkeypatch):
    # gunakan tmp dir supaya tidak ganggu artefak lokal
    monkeypatch.setattr(config, "model_dir", tmp_path)
    monkeypatch.setattr(config, "model_path", tmp_path / "model.pkl")
    monkeypatch.setattr(config, "metrics_path", tmp_path / "metrics.json")

    train()

    assert config.model_path.exists()
    assert config.metrics_path.exists()

    metrics = json.loads(config.metrics_path.read_text())
    assert metrics["accuracy"] >= 0.9
