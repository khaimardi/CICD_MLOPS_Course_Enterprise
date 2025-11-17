from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import load_model

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_endpoint(monkeypatch):
    # mock supaya tidak perlu load model beneran
    class DummyModel:
        def predict(self, X):
            return [1]

        def predict_proba(self, X):
            return [[0.1, 0.8, 0.1]]

    # patch load_model to return our dummy model
    monkeypatch.setattr("app.main.load_model", lambda: DummyModel())

    payload = {"features": [1.0, 2.0, 3.0, 4.0]}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
    assert len(data["probabilities"]) == 3
