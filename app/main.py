from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List

app = FastAPI(title="Enterprise MLOps API", version="1.0.0")

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: int
    probabilities: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

# define a module-level wrapper that lazily imports the real loader
def load_model():
    """
    Lazy wrapper that imports the real load_model from app.model_loader
    only when this function is called. This avoids import-time errors
    (useful for tests that monkeypatch app.main.load_model).
    """
    from .model_loader import load_model as _real_load_model
    return _real_load_model()

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 4:
        # Contoh bug yang sering muncul: return 500 → ini nanti kamu bahas di skenario hotfix
        raise HTTPException(status_code=400, detail="features must have length 4")

    model = load_model()
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)[0]

    # robust handling: model.predict_proba may return numpy arrays or plain lists
    probs_raw = model.predict_proba(X)[0]
    if hasattr(probs_raw, "tolist"):
        probs = probs_raw.tolist()
    else:
        probs = list(probs_raw)

    return PredictResponse(prediction=int(pred), probabilities=probs)