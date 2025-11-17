cat > ml/train.py <<'PY'
import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# use absolute import so both `python ml/train.py` and `python -m ml.train` work
from ml.config import config

def train():
    config.model_dir.mkdir(parents=True, exist_ok=True)

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data.target,
    )

    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(clf, config.model_path)

    metrics = {
        "accuracy": acc,
        "n_estimators": config.n_estimators,
    }

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Model saved to {config.model_path}")
    print(f"Metrics: {metrics}")

    # simple "gate"
    if acc < 0.9:
        raise ValueError(f"Accuracy too low: {acc:.3f}")

if __name__ == "__main__":
    train()
PY
