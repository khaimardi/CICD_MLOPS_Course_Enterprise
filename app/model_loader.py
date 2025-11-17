import os
from functools import lru_cache
from pathlib import Path
import joblib

LOCAL_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
GCS_MODEL_URI = os.getenv("GCS_MODEL_URI", "")  # contoh: gs://bucket/path/model.pkl

def _download_from_gcs(gcs_uri: str, local_path: Path):
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS_MODEL_URI must start with gs://")

    # lazy import supaya module tetap importable di environment tanpa google-cloud-storage
    try:
        from google.cloud import storage
    except Exception as e:
        raise RuntimeError("google-cloud-storage is required to download from GCS") from e

    _, path_part = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path_part.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))

@lru_cache(maxsize=1)
def load_model():
    """
    Prioritas:
    1. Kalau GCS_MODEL_URI di-set → download dari GCS ke LOCAL_MODEL_PATH.
    2. Kalau tidak → langsung load dari LOCAL_MODEL_PATH (sudah dibawa di Docker image).
    """
    if GCS_MODEL_URI:
        print(f"[model_loader] Downloading model from {GCS_MODEL_URI}")
        _download_from_gcs(GCS_MODEL_URI, LOCAL_MODEL_PATH)

    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}")

    print(f"[model_loader] Loading model from {LOCAL_MODEL_PATH}")
    return joblib.load(LOCAL_MODEL_PATH)