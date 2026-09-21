"""
Loads the trained attrition model and metadata once at import time.
Every view imports from here, so we never hit disk repeatedly.
"""
from pathlib import Path
import json
import joblib

ML_DIR = Path(__file__).resolve().parent

MODEL_PATH = ML_DIR / "attrition_model.joblib"
META_PATH  = ML_DIR / "model_metadata.json"

_model = None
_meta  = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metadata():
    global _meta
    if _meta is None:
        _meta = json.loads(META_PATH.read_text())
    return _meta


def get_feature_names():
    return get_metadata()["feature_names"]