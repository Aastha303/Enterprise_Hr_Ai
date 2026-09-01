"""Loads the versioned attrition model + its metadata + the expected feature columns."""
import json
import joblib
import pandas as pd
from app.utils.config import MODEL_PATH, METADATA_PATH, DATA_PROCESSED
from app.utils.logger import logger

_model = None
_metadata = None
_expected_columns = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    return _model


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _metadata


def get_expected_columns():
    """The exact column set/order the model was trained on (post one-hot-encoding)."""
    global _expected_columns
    if _expected_columns is None:
        df = pd.read_csv(f"{DATA_PROCESSED}/attrition_features.csv", nrows=1)
        _expected_columns = [c for c in df.columns if c != "Attrition"]
    return _expected_columns
