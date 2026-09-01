"""Business logic for attrition prediction + prediction logging."""
import os
import csv
from datetime import datetime, timezone
from app.ml.predictor import predict_attrition
from app.ml.model_loader import get_metadata
from app.utils.config import DATA_PREDICTIONS
from app.utils.logger import logger

PREDICTIONS_LOG = os.path.join(DATA_PREDICTIONS, "prediction_log.csv")


def log_prediction(employee_number, prob, risk):
    """Separate from application logs: a durable record of every prediction made,
    for later checking the prediction distribution against training-time expectations
    (an early warning sign of model drift)."""
    os.makedirs(DATA_PREDICTIONS, exist_ok=True)
    file_exists = os.path.isfile(PREDICTIONS_LOG)
    with open(PREDICTIONS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "employee_number", "model_version", "probability", "risk_level"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), employee_number,
                          get_metadata()["version"], prob, risk])


def predict_for_employee(record: dict) -> dict:
    logger.info("Prediction request received")
    result = predict_attrition(record)
    log_prediction(record.get("EmployeeNumber"), result["attrition_probability"], result["risk_level"])
    return {
        **result,
        "model_version": get_metadata()["version"],
    }
