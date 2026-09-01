"""Central config - paths and constants used across the app."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "..", "data", "processed")
DATA_PREDICTIONS = os.path.join(BASE_DIR, "..", "data", "predictions")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

ACTIVE_MODEL_VERSION = "v1"
MODEL_PATH = os.path.join(MODELS_DIR, ACTIVE_MODEL_VERSION, "attrition_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, ACTIVE_MODEL_VERSION, "metadata.json")

EMPLOYEE_INTELLIGENCE_PATH = os.path.join(DATA_PROCESSED, "employee_intelligence.csv")
ENGAGEMENT_PATH = os.path.join(DATA_PROCESSED, "engagement_processed.csv")
SKILL_GAP_PATH = os.path.join(DATA_PROCESSED, "employee_skill_gaps.csv")
ORG_SKILL_GAP_PATH = os.path.join(BASE_DIR, "..", "docs", "organization_skill_gap.csv")
ATTRITION_PROCESSED_PATH = os.path.join(DATA_PROCESSED, "employee_attrition_processed.csv")
EMPLOYEE_NAMES_PATH = os.path.join(DATA_PROCESSED, "employee_names_SYNTHETIC.csv")

RISK_THRESHOLDS = {"HIGH": 0.6, "MEDIUM": 0.3}
