"""Plain Python logging for app lifecycle: startup, prediction requests, errors."""
import logging
import sys

logger = logging.getLogger("enterprise_hr_ai")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
