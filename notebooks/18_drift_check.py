"""
18 - Drift Check (lightweight MLOps monitoring)
Deck's MLOps slide describes retraining triggered automatically when
monitoring detects data or prediction drift. Full MLflow + automated
retraining triggers are out of scope for this MVP (see README), but the
actual comparison logic is real and runs today: compares the distribution
of live predictions (data/predictions/prediction_log.csv, written by the
API on every request) against the training-time baseline rate.

Run periodically (e.g. daily) once the API has served real traffic:
    python3 notebooks/18_drift_check.py
"""
import os
import pandas as pd

PRED_LOG = "../data/predictions/prediction_log.csv"
TRAINING_DATA = "../data/processed/employee_attrition_processed.csv"
ALERT_THRESHOLD = 0.15  # absolute difference in mean predicted probability vs training attrition rate


def main():
    if not os.path.isfile(PRED_LOG):
        print("No predictions logged yet - nothing to check. Run the API and make some /predict/attrition "
              "calls first, then re-run this script.")
        return

    baseline_rate = pd.read_csv(TRAINING_DATA)["Attrition"].mean()
    live = pd.read_csv(PRED_LOG)

    if len(live) < 10:
        print(f"Only {len(live)} logged predictions so far - too few to check drift reliably (need 10+).")
        return

    live_mean_prob = live["probability"].mean()
    diff = abs(live_mean_prob - baseline_rate)

    print(f"Training-time attrition rate: {baseline_rate:.3f}")
    print(f"Live mean predicted probability ({len(live)} predictions): {live_mean_prob:.3f}")
    print(f"Absolute difference: {diff:.3f} (alert threshold: {ALERT_THRESHOLD})")

    if diff > ALERT_THRESHOLD:
        print("\n⚠️  DRIFT ALERT: live predictions have drifted meaningfully from the training baseline. "
              "Consider investigating input data changes and retraining (09_model_versioning.py -> bump to v2).")
    else:
        print("\nNo significant drift detected.")


if __name__ == "__main__":
    main()
