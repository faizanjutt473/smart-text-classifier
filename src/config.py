"""Central place for paths and settings so they are not repeated in every file."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "text_classification_dataset.csv"
EXTERNAL_TEST_PATH = ROOT / "data" / "external_test_samples.csv"

MODEL_PATH = ROOT / "models" / "text_classifier.joblib"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
SAMPLE_PREDICTIONS_PATH = REPORTS_DIR / "sample_predictions.csv"

LABELS = ["Complaint", "Inquiry", "Feedback", "Other"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Predictions below this confidence are flagged as "low confidence" in the API.
LOW_CONFIDENCE_THRESHOLD = 0.50
