"""Load the trained model and predict the category of new text."""

import joblib

from src import config
from src.preprocess import clean_text


class TextClassifier:
    """Small wrapper around the saved scikit-learn pipeline."""

    def __init__(self, model_path=config.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run 'python -m src.train' first to create it."
            )
        self.model = joblib.load(model_path)
        self.classes = list(self.model.classes_)

    def predict(self, text: str) -> dict:
        """Return the predicted category, confidence and all class probabilities."""
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Text is empty after cleaning. Please enter some words.")
        probabilities = self.model.predict_proba([cleaned])[0]
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        return {
            "category": self.classes[best],
            "confidence": round(confidence, 4),
            "low_confidence": confidence < config.LOW_CONFIDENCE_THRESHOLD,
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probabilities)},
        }
