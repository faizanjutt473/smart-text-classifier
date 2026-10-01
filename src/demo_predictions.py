"""Print and save sample predictions for demonstration (assignment asks for 10+).

Run from the project root:
    python -m src.demo_predictions
"""

import pandas as pd

from src import config
from src.predict import TextClassifier

DEMO_MESSAGES = [
    "My order arrived broken and nobody is answering my emails.",
    "I was charged twice this month, please refund the extra payment.",
    "The app crashes every time I open the settings page.",
    "What are your delivery charges for Karachi?",
    "How can I change the phone number on my account?",
    "Do you offer a discount for students?",
    "I really like the new design, it feels very clean.",
    "It would be great to have an option to schedule deliveries.",
    "The support team was polite and solved my issue quickly.",
    "Good morning, hope you are doing well.",
    "Congratulations! You won a free phone, click here to claim.",
    "The meeting is rescheduled to Thursday at 3 pm.",
]


def main():
    classifier = TextClassifier()
    rows = []
    for text in DEMO_MESSAGES:
        result = classifier.predict(text)
        rows.append({"text": text, "predicted_category": result["category"],
                     "confidence": result["confidence"]})

    df = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SAMPLE_PREDICTIONS_PATH, index=False)

    print(f"{'#':<3}{'Predicted':<12}{'Conf.':<8}Text")
    print("-" * 90)
    for i, r in df.iterrows():
        print(f"{i + 1:<3}{r['predicted_category']:<12}{r['confidence']:<8.2f}{r['text']}")
    print(f"\nSaved to {config.SAMPLE_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
