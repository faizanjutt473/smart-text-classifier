"""Train, evaluate and save the Smart Text Classifier.

Pipeline:  clean text -> TF-IDF (word 1-2 grams) -> Logistic Regression

Run from the project root:
    python -m src.train
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.preprocess import load_dataset


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),          # single words and word pairs
            sublinear_tf=True,           # dampen very frequent words
            min_df=2,                    # ignore words seen only once
            token_pattern=r"(?u)\b\w+\b|\?",  # keep single letters and '?'
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )),
    ])


def summarise(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=config.LABELS)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.LABELS, yticklabels=config.LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    # 1. Load and clean data
    df = load_dataset(config.DATA_PATH)
    print(f"Loaded {len(df)} rows. Class counts:\n{df['category'].value_counts().to_string()}\n")

    # 2. Train / test split (stratified keeps class balance equal in both parts)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"],
        test_size=config.TEST_SIZE, stratify=df["category"], random_state=config.RANDOM_STATE,
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Cross-validation on the training part only (the test set stays untouched)
    model = build_pipeline()
    cv_scores = cross_val_score(model, X_train, y_train, cv=config.CV_FOLDS, scoring="f1_macro")
    print(f"{config.CV_FOLDS}-fold CV macro F1 (train only): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 4. Final training and evaluation on the held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_metrics = summarise(y_test, y_pred)
    print("\nHeld-out test set results")
    print(classification_report(y_test, y_pred, labels=config.LABELS, digits=3))

    # 5. Extra check on independent, hand-written messages that do not come from templates
    ext = load_dataset(config.EXTERNAL_TEST_PATH)
    ext_pred = model.predict(ext["clean_text"])
    ext_metrics = summarise(ext["category"], ext_pred)
    print("Independent hand-written test set results")
    print(classification_report(ext["category"], ext_pred, labels=config.LABELS, digits=3))

    # 6. Save artefacts
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    save_confusion_matrix(y_test, y_pred, "Confusion Matrix - Held-out Test Set", config.CONFUSION_MATRIX_PATH)
    save_confusion_matrix(ext["category"], ext_pred, "Confusion Matrix - Hand-written Set",
                          config.REPORTS_DIR / "confusion_matrix_external.png")

    results = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "external_rows": int(len(ext)),
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        "test": test_metrics,
        "external": ext_metrics,
        "test_per_class": classification_report(y_test, y_pred, labels=config.LABELS,
                                                output_dict=True, zero_division=0),
    }
    config.METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Metrics saved to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
