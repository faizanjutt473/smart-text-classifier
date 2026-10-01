"""Text cleaning and dataset loading helpers."""

import re

import pandas as pd

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
NUMBER_PATTERN = re.compile(r"\d+")
# Keep letters, digits, apostrophes and the question mark (a strong Inquiry signal).
UNWANTED_CHARS = re.compile(r"[^a-z0-9?' ]+")
SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise one message.

    Steps: lower-case -> remove URLs -> replace numbers with a token ->
    drop unwanted punctuation (but keep '?') -> collapse extra spaces.

    Stop-words such as "how", "do" or "please" are deliberately KEPT because
    they carry the intent of a message (e.g. "how do I ..." signals an Inquiry).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" num ", text)
    text = UNWANTED_CHARS.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def load_dataset(path) -> pd.DataFrame:
    """Load a CSV, drop empty/duplicate rows and add a 'clean_text' column."""
    df = pd.read_csv(path)
    missing = {"text", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {sorted(missing)}")
    df = df.dropna(subset=["text", "category"]).drop_duplicates(subset="text").copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""]
    return df.reset_index(drop=True)
