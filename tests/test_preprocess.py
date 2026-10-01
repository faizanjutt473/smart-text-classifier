import pandas as pd
import pytest

from src.preprocess import clean_text, load_dataset


def test_lowercases_and_removes_punctuation():
    assert clean_text("Hello, World!!!") == "hello world"


def test_keeps_question_mark():
    assert "?" in clean_text("Do you deliver to Lahore?")


def test_removes_urls_and_replaces_numbers():
    cleaned = clean_text("Visit https://example.com now, code 123456")
    assert "http" not in cleaned
    assert "123456" not in cleaned
    assert "num" in cleaned


def test_non_string_returns_empty():
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_load_dataset_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hi"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_dataset(bad)


def test_load_dataset_drops_empty_and_duplicate_rows(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "text": ["Hello there", "Hello there", None, "!!!"],
        "category": ["Other", "Other", "Other", "Other"],
    }).to_csv(path, index=False)
    df = load_dataset(path)
    assert len(df) == 1
