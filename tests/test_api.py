import pytest
from fastapi.testclient import TestClient

from src import config
from src.app import app

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="Run 'python -m src.train' before testing the API."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Smart Text Classifier" in response.text


@pytest.mark.parametrize("text, expected", [
    ("What are your working hours?", "Inquiry"),
    ("I was charged twice and I want my money back.", "Complaint"),
    ("Good morning everyone.", "Other"),
])
def test_predict_known_examples(client, text, expected):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == expected
    assert set(body["probabilities"]) == set(config.LABELS)
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": "x" * 1001}])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_text_with_only_symbols_is_rejected(client):
    assert client.post("/predict", json={"text": "!!! ???".replace("?", "")}).status_code == 422
