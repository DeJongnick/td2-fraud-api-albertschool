import base64
import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import needs_data

pytestmark = needs_data


@pytest.fixture
def client():
    import app

    with TestClient(app.app) as c:
        yield c


def envelope(message: dict | str, attempt: int = 1) -> dict:
    raw = message if isinstance(message, str) else json.dumps(message)
    return {
        "message": {"data": base64.b64encode(raw.encode()).decode(), "messageId": "42"},
        "subscription": "projects/advanced-cloud-ai/subscriptions/team3-tx-push",
        "deliveryAttempt": attempt,
    }


def fraudfinder(tx: dict) -> dict:
    """Une transaction du rejeu au format du flux FraudFinder (clés en majuscules)."""
    return {
        "TX_ID": tx["tx_id"],
        "TX_TS": tx["tx_ts"][:19].replace("T", " "),
        "CUSTOMER_ID": tx["customer_id"],
        "TERMINAL_ID": tx["terminal_id"],
        "TX_AMOUNT": tx["tx_amount"],
    }


def log_lines(capsys) -> list[dict]:
    lines = capsys.readouterr().out.splitlines()
    return [json.loads(line) for line in lines if line.startswith("{")]


def test_valid_message_is_scored_then_acknowledged(client, replay, golden, capsys):
    response = client.post("/pubsub", json=envelope(fraudfinder(replay[0]), attempt=2))
    assert response.status_code == 204
    assert response.content == b""

    [line] = log_lines(capsys)
    assert line["severity"] == "INFO" and line["message"] == "scored"
    assert line["tx_id"] == replay[0]["tx_id"]
    assert line["score"] == pytest.approx(golden["scores"][0], abs=1e-6)
    assert line["delivery_attempt"] == 2


def test_redelivery_gives_the_same_score(client, replay, capsys):
    message = envelope(fraudfinder(replay[0]))
    client.post("/pubsub", json=message)
    client.post("/pubsub", json=message)
    first, second = log_lines(capsys)
    assert first["score"] == second["score"]


def test_negative_amount_is_rejected_for_redelivery(client, replay, capsys):
    message = fraudfinder(replay[0]) | {"TX_ID": "poison", "TX_AMOUNT": -1}
    response = client.post("/pubsub", json=envelope(message, attempt=3))
    assert response.status_code == 422

    [line] = log_lines(capsys)
    assert line == {
        "severity": "ERROR",
        "message": "rejected",
        "tx_id": "poison",
        "delivery_attempt": 3,
        "message_id": "42",
    }


@pytest.mark.parametrize("data", ["not json", "[1, 2]"])
def test_unreadable_message_is_rejected(client, data, capsys):
    assert client.post("/pubsub", json=envelope(data)).status_code == 422
    [line] = log_lines(capsys)
    assert line["message"] == "rejected" and line["tx_id"] is None


def test_pubsub_does_not_change_the_state(client, replay):
    before = client.post("/predict", json={"transactions": replay}).json()["scores"]
    for tx in replay[:20]:
        client.post("/pubsub", json=envelope(fraudfinder(tx)))
    assert client.post("/predict", json={"transactions": replay}).json()["scores"] == before
