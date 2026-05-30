"""Integration tests for the HTTP API (via FastAPI TestClient)."""

from __future__ import annotations

VALID_PAYLOAD = {
    "Age": 35, "SystolicBP": 140, "DiastolicBP": 90,
    "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 86,
}


def test_health_ok(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_meta_shape(client):
    res = client.get("/api/v1/meta")
    assert res.status_code == 200
    body = res.json()
    assert body["feature_names"] == [
        "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    ]
    assert len(body["feature_meta"]) == 6
    assert set(body["classes"]) == {"low risk", "mid risk", "high risk"}


def test_predict_ok(client):
    res = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert body["risk_level"] in {"low risk", "mid risk", "high risk"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert "X-Process-Time-ms" in res.headers


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Age"}
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422


def test_predict_out_of_range_returns_422(client):
    payload = {**VALID_PAYLOAD, "Age": 5000}
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422


def test_root_serves_frontend(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Maternal Health Risk" in res.text
