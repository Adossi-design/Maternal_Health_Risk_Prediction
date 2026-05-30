"""Unit tests for the model serving layer."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from maternal_risk.core import FEATURE_NAMES, ModelNotLoadedError, ModelService

VALID_INPUT = {
    "Age": 35, "SystolicBP": 140, "DiastolicBP": 90,
    "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 86,
}


def test_load_missing_artifact_raises(tmp_path: Path):
    service = ModelService(model_path=tmp_path / "nope.joblib")
    with pytest.raises(ModelNotLoadedError):
        service.load()


def test_predict_before_load_raises(artifact_path: Path):
    service = ModelService(model_path=artifact_path)
    with pytest.raises(ModelNotLoadedError):
        service.predict(VALID_INPUT)


def test_predict_returns_valid_distribution(artifact_path: Path):
    service = ModelService(model_path=artifact_path).load()
    result = service.predict(VALID_INPUT)

    assert result.risk_level in service.classes
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == set(service.classes)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    # Confidence is the probability of the predicted class.
    assert result.confidence == pytest.approx(result.probabilities[result.risk_level])


def test_predict_missing_feature_raises(artifact_path: Path):
    service = ModelService(model_path=artifact_path).load()
    incomplete = {k: v for k, v in VALID_INPUT.items() if k != "Age"}
    with pytest.raises(ValueError, match="Missing required features"):
        service.predict(incomplete)


def test_artifact_feature_mismatch_rejected(tmp_path: Path, artifact_path: Path):
    artifact = joblib.load(artifact_path)
    artifact["feature_names"] = list(reversed(FEATURE_NAMES))
    bad = tmp_path / "bad.joblib"
    joblib.dump(artifact, bad)

    with pytest.raises(ModelNotLoadedError, match="feature order"):
        ModelService(model_path=bad).load()
