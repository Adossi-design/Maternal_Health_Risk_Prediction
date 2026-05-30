"""Shared pytest fixtures.

Tests are hermetic: instead of downloading the UCI dataset, we fit a small
pipeline on synthetic data that respects the real feature contract and class
labels, persist it in the artifact format, and point the service/app at it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maternal_risk.core import FEATURE_NAMES

# Alphabetical, matching sklearn LabelEncoder on the real targets.
CLASSES = ["high risk", "low risk", "mid risk"]


def _synthetic_artifact() -> dict:
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    # Make the label loosely depend on the first feature so the model learns.
    y = np.clip((X[:, 0] + rng.normal(scale=0.3, size=n) + 1.5).astype(int), 0, 2)

    pipeline = Pipeline(
        [("scaler", StandardScaler()),
         ("rf", RandomForestClassifier(n_estimators=25, random_state=0))]
    )
    pipeline.fit(X, y)

    return {
        "schema_version": 2,
        "package_version": "test",
        "pipeline": pipeline,
        "classes": CLASSES,
        "feature_names": list(FEATURE_NAMES),
        "feature_importances": dict(
            zip(FEATURE_NAMES, pipeline.named_steps["rf"].feature_importances_)
        ),
        "metrics": {"accuracy": 0.9, "macro_f1": 0.9},
        "model_card": {"name": "test"},
    }


@pytest.fixture()
def artifact_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.joblib"
    joblib.dump(_synthetic_artifact(), path)
    return path


@pytest.fixture()
def client(artifact_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A TestClient whose app loads the synthetic artifact via settings."""
    from fastapi.testclient import TestClient

    from maternal_risk.core import settings

    monkeypatch.setattr(settings, "models_dir", artifact_path.parent)
    monkeypatch.setattr(settings, "model_filename", artifact_path.name)

    from maternal_risk.api import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
