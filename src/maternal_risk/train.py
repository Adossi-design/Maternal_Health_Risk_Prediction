"""Training pipeline.

Trains the production model (a scaled Random Forest) with a reproducible
train/validation/test split, reports cross-validated and held-out metrics, and
persists a self-describing artifact plus a human-readable model card.

The artifact bundles everything the serving layer needs — the fitted pipeline,
class labels, the feature contract and rich provenance metadata — so the API
never has to reconstruct preprocessing or guess feature order.

CLI:  ``python -m maternal_risk.train``  /  ``maternal-risk-train``
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from maternal_risk import __version__
from maternal_risk.core import (
    FEATURE_NAMES,
    Dataset,
    configure_logging,
    get_logger,
    load_dataset,
    settings,
)

logger = get_logger(__name__)

ARTIFACT_SCHEMA_VERSION = 2


@dataclass(slots=True)
class TrainingResult:
    """Outcome of a training run."""

    pipeline: Pipeline
    classes: list[str]
    metrics: dict[str, Any]
    model_card: dict[str, Any]
    feature_importances: dict[str, float] = field(default_factory=dict)


def build_pipeline() -> Pipeline:
    """Construct the inference pipeline (scaling + Random Forest)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=settings.rf_n_estimators,
                    max_depth=settings.rf_max_depth,
                    class_weight=settings.rf_class_weight,
                    random_state=settings.random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _split(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """Stratified train/validation/test split driven by config."""
    seed = settings.random_seed
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=seed, stratify=y
    )
    # validation_size is expressed as a fraction of the *full* dataset.
    val_fraction = settings.validation_size / (1.0 - settings.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=seed, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    """Compute held-out metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y, y_pred, target_names=classes, output_dict=True, zero_division=0
        ),
    }


def train(dataset: Dataset | None = None) -> TrainingResult:
    """Run the full training + evaluation pipeline and return the result."""
    configure_logging(settings.log_level)
    dataset = dataset or load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(dataset.target.values)
    classes = [str(c) for c in encoder.classes_]
    X = dataset.features.values.astype(float)
    logger.info("Classes: %s", classes)

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    logger.info(
        "Split sizes — train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test)
    )

    pipeline = build_pipeline()

    # Cross-validation on the training set for an honest variance estimate.
    cv = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    logger.info("CV macro-F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    pipeline.fit(X_train, y_train)
    val_metrics = _evaluate(pipeline, X_val, y_val, classes)
    test_metrics = _evaluate(pipeline, X_test, y_test, classes)
    logger.info(
        "Validation: acc=%.4f f1=%.4f | Test: acc=%.4f f1=%.4f",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
    )

    importances = dict(
        sorted(
            zip(FEATURE_NAMES, pipeline.named_steps["rf"].feature_importances_, strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    metrics = {
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "validation": val_metrics,
        "test": test_metrics,
        # Convenience top-level values for the UI.
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
    }

    model_card = {
        "name": "Maternal Health Risk Classifier",
        "package_version": __version__,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": "Pipeline(StandardScaler -> RandomForestClassifier)",
        "hyperparameters": {
            "n_estimators": settings.rf_n_estimators,
            "max_depth": settings.rf_max_depth,
            "class_weight": settings.rf_class_weight,
            "random_state": settings.random_seed,
        },
        "features": list(FEATURE_NAMES),
        "classes": classes,
        "dataset": {
            "source": f"UCI id={settings.uci_dataset_id}",
            "n_samples": dataset.n_samples,
            "signature_sha256": dataset.signature,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "metrics": metrics,
        "feature_importances": {k: float(v) for k, v in importances.items()},
        "intended_use": (
            "Educational decision-support demonstration. Not a medical device; "
            "not for clinical use without validation and regulatory clearance."
        ),
    }

    return TrainingResult(
        pipeline=pipeline,
        classes=classes,
        metrics=metrics,
        model_card=model_card,
        feature_importances=model_card["feature_importances"],
    )


def save_artifact(result: TrainingResult) -> None:
    """Persist the model artifact and the model card to ``settings.models_dir``."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "package_version": __version__,
        "pipeline": result.pipeline,
        "classes": result.classes,
        "feature_names": list(FEATURE_NAMES),
        "feature_importances": result.feature_importances,
        "metrics": result.metrics,
        "model_card": result.model_card,
    }
    joblib.dump(artifact, settings.model_path)
    settings.model_card_path.write_text(json.dumps(result.model_card, indent=2), encoding="utf-8")

    logger.info("Saved artifact   -> %s", settings.model_path)
    logger.info("Saved model card -> %s", settings.model_card_path)


def main() -> None:
    """CLI entry point: train and persist the production artifact."""
    configure_logging(settings.log_level)
    result = train()
    save_artifact(result)
    print(
        f"\nDone. Test accuracy={result.metrics['accuracy']:.4f} "
        f"macro-F1={result.metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
