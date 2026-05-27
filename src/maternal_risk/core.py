"""Core domain logic: configuration, feature contract, data access and model serving.

Everything the application needs except training and the web layer lives here:

* logging setup
* :data:`settings` — centralised, environment-overridable configuration
* the feature contract (:data:`FEATURE_SPECS`) — the single source of truth for
  feature names, order, units, plausible bounds and healthy reference ranges
* :func:`load_dataset` — dataset access with a content signature
* :class:`ModelService` — loads a trained artifact, validates it and predicts
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Initialise root logging once with a consistent format (idempotent)."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name)


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]  # .../Maternal_Health_Risk_Prediction


class Settings(BaseSettings):
    """Application settings, overridable via ``MRISK_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MRISK_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    # Reproducibility
    random_seed: int = 42

    # Data
    uci_dataset_id: int = 863  # UCI "Maternal Health Risk" dataset

    # Train / evaluation
    test_size: float = 0.15
    validation_size: float = 0.15  # fraction of the full dataset
    cv_folds: int = 5
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    rf_class_weight: str = "balanced"

    # Artifacts
    models_dir: Path = PROJECT_ROOT / "models"
    model_filename: str = "maternal_risk_model.joblib"
    model_card_filename: str = "model_card.json"

    # Frontend
    static_dir: Path = PROJECT_ROOT / "static"

    # API server
    api_title: str = "Maternal Health Risk Prediction API"
    api_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def model_card_path(self) -> Path:
        return self.models_dir / self.model_card_filename


settings = Settings()
"""Process-wide settings singleton."""

# ---------------------------------------------------------------------------
# Feature contract (single source of truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Specification for a single model input feature."""

    name: str  # dataset column name (exact)
    label: str  # human-readable label for the UI
    unit: str  # measurement unit
    minimum: float  # plausible lower bound (validation + UI)
    maximum: float  # plausible upper bound
    default: float  # sensible default / placeholder for the UI
    # Healthy reference range. A reading outside [normal_low, normal_high] is
    # flagged as "off" in the result panel. General adult reference values, not
    # pregnancy-specific clinical thresholds.
    normal_low: float
    normal_high: float
    integer: bool = False  # whether the field is conceptually integer-valued

    def as_meta(self) -> dict:
        """Serialisable form for the ``/meta`` endpoint and frontend."""
        d = asdict(self)
        return {
            "name": d["name"],
            "label": d["label"],
            "unit": d["unit"],
            "min": d["minimum"],
            "max": d["maximum"],
            "default": d["default"],
            "normal_low": d["normal_low"],
            "normal_high": d["normal_high"],
            "integer": d["integer"],
        }


# Order matters: this is the exact column order the model expects.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("Age", "Age", "years", 10, 70, 30, normal_low=18, normal_high=35, integer=True),
    FeatureSpec("SystolicBP", "Systolic BP", "mmHg", 70, 180, 120,
                normal_low=90, normal_high=120, integer=True),
    FeatureSpec("DiastolicBP", "Diastolic BP", "mmHg", 40, 120, 80,
                normal_low=60, normal_high=80, integer=True),
    FeatureSpec("BS", "Blood Sugar", "mmol/L", 4.0, 20.0, 7.0, normal_low=4.0, normal_high=7.8),
    FeatureSpec("BodyTemp", "Body Temperature", "°F", 95.0, 105.0, 98.0,
                normal_low=97.0, normal_high=99.0),
    FeatureSpec("HeartRate", "Heart Rate", "bpm", 40, 130, 76,
                normal_low=60, normal_high=100, integer=True),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

# Canonical display ordering (the dataset's encoded order is alphabetical).
RISK_DISPLAY_ORDER: tuple[str, ...] = ("low risk", "mid risk", "high risk")


def feature_meta() -> list[dict]:
    """Return the serialisable feature metadata, in model order."""
    return [spec.as_meta() for spec in FEATURE_SPECS]


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataset:
    """Loaded, validated dataset."""

    features: pd.DataFrame
    target: pd.Series
    signature: str  # sha256 over feature+target content

    @property
    def n_samples(self) -> int:
        return len(self.features)


def _content_signature(features: pd.DataFrame, target: pd.Series) -> str:
    """Deterministic hash of dataset contents, for provenance."""
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(features, index=True).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(target, index=True).values.tobytes())
    return hasher.hexdigest()


def load_dataset(dataset_id: int | None = None) -> Dataset:
    """Fetch the dataset and validate it against the feature contract."""
    dataset_id = dataset_id if dataset_id is not None else settings.uci_dataset_id
    logger.info("Fetching UCI dataset id=%d", dataset_id)
    raw = fetch_ucirepo(id=dataset_id)

    features: pd.DataFrame = raw.data.features
    target: pd.Series = raw.data.targets.iloc[:, 0]

    columns = tuple(features.columns)
    if columns != FEATURE_NAMES:
        raise ValueError(
            f"Dataset columns {columns} do not match the feature contract "
            f"{FEATURE_NAMES}. Update FEATURE_SPECS if the source changed."
        )

    n_missing = int(features.isnull().sum().sum())
    if n_missing:
        logger.warning("Dataset contains %d missing feature values", n_missing)

    signature = _content_signature(features, target)
    logger.info("Loaded %d samples (signature %s…)", len(features), signature[:12])
    return Dataset(features=features, target=target, signature=signature)


# ---------------------------------------------------------------------------
# Model serving
# ---------------------------------------------------------------------------


class ModelNotLoadedError(RuntimeError):
    """Raised when an artifact is missing or fails validation."""


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """A single prediction with class probabilities."""

    risk_level: str
    confidence: float
    probabilities: dict[str, float]


class ModelService:
    """Loads a trained artifact and serves predictions."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or settings.model_path
        self._pipeline: Any = None
        self._classes: list[str] = []
        self._feature_names: list[str] = []
        self._metrics: dict[str, Any] = {}
        self._feature_importances: dict[str, float] = {}
        self._model_card: dict[str, Any] = {}

    def load(self) -> ModelService:
        """Load and validate the artifact. Returns self for chaining."""
        if not self._model_path.exists():
            raise ModelNotLoadedError(
                f"Model artifact not found at {self._model_path}. "
                "Train it first: `python -m maternal_risk.train`."
            )
        logger.info("Loading model artifact from %s", self._model_path)
        artifact = joblib.load(self._model_path)
        self._validate(artifact)

        self._pipeline = artifact["pipeline"]
        self._classes = list(artifact["classes"])
        self._feature_names = list(artifact["feature_names"])
        self._metrics = dict(artifact.get("metrics", {}))
        self._feature_importances = dict(artifact.get("feature_importances", {}))
        self._model_card = dict(artifact.get("model_card", {}))
        logger.info("Model loaded (classes=%s)", self._classes)
        return self

    @staticmethod
    def _validate(artifact: dict[str, Any]) -> None:
        """Guard against silently loading an incompatible artifact."""
        required = {"pipeline", "classes", "feature_names"}
        missing = required - artifact.keys()
        if missing:
            raise ModelNotLoadedError(f"Artifact is missing required keys: {sorted(missing)}")
        if tuple(artifact["feature_names"]) != FEATURE_NAMES:
            raise ModelNotLoadedError(
                "Artifact feature order does not match the current feature contract. "
                f"Artifact={tuple(artifact['feature_names'])} expected={FEATURE_NAMES}. "
                "Retrain the model."
            )
        if not hasattr(artifact["pipeline"], "predict_proba"):
            raise ModelNotLoadedError("Loaded pipeline does not support predict_proba.")

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    def predict(self, features: dict[str, float]) -> PredictionResult:
        """Predict the risk level for one patient."""
        if not self.is_ready:
            raise ModelNotLoadedError("Model is not loaded. Call load() first.")

        missing = [name for name in self._feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = np.array([[float(features[name]) for name in self._feature_names]], dtype=float)
        proba = self._pipeline.predict_proba(row)[0]
        top = int(np.argmax(proba))
        return PredictionResult(
            risk_level=self._classes[top],
            confidence=float(proba[top]),
            probabilities={cls: float(p) for cls, p in zip(self._classes, proba, strict=True)},
        )

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    @property
    def feature_importances(self) -> dict[str, float]:
        return dict(self._feature_importances)

    @property
    def model_card(self) -> dict[str, Any]:
        return dict(self._model_card)
