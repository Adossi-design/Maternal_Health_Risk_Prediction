"""HTTP API: FastAPI application, request/response schemas and routes.

Exposes the model behind a small, versioned JSON API and serves the static
frontend:

    GET  /                -> single-page frontend
    GET  /api/v1/health   -> liveness/readiness + model status
    GET  /api/v1/meta     -> feature metadata, classes, importances, metrics
    POST /api/v1/predict  -> risk prediction for one patient
    GET  /docs            -> interactive OpenAPI docs (built-in)

Run:  python -m uvicorn maternal_risk.api:app   /   maternal-risk-serve
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model

from maternal_risk import __version__
from maternal_risk.core import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    ModelNotLoadedError,
    ModelService,
    feature_meta,
    get_logger,
    settings,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _build_patient_model() -> type[BaseModel]:
    """Generate the request model from the feature contract."""
    fields: dict[str, tuple] = {}
    for spec in FEATURE_SPECS:
        # Allow a margin beyond UI bounds so genuine outliers aren't rejected.
        lower = spec.minimum - abs(spec.minimum) * 0.5 - 1
        upper = spec.maximum + abs(spec.maximum) * 0.5 + 1
        fields[spec.name] = (
            float,
            Field(..., ge=lower, le=upper, description=f"{spec.label} ({spec.unit})"),
        )
    model = create_model("PatientFeatures", __base__=BaseModel, **fields)
    model.model_config = {
        "json_schema_extra": {"example": {s.name: s.default for s in FEATURE_SPECS}}
    }
    return model


PatientFeatures = _build_patient_model()


class PredictionResponse(BaseModel):
    risk_level: str = Field(..., description="Predicted risk class")
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    probabilities: dict[str, float] = Field(..., description="Probability per class")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetaResponse(BaseModel):
    feature_names: list[str]
    feature_meta: list[dict]
    classes: list[str]
    feature_importances: dict[str, float]
    metrics: dict


# ---------------------------------------------------------------------------
# Dependencies + routes
# ---------------------------------------------------------------------------


def get_model_service(request: Request) -> ModelService:
    """Return the loaded ModelService from application state."""
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_ready:
        raise ModelNotLoadedError("Model service is not initialised.")
    return service


router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Liveness/readiness probe."""
    return HealthResponse(status="ok", model_loaded=service.is_ready, version=__version__)


@router.get("/meta", response_model=MetaResponse)
def meta(service: ModelService = Depends(get_model_service)) -> MetaResponse:
    """Feature metadata, class labels, importances and model metrics."""
    return MetaResponse(
        feature_names=list(FEATURE_NAMES),
        feature_meta=feature_meta(),
        classes=service.classes,
        feature_importances=service.feature_importances,
        metrics=service.metrics,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    features: PatientFeatures,  # type: ignore[valid-type]
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Predict the maternal health risk level for one patient."""
    result = service.predict(features.model_dump())
    return PredictionResponse(
        risk_level=result.risk_level,
        confidence=result.confidence,
        probabilities=result.probabilities,
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; release references at shutdown."""
    service = ModelService()
    try:
        service.load()
    except ModelNotLoadedError:
        logger.exception("Model failed to load at startup")
    app.state.model_service = service
    logger.info("API startup complete (model_loaded=%s)", service.is_ready)
    yield
    app.state.model_service = None
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Predicts maternal health risk (low / mid / high) from six vital "
            "signs using a Random Forest classifier. For educational use only."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        logger.error("Model unavailable for %s: %s", request.url.path, exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(settings.static_dir / "index.html")

    if settings.static_dir.exists():
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    logger.info("Application created (version %s)", __version__)
    return app


app = create_app()


def run() -> None:
    """Console-script entry point: ``maternal-risk-serve``."""
    import uvicorn

    uvicorn.run(
        "maternal_risk.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
