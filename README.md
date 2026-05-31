# Maternal Health Risk Prediction

Production-style machine-learning service that predicts a maternal health
**risk level** (`low risk` / `mid risk` / `high risk`) from six vital signs.
A scaled Random Forest is trained on the
[UCI Maternal Health Risk dataset](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
and served through a layered **FastAPI** application with a lightweight
HTML/JS frontend.

The original exploratory analysis: EDA, classical models, and a neural network
implemented from scratch, remains in `Assignment1_AdossiFredWilliam (1).ipynb`.
This repository productionises the strongest model from that study.

| | |
|---|---|
| Estimator | `Pipeline(StandardScaler → RandomForestClassifier)` |
| Test accuracy | ~0.85 |
| Macro F1 | ~0.86 |
| Serving | FastAPI + Uvicorn, versioned `/api/v1` |

---

## Why this project matters

Maternal health remains one of the most urgent concerns in global healthcare. Every day many women lose their lives because of complications during pregnancy and childbirth, and a large share of those deaths are preventable when risk is recognised early. The real difficulty is that warning signs such as raised blood pressure, abnormal blood sugar, or an unusual heart rate often go unnoticed in places where doctors and specialists are few.

This project speaks directly to that problem. It relies only on simple vital signs that any clinic or community health worker can measure, and it turns those readings into a clear estimate of whether a mother faces low, mid, or high risk. A few routine numbers become an instant and understandable assessment, which helps a health worker judge who needs urgent care and who can be watched safely.

The benefit is largest in rural and poorly served regions, where access to obstetric expertise is scarce. A lightweight model that runs on an ordinary laptop or phone can bring a layer of decision support to the communities that need it most. Earlier identification leads to earlier referral, earlier treatment, and in the end more mothers and babies coming through pregnancy in good health.

For all of these reasons the project deserves serious attention. It confronts a real and measurable problem, it depends on data that is cheap and easy to gather, and it carries genuine potential to save lives at very low cost.

---

## Architecture

The backend is a small, installable package with a clear separation between
core domain logic, training, and the web layer.

```
src/maternal_risk/
├── core.py     # Config, feature contract, data access, ModelService
├── train.py    # Reproducible train/eval pipeline
└── api.py      # FastAPI app, request/response schemas, /api/v1 routes

tests/          # Hermetic pytest suite (model + API)
static/         # Frontend (index.html, style.css, app.js, favicon.svg)
models/         # Trained artifact + model_card.json (built by train.py)
```

### Design notes

- **One feature contract.** `core.py` defines feature order, ranges and units
  once; training asserts the dataset matches it, the API validates against it,
  and the model service refuses to load an artifact whose feature order drifts, 
  eliminating the most common train/serve skew bug.
- **Self-describing artifact.** Training persists the fitted pipeline *and* a
  `model_card.json` with metrics, hyper-parameters, dataset signature, and the
  library versions used, full provenance for every model file.
- **Application factory + DI.** `create_app()` has no import-time side effects;
  the model is loaded once in the lifespan handler and injected into routes,
  so tests can swap in a synthetic model trivially.

---

## Quickstart

```bash
# 1. Install (editable, with dev tooling)
pip install -e ".[dev]"         

# 2. Train - downloads the dataset, writes models/maternal_risk_model.joblib
python -m maternal_risk.train

# 3. Serve the API + frontend
python -m uvicorn maternal_risk.api:app --reload
```

Open **http://127.0.0.1:8000** for the UI and **/docs** for interactive API docs.

Common tasks are also wrapped in the `Makefile` (`make train`, `make serve`,
`make test`, `make lint`, `make docker`).

### Configuration

Every setting in `core.py` (`Settings`) can be overridden via environment
variables prefixed with `MRISK_`, e.g.:

```bash
MRISK_PORT=9000 MRISK_LOG_LEVEL=DEBUG python -m maternal_risk.api
```

---

## API

Base path: `/api/v1`

| Method | Path        | Description                                  |
|--------|-------------|----------------------------------------------|
| GET    | `/health`   | Liveness/readiness + model status            |
| GET    | `/meta`     | Feature metadata, classes, importances, metrics |
| POST   | `/predict`  | Risk prediction for one patient              |

### `POST /api/v1/predict`

```json
{
  "Age": 35, "SystolicBP": 140, "DiastolicBP": 90,
  "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 86
}
```

```json
{
  "risk_level": "high risk",
  "confidence": 0.87,
  "probabilities": { "low risk": 0.05, "mid risk": 0.08, "high risk": 0.87 }
}
```

Invalid or out-of-range input returns `422`; an unloaded model returns `503`.

---

## Testing

```bash
pytest          
```

The suite covers the model service (loading, validation, prediction
invariants) and the API (health, meta, predict, validation, frontend).

---

## Docker

```bash
docker build -t maternal-risk:latest .
docker run -p 8000:8000 maternal-risk:latest
```

The image installs the package, bundles the trained artifact and frontend, and
exposes a container `HEALTHCHECK` against `/api/v1/health`.

---

