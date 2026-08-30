import os
import time
from contextlib import asynccontextmanager
from typing import cast

import mlflow
import pandas as pd
import pybreaker
from fastapi import FastAPI, HTTPException
from mlflow.pyfunc import PyFuncModel, load_model
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from api.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------
# CONFIGURACIÓN DEL MODEL REGISTRY (MLflow)
# ---------------------------------------------------------
# El modelo NUNCA se lee como un .pth/.pkl suelto: se resuelve por nombre +
# alias contra el MLflow Model Registry, una referencia inmutable y
# versionada que ya incluye la red DLinear y sus scalers empaquetados
# (ver core_ml/src/mlflow_utils.py::DLinearForecaster).
MODEL_NAME = os.getenv("MODEL_NAME", "dlinear-ett-forecaster")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
SEQ_LEN = 48

model: PyFuncModel | None = None

# Circuit breaker: si el Model Registry falla 5 veces seguidas, deja de
# intentar durante 60s en vez de seguir golpeándolo (evita un loop de
# reintentos infinito contra una dependencia caída -- Regla Absoluta Final).
# Cada intento individual, mientras el circuito está cerrado, ya trae su
# propio retry con backoff exponencial acotado (máx. 3 intentos).
MODEL_REGISTRY_BREAKER = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="mlflow-model-registry",
)


@MODEL_REGISTRY_BREAKER
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _load_model_resilient(model_uri: str) -> PyFuncModel:
    return cast(PyFuncModel, load_model(model_uri))


# ---------------------------------------------------------
# FASTAPI APP & ENDPOINTS
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Se ejecuta antes de recibir peticiones
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info("model_load_started", model_uri=model_uri)
    try:
        model = _load_model_resilient(model_uri)
        logger.info("model_load_succeeded", model_uri=model_uri)
    except Exception:
        # No tumba el proceso: un pod recién desplegado antes del primer
        # `quality_gate.py` (o mientras el Model Registry está temporalmente
        # inalcanzable) debe seguir vivo -- /health responde 200 (liveness),
        # /predict responde 503 (not ready) hasta que el modelo cargue.
        # Requiere un reinicio/rollout del pod para reintentar la carga.
        logger.error("model_load_failed", model_uri=model_uri, exc_info=True)
        model = None

    yield  # Aquí corre la aplicación

    # Teardown: Liberar recursos si es necesario
    model = None


app = FastAPI(
    title="ETTh1 DLinear Time Series Prediction API",
    description="API de inferencia productiva en EKS para predicción multivariada.",
    version="1.0.0",
    lifespan=lifespan,
)


class FeatureReading(BaseModel):
    """Contrato de una lectura horaria de sensores ETT + componentes
    temporales derivados. Rangos alineados con
    core_ml/src/data_contracts.py::ETTDatasetContract -- ese módulo es la
    fuente de verdad; aquí solo se refleja para fallar rápido (422) en el
    borde de la API, antes de invocar al modelo.
    """

    HUFL: float = Field(ge=-40.0, le=40.0)
    HULL: float = Field(ge=-15.0, le=20.0)
    MUFL: float = Field(ge=-40.0, le=30.0)
    MULL: float = Field(ge=-15.0, le=15.0)
    LUFL: float = Field(ge=-10.0, le=15.0)
    LULL: float = Field(ge=-5.0, le=8.0)
    OT: float = Field(ge=-15.0, le=60.0)
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    hour: int = Field(ge=0, le=23)


class PredictionRequest(BaseModel):
    features: list[FeatureReading] = Field(
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"Exactamente {SEQ_LEN} lecturas horarias consecutivas (seq_len del modelo).",
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Distinto de /health (liveness): un pod puede estar vivo pero sin
    modelo cargado todavía (ver lifespan). Un readinessProbe apuntado aquí
    evita que el Service le mande tráfico a un pod que solo respondería 503.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está inicializado.")
    return {"status": "ready"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        logger.warning("predict_rejected_model_not_ready")
        raise HTTPException(status_code=503, detail="El modelo aún no está inicializado.")

    start = time.monotonic()
    try:
        # El orden de columnas debe coincidir con feature_cols en
        # data_processing.py: HUFL,HULL,MUFL,MULL,LUFL,LULL,OT,month,day,hour.
        # `model_dump()` preserva el orden de declaración de FeatureReading,
        # que ya está alineado con ese orden.
        df_input = pd.DataFrame([reading.model_dump() for reading in request.features])
        prediction = model.predict(df_input)
        # prediction tiene forma (1, pred_len): una sola ventana de entrada,
        # pred_len valores futuros (1 en el caso histórico de un solo paso,
        # >1 para un horizonte multi-step -- ver
        # core_ml/src/model_architecture.py::DLinear).
        predictions = [float(v) for v in prediction[0]]

        logger.info(
            "predict_succeeded",
            model_version=f"{MODEL_NAME}@{MODEL_ALIAS}",
            horizon=len(predictions),
            duration_ms=round((time.monotonic() - start) * 1000, 2),
        )
        return {
            "model_version": f"{MODEL_NAME}@{MODEL_ALIAS}",
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(
            "predict_failed",
            error=str(e),
            duration_ms=round((time.monotonic() - start) * 1000, 2),
        )
        raise HTTPException(status_code=400, detail=f"Error en la inferencia: {str(e)}") from e
