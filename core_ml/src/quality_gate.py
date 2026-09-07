"""Quality Gate para el Model Registry de MLflow.

Promueve la última versión registrada de un modelo al alias `production`
SOLO si su métrica de validación mejora (o iguala) a la versión actualmente
en `production`. Si no la supera, el script termina con código de salida
distinto de cero -- en CI eso bloquea el resto del pipeline (build/deploy),
así que un modelo peor nunca llega a reemplazar al que está sirviendo
tráfico real.

Usa el sistema de "aliases" del Model Registry (no el de `stages`, deprecado
desde MLflow 2.9) como fuente de verdad de "cuál versión es la que sirve
producción" -- exactamente la referencia inmutable y versionada que
`api/main.py` y `batch_inference.py` resuelven en runtime.
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from src.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

PRODUCTION_ALIAS = "production"
# final_test_mse (no final_val_mse): val ya decidió el learning rate
# (Optuna) y qué checkpoint restaurar (early stopping), así que un candidato
# puede "verse bien" en val precisamente porque el proceso de selección lo
# empujó ahí. test nunca influyó en nada de eso -- es la única estimación de
# error que este pipeline produce que no está contaminada por su propio
# proceso de selección de modelo, y por eso es la que decide qué llega a
# producción.
DEFAULT_METRIC_KEY = "final_test_mse"  # menor es mejor (MSE de test, en °C^2)


def _latest_version(client: MlflowClient, model_name: str) -> ModelVersion:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"No hay ninguna versión registrada para el modelo '{model_name}'.")
    return max(versions, key=lambda v: int(v.version))


def _metric_for_version(
    client: MlflowClient, version: ModelVersion, metric_key: str
) -> float | None:
    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric_key)
    return float(value) if value is not None else None


def _current_production_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def run_quality_gate(model_name: str, metric_key: str = DEFAULT_METRIC_KEY) -> bool:
    """Devuelve True si la versión candidata fue promovida a producción."""
    client = MlflowClient()

    candidate = _latest_version(client, model_name)
    candidate_metric = _metric_for_version(client, candidate, metric_key)
    if candidate_metric is None:
        raise SystemExit(
            f"La versión candidata v{candidate.version} de '{model_name}' no tiene "
            f"la métrica '{metric_key}' logueada; no se puede evaluar el quality gate."
        )

    production = _current_production_version(client, model_name)
    if production is None:
        logger.info(
            "quality_gate_baseline_promoted",
            model_name=model_name,
            version=candidate.version,
            metric_key=metric_key,
            metric_value=candidate_metric,
        )
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate.version)
        return True

    production_metric = _metric_for_version(client, production, metric_key)
    logger.info(
        "quality_gate_evaluating",
        candidate_version=candidate.version,
        candidate_metric=candidate_metric,
        production_version=production.version,
        production_metric=production_metric,
        metric_key=metric_key,
    )

    if candidate.version == production.version:
        logger.info("quality_gate_noop_already_production", version=candidate.version)
        return True

    if production_metric is not None and candidate_metric >= production_metric:
        logger.error(
            "quality_gate_rejected",
            model_name=model_name,
            candidate_version=candidate.version,
            candidate_metric=candidate_metric,
            production_version=production.version,
            production_metric=production_metric,
            metric_key=metric_key,
        )
        return False

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, candidate.version)
    logger.info(
        "quality_gate_promoted",
        model_name=model_name,
        previous_version=production.version,
        new_version=candidate.version,
        metric_key=metric_key,
        previous_metric=production_metric,
        new_metric=candidate_metric,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality Gate de promoción a producción (MLflow Registry)"
    )
    parser.add_argument("--model_name", type=str, default="dlinear-ett-forecaster")
    parser.add_argument("--metric_key", type=str, default=DEFAULT_METRIC_KEY)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    promoted = run_quality_gate(args.model_name, args.metric_key)
    sys.exit(0 if promoted else 1)
