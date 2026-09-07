"""Quality Gate para el Model Registry de MLflow.

Promueve la última versión registrada de un modelo al alias `production`
SOLO si su métrica de validación mejora (o iguala, dentro de `tolerance`) a
la versión actualmente en `production`. Si no la supera, el script termina
con código de salida distinto de cero -- en CI eso bloquea el resto del
pipeline (build/deploy), así que un modelo peor nunca llega a reemplazar al
que está sirviendo tráfico real. Esta es la comparación canary/shadow de
este proyecto: no un re-scoring en vivo contra tráfico replicado (eso
necesitaría infraestructura que el volumen de este proyecto no justifica,
el mismo trade-off ya documentado en events.py para Kafka), sino la
métrica held-out ya logueada de cada versión -- el mismo estándar al que
core_ml/src/monitoring/stream_consumer.py ya somete al modelo en
producción (su CUSUM de concept drift compara contra este mismo
baseline_test_mae).

`tolerance` (--tolerance, default 0.0) permite que el candidato sea hasta
esa cantidad PEOR (MSE más alto) que producción y aún así promover -- un
número positivo pequeño, no cero, por la misma razón que
610-hotel-booking-mlops's promote_model.py acepta una pequeña regresión:
exigir una mejora estricta en cada reentrenamiento automático rechazaría un
modelo que recuperó la mayor parte, pero no toda, una regresión real.

Cada promoción etiqueta la nueva versión con `promoted_from_version` (la
que reemplazó, o "none" en el primer despliegue), que `--rollback` lee
para mover el alias de vuelta exactamente un paso -- no un recorrido
completo del historial; deshacer más de una promoción sin que un humano
elija cuál versión anterior es segura es exactamente el tipo de autoridad
desatendida que core_ml/src/monitoring/mitigation.py's propio docstring
argumenta en contra de otorgarle a este sistema.

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


def _promote(
    client: MlflowClient, model_name: str, version: ModelVersion, previous_version: str | None
) -> None:
    client.set_model_version_tag(
        model_name, version.version, "promoted_from_version", previous_version or "none"
    )
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version.version)


def run_quality_gate(
    model_name: str, metric_key: str = DEFAULT_METRIC_KEY, tolerance: float = 0.0
) -> bool:
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
        _promote(client, model_name, candidate, previous_version=None)
        return True

    production_metric = _metric_for_version(client, production, metric_key)
    logger.info(
        "quality_gate_evaluating",
        candidate_version=candidate.version,
        candidate_metric=candidate_metric,
        production_version=production.version,
        production_metric=production_metric,
        metric_key=metric_key,
        tolerance=tolerance,
    )

    if candidate.version == production.version:
        logger.info("quality_gate_noop_already_production", version=candidate.version)
        return True

    if production_metric is not None and candidate_metric > production_metric + tolerance:
        logger.error(
            "quality_gate_rejected",
            model_name=model_name,
            candidate_version=candidate.version,
            candidate_metric=candidate_metric,
            production_version=production.version,
            production_metric=production_metric,
            metric_key=metric_key,
            tolerance=tolerance,
        )
        return False

    _promote(client, model_name, candidate, previous_version=production.version)
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


def rollback_production(model_name: str) -> str:
    """Mueve el alias `production` de vuelta exactamente una promoción."""
    client = MlflowClient()
    current = _current_production_version(client, model_name)
    if current is None:
        raise SystemExit(f"'{model_name}' no tiene ninguna versión aliasada '{PRODUCTION_ALIAS}'.")

    previous_version = current.tags.get("promoted_from_version")
    if not previous_version or previous_version == "none":
        raise SystemExit(
            f"La versión {current.version} (actualmente '{PRODUCTION_ALIAS}') no tiene "
            "una versión anterior registrada -- o nunca se promovió con este script, o es "
            "la primera versión jamás promovida. Nada a lo que hacer rollback."
        )

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, previous_version)
    logger.info(
        "quality_gate_rolled_back",
        model_name=model_name,
        from_version=current.version,
        to_version=previous_version,
    )
    return str(previous_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality Gate de promoción a producción (MLflow Registry)"
    )
    parser.add_argument("--model_name", type=str, default="dlinear-ett-forecaster")
    parser.add_argument("--metric_key", type=str, default=DEFAULT_METRIC_KEY)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Cuánto puede empeorar (MSE más alto) el candidato frente a producción y "
        "aún así promover. 0.0 exige mejora estricta o empate.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Mueve el alias 'production' de vuelta a la versión previamente promovida, "
        "ignorando --model_name's candidata actual.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    if args.rollback:
        rollback_production(args.model_name)
        sys.exit(0)

    promoted = run_quality_gate(args.model_name, args.metric_key, args.tolerance)
    sys.exit(0 if promoted else 1)
