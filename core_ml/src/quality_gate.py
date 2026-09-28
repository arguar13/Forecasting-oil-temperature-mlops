"""Quality Gate para el Model Registry de MLflow.

Promueve la última versión registrada de un modelo al alias `production`
SOLO si cumple dos condiciones:

1. Le gana a la línea base ingenua (persistencia: repetir la última lectura)
   sobre el mismo split de test -- también la primera versión. Un modelo que
   no mejora a "no hacer nada" no justifica su costo de operación.
2. Su métrica de test mejora (o iguala) a la versión actualmente en
   `production`, y ambas son comparables (mismo dataset, seq_len y pred_len
   -- ver COMPARABILITY_PARAMS).

Si no las cumple, el script termina con código de salida distinto de cero --
en CI eso bloquea el resto del pipeline (deploy), así que un modelo peor
nunca reemplaza al que está sirviendo tráfico real.

Usa el sistema de "aliases" del Model Registry (no el de `stages`,
deprecado desde MLflow 2.9) como fuente de verdad de "cuál versión sirve
producción" -- la misma referencia que `api/main.py` y `batch_inference.py`
resuelven en runtime.
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
# empujó ahí. test nunca influyó en nada de eso.
DEFAULT_METRIC_KEY = "final_test_mse"  # menor es mejor (MSE de test, en °C^2)
# train.py loguea, para cada métrica de test, la misma métrica de la línea
# base de persistencia con este sufijo (final_test_mse_persistence).
BASELINE_METRIC_SUFFIX = "_persistence"


def _latest_version(client: MlflowClient, model_name: str) -> ModelVersion:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"No hay ninguna versión registrada para el modelo '{model_name}'.")
    return max(versions, key=lambda v: int(v.version))


# Params de train.py que definen QUÉ mide la métrica de test: dos versiones
# solo son comparables si coinciden en todos. Un MSE sobre el split de test
# del dataset toy (1000 filas) no dice nada frente a uno sobre ETTh1
# completo, ni un MSE a 1 paso frente a uno a 48 pasos.
COMPARABILITY_PARAMS = ("dataset", "seq_len", "pred_len")
SMOKE_TEST_DATASET = "toy"


def _metric_for_version(
    client: MlflowClient, version: ModelVersion, metric_key: str
) -> tuple[float | None, dict[str, str]]:
    """Devuelve (métrica, params del run) de la versión."""
    if version.run_id is None:
        raise SystemExit(f"La versión {version.version} no tiene un run_id asociado.")
    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric_key)
    return (float(value) if value is not None else None), dict(run.data.params)


def _current_production_version(client: MlflowClient, model_name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException:
        return None


def run_quality_gate(model_name: str, metric_key: str = DEFAULT_METRIC_KEY) -> bool:
    """Devuelve True si la versión candidata fue promovida a producción."""
    client = MlflowClient()

    candidate = _latest_version(client, model_name)
    candidate_metric, candidate_params = _metric_for_version(client, candidate, metric_key)
    if candidate_metric is None:
        raise SystemExit(
            f"La versión candidata v{candidate.version} de '{model_name}' no tiene "
            f"la métrica '{metric_key}' logueada; no se puede evaluar el quality gate."
        )

    # Vara mínima, antes de mirar producción: un candidato entrenado con datos
    # reales tiene que ganarle a la persistencia en su propio split de test.
    # El toy queda exento: su test (~100 ventanas) existe para ejercitar la
    # plomería del pipeline, no para medir skill, y nunca puede reemplazar a
    # un modelo real (ver más abajo).
    if candidate_params.get("dataset") != SMOKE_TEST_DATASET:
        baseline_key = metric_key + BASELINE_METRIC_SUFFIX
        baseline_metric, _ = _metric_for_version(client, candidate, baseline_key)
        if baseline_metric is None:
            raise SystemExit(
                f"La versión candidata v{candidate.version} de '{model_name}' no tiene "
                f"la métrica '{baseline_key}' logueada; no se puede comparar contra la "
                "línea base."
            )
        if candidate_metric >= baseline_metric:
            logger.error(
                "quality_gate_rejected_not_better_than_persistence",
                model_name=model_name,
                candidate_version=candidate.version,
                candidate_metric=candidate_metric,
                persistence_metric=baseline_metric,
                metric_key=metric_key,
            )
            return False

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

    if candidate.version == production.version:
        logger.info("quality_gate_noop_already_production", version=candidate.version)
        return True

    production_metric, production_params = _metric_for_version(client, production, metric_key)
    logger.info(
        "quality_gate_evaluating",
        candidate_version=candidate.version,
        candidate_metric=candidate_metric,
        production_version=production.version,
        production_metric=production_metric,
        metric_key=metric_key,
    )

    candidate_dataset = candidate_params.get("dataset")
    production_dataset = production_params.get("dataset")
    if candidate_dataset == SMOKE_TEST_DATASET and production_dataset != SMOKE_TEST_DATASET:
        # Un smoke test nunca reemplaza a un modelo entrenado con datos reales,
        # por buena que parezca su métrica (se mide sobre otro split de test).
        logger.error(
            "quality_gate_rejected_smoke_test_candidate",
            candidate_version=candidate.version,
            production_version=production.version,
            production_dataset=production_dataset,
        )
        return False
    if production_dataset == SMOKE_TEST_DATASET and candidate_dataset != SMOKE_TEST_DATASET:
        # Caso inverso: producción solo tiene un modelo de humo; cualquier
        # modelo entrenado con datos reales pasa a ser la nueva línea base.
        return _promote(
            client,
            model_name,
            candidate,
            production,
            metric_key,
            candidate_metric,
            production_metric,
        )

    mismatched = {
        key: (candidate_params.get(key), production_params.get(key))
        for key in COMPARABILITY_PARAMS
        if candidate_params.get(key) != production_params.get(key)
    }
    if mismatched:
        logger.error(
            "quality_gate_rejected_not_comparable",
            candidate_version=candidate.version,
            production_version=production.version,
            mismatched_params=mismatched,
            hint="Promover a mano si el cambio de configuración es intencional.",
        )
        return False

    if production_metric is not None and candidate_metric > production_metric:
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

    return _promote(
        client, model_name, candidate, production, metric_key, candidate_metric, production_metric
    )


def _promote(
    client: MlflowClient,
    model_name: str,
    candidate: ModelVersion,
    production: ModelVersion,
    metric_key: str,
    candidate_metric: float,
    production_metric: float | None,
) -> bool:
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
