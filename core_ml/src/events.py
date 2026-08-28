"""Publicación de eventos de dominio a Kafka.

No es una dependencia crítica del pipeline: si el broker no está disponible,
se loguea un warning y el flujo que originó el evento (p. ej. batch
inference) continúa sin verse afectado. Estos eventos son observabilidad/
auditoría ("¿cuándo corrió el último batch y con qué modelo?"), nunca una
ruta obligatoria del negocio.
"""

from __future__ import annotations

import json
import os

# datetime.UTC solo existe desde Python 3.11 -- el runtime real del
# proyecto es 3.10 (ver pyproject.toml y la imagen `python:3.10` de CI);
# probarlo con un venv local en 3.12 (donde datetime.UTC sí existe) no
# lo detecta, timezone.utc es el equivalente compatible con 3.10+.
from datetime import datetime, timezone
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_BOOTSTRAP_SERVERS = "localhost:29092"
BATCH_INFERENCE_TOPIC = "batch-inference-events"


def publish_event(topic: str, payload: dict[str, Any]) -> bool:
    """Publica `payload` como JSON en `topic`. Devuelve True si tuvo éxito.

    Falla de forma silenciosa (solo loguea): publicar un evento de
    observabilidad nunca debe tumbar un pipeline de datos/ML.
    """
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS)
    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=5000,
            max_block_ms=5000,
        )
        try:
            producer.send(topic, value=payload)
            producer.flush(timeout=5)
        finally:
            producer.close(timeout=5)
        logger.info("event_published", topic=topic, payload=payload)
        return True
    except Exception as exc:
        logger.warning("event_publish_failed", topic=topic, error=str(exc))
        return False


def publish_batch_inference_completed(
    *,
    bucket: str,
    output_key: str,
    n_predictions: int,
    model_name: str,
    model_alias: str,
) -> bool:
    """Notifica que un ciclo de batch inference terminó exitosamente."""
    payload = {
        "event": "batch_inference_completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "output_key": output_key,
        "n_predictions": n_predictions,
        "model_name": model_name,
        "model_alias": model_alias,
    }
    return publish_event(BATCH_INFERENCE_TOPIC, payload)
