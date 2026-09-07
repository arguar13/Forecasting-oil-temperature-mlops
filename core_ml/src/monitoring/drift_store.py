"""1-step-ahead prediction log, and its reconciliation against real readings.

There is no equivalent of this in 612 before Section 4: the FastAPI service
answers a prediction and forgets it (no correlation id, no persisted
record), and the daily batch job scores a static S3 CSV and writes the
output back to S3 - neither leaves behind a queryable "what did the model
predict for timestamp T" that a concept-drift check could compare against
real readings later. stream_consumer.py is what now produces that record:
every time it has accumulated a full 48-hour window, it asks the served
model for the *next* hour's OT and inserts the row here; every new sensor
reading it receives is also a chance to reconcile an older row whose target
hour has finally arrived.

Deliberately 1-step-ahead only, not the full 48-hour horizon
batch_inference.py also produces: a 1-hour-ahead prediction reconciles
within an hour of being made, which is what makes it useful as a fast,
continuously-updating concept-drift signal - a 48-hour-ahead prediction
would not reconcile until two days later, adding a detection lag this
system does not need to accept. The 48-hour horizon is already audited on
its own timescale by the batch job.

Same table-ownership pattern 610-hotel-booking-mlops's api/monitoring.py
uses for its own prediction log: no migration tool for a schema this size,
so the DDL is idempotent and applied by whichever process uses it first -
here, that is stream_consumer.py itself, since nothing else writes to this
table.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

SCHEMA = "monitoring"

SCHEMA_DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.predictions (
    prediction_id    UUID PRIMARY KEY,
    predicted_at     TIMESTAMPTZ      NOT NULL,
    target_timestamp TIMESTAMPTZ      NOT NULL,
    model_version    TEXT             NOT NULL,
    predicted_ot     DOUBLE PRECISION NOT NULL,
    actual_ot        DOUBLE PRECISION,
    reconciled_at    TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_target_timestamp
    ON {SCHEMA}.predictions (target_timestamp);
"""  # SCHEMA is a module constant, never user input - no dynamic SQL here

_INSERT_PREDICTION = f"""
INSERT INTO {SCHEMA}.predictions
    (prediction_id, predicted_at, target_timestamp, model_version, predicted_ot)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (target_timestamp) DO NOTHING
"""  # nosec B608

# UPDATE ... RETURNING: one round trip, atomic. `actual_ot IS NULL` in the
# WHERE clause is what makes this idempotent against a replayed reading -
# a target_timestamp already reconciled is left untouched rather than
# silently overwritten by a second arrival of the same hour.
_RECONCILE = f"""
UPDATE {SCHEMA}.predictions
SET actual_ot = %s, reconciled_at = %s
WHERE target_timestamp = %s AND actual_ot IS NULL
RETURNING predicted_ot, model_version
"""  # nosec B608


@dataclass
class ReconciledPrediction:
    target_timestamp: datetime
    predicted_ot: float
    actual_ot: float
    model_version: str

    @property
    def residual(self) -> float:
        """actual - predicted, in degrees C. Positive: the model under-predicted."""
        return self.actual_ot - self.predicted_ot


def dsn_from_env() -> str:
    """Same RDS instance MLflow's own backend store already uses, a separate schema.

    RDS_ENDPOINT (host:port) comes from the dlinear-config ConfigMap;
    MLFLOW_DB_USERNAME/MLFLOW_DB_PASSWORD from the same `mlflow-db-credentials`
    Secret the mlflow Deployment already mounts (kubernetes/base/mlflow.yaml)
    - reusing that credential rather than provisioning a second one for a
    second schema in the same database.
    """
    endpoint = os.getenv("RDS_ENDPOINT")
    username = os.getenv("MLFLOW_DB_USERNAME")
    password = os.getenv("MLFLOW_DB_PASSWORD")
    if not endpoint or not username or not password:
        raise RuntimeError(
            "RDS_ENDPOINT / MLFLOW_DB_USERNAME / MLFLOW_DB_PASSWORD are not all set; "
            "the drift store is unreachable."
        )
    host, _, port = endpoint.partition(":")
    dbname = os.getenv("DB_NAME", "mlflow_db")
    return (
        f"host={host} "
        f"port={port or '5432'} "
        f"dbname={dbname} "
        f"user={username} "
        f"password={password} "
        f"connect_timeout={os.getenv('DRIFT_STORE_CONNECT_TIMEOUT', '10')} "
        f"application_name=dlinear-stream-consumer"
    )


def ensure_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_DDL)
    conn.commit()


def record_prediction(
    conn: Any, target_timestamp: datetime, model_version: str, predicted_ot: float
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_PREDICTION,
            (str(uuid.uuid4()), datetime.now(UTC), target_timestamp, model_version, predicted_ot),
        )
    conn.commit()


def reconcile_actual(
    conn: Any, target_timestamp: datetime, actual_ot: float
) -> ReconciledPrediction | None:
    """Match `actual_ot` against a pending prediction for this exact hour, if one exists.

    Returns None when no prediction was ever made for this timestamp (the
    consumer had not yet accumulated a full window when that hour would
    have been predicted) or it was already reconciled - both are ordinary,
    not errors.
    """
    with conn.cursor() as cur:
        cur.execute(_RECONCILE, (actual_ot, datetime.now(UTC), target_timestamp))
        row = cur.fetchone()
    conn.commit()
    if row is None:
        return None
    predicted_ot, model_version = row
    return ReconciledPrediction(
        target_timestamp=target_timestamp,
        predicted_ot=float(predicted_ot),
        actual_ot=actual_ot,
        model_version=model_version,
    )
