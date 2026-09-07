"""Kinesis-driven online inference + drift detection for the served DLinear model.

Neither of this project's two existing inference paths leaves behind a
queryable "what did the model predict for timestamp T": the FastAPI service
answers and forgets (no correlation id, nothing persisted), and the daily
batch job scores a static S3 CSV and writes the result back to S3. Building
concept drift here therefore needs more than a reader over an existing log
(the situation 610-hotel-booking-mlops's stream_consumer.py is in, reading
api/monitoring.py's Postgres table) - it needs the log to exist first. That
is most of what this module does: for every 48 consecutive hourly readings
it has accumulated from Kinesis, it asks the served model for the *next*
hour's oil temperature and records that 1-step-ahead prediction
(drift_store.py); every new reading is also a chance to reconcile an older
prediction whose target hour has just arrived, turning the pair into a
residual.

Two independent CUSUM trackers (changepoint.py), not one:

    data drift     MultiFeatureCusum over the seven raw sensor readings -
                    has the sensors' own physical level shifted, regardless
                    of what the model predicts.
    concept drift  a single CusumDetector over the reconciled residuals
                    (actual - predicted) - has the model's relationship to
                    the physical process it predicts moved, baselined
                    against a mean of 0 (a well-calibrated model has no
                    systematic bias) and a standard deviation derived from
                    the held-out test MAE logged with this model version -
                    an approximation, not a re-derivation of the residuals'
                    true std, accepted because it needs no second pass over
                    the test set and MAE is already the right order of
                    magnitude for "how far off is this model normally".

A confirmed ALERT on either tracker reaches mitigation.trigger_retrain()
exactly like 610's stream_consumer.py does for its own monitors - this
module adds no separate mitigation path, only the detection that feeds the
one that exists.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from datetime import UTC, datetime, timedelta

import boto3
import mlflow
import pandas as pd
import psycopg2
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_contracts import FEATURE_COLUMNS
from src.logging_config import configure_logging, get_logger
from src.monitoring import drift_store, mitigation
from src.monitoring.changepoint import CusumDetector, MultiFeatureCusum
from src.monitoring.reference_profile import SENSOR_COLUMNS, ReferenceProfile

configure_logging()
logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "dlinear-ett-forecaster"
DEFAULT_MODEL_ALIAS = "production"
SEQ_LEN = 48

# /tmp, not a freely chosen path: this Deployment runs with a read-only
# root filesystem, matching every other workload in this project, and
# /tmp is the one writable emptyDir it gets.
HEARTBEAT_PATH = os.getenv(
    "STREAM_CONSUMER_HEARTBEAT_PATH", "/tmp/stream_consumer_heartbeat"  # nosec B108
)

# One retry pause after a transient Kinesis error, not an unbounded retry
# loop - without a floor, a throttled shard would spin this process at
# 100% CPU issuing GetRecords as fast as the client library allows.
KINESIS_ERROR_BACKOFF_SECONDS = 5.0

_RESILIENT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)


def _touch_heartbeat() -> None:
    with open(HEARTBEAT_PATH, "w", encoding="utf-8") as f:
        f.write(datetime.now(UTC).isoformat())


@_RESILIENT_RETRY
def _load_production_model():
    """The served version, plus enough MLflow metadata to fetch its reference profile."""
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    model_alias = os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, model_alias)
    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info("model_load_started", model_uri=model_uri, version=version.version)
    model = load_model(model_uri)
    logger.info("model_load_succeeded", model_uri=model_uri, version=version.version)
    return model, version


def _load_reference_profile(run_id: str) -> ReferenceProfile:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="monitoring/reference_profile.json"
    )
    return ReferenceProfile.read(local_path)


def _kinesis_shard_iterator(client, stream_name: str) -> str:
    shards = client.list_shards(StreamName=stream_name)["Shards"]
    if not shards:
        raise RuntimeError(f"Stream '{stream_name}' has no shards.")
    # ON_DEMAND streams (terraform/kinesis.tf) reshard on their own as
    # throughput grows; at this project's hourly cadence that has never
    # happened in practice, so the first shard is always the only one.
    # TRIM_HORIZON, not LATEST: a restarted pod should pick up readings
    # published while it was down (within the 24h retention window)
    # rather than silently skip them.
    return str(
        client.get_shard_iterator(
            StreamName=stream_name,
            ShardId=shards[0]["ShardId"],
            ShardIteratorType="TRIM_HORIZON",
        )["ShardIterator"]
    )


class StreamConsumer:
    def __init__(self) -> None:
        self.model, self.model_version = _load_production_model()
        self.reference_profile = _load_reference_profile(self.model_version.run_id)

        self.data_drift = MultiFeatureCusum(baselines=self.reference_profile.cusum_baselines())

        baseline_mae = self.reference_profile.baseline_test_mae
        if not baseline_mae or baseline_mae <= 0:
            raise RuntimeError(
                f"Model version {self.model_version.version}'s reference profile has no "
                "usable baseline_test_mae - cannot size the concept-drift CUSUM's std."
            )
        self.concept_drift = CusumDetector(mean=0.0, std=baseline_mae)

        self.window: deque[dict] = deque(maxlen=SEQ_LEN)

        # Same AWS_ENDPOINT_URL convention batch_inference.py already uses:
        # unset in production (real Kinesis), pointed at LocalStack
        # (http://localstack:4566) for the local/docker-compose stack and
        # for integration tests - the only way this consumer's Kinesis
        # calls can be exercised without a real AWS account.
        self.kinesis = boto3.client("kinesis", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
        self.stream_name = _stream_name()

        self.db_conn = psycopg2.connect(drift_store.dsn_from_env())
        drift_store.ensure_schema(self.db_conn)

    def _predict_next_hour(self, target_timestamp: datetime) -> None:
        frame = pd.DataFrame(list(self.window), columns=list(FEATURE_COLUMNS))
        prediction = self.model.predict(frame)
        # The served model's own pred_len (48 in production) is irrelevant
        # here - only the first step is a fast-reconciling, 1-hour-ahead
        # signal (see this module's docstring for why). `prediction` may be
        # a 1-D array (pred_len == 1) or 2-D (pred_len > 1, one row); either
        # way the first predicted value is index [0] of the flattened output.
        first_step = float(pd.Series(prediction).to_numpy().reshape(-1)[0])
        drift_store.record_prediction(
            self.db_conn,
            target_timestamp=target_timestamp,
            model_version=str(self.model_version.version),
            predicted_ot=first_step,
        )

    def handle_reading(self, reading: dict) -> None:
        timestamp = datetime.fromisoformat(reading["date"]).replace(tzinfo=UTC)

        data_verdict = self.data_drift.update({col: float(reading[col]) for col in SENSOR_COLUMNS})

        reconciled = drift_store.reconcile_actual(
            self.db_conn, target_timestamp=timestamp, actual_ot=float(reading["OT"])
        )
        concept_verdict = None
        if reconciled is not None:
            concept_verdict = self.concept_drift.update(reconciled.residual)

        alert_reasons = []
        if data_verdict.status == "ALERT":
            alert_reasons.append(f"data drift on {', '.join(data_verdict.alerting_features)}")
        if concept_verdict is not None and concept_verdict.status == "ALERT":
            alert_reasons.append(f"concept drift ({concept_verdict.direction})")

        if alert_reasons:
            reason = "; ".join(alert_reasons)
            logger.error("drift_alert", reason=reason, timestamp=timestamp.isoformat())
            outcome = mitigation.trigger_retrain(
                source="stream",
                reason=reason,
                model_version=str(self.model_version.version),
            )
            logger.info(
                "mitigation_requested",
                triggered=outcome.triggered,
                reason=outcome.reason,
                pipeline_id=outcome.pipeline_id,
            )

        self.window.append(
            {
                **{col: float(reading[col]) for col in SENSOR_COLUMNS},
                "month": timestamp.month,
                "day": timestamp.day,
                "hour": timestamp.hour,
            }
        )
        if len(self.window) == SEQ_LEN:
            self._predict_next_hour(target_timestamp=timestamp + timedelta(hours=1))

    def run_forever(self) -> None:
        shard_iterator = _kinesis_shard_iterator(self.kinesis, self.stream_name)
        logger.info("stream_consumer_started", stream_name=self.stream_name)

        while True:
            _touch_heartbeat()
            try:
                response = self.kinesis.get_records(ShardIterator=shard_iterator, Limit=100)
            except Exception as e:  # noqa: BLE001
                logger.warning("kinesis_get_records_failed", error=str(e))
                time.sleep(KINESIS_ERROR_BACKOFF_SECONDS)
                shard_iterator = _kinesis_shard_iterator(self.kinesis, self.stream_name)
                continue

            for record in response["Records"]:
                reading = json.loads(record["Data"])
                try:
                    self.handle_reading(reading)
                except Exception as e:  # noqa: BLE001
                    # One malformed/unreconcilable reading must not take the
                    # whole consumer down - the next record still gets a
                    # chance. Same posture as drift_monitor.py's evaluation
                    # failure handling in 610.
                    logger.error("reading_processing_failed", error=str(e), reading=reading)

            shard_iterator = response["NextShardIterator"]
            if not response["Records"]:
                # No new data: idle briefly instead of hammering GetRecords
                # (AWS throttles at 5 calls/second/shard regardless).
                time.sleep(2.0)


def _stream_name() -> str:
    name = os.getenv("SENSOR_STREAM_NAME")
    if not name:
        raise RuntimeError(
            "SENSOR_STREAM_NAME is not set - should come from the dlinear-config ConfigMap, "
            "set from terraform/kinesis.tf's sensor_telemetry_stream_name output."
        )
    return name


def main() -> int:
    try:
        consumer = StreamConsumer()
    except MlflowException as e:
        logger.error("stream_consumer_startup_failed", error=str(e))
        return 1
    consumer.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
