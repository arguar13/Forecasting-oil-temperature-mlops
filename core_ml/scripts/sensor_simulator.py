"""Replays historical ETT sensor readings onto Kinesis as if they were live.

ETTh1.csv is a fixed historical recording (2016-07 to 2018-06) of one
transformer's sensors - there is no real sensor in this portfolio to stream
from, the same situation 610-hotel-booking-mlops solves for its own domain
with scripts/replay_bookings.py. This script is that solution's counterpart
here: it validates the source CSV against the exact same data contract
data_processing.py and batch_inference.py already enforce
(data_contracts.py's ETT_CONTRACT), then PutRecords one row at a time onto
the sensor-telemetry Kinesis stream, each record's PartitionKey the row's
own ISO timestamp so replay order is stable and reproducible.

Two uses:

    A fast backtest (--delay-seconds 0, the default) drains a whole slice
    in seconds - what CI or a local smoke test wants.

    A slow, wall-clock-paced replay (--delay-seconds 3600, or any other
    value) is what actually exercises stream_consumer.py's micro-batch
    trigger and change-point detectors the way they would behave against a
    real, slowly-arriving feed - see that module's own docstring for how
    the trigger is paced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime

import boto3

from src.data_contracts import ETT_CONTRACT, TOY_ETT_CONTRACT, validate_ett_csv
from src.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

RAW_DATA_PATH = "data/raw/ETTh1.csv"
TOY_DATA_PATH = "data/toy/ETTh1_toy.csv"

SENSOR_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


def _stream_name() -> str:
    name = os.getenv("SENSOR_STREAM_NAME")
    if not name:
        raise RuntimeError(
            "SENSOR_STREAM_NAME is not set - this should come from the dlinear-config "
            "ConfigMap (kubernetes/base/kustomization.yaml), set from "
            "terraform/kinesis.tf's sensor_telemetry_stream_name output."
        )
    return name


def replay(
    dataset: str,
    start_row: int,
    limit: int | None,
    delay_seconds: float,
    kinesis_client=None,
) -> int:
    path = RAW_DATA_PATH if dataset == "raw" else TOY_DATA_PATH
    contract = ETT_CONTRACT if dataset == "raw" else TOY_ETT_CONTRACT
    df = validate_ett_csv(path, contract=contract)

    end_row = len(df) if limit is None else min(start_row + limit, len(df))
    window = df.iloc[start_row:end_row]

    # Same AWS_ENDPOINT_URL convention batch_inference.py uses for S3 -
    # unset against real Kinesis in production, pointed at LocalStack for
    # local runs and integration tests.
    client = kinesis_client or boto3.client("kinesis", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
    stream_name = _stream_name()

    published = 0
    for _, row in window.iterrows():
        payload = {
            "date": str(row["date"]),
            "replayed_at": datetime.now(UTC).isoformat(),
            **{col: float(row[col]) for col in SENSOR_COLUMNS},
        }
        client.put_record(
            StreamName=stream_name,
            Data=json.dumps(payload).encode("utf-8"),
            PartitionKey=payload["date"],
        )
        published += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    log.info(
        "replay_completed",
        dataset=dataset,
        start_row=start_row,
        end_row=end_row,
        published=published,
        stream_name=stream_name,
    )
    return published


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["raw", "toy"], default="toy")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Rows to replay; default: all")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Pause between records - 0 for a fast backtest, 3600 for a real-time-paced replay",
    )
    args = parser.parse_args()

    replay(
        dataset=args.dataset,
        start_row=args.start_row,
        limit=args.limit,
        delay_seconds=args.delay_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
