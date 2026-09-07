from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import sensor_simulator


class _FakeKinesisClient:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def put_record(self, *, StreamName: str, Data: bytes, PartitionKey: str) -> dict:
        self.records.append(
            {
                "StreamName": StreamName,
                "Data": json.loads(Data.decode("utf-8")),
                "PartitionKey": PartitionKey,
            }
        )
        return {"ShardId": "shardId-000000000000", "SequenceNumber": str(len(self.records))}


def _write_toy_csv(path, n_rows: int = 300) -> None:
    dates = pd.date_range("2016-07-01", periods=n_rows, freq="h")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "HUFL": 5.0,
            "HULL": 2.0,
            "MUFL": 1.5,
            "MULL": 0.5,
            "LUFL": 1.0,
            "LULL": 0.2,
            "OT": 20.0,
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture(autouse=True)
def _stream_name_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENSOR_STREAM_NAME", "test-sensor-telemetry")


@pytest.fixture
def toy_csv(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ETTh1_toy.csv"
    _write_toy_csv(path)
    monkeypatch.setattr(sensor_simulator, "TOY_DATA_PATH", str(path))
    return path


def test_replay_publishes_one_record_per_row(toy_csv) -> None:
    client = _FakeKinesisClient()

    published = sensor_simulator.replay(
        dataset="toy", start_row=0, limit=None, delay_seconds=0.0, kinesis_client=client
    )

    assert published == 300
    assert len(client.records) == 300
    assert client.records[0]["StreamName"] == "test-sensor-telemetry"


def test_replay_respects_start_row_and_limit(toy_csv) -> None:
    client = _FakeKinesisClient()

    published = sensor_simulator.replay(
        dataset="toy", start_row=10, limit=5, delay_seconds=0.0, kinesis_client=client
    )

    assert published == 5


def test_replay_record_payload_carries_sensor_readings_and_timestamp(toy_csv) -> None:
    client = _FakeKinesisClient()

    sensor_simulator.replay(
        dataset="toy", start_row=0, limit=1, delay_seconds=0.0, kinesis_client=client
    )

    payload = client.records[0]["Data"]
    assert payload["date"] == "2016-07-01 00:00:00"
    assert payload["OT"] == 20.0
    assert "replayed_at" in payload
    assert client.records[0]["PartitionKey"] == "2016-07-01 00:00:00"


def test_replay_requires_sensor_stream_name(toy_csv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENSOR_STREAM_NAME", raising=False)
    client = _FakeKinesisClient()

    with pytest.raises(RuntimeError, match="SENSOR_STREAM_NAME"):
        sensor_simulator.replay(
            dataset="toy", start_row=0, limit=1, delay_seconds=0.0, kinesis_client=client
        )
