"""stream_consumer.py's per-reading logic, against a hand-built StreamConsumer.

__init__ talks to MLflow, Kinesis and Postgres directly, so these tests
build a StreamConsumer via __new__ and set only the attributes
handle_reading()/​_predict_next_hour() actually touch - the same pattern
used to unit-test a class whose constructor is all I/O.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from src.monitoring import stream_consumer
from src.monitoring.changepoint import CusumDetector, MultiFeatureCusum
from src.monitoring.reference_profile import SENSOR_COLUMNS


@dataclass
class _FakeModelVersion:
    version: str = "5"
    run_id: str = "run-abc"


class _ConstantModel:
    """Always predicts a fixed OT value - makes the residual math in tests exact."""

    def __init__(self, value: float = 20.0) -> None:
        self.value = value

    def predict(self, frame):
        return np.array([self.value])


class _FakeDbConn:
    def cursor(self):
        raise AssertionError("tests stub drift_store functions directly, not the DB layer")


def _baselines() -> dict[str, tuple[float, float]]:
    return {col: (10.0, 2.0) for col in SENSOR_COLUMNS}


def _reading(date: str, ot: float = 10.0) -> dict:
    return {
        "date": date,
        "HUFL": 10.0,
        "HULL": 10.0,
        "MUFL": 10.0,
        "MULL": 10.0,
        "LUFL": 10.0,
        "LULL": 10.0,
        "OT": ot,
    }


def _make_consumer(model=None) -> stream_consumer.StreamConsumer:
    consumer = stream_consumer.StreamConsumer.__new__(stream_consumer.StreamConsumer)
    consumer.model = model or _ConstantModel(value=20.0)
    consumer.model_version = _FakeModelVersion()
    consumer.data_drift = MultiFeatureCusum(baselines=_baselines())
    consumer.concept_drift = CusumDetector(mean=0.0, std=1.0)
    consumer.window = deque(maxlen=stream_consumer.SEQ_LEN)
    consumer.db_conn = _FakeDbConn()
    return consumer


def test_handle_reading_updates_the_data_drift_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = _make_consumer()
    monkeypatch.setattr(stream_consumer.drift_store, "reconcile_actual", lambda *a, **k: None)
    monkeypatch.setattr(
        stream_consumer.mitigation,
        "trigger_retrain",
        lambda **k: (_ for _ in ()).throw(
            AssertionError("mitigation must not be called without an ALERT")
        ),
    )

    consumer.handle_reading(_reading("2026-01-01T00:00:00"))

    assert consumer.data_drift.detectors["OT"].state.n_observations == 1


def test_handle_reading_predicts_the_next_hour_once_the_window_fills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _make_consumer()
    monkeypatch.setattr(stream_consumer.drift_store, "reconcile_actual", lambda *a, **k: None)

    recorded: list[dict] = []
    monkeypatch.setattr(
        stream_consumer.drift_store,
        "record_prediction",
        lambda conn, target_timestamp, model_version, predicted_ot: recorded.append(
            {
                "target_timestamp": target_timestamp,
                "model_version": model_version,
                "predicted_ot": predicted_ot,
            }
        ),
    )

    for i in range(stream_consumer.SEQ_LEN):
        iso = f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00"
        consumer.handle_reading(_reading(iso))

    assert len(recorded) == 1
    assert recorded[0]["model_version"] == "5"
    assert recorded[0]["predicted_ot"] == pytest.approx(20.0)


def test_handle_reading_triggers_mitigation_on_a_confirmed_data_drift_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _make_consumer()
    monkeypatch.setattr(stream_consumer.drift_store, "reconcile_actual", lambda *a, **k: None)

    triggered: list[dict] = []
    monkeypatch.setattr(
        stream_consumer.mitigation,
        "trigger_retrain",
        lambda source, reason, model_version: (
            triggered.append({"source": source, "reason": reason, "model_version": model_version})
            or stream_consumer.mitigation.MitigationOutcome(triggered=True, reason="ok")
        ),
    )

    # OT baseline is (10.0, 2.0); a sustained reading of 30.0 (10 sigma away)
    # crosses the CUSUM threshold within a handful of readings.
    for i in range(15):
        consumer.handle_reading(_reading(f"2026-01-01T{i:02d}:00:00", ot=30.0))

    assert triggered
    assert triggered[0]["source"] == "stream"
    assert "data drift" in triggered[0]["reason"]


def test_handle_reading_reconciles_and_feeds_the_concept_drift_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _make_consumer()

    reconciled = stream_consumer.drift_store.ReconciledPrediction(
        target_timestamp=datetime(2026, 1, 1, 5, tzinfo=UTC),
        predicted_ot=20.0,
        actual_ot=25.0,
        model_version="5",
    )
    monkeypatch.setattr(stream_consumer.drift_store, "reconcile_actual", lambda *a, **k: reconciled)
    monkeypatch.setattr(
        stream_consumer.mitigation,
        "trigger_retrain",
        lambda **k: stream_consumer.mitigation.MitigationOutcome(triggered=True, reason="ok"),
    )

    consumer.handle_reading(_reading("2026-01-01T05:00:00", ot=25.0))

    assert consumer.concept_drift.state.n_observations == 1
