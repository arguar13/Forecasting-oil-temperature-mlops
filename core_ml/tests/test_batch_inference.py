import io

import numpy as np
import pandas as pd
import pytest

from src.batch_inference import BatchInferenceService
from src.data_contracts import FEATURE_COLUMNS, DataContractError


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload


class _FakePyfuncModel:
    def predict(self, model_input):
        n_windows = model_input.shape[0] if model_input.ndim == 3 else 1
        return np.full((n_windows, 1), 42.5)


class _FakeMultiStepPyfuncModel:
    """Simula un modelo con horizonte multi-step (pred_len > 1)."""

    def __init__(self, pred_len: int):
        self.pred_len = pred_len

    def predict(self, model_input):
        n_windows = model_input.shape[0] if model_input.ndim == 3 else 1
        return np.full((n_windows, self.pred_len), 42.5)


def _valid_batch_df(n_rows: int) -> pd.DataFrame:
    row = {
        "HUFL": 5.8,
        "HULL": 2.0,
        "MUFL": 1.6,
        "MULL": 0.5,
        "LUFL": 4.2,
        "LULL": 1.3,
        "OT": 30.5,
        "month": 7,
        "day": 1,
        "hour": 0,
    }
    return pd.DataFrame([row] * n_rows, columns=list(FEATURE_COLUMNS))


@pytest.fixture()
def service(monkeypatch):
    # Los tests unitarios nunca deben tocar red: publish_batch_inference_completed
    # intenta conectarse a un broker Kafka real (best-effort, no bloqueante en
    # producción, pero sin mockear haría esperar ~5s de timeout por test aquí).
    monkeypatch.setattr("src.batch_inference.publish_batch_inference_completed", lambda **_: True)

    svc = BatchInferenceService()
    svc.model = _FakePyfuncModel()
    return svc


def test_process_batch_rejects_input_shorter_than_seq_len(monkeypatch, service):
    short_df = _valid_batch_df(service.seq_len - 1)
    payload = short_df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    with pytest.raises(DataContractError, match="al menos"):
        service.process_batch()


def test_process_batch_rejects_out_of_range_values(monkeypatch, service):
    df = _valid_batch_df(service.seq_len + 5)
    df.loc[0, "OT"] = 999.0
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    with pytest.raises(DataContractError):
        service.process_batch()


def test_process_batch_scores_and_uploads_predictions(monkeypatch, service):
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    payload = df.to_csv(index=False).encode()

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    uploaded = {}

    def _fake_put_object(Bucket, Key, Body):
        uploaded["bucket"] = Bucket
        uploaded["key"] = Key
        uploaded["body"] = Body

    monkeypatch.setattr(service.s3_client, "put_object", _fake_put_object)

    service.process_batch()

    assert uploaded["bucket"] == service.bucket
    assert uploaded["key"] == service.output_key
    result_df = pd.read_csv(io.StringIO(uploaded["body"]))
    assert list(result_df.columns) == ["Prediction"]
    assert len(result_df) == n_rows - service.seq_len + 1
    assert (result_df["Prediction"] == 42.5).all()


def test_process_batch_names_columns_per_horizon_step_for_multi_step_model(monkeypatch, service):
    n_rows = service.seq_len + 5
    df = _valid_batch_df(n_rows)
    payload = df.to_csv(index=False).encode()
    service.model = _FakeMultiStepPyfuncModel(pred_len=3)

    monkeypatch.setattr(
        service.s3_client,
        "get_object",
        lambda Bucket, Key: {"Body": _FakeBody(payload)},
    )

    uploaded = {}
    monkeypatch.setattr(
        service.s3_client,
        "put_object",
        lambda Bucket, Key, Body: uploaded.update(bucket=Bucket, key=Key, body=Body),
    )

    service.process_batch()

    result_df = pd.read_csv(io.StringIO(uploaded["body"]))
    assert list(result_df.columns) == ["Prediction_h1", "Prediction_h2", "Prediction_h3"]
    assert len(result_df) == n_rows - service.seq_len + 1
    assert (result_df == 42.5).all().all()
