from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


class _FakePyfuncModel:
    """Sustituye a `mlflow.pyfunc.load_model(...)` en los tests: evita
    depender de un MLflow Tracking Server real ni de artefactos en S3.
    """

    def predict(self, model_input):
        return np.array([[42.5]])


@pytest.fixture()
def client(monkeypatch):
    import api.main as main_module

    monkeypatch.setattr(main_module, "load_model", lambda uri: _FakePyfuncModel())

    with TestClient(main_module.app) as test_client:
        yield test_client


def _valid_features(n: int = 48) -> list[dict]:
    return [
        {
            "HUFL": 5.8,
            "HULL": 2.0,
            "MUFL": 1.6,
            "MULL": 0.5,
            "LUFL": 4.2,
            "LULL": 1.3,
            "OT": 30.5,
            "month": 7,
            "day": 1,
            "hour": h % 24,
        }
        for h in range(n)
    ]


def test_health_check_returns_healthy(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_a_numeric_prediction(client):
    response = client.post("/predict", json={"features": _valid_features()})

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "dlinear-ett-forecaster@production"
    # predictions es una lista (horizonte de predicción) -- el modelo fake
    # de este test devuelve un solo paso, [[42.5]], pero el contrato ya
    # soporta N pasos (ver core_ml/src/model_architecture.py::DLinear).
    assert body["predictions"] == pytest.approx([42.5])


def test_predict_returns_full_horizon_for_multi_step_model(monkeypatch):
    import api.main as main_module

    class _FakeMultiStepModel:
        def predict(self, model_input):
            return np.array([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(main_module, "load_model", lambda uri: _FakeMultiStepModel())

    with TestClient(main_module.app) as test_client:
        response = test_client.post("/predict", json={"features": _valid_features()})

    assert response.status_code == 200
    assert response.json()["predictions"] == pytest.approx([1.0, 2.0, 3.0])


def test_predict_rejects_wrong_sequence_length(client):
    response = client.post("/predict", json={"features": _valid_features(n=10)})

    assert response.status_code == 422


def test_predict_rejects_out_of_range_feature(client):
    features = _valid_features()
    features[0]["OT"] = 999.0  # fuera del rango del contrato de datos

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422


def test_predict_rejects_malformed_payload(client):
    response = client.post("/predict", json={"features": "not-a-list"})

    assert response.status_code == 422


def test_health_and_predict_degrade_gracefully_when_registry_unavailable(monkeypatch):
    """Un pod recién desplegado (antes del primer quality_gate.py, o con el
    Model Registry momentáneamente inalcanzable) no debe crashear: /health
    sigue en 200 (liveness), /ready y /predict responden 503 (not ready).

    Monkeypatchea `_load_model_resilient` (no `load_model`): así se prueba
    el manejo de errores del lifespan sin ejecutar los reintentos reales de
    tenacity (que sí esperan con backoff exponencial -- ver
    test_load_model_resilient_is_configured_with_bounded_retries_and_breaker).
    """
    import api.main as main_module

    def _raise(uri):
        raise RuntimeError("Registered Model with name=... not found")

    monkeypatch.setattr(main_module, "_load_model_resilient", _raise)

    with TestClient(main_module.app) as test_client:
        health_response = test_client.get("/health")
        assert health_response.status_code == 200

        ready_response = test_client.get("/ready")
        assert ready_response.status_code == 503

        predict_response = test_client.post("/predict", json={"features": _valid_features()})
        assert predict_response.status_code == 503


def test_ready_returns_200_once_model_is_loaded(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_load_model_resilient_is_configured_with_bounded_retries_and_breaker():
    """Verifica la configuración de resiliencia por introspección (tenacity
    expone `.retry` en la función decorada) en vez de ejecutar reintentos
    reales -- que sí duermen con backoff exponencial y harían este test lento.
    """
    import api.main as main_module

    retry_config = main_module._load_model_resilient.retry
    assert retry_config.stop.max_attempt_number == 3  # nunca reintentos infinitos

    assert main_module.MODEL_REGISTRY_BREAKER.fail_max == 5
    assert main_module.MODEL_REGISTRY_BREAKER.reset_timeout == 60


def test_predict_returns_500_without_leaking_internal_errors(monkeypatch):
    import api.main as main_module

    class _BrokenModel:
        def predict(self, model_input):
            raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main_module, "load_model", lambda uri: _BrokenModel())

    with TestClient(main_module.app) as test_client:
        response = test_client.post("/predict", json={"features": _valid_features()})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_model_with_mismatched_seq_len_is_never_marked_ready(monkeypatch):
    """Un modelo reentrenado con otro seq_len no debe pasar /ready: cada
    /predict fallaría contra el contrato HTTP de 48 lecturas."""
    import api.main as main_module

    class _OtherWindowModel(_FakePyfuncModel):
        metadata = SimpleNamespace(
            flavors={"python_function": {"model_config": {"seq_len": 96, "n_features": 10}}}
        )

    monkeypatch.setattr(main_module, "load_model", lambda uri: _OtherWindowModel())

    with TestClient(main_module.app) as test_client:
        assert test_client.get("/ready").status_code == 503
        assert test_client.get("/health").status_code == 200
