import mlflow
import mlflow.pyfunc
import pytest

from src.quality_gate import run_quality_gate


class _DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture()
def mlflow_local_registry(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("quality-gate-test")
    yield


def _log_and_register(model_name: str, metric_value: float, **params) -> str:
    with mlflow.start_run():
        mlflow.log_metric("final_test_mse", metric_value)
        if params:
            mlflow.log_params(params)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_DummyModel(),
            registered_model_name=model_name,
        )
    return info.registered_model_version


def test_first_version_is_promoted_as_baseline(mlflow_local_registry):
    _log_and_register("baseline-model", metric_value=1.0)

    promoted = run_quality_gate("baseline-model")

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("baseline-model", "production")
    assert str(production.version) == "1"


def test_better_candidate_is_promoted(mlflow_local_registry):
    _log_and_register("improving-model", metric_value=1.0)
    run_quality_gate("improving-model")

    _log_and_register("improving-model", metric_value=0.5)  # menor MSE = mejor
    promoted = run_quality_gate("improving-model")

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("improving-model", "production")
    assert str(production.version) == "2"


def test_worse_candidate_is_not_promoted(mlflow_local_registry):
    _log_and_register("regressing-model", metric_value=0.5)
    run_quality_gate("regressing-model")

    _log_and_register("regressing-model", metric_value=1.0)  # peor MSE
    promoted = run_quality_gate("regressing-model")

    assert promoted is False
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("regressing-model", "production")
    assert str(production.version) == "1"  # sigue siendo la v1, no se degradó


def test_equal_metric_still_promotes(mlflow_local_registry):
    _log_and_register("tied-model", metric_value=1.0)
    run_quality_gate("tied-model")

    _log_and_register("tied-model", metric_value=1.0)  # empate exacto
    promoted = run_quality_gate("tied-model")

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("tied-model", "production")
    assert str(production.version) == "2"


def test_no_registered_versions_raises(mlflow_local_registry):
    with pytest.raises(SystemExit, match="No hay ninguna versión"):
        run_quality_gate("nonexistent-model")


def _production_version(model_name: str) -> str:
    client = mlflow.MlflowClient()
    return str(client.get_model_version_by_alias(model_name, "production").version)


REAL = {"dataset": "raw", "seq_len": 48, "pred_len": 48}
TOY = {"dataset": "toy", "seq_len": 48, "pred_len": 48}


def test_toy_candidate_never_replaces_a_real_production_model(mlflow_local_registry):
    _log_and_register("guarded-model", metric_value=5.0, **REAL)
    run_quality_gate("guarded-model")

    # Mejor métrica, pero medida sobre el split de test del dataset toy.
    _log_and_register("guarded-model", metric_value=0.1, **TOY)
    promoted = run_quality_gate("guarded-model")

    assert promoted is False
    assert _production_version("guarded-model") == "1"


def test_real_candidate_replaces_a_toy_production_model(mlflow_local_registry):
    _log_and_register("smoke-model", metric_value=0.1, **TOY)
    run_quality_gate("smoke-model")

    _log_and_register("smoke-model", metric_value=5.0, **REAL)
    promoted = run_quality_gate("smoke-model")

    assert promoted is True
    assert _production_version("smoke-model") == "2"


def test_candidate_with_a_different_horizon_is_not_comparable(mlflow_local_registry):
    _log_and_register("horizon-model", metric_value=5.0, **REAL)
    run_quality_gate("horizon-model")

    _log_and_register("horizon-model", metric_value=0.1, **{**REAL, "pred_len": 1})
    promoted = run_quality_gate("horizon-model")

    assert promoted is False
    assert _production_version("horizon-model") == "1"
