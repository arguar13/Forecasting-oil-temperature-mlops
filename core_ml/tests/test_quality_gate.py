import mlflow
import mlflow.pyfunc
import pytest

from src.quality_gate import run_quality_gate


class _DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture()
def mlflow_local_registry(tmp_path):
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("quality-gate-test")
    yield


def _log_and_register(model_name: str, metric_value: float) -> str:
    with mlflow.start_run():
        mlflow.log_metric("final_test_mse", metric_value)
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


def test_no_registered_versions_raises(mlflow_local_registry):
    with pytest.raises(SystemExit, match="No hay ninguna versión"):
        run_quality_gate("nonexistent-model")
