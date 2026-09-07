import mlflow
import mlflow.pyfunc
import pytest

from src.quality_gate import rollback_production, run_quality_gate


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


def test_a_small_regression_within_tolerance_still_promotes(mlflow_local_registry):
    _log_and_register("tolerant-model", metric_value=1.0)
    run_quality_gate("tolerant-model")

    _log_and_register("tolerant-model", metric_value=1.05)  # 0.05 worse
    promoted = run_quality_gate("tolerant-model", tolerance=0.1)

    assert promoted is True
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("tolerant-model", "production")
    assert str(production.version) == "2"


def test_a_regression_beyond_tolerance_is_still_rejected(mlflow_local_registry):
    _log_and_register("strict-model", metric_value=1.0)
    run_quality_gate("strict-model")

    _log_and_register("strict-model", metric_value=1.5)  # 0.5 worse
    promoted = run_quality_gate("strict-model", tolerance=0.1)

    assert promoted is False
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("strict-model", "production")
    assert str(production.version) == "1"


def test_rollback_moves_production_back_exactly_one_promotion(mlflow_local_registry):
    _log_and_register("rollback-model", metric_value=1.0)
    run_quality_gate("rollback-model")
    _log_and_register("rollback-model", metric_value=0.5)
    run_quality_gate("rollback-model")

    restored_version = rollback_production("rollback-model")

    assert restored_version == "1"
    client = mlflow.MlflowClient()
    production = client.get_model_version_by_alias("rollback-model", "production")
    assert str(production.version) == "1"


def test_rollback_with_only_a_bootstrap_promotion_fails_loudly(mlflow_local_registry):
    _log_and_register("bootstrap-only-model", metric_value=1.0)
    run_quality_gate("bootstrap-only-model")  # promoted_from_version tag is "none"

    with pytest.raises(SystemExit, match="Nada a lo que hacer rollback"):
        rollback_production("bootstrap-only-model")


def test_rollback_without_any_production_alias_fails_loudly(mlflow_local_registry):
    _log_and_register("never-promoted-model", metric_value=99.0)  # never quality-gated

    with pytest.raises(SystemExit, match="no tiene ninguna versión aliasada"):
        rollback_production("never-promoted-model")
