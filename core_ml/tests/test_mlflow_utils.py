import numpy as np

from src.mlflow_utils import (
    DLinearForecaster,
    build_reproducibility_tags,
    get_container_image_tag,
    get_dvc_data_hash,
    get_git_commit_hash,
)
from src.model_architecture import DLinear


class _IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)


def test_get_git_commit_hash_prefers_ci_env_var(monkeypatch):
    monkeypatch.setenv("CI_COMMIT_SHA", "abc123")

    assert get_git_commit_hash() == "abc123"


def test_get_git_commit_hash_falls_back_to_git(monkeypatch):
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)

    result = get_git_commit_hash()

    # En este repo (con git init) debe resolver un SHA real de 40 hex chars,
    # o degradar a "unknown" si no hay commits todavía -- nunca lanzar.
    assert result == "unknown" or (
        len(result) == 40 and all(c in "0123456789abcdef" for c in result)
    )


def test_get_container_image_tag_prefers_ci_commit_sha(monkeypatch):
    monkeypatch.setenv("CI_COMMIT_SHA", "sha-from-ci")
    monkeypatch.delenv("CONTAINER_IMAGE_TAG", raising=False)

    assert get_container_image_tag() == "sha-from-ci"


def test_get_container_image_tag_defaults_to_local_dev(monkeypatch):
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    monkeypatch.delenv("CONTAINER_IMAGE_TAG", raising=False)

    assert get_container_image_tag() == "local-dev"


def test_get_dvc_data_hash_reads_md5_from_dvc_file(tmp_path):
    dvc_file = tmp_path / "data.csv.dvc"
    fake_md5 = "not-a-real-hash-just-for-testing"
    dvc_file.write_text(f"outs:\n- md5: {fake_md5}\n  size: 10\n  path: data.csv\n")

    assert get_dvc_data_hash(dvc_file) == fake_md5


def test_get_dvc_data_hash_missing_file_returns_unknown(tmp_path):
    assert get_dvc_data_hash(tmp_path / "nope.csv.dvc") == "unknown"


def test_build_reproducibility_tags_has_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_COMMIT_SHA", "commit-sha")
    dvc_file = tmp_path / "data.csv.dvc"
    dvc_file.write_text("outs:\n- md5: hash123\n  size: 10\n  path: data.csv\n")

    tags = build_reproducibility_tags(dvc_file)

    assert tags["git_commit_hash"] == "commit-sha"
    assert tags["dvc_data_hash"] == "hash123"
    assert tags["container_image_tag"] == "commit-sha"


def test_dlinear_forecaster_predict_handles_single_sequence():
    forecaster = DLinearForecaster()
    forecaster.model = DLinear(seq_len=48, n_features=10)
    forecaster.model.eval()
    forecaster.scaler_X = _IdentityScaler()
    forecaster.scaler_y = _IdentityScaler()

    single_sequence = np.random.rand(48, 10)

    result = forecaster.predict(context=None, model_input=single_sequence)

    assert result.shape == (1, 1)


def test_dlinear_forecaster_predict_handles_batch_of_sequences():
    forecaster = DLinearForecaster()
    forecaster.model = DLinear(seq_len=48, n_features=10)
    forecaster.model.eval()
    forecaster.scaler_X = _IdentityScaler()
    forecaster.scaler_y = _IdentityScaler()

    batch = np.random.rand(5, 48, 10)

    result = forecaster.predict(context=None, model_input=batch)

    assert result.shape == (5, 1)


def test_dlinear_forecaster_predict_handles_multi_step_horizon():
    forecaster = DLinearForecaster()
    forecaster.model = DLinear(seq_len=48, n_features=10, pred_len=48)
    forecaster.model.eval()
    forecaster.scaler_X = _IdentityScaler()
    forecaster.scaler_y = _IdentityScaler()

    batch = np.random.rand(5, 48, 10)

    result = forecaster.predict(context=None, model_input=batch)

    assert result.shape == (5, 48)
