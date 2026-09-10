from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_check import (
    SENSOR_COLUMNS,
    ReferenceProfile,
    build_feature_baselines,
    check_batch_for_drift,
)


def _raw_train_df(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {col: rng.normal(loc=10.0, scale=2.0, size=n_rows) for col in SENSOR_COLUMNS}
    return pd.DataFrame(data)


def test_build_feature_baselines_covers_every_sensor_column() -> None:
    profile = build_feature_baselines(_raw_train_df())

    assert set(profile.features.keys()) == set(SENSOR_COLUMNS)
    assert profile.n_rows == 200
    assert profile.baseline_test_mse is None
    assert profile.baseline_test_mae is None


def test_build_feature_baselines_matches_numpy_mean_and_std() -> None:
    df = _raw_train_df()
    profile = build_feature_baselines(df)

    baseline = profile.features["OT"]
    assert baseline.mean == pytest.approx(float(df["OT"].mean()), abs=1e-9)
    assert baseline.std == pytest.approx(float(df["OT"].std(ddof=0)), abs=1e-9)


def test_build_feature_baselines_floors_a_constant_column_std() -> None:
    df = _raw_train_df()
    df["OT"] = 42.0  # perfectly constant - std would otherwise be exactly 0

    profile = build_feature_baselines(df)

    assert profile.features["OT"].std > 0.0


def test_reference_profile_round_trips_through_json(tmp_path) -> None:
    profile = build_feature_baselines(_raw_train_df())
    profile.baseline_test_mse = 1.23
    profile.baseline_test_mae = 0.98

    path = profile.write(tmp_path / "reference_profile.json")
    restored = ReferenceProfile.read(path)

    assert restored.n_rows == profile.n_rows
    assert restored.baseline_test_mse == pytest.approx(1.23)
    assert restored.baseline_test_mae == pytest.approx(0.98)
    for column in SENSOR_COLUMNS:
        assert restored.features[column].mean == pytest.approx(profile.features[column].mean)
        assert restored.features[column].std == pytest.approx(profile.features[column].std)


def test_check_batch_for_drift_reports_ok_when_batch_matches_reference() -> None:
    reference = build_feature_baselines(_raw_train_df())
    matching_batch = _raw_train_df(n_rows=50)

    report = check_batch_for_drift(matching_batch, reference)

    assert report.status == "ok"
    assert report.drifted_features == []


def test_check_batch_for_drift_flags_a_shifted_feature() -> None:
    reference = build_feature_baselines(_raw_train_df())
    shifted_batch = _raw_train_df(n_rows=50)
    shifted_batch["OT"] = shifted_batch["OT"] + 100.0  # far outside the reference distribution

    report = check_batch_for_drift(shifted_batch, reference)

    assert report.status == "drift"
    assert "OT" in report.drifted_features
    assert report.per_feature["OT"].drifted is True


def test_check_batch_for_drift_ignores_columns_missing_from_the_batch() -> None:
    reference = build_feature_baselines(_raw_train_df())
    partial_batch = _raw_train_df(n_rows=50).drop(columns=["OT"])

    report = check_batch_for_drift(partial_batch, reference)

    assert "OT" not in report.per_feature
