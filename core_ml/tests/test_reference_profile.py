from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.reference_profile import (
    SENSOR_COLUMNS,
    ReferenceProfile,
    build_feature_baselines,
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


def test_cusum_baselines_matches_the_shape_multifeaturecusum_expects() -> None:
    profile = build_feature_baselines(_raw_train_df())

    baselines = profile.cusum_baselines()

    assert set(baselines.keys()) == set(SENSOR_COLUMNS)
    mean, std = baselines["OT"]
    assert mean == pytest.approx(profile.features["OT"].mean)
    assert std == pytest.approx(profile.features["OT"].std)
