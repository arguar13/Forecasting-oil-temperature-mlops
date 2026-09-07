from __future__ import annotations

import pytest

from src.monitoring.changepoint import CusumDetector, MultiFeatureCusum


def test_cusum_stays_ok_on_values_near_the_baseline_mean() -> None:
    detector = CusumDetector(mean=20.0, std=2.0, k=0.5, h=5.0)

    for value in [19.5, 20.2, 20.1, 19.8, 20.3, 19.9]:
        verdict = detector.update(value)
        assert verdict.status == "OK"


def test_cusum_alerts_on_a_sustained_upward_shift() -> None:
    """No single observation is an outlier - the accumulation is what alerts."""
    detector = CusumDetector(mean=20.0, std=2.0, k=0.5, h=5.0)

    # +1 sigma sustained: z=1.0 each step, s_pos accumulates by (1.0 - 0.5)
    # = 0.5/step, crossing h=5.0 after 11 observations.
    statuses = [detector.update(22.0).status for _ in range(15)]

    assert "ALERT" in statuses
    assert statuses[:10] == ["OK"] * 10


def test_cusum_alerts_on_a_sustained_downward_shift() -> None:
    detector = CusumDetector(mean=20.0, std=2.0, k=0.5, h=5.0)

    verdicts = [detector.update(18.0) for _ in range(15)]

    alerted = [v for v in verdicts if v.status == "ALERT"]
    assert alerted
    assert alerted[0].direction == "down"


def test_cusum_resets_immediately_after_the_alert_that_fires() -> None:
    detector = CusumDetector(mean=20.0, std=2.0, k=0.5, h=5.0)

    verdict = None
    for _ in range(15):
        verdict = detector.update(22.0)
        if verdict.status == "ALERT":
            break

    assert verdict is not None
    assert verdict.status == "ALERT"
    assert detector.state.s_pos == 0.0
    assert detector.state.n_alerts == 1


def test_cusum_rejects_a_non_positive_baseline_std() -> None:
    with pytest.raises(ValueError, match="positive"):
        CusumDetector(mean=20.0, std=0.0)


def test_multi_feature_cusum_alerts_only_on_the_feature_that_shifted() -> None:
    tracker = MultiFeatureCusum(
        baselines={"OT": (20.0, 2.0), "HUFL": (5.0, 1.0)},
        k=0.5,
        h=5.0,
    )

    verdicts = [tracker.update({"OT": 22.0, "HUFL": 5.1}) for _ in range(15)]
    alerts = [v for v in verdicts if v.status == "ALERT"]

    assert alerts
    assert alerts[0].alerting_features == ["OT"]
    assert all("HUFL" not in v.alerting_features for v in verdicts)


def test_multi_feature_cusum_ignores_unknown_features_in_a_reading() -> None:
    tracker = MultiFeatureCusum(baselines={"OT": (20.0, 2.0)})

    verdict = tracker.update({"OT": 20.1, "not_tracked": 999.0})

    assert verdict.status == "OK"
    assert "not_tracked" not in verdict.per_feature
