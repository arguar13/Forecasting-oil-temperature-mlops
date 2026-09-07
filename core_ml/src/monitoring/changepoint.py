"""Two-sided CUSUM (Page's test) for level-shift detection in continuous series.

Why CUSUM/Page-Hinkley here and not PSI/histograms (the approach the other
project in this portfolio, 610-hotel-booking-mlops, uses for its own drift
detection): that project's features are categorical/tabular attributes of
thousands of independent bookings per window, where comparing the *shape* of
a distribution (PSI/Jensen-Shannon over bins) is the right question. The
features here are continuous physical readings from ONE transformer's
sensors over time - the question is not "did the distribution's shape
change" but "did the series' level shift", and that is exactly what a
change-point detector answers, with decades of use in statistical process
control (SPC) - the same physical domain (industrial sensor monitoring) this
project's own subject matter belongs to.

CUSUM accumulates each observation's standardized deviation from a baseline
(mean/std logged with the model - see reference_profile.py) into two
one-sided running sums, S+ for upward drift and S- for downward. A run of
small deviations in the same direction accumulates until it crosses the
decision threshold `h`, even when no single observation is an outlier -
that is the property that distinguishes CUSUM from a bare "|z| > 3" check:
it is sensitive to a small, sustained shift, exactly the pattern of a
sensor that has started to drift, not a single spike.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CusumState:
    """Persistent state of one CUSUM detector - updated one observation at a time."""

    s_pos: float = 0.0
    s_neg: float = 0.0
    n_observations: int = 0
    n_alerts: int = 0


@dataclass
class CusumVerdict:
    status: str  # "OK" | "ALERT"
    z_score: float
    s_pos: float
    s_neg: float
    direction: str  # "" | "up" | "down"


class CusumDetector:
    """Two-sided CUSUM over a series standardized against a fixed baseline (mean, std).

    `k` (slack, in standard deviations): how far an observation must sit
    from the baseline before it starts accumulating. 0.5 is the standard
    reference value for detecting a ~1-sigma sustained shift with good
    sensitivity without tripping on ordinary noise.

    `h` (decision threshold, in accumulated standard deviations): 5.0 is
    the standard reference value (Page, 1954) balancing mean time to a
    false alarm against detection latency for a real shift.
    """

    def __init__(self, mean: float, std: float, k: float = 0.5, h: float = 5.0) -> None:
        if std <= 0:
            raise ValueError(f"baseline std must be positive, got {std}")
        self.mean = mean
        self.std = std
        self.k = k
        self.h = h
        self.state = CusumState()

    def update(self, value: float) -> CusumVerdict:
        z = (value - self.mean) / self.std
        self.state.n_observations += 1

        self.state.s_pos = max(0.0, self.state.s_pos + z - self.k)
        self.state.s_neg = min(0.0, self.state.s_neg + z + self.k)

        status = "OK"
        direction = ""
        if self.state.s_pos > self.h:
            status = "ALERT"
            direction = "up"
        elif self.state.s_neg < -self.h:
            status = "ALERT"
            direction = "down"

        if status == "ALERT":
            self.state.n_alerts += 1
            # Reset after an alert: without this, once the sum crosses the
            # threshold it stays saturated and every following observation
            # keeps reading ALERT even after the series has returned to
            # baseline - the same failure mode as a circuit breaker that
            # never closes.
            self.state.s_pos = 0.0
            self.state.s_neg = 0.0

        return CusumVerdict(
            status=status,
            z_score=z,
            s_pos=self.state.s_pos,
            s_neg=self.state.s_neg,
            direction=direction,
        )


@dataclass
class MultiCusumVerdict:
    """One window's verdict across every feature being tracked."""

    status: str  # "OK" | "ALERT"
    alerting_features: list[str] = field(default_factory=list)
    per_feature: dict[str, CusumVerdict] = field(default_factory=dict)


class MultiFeatureCusum:
    """One CusumDetector per tracked feature, sharing a single verdict.

    A single feature crossing its threshold is enough to alert - unlike
    610-hotel-booking-mlops's *share*-of-features-drifted rule (appropriate
    there because thirty independent booking attributes drifting together
    signals a population change, while one drifting alone is ordinary
    seasonality), a single sensor on one physical machine drifting is
    already the finding: transformers do not have "ordinary seasonality"
    in their internal load/temperature relationship the way a booking mix
    does.
    """

    def __init__(self, baselines: dict[str, tuple[float, float]], k: float = 0.5, h: float = 5.0):
        self.detectors = {
            feature: CusumDetector(mean=mean, std=std, k=k, h=h)
            for feature, (mean, std) in baselines.items()
        }

    def update(self, readings: dict[str, float]) -> MultiCusumVerdict:
        per_feature: dict[str, CusumVerdict] = {}
        alerting: list[str] = []
        for feature, detector in self.detectors.items():
            if feature not in readings:
                continue
            verdict = detector.update(readings[feature])
            per_feature[feature] = verdict
            if verdict.status == "ALERT":
                alerting.append(feature)

        return MultiCusumVerdict(
            status="ALERT" if alerting else "OK",
            alerting_features=alerting,
            per_feature=per_feature,
        )
