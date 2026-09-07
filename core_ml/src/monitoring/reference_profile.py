"""The statistical baseline logged alongside the model.

Same principle as 610-hotel-booking-mlops's profile.py: the reference is
computed ONCE, from the exact training split, and logged as an immutable
artifact of that model's own MLflow run - never re-read from the raw CSV
later (which would need DVC and bucket credentials in a streaming pod, and
would let the baseline silently change under a model that never
retrained).

Built in two stages, unlike 610's single-pass profile, because the two
halves come from two different pipeline stages that do not share a
process: data_processing.py computes per-sensor mean/std from the raw
training split (`build_feature_baselines`, written to
artifacts/reference_profile.json) - it has the raw dataframe but has not
trained anything yet. train.py then reads that file, adds the held-out
test MSE/MAE evaluate_test_set() just computed - it has never seen the raw
dataframe, only tensors - and logs the completed profile as
`monitoring/reference_profile.json` on its own run
(`attach_model_performance`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SENSOR_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


@dataclass
class FeatureBaseline:
    mean: float
    std: float


@dataclass
class ReferenceProfile:
    created_at: str
    n_rows: int
    features: dict[str, FeatureBaseline] = field(default_factory=dict)
    baseline_test_mse: float | None = None
    baseline_test_mae: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceProfile:
        return cls(
            created_at=payload["created_at"],
            n_rows=int(payload["n_rows"]),
            features={k: FeatureBaseline(**v) for k, v in payload.get("features", {}).items()},
            baseline_test_mse=payload.get("baseline_test_mse"),
            baseline_test_mae=payload.get("baseline_test_mae"),
        )

    @classmethod
    def read(cls, path: str | Path) -> ReferenceProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def cusum_baselines(self) -> dict[str, tuple[float, float]]:
        """`{feature: (mean, std)}`, exactly what MultiFeatureCusum's constructor needs."""
        return {name: (baseline.mean, baseline.std) for name, baseline in self.features.items()}


def build_feature_baselines(raw_train_df: pd.DataFrame) -> ReferenceProfile:
    """Mean/std per sensor from the raw (pre-scaling) training split.

    Called from data_processing.py, which has `train_data` - the same
    70% temporal split train.py's tensors are derived from - already in
    hand before it hands the frame off to StandardScaler. Performance
    fields are left None here; train.py fills them in after evaluating the
    held-out test set, since data_processing.py has no model to evaluate
    yet.
    """
    features: dict[str, FeatureBaseline] = {}
    for column in SENSOR_COLUMNS:
        values = raw_train_df[column].dropna().to_numpy(dtype=float)
        std = float(np.std(values))
        # Floored, not left at zero: real ETTh1 data never produces a
        # constant sensor column, but a synthetic/smoke-test dataset could,
        # and a zero std would make CusumDetector's constructor reject the
        # baseline outright rather than degrade gracefully.
        features[column] = FeatureBaseline(mean=float(np.mean(values)), std=max(std, 1e-6))

    return ReferenceProfile(
        created_at=datetime.now(UTC).isoformat(),
        n_rows=int(len(raw_train_df)),
        features=features,
    )
