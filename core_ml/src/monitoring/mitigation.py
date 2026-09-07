"""Bounded, cooldown-guarded automatic retraining.

Same three bounding mechanisms as 610-hotel-booking-mlops's mitigation.py,
for the same reasons - see that module's docstring for the full argument
(and for why "may trigger a retrain" is not the same authority as "may
promote one": promotion here stays gated behind quality_gate.py's own
production-vs-candidate comparison - train.py never moves the `production`
alias itself, and this module launches train.py's pipeline, never
quality_gate.py's). The one structural difference is configuration: this
project has no config/*.yaml (every other module here reads plain
environment variables, ConfigMap-injected - see data_processing.py,
train.py, batch_inference.py), so this module follows that convention
rather than introducing a YAML file for a single caller.

1. **Cooldown.** No new trigger within MITIGATION_COOLDOWN_HOURS of the
   last one, regardless of which caller asked (stream_consumer.py is the
   only caller today, but the mechanism does not assume that stays true).
2. **A hard ceiling.** At most MITIGATION_MAX_AUTO_RETRAINS triggers in a
   rolling MITIGATION_MAX_AUTO_RETRAINS_WINDOW_DAYS window. A drift source
   a retrain cannot fix (a genuine sensor fault, not a distribution shift)
   would otherwise retrigger every cooldown window forever. Hitting the
   ceiling fails loudly - logged at ERROR - rather than looping.
3. **A bounded HTTP call.** The GitLab Pipeline Trigger request has a
   fixed timeout and is never retried in a loop.

State lives in `monitoring.mitigation_events`, a second table alongside
drift_store.py's `monitoring.predictions` in the same RDS instance -
same "no migration tool, DDL owned by whichever process uses it first"
reasoning as that module.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import psycopg2
import requests

from src.logging_config import configure_logging, get_logger
from src.monitoring import drift_store

configure_logging()
logger = get_logger(__name__)

SCHEMA = "monitoring"

SCHEMA_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.mitigation_events (
    event_id      UUID PRIMARY KEY,
    triggered_at  TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL,
    reason        TEXT NOT NULL,
    model_version TEXT,
    pipeline_id   TEXT,
    outcome       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_mitigation_events_triggered_at
    ON {SCHEMA}.mitigation_events (triggered_at DESC);
"""  # SCHEMA is a module constant, never user input - no dynamic SQL here

_INSERT_EVENT = f"""
INSERT INTO {SCHEMA}.mitigation_events
    (event_id, triggered_at, source, reason, model_version, pipeline_id, outcome)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""  # nosec B608

_SELECT_SINCE = f"""
SELECT triggered_at, outcome
FROM {SCHEMA}.mitigation_events
WHERE triggered_at >= %s
ORDER BY triggered_at DESC
"""  # nosec B608


@dataclass
class MitigationOutcome:
    triggered: bool
    reason: str
    pipeline_id: str | None = None


def _ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_DDL)
    conn.commit()


def _events_since(conn, since: datetime) -> list[tuple[datetime, str]]:
    with conn.cursor() as cur:
        cur.execute(_SELECT_SINCE, (since,))
        return list(cur.fetchall())


def _record_event(
    conn, source: str, reason: str, model_version: str | None, pipeline_id: str | None, outcome: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_EVENT,
            (
                str(uuid.uuid4()),
                datetime.now(UTC),
                source,
                reason,
                model_version,
                pipeline_id,
                outcome,
            ),
        )
    conn.commit()


def _trigger_gitlab_pipeline(reason: str, source: str, model_version: str | None) -> str:
    """POST to GitLab's Pipeline Trigger API. Bounded: one call, one timeout, no retry loop."""
    api_url = os.getenv("GITLAB_API_URL")
    project_path = os.getenv("GITLAB_PROJECT_PATH")
    token = os.getenv("GITLAB_TRIGGER_TOKEN")
    ref = os.getenv("GITLAB_REF", "main")

    if not api_url or not project_path or not token:
        raise RuntimeError(
            "GitLab trigger is not configured (GITLAB_API_URL / GITLAB_PROJECT_PATH / "
            "GITLAB_TRIGGER_TOKEN) - cannot launch an automatic retrain."
        )

    timeout_seconds = float(os.getenv("MITIGATION_TRIGGER_TIMEOUT_SECONDS", "10"))
    url = f"{api_url}/projects/{quote(project_path, safe='')}/trigger/pipeline"
    response = requests.post(
        url,
        data={
            "token": token,
            "ref": ref,
            "variables[TRIGGER_SOURCE]": source,
            "variables[TRIGGER_REASON]": reason[:500],
            "variables[TRIGGER_MODEL_VERSION]": model_version or "unknown",
            "variables[AUTO_RETRAIN]": "true",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return str(response.json()["id"])


def trigger_retrain(source: str, reason: str, model_version: str | None) -> MitigationOutcome:
    """Ask mitigation to launch a retrain for a confirmed drift alert.

    Returns a MitigationOutcome rather than raising for the expected
    non-triggering paths (cooldown active, ceiling reached, mitigation
    disabled) - ordinary outcomes a caller logs and moves on from. A
    GitLab API error is caught and recorded the same way: this function's
    contract is "tell me what happened", never "crash the monitor that
    called you".
    """
    if os.getenv("MITIGATION_ENABLED", "true").lower() != "true":
        return MitigationOutcome(triggered=False, reason="MITIGATION_ENABLED is not true")

    conn = psycopg2.connect(drift_store.dsn_from_env())
    try:
        _ensure_schema(conn)
        now = datetime.now(UTC)

        cooldown_hours = float(os.getenv("MITIGATION_COOLDOWN_HOURS", "6"))
        recent = _events_since(conn, now - timedelta(hours=cooldown_hours))
        if recent:
            last_triggered_at, _ = recent[0]
            outcome = MitigationOutcome(
                triggered=False,
                reason=(
                    f"cooldown active - last mitigation event at {last_triggered_at.isoformat()}, "
                    f"{cooldown_hours}h must pass before another automatic trigger"
                ),
            )
            logger.info("mitigation_skipped_cooldown", source=source, reason=outcome.reason)
            return outcome

        max_retrains = int(os.getenv("MITIGATION_MAX_AUTO_RETRAINS", "3"))
        window_days = float(os.getenv("MITIGATION_MAX_AUTO_RETRAINS_WINDOW_DAYS", "7"))
        window_events = _events_since(conn, now - timedelta(days=window_days))
        if len(window_events) >= max_retrains:
            ceiling_reason = (
                f"{len(window_events)} automatic retrain(s) already triggered in the last "
                f"{window_days} day(s) (limit {max_retrains}) - refusing to trigger another "
                "automatically; a drift source a retrain cannot fix needs a human, not "
                "another retrain"
            )
            logger.error("mitigation_ceiling_reached", source=source, drift_reason=reason)
            _record_event(conn, source, reason, model_version, None, "ceiling_reached")
            return MitigationOutcome(triggered=False, reason=ceiling_reason)

        try:
            pipeline_id = _trigger_gitlab_pipeline(reason, source, model_version)
        except (requests.RequestException, RuntimeError) as e:
            logger.error("mitigation_trigger_failed", source=source, error=str(e))
            _record_event(conn, source, reason, model_version, None, "failed")
            return MitigationOutcome(triggered=False, reason=f"GitLab trigger call failed: {e}")

        _record_event(conn, source, reason, model_version, pipeline_id, "triggered")
        logger.info(
            "mitigation_retrain_triggered",
            source=source,
            reason=reason,
            model_version=model_version,
            pipeline_id=pipeline_id,
        )
        return MitigationOutcome(
            triggered=True, reason="retrain triggered", pipeline_id=pipeline_id
        )
    finally:
        conn.close()
