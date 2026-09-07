from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.monitoring import drift_store


class _FakeCursor:
    def __init__(self, rows: dict[datetime, tuple[float, float | None, str]]) -> None:
        # rows: target_timestamp -> (predicted_ot, actual_ot, model_version)
        self._rows = rows
        self._result: tuple | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        statement = sql.strip()
        if statement.startswith("CREATE SCHEMA") or statement.startswith("CREATE TABLE"):
            return
        if statement.startswith("INSERT INTO"):
            assert params is not None
            _, _, target_timestamp, model_version, predicted_ot = params
            if target_timestamp not in self._rows:
                self._rows[target_timestamp] = (predicted_ot, None, model_version)
            return
        if statement.startswith("UPDATE"):
            assert params is not None
            actual_ot, _reconciled_at, target_timestamp = params
            existing = self._rows.get(target_timestamp)
            if existing is None or existing[1] is not None:
                self._result = None
                return
            predicted_ot, _, model_version = existing
            self._rows[target_timestamp] = (predicted_ot, actual_ot, model_version)
            self._result = (predicted_ot, model_version)
            return
        raise AssertionError(f"unexpected SQL in fake cursor: {sql}")

    def fetchone(self) -> tuple | None:
        return self._result


class _FakeConnection:
    def __init__(self) -> None:
        self.rows: dict[datetime, tuple[float, float | None, str]] = {}

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.rows)

    def commit(self) -> None:
        pass


@pytest.fixture
def fake_conn() -> _FakeConnection:
    return _FakeConnection()


def test_record_prediction_stores_a_row(fake_conn: _FakeConnection) -> None:
    ts = datetime(2026, 1, 1, 12, tzinfo=UTC)

    drift_store.record_prediction(
        fake_conn, target_timestamp=ts, model_version="3", predicted_ot=21.5
    )

    predicted_ot, actual_ot, model_version = fake_conn.rows[ts]
    assert predicted_ot == 21.5
    assert actual_ot is None
    assert model_version == "3"


def test_reconcile_actual_matches_a_pending_prediction(fake_conn: _FakeConnection) -> None:
    ts = datetime(2026, 1, 1, 12, tzinfo=UTC)
    drift_store.record_prediction(
        fake_conn, target_timestamp=ts, model_version="3", predicted_ot=21.5
    )

    reconciled = drift_store.reconcile_actual(fake_conn, target_timestamp=ts, actual_ot=22.0)

    assert reconciled is not None
    assert reconciled.predicted_ot == 21.5
    assert reconciled.actual_ot == 22.0
    assert reconciled.model_version == "3"
    assert reconciled.residual == pytest.approx(0.5)


def test_reconcile_actual_returns_none_when_no_prediction_was_ever_made(
    fake_conn: _FakeConnection,
) -> None:
    ts = datetime(2026, 1, 1, 12, tzinfo=UTC)

    reconciled = drift_store.reconcile_actual(fake_conn, target_timestamp=ts, actual_ot=22.0)

    assert reconciled is None


def test_reconcile_actual_is_idempotent_against_a_replayed_reading(
    fake_conn: _FakeConnection,
) -> None:
    ts = datetime(2026, 1, 1, 12, tzinfo=UTC)
    drift_store.record_prediction(
        fake_conn, target_timestamp=ts, model_version="3", predicted_ot=21.5
    )
    drift_store.reconcile_actual(fake_conn, target_timestamp=ts, actual_ot=22.0)

    second = drift_store.reconcile_actual(fake_conn, target_timestamp=ts, actual_ot=99.0)

    assert second is None
    # The original reconciliation must not be overwritten.
    assert fake_conn.rows[ts][1] == 22.0


def test_dsn_from_env_requires_all_three_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RDS_ENDPOINT", raising=False)
    monkeypatch.delenv("MLFLOW_DB_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_DB_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="RDS_ENDPOINT"):
        drift_store.dsn_from_env()


def test_dsn_from_env_parses_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDS_ENDPOINT", "db.example.internal:5432")
    monkeypatch.setenv("MLFLOW_DB_USERNAME", "mlopsadmin")
    monkeypatch.setenv("MLFLOW_DB_PASSWORD", "secret")  # pragma: allowlist secret

    dsn = drift_store.dsn_from_env()

    assert "host=db.example.internal" in dsn
    assert "port=5432" in dsn
    assert "user=mlopsadmin" in dsn
