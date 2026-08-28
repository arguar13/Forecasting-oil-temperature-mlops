"""Prueba que un Postgres efímero (Testcontainers) acepta conexiones y
consultas -- exactamente el mismo motor que MLflow usa como backend store
en docker-compose.yml y como RDS lo hará en AWS real (terraform/rds.tf).
"""

import psycopg2
import pytest
from testcontainers.community.postgres import PostgresContainer

pytestmark = pytest.mark.integration


def test_postgres_container_accepts_connections_and_queries():
    with PostgresContainer("postgres:18.3-alpine") as postgres:
        dsn = postgres.get_connection_url(driver=None)

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                assert cur.fetchone() == (1,)

                cur.execute("CREATE TABLE smoke_test (id SERIAL PRIMARY KEY, name TEXT);")
                cur.execute("INSERT INTO smoke_test (name) VALUES (%s);", ("dlinear",))
                conn.commit()

                cur.execute("SELECT name FROM smoke_test WHERE id = 1;")
                assert cur.fetchone() == ("dlinear",)
        finally:
            conn.close()
