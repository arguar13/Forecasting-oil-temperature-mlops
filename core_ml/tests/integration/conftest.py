"""Fixtures y hooks compartidos para los tests de integración (Fase 3).

Estos tests levantan contenedores efímeros reales (Testcontainers) y, en
`test_docker_compose_stack.py`, el `docker-compose.yml` completo del repo.
Requieren Docker corriendo localmente -- se saltan con un mensaje claro (vía
`pytest_collection_modifyitems`) en vez de fallar de forma confusa, o
colgarse, si el daemon no está disponible.
"""

from __future__ import annotations

import docker
import pytest


def _docker_available() -> bool:
    try:
        client = docker.from_env(timeout=3)
        client.ping()
        return True
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if DOCKER_AVAILABLE:
        return
    skip_marker = pytest.mark.skip(
        reason="Docker no está disponible/corriendo -- requerido para tests de integración (Testcontainers)."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
