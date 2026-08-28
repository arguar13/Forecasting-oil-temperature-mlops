"""Logging estructurado (Fase 4): JSON en producción, legible en un TTY local.

Un log en JSON de una sola línea por evento es lo que CloudWatch Logs
Insights / Elasticsearch necesitan para poder filtrar y agregar por campo
(`event`, `level`, `logger`, timestamp) en vez de tener que parsear texto
libre. En una terminal interactiva se renderiza en color para que siga
siendo legible por humanos.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configura structlog una vez, al inicio del proceso (entrypoint).

    Idempotente: llamarla más de una vez simplemente reemplaza la
    configuración global con la misma, sin efectos secundarios.
    """
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return cast("structlog.typing.FilteringBoundLogger", structlog.get_logger(name))
