# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.10

# Evitar escritura de bytecodes y configurar logs en crudo
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3

# Instalar Poetry a nivel del sistema para que quede en la caché de Docker
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 -
ENV PATH="/root/.local/bin:$PATH"

# make: interfaz única de comandos (local == CI). postgresql-client: debug de RDS/MLflow.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends make postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# pre-commit: instalado a nivel de sistema para no depender del venv de
# ningún subproyecto al ejecutar los git hooks.
RUN pip install --no-cache-dir pre-commit
