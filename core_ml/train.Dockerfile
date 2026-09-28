# Entrenamiento local -- ver docker-compose.yml::trainer.
#
# mlflow.pyfunc.log_model() persiste la ruta relativa de cada artefacto
# ("model_state_dict", "scaler_X", ...) usando el separador nativo del SO
# donde corre `src.train` en el momento de guardar (mlflow/pyfunc/model.py,
# _save_model_with_class_artifacts_params: os.path.join(...)) -- ese literal
# queda grabado en el manifiesto del modelo y NO se recalcula al servirlo.
# Entrenar directo en el host Windows (`poetry run python -m src.train`)
# graba "artifacts\dlinear_model.pth"; el contenedor de la API (Linux)
# busca "artifacts/dlinear_model.pth" y falla con FileNotFoundError.
# CI ya entrena dentro de `image: python:3.10` (.gitlab-ci.yml::train)
# -- esta imagen replica ese mismo entorno para desarrollo local, sea cual
# sea el SO del host.
FROM python:3.10-slim

WORKDIR /workspace

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencias horneadas en la imagen, no instaladas en cada arranque: antes
# un entrypoint corría `pip install` en cada `docker compose run`, y cada
# `make train`/`train-toy`/`quality-gate` pagaba varios minutos de
# instalación. Esta capa solo se reconstruye cuando cambia el lockfile
# (`docker compose build trainer`; `make up` no: trainer está en el profile tools).
#
# "poetry export" + pip, no "poetry install": en este entorno (Docker
# Desktop/WSL2, python:3.10-slim), `poetry install` con el lockfile de
# core_ml muere en silencio (exit 1, sin traza) al resolver el entorno de
# virtualenv; "poetry export" no toca esa ruta de código. --timeout 120
# (default de pip: 15s) da margen para bajar torch en un host con red
# compartida.
COPY core_ml/pyproject.toml core_ml/poetry.lock /tmp/deps/
RUN pip install --no-cache-dir "poetry==1.8.3" "poetry-plugin-export==1.8.0" \
    && poetry export -C /tmp/deps -f requirements.txt --without-hashes --only main -o /tmp/deps/requirements.txt \
    && pip install --no-cache-dir --timeout 120 -r /tmp/deps/requirements.txt \
    && pip uninstall -y poetry poetry-plugin-export \
    && rm -rf /tmp/deps

# El código (core_ml/) llega por bind mount (docker-compose.yml), así que
# no hace falta reconstruir la imagen al editar src/.
