# Dockerfile para la API de DLinear
FROM python:3.10-slim

WORKDIR /app

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION" "poetry-plugin-export==1.8.0"

# poetry.lock se copia sin comodín: el build falla si el lockfile no está
# commiteado, en vez de degradar silenciosamente a una resolución no
# determinista de dependencias.
COPY api/pyproject.toml api/poetry.lock ./
# "poetry install --sync" (no "poetry export" + "pip install") moría en
# silencio y de forma intermitente en este host (exit code 1, sin traceback
# -- ver el mismo diagnóstico ya hecho para core_ml/train.Dockerfile y los
# jobs train_model/quality_gate/integration_tests de .gitlab-ci.yml,
# torch/DVC traen dependencias con C-extensions que disparan el mismo bug
# del instalador de poetry en Linux/Docker Desktop). Mismo fix aquí: exportar
# a requirements.txt y usar pip, que sí resuelve estable.
#
# --timeout 120 (default de pip: 15s): con ese fix aplicado, el build real
# siguiente falló igual, pero con un error distinto y explícito esta vez --
# "ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', ...)
# Read timed out" bajando un wheel grande (torch) bajo contención del host
# (verificado en el trace real del job). 15s alcanza en una red normal, no
# en este host compartido con otros 2 proyectos corriendo contenedores en
# paralelo -- pip ya reintenta solo (5 veces por defecto), lo que faltaba
# era margen por intento.
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt --only main \
    && pip install --no-cache-dir --timeout 120 -r requirements.txt \
    && rm requirements.txt

# Copiar la API
COPY api/ ./api/

# El wrapper mlflow.pyfunc registrado por core_ml/src/train.py referencia
# estos módulos por ruta (`src.model_architecture`, `src.mlflow_utils`); deben
# existir aquí, con el mismo nombre de paquete, para poder deserializarlo al
# cargar el modelo desde el MLflow Model Registry. batch_inference.py es el
# entrypoint que usa el CronJob de scoring por lotes, y
# core_ml/src/monitoring/stream_consumer.py el que usa el Deployment de
# streaming (kubernetes/base/stream-consumer.yaml) -- mismo contenedor para
# los tres.
COPY core_ml/src/__init__.py ./src/__init__.py
COPY core_ml/src/model_architecture.py ./src/model_architecture.py
COPY core_ml/src/mlflow_utils.py ./src/mlflow_utils.py
COPY core_ml/src/data_contracts.py ./src/data_contracts.py
COPY core_ml/src/events.py ./src/events.py
COPY core_ml/src/logging_config.py ./src/logging_config.py
COPY core_ml/src/batch_inference.py ./src/batch_inference.py
COPY core_ml/src/monitoring/ ./src/monitoring/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
