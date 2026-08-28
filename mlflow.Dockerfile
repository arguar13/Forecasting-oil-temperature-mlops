# Dockerfile para el Tracking Server de MLflow (docker-compose.yml).
#
# La imagen oficial ghcr.io/mlflow/mlflow no trae driver de Postgres: arranca
# con "ModuleNotFoundError: No module named 'psycopg2'" en cuanto
# --backend-store-uri es postgresql://... (ver docker-compose.yml::mlflow).
# psycopg2-binary lo resuelve sin recompilar libpq en la imagen.
FROM ghcr.io/mlflow/mlflow:v2.6.0

RUN pip install --no-cache-dir psycopg2-binary==2.9.9
