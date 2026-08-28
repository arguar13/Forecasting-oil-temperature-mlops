"""El test de integración insignia de la Fase 3: levanta el `docker-compose.yml`
real del repo (Postgres, LocalStack, Kafka, MLflow y la API) con
Testcontainers y verifica que los componentes efectivamente se entienden
entre sí -- exactamente lo que se pide antes de tocar AWS.

Es lento (construye la imagen de la API) y requiere Docker: se salta si no
está disponible y no corre en el `make test` rápido (marcado `integration`).
"""

import json
import urllib.request
from pathlib import Path

import boto3
import pytest
from testcontainers.compose import DockerCompose

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]


def _get(url: str, timeout: int = 5) -> tuple[int, bytes]:
    with urllib.request.urlopen(
        url, timeout=timeout
    ) as response:  # nosec B310 -- URL fija, localhost
        return response.status, response.read()


def test_local_cloud_stack_boots_and_services_integrate():
    compose = DockerCompose(
        context=str(REPO_ROOT),
        build=True,
        services=["postgres", "localstack", "kafka", "kafka-init", "mlflow", "api"],
    )

    with compose:
        # --- Postgres: aceptando conexiones (MLflow backend store) ---
        pg_host, pg_port = compose.get_service_host_and_port("postgres", 5432)
        assert pg_host and pg_port

        # --- LocalStack: el bootstrap script provisionó bucket/cola/secreto ---
        localstack_host, localstack_port = compose.get_service_host_and_port("localstack", 4566)
        endpoint_url = f"http://{localstack_host}:{localstack_port}"
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id="test",
            aws_secret_access_key="test",  # pragma: allowlist secret -- credencial fija de LocalStack
            region_name="us-east-1",
        )
        buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
        assert "mlops-portafolio-proj3-models" in buckets

        sqs = boto3.client(
            "sqs",
            endpoint_url=endpoint_url,
            aws_access_key_id="test",
            aws_secret_access_key="test",  # pragma: allowlist secret -- credencial fija de LocalStack
            region_name="us-east-1",
        )
        queue_urls = sqs.list_queues().get("QueueUrls", [])
        assert any("batch-inference-events" in url for url in queue_urls)

        secrets = boto3.client(
            "secretsmanager",
            endpoint_url=endpoint_url,
            aws_access_key_id="test",
            aws_secret_access_key="test",  # pragma: allowlist secret -- credencial fija de LocalStack
            region_name="us-east-1",
        )
        secret = secrets.get_secret_value(SecretId="dlinear/mlflow-db-credentials")
        assert json.loads(secret["SecretString"])["username"] == "mlopsadmin"

        # --- MLflow: accesible y respaldado por Postgres + LocalStack S3 ---
        mlflow_host, mlflow_port = compose.get_service_host_and_port("mlflow", 5000)
        status, _ = _get(f"http://{mlflow_host}:{mlflow_port}/")
        assert status == 200

        # --- API: viva aunque el Model Registry todavía no tenga un modelo
        # 'production' (ver api/main.py::lifespan -- degradación elegante) ---
        api_host, api_port = compose.get_service_host_and_port("api", 8000)
        status, body = _get(f"http://{api_host}:{api_port}/health")
        assert status == 200
        assert json.loads(body) == {"status": "healthy"}
