"""Prueba que LocalStack ("la falsa nube") emula S3, SQS y Secrets Manager
con el mismo API de boto3 que usaríamos contra AWS real -- así el código de
la aplicación (y estos mismos tests) nunca necesitan tocar una cuenta AWS
real ni gastar dinero para validarse.
"""

import json

import boto3
import pytest
from testcontainers.community.localstack import LocalStackContainer

pytestmark = pytest.mark.integration


def _client(localstack: LocalStackContainer, service: str):
    return boto3.client(
        service,
        endpoint_url=localstack.get_url(),
        aws_access_key_id="test",
        aws_secret_access_key="test",  # pragma: allowlist secret -- credencial fija de LocalStack
        region_name="us-east-1",
    )


def test_localstack_s3_bucket_roundtrip():
    with LocalStackContainer("localstack/localstack:4.14.0").with_services("s3") as localstack:
        s3 = _client(localstack, "s3")
        bucket = "integration-test-bucket"

        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key="hello.txt", Body=b"hola mundo")

        obj = s3.get_object(Bucket=bucket, Key="hello.txt")
        assert obj["Body"].read() == b"hola mundo"


def test_localstack_sqs_send_and_receive_message():
    with LocalStackContainer("localstack/localstack:4.14.0").with_services("sqs") as localstack:
        sqs = _client(localstack, "sqs")

        queue_url = sqs.create_queue(QueueName="batch-inference-events")["QueueUrl"]
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps({"event": "smoke-test"}))

        response = sqs.receive_message(QueueUrl=queue_url, WaitTimeSeconds=5)
        messages = response.get("Messages", [])
        assert len(messages) == 1
        assert json.loads(messages[0]["Body"]) == {"event": "smoke-test"}


def test_localstack_secrets_manager_stores_and_retrieves_secret():
    with LocalStackContainer("localstack/localstack:4.14.0").with_services(
        "secretsmanager"
    ) as localstack:
        secrets = _client(localstack, "secretsmanager")
        secret_value = {
            "username": "mlopsadmin",
            "password": "localpassword123",  # pragma: allowlist secret -- credencial fija de prueba, nunca sale de LocalStack
        }

        secrets.create_secret(
            Name="dlinear/mlflow-db-credentials", SecretString=json.dumps(secret_value)
        )

        response = secrets.get_secret_value(SecretId="dlinear/mlflow-db-credentials")
        assert json.loads(response["SecretString"]) == secret_value
