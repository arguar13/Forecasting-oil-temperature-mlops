#!/bin/sh
# Se ejecuta automáticamente una vez que LocalStack está listo (montado en
# /etc/localstack/init/ready.d/, ver docker-compose.yml). Provisiona los
# mismos recursos "AWS" que Terraform crearía en la nube real -- así la app
# y sus tests de integración encuentran el bucket/cola/secreto esperados
# desde el primer arranque, sin pasos manuales.
set -e

BUCKET="mlops-portafolio-proj3-models"
QUEUE="batch-inference-events"
SECRET="dlinear/mlflow-db-credentials"  # pragma: allowlist secret -- nombre del secreto, no un valor

echo "[localstack-init] Creando bucket S3 s3://${BUCKET} ..."
awslocal s3 mb "s3://${BUCKET}"
awslocal s3api put-bucket-versioning --bucket "${BUCKET}" --versioning-configuration Status=Enabled

echo "[localstack-init] Creando cola SQS ${QUEUE} ..."
awslocal sqs create-queue --queue-name "${QUEUE}"

echo "[localstack-init] Creando secreto ${SECRET} en Secrets Manager ..."
awslocal secretsmanager create-secret \
  --name "${SECRET}" \
  --secret-string '{"username":"mlopsadmin","password":"localpassword123"}'  # pragma: allowlist secret -- credencial fija del stack local, nunca se usa fuera de docker-compose

echo "[localstack-init] Bootstrap completado."
