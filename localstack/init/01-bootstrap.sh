#!/bin/sh
# Se ejecuta automáticamente una vez que LocalStack está listo (montado en
# /etc/localstack/init/ready.d/, ver docker-compose.yml). Crea el mismo
# bucket S3 que Terraform crearía en la nube real, para que la app encuentre
# el bucket esperado desde el primer arranque, sin pasos manuales.
set -e

BUCKET="mlops-portafolio-proj3-models"

# Idempotente: con `set -e`, un `s3 mb` sobre un bucket que ya existe (p. ej.
# si el estado persistió entre reinicios) abortaría el script.
if awslocal s3api head-bucket --bucket "${BUCKET}" >/dev/null 2>&1; then
  echo "[localstack-init] El bucket s3://${BUCKET} ya existe, no se recrea."
else
  echo "[localstack-init] Creando bucket S3 s3://${BUCKET} ..."
  awslocal s3 mb "s3://${BUCKET}"
fi
awslocal s3api put-bucket-versioning --bucket "${BUCKET}" --versioning-configuration Status=Enabled

echo "[localstack-init] Bootstrap completado."
