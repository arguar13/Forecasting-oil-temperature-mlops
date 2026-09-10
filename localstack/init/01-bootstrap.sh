#!/bin/sh
# Se ejecuta automáticamente una vez que LocalStack está listo (montado en
# /etc/localstack/init/ready.d/, ver docker-compose.yml). Crea el mismo
# bucket S3 que Terraform crearía en la nube real, para que la app encuentre
# el bucket esperado desde el primer arranque, sin pasos manuales.
set -e

BUCKET="mlops-portafolio-proj3-models"

echo "[localstack-init] Creando bucket S3 s3://${BUCKET} ..."
awslocal s3 mb "s3://${BUCKET}"
awslocal s3api put-bucket-versioning --bucket "${BUCKET}" --versioning-configuration Status=Enabled

echo "[localstack-init] Bootstrap completado."
