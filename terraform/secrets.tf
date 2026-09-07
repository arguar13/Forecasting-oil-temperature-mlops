# Token de GitLab Pipeline Trigger, leido por
# core_ml/src/monitoring/mitigation.py (corre dentro del pod
# stream-consumer) para disparar un reentrenamiento automatico ante un
# ALERT de drift confirmado.
#
# Mismo patron ya establecido en este proyecto para credenciales externas
# (ver rds.tf::db_password): el secreto se crea A MANO en Secrets Manager
# una sola vez -- GitLab, no Terraform, es quien emite el valor (Settings >
# CI/CD > Pipeline triggers de este proyecto), asi que no hay nada que
# generar. Terraform solo lo referencia como dato, nunca lo gestiona como
# resource.
#
#   aws secretsmanager create-secret \
#     --name dlinear-forecast-cluster/gitlab-pipeline-trigger-token \
#     --secret-string '{"token":"<el token que genera GitLab>"}'
#
# El pod stream-consumer no lee Secrets Manager directamente -- igual que
# mlflow-db-credentials (kubernetes/base/mlflow.yaml), el valor se
# materializa a mano como un Secret de Kubernetes (ver
# kubernetes/base/stream-consumer.yaml para el comando exacto), asi que
# este data source no habilita ningun permiso IAM nuevo para ningun pod.
data "aws_secretsmanager_secret" "gitlab_pipeline_trigger_token" {
  name = "${var.project_name}/gitlab-pipeline-trigger-token"
}

data "aws_secretsmanager_secret_version" "gitlab_pipeline_trigger_token" {
  secret_id = data.aws_secretsmanager_secret.gitlab_pipeline_trigger_token.id
}

output "gitlab_pipeline_trigger_token_secret_name" {
  description = "Nombre en Secrets Manager. `aws secretsmanager get-secret-value --secret-id <este nombre>` es como se materializa el Secret de Kubernetes gitlab-trigger-credentials."
  value       = data.aws_secretsmanager_secret.gitlab_pipeline_trigger_token.name
}
