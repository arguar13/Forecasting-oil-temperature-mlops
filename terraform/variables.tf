variable "aws_region" {
  description = "Región de AWS"
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre del proyecto para etiquetado"
  default     = "dlinear-forecast-cluster"
}

# Antes referenciadas en rds.tf (var.environment, var.db_username) sin
# estar declaradas -- `terraform plan` fallaba con "Reference to
# undeclared input variable".
variable "environment" {
  description = "Entorno de despliegue (dev | staging | prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment debe ser uno de: dev, staging, prod."
  }
}

variable "db_username" {
  description = "Usuario administrador de la base de datos RDS (MLflow backend store)"
  type        = string
  default     = "mlopsadmin"
}

variable "gitlab_project_path" {
  description = "Ruta del proyecto en GitLab (namespace/proyecto), usada en el trust policy OIDC de GitLabCIRole"
  type        = string
  default     = "personal-group7745334/612-forecasting-oil-temperature-mlops"
}

# Debe coincidir con operator_user_name en terraform/bootstrap/variables.tf.
# Usado (no data.aws_caller_identity.current.arn) para el key administrator
# de la KMS key de EKS en eks.tf -- ese data source es dinámico, cambia
# según quién corra el apply en ese momento (el usuario operador local, o
# TerraformCI_OIDC_Role en el pipeline). Usarlo directamente hace que cada
# apply SOBREESCRIBA la política de la key con la identidad de quien la
# corrió último, dejando afuera a cualquier otro administrador legítimo --
# verificado en la cuenta real: un apply del pipeline dejó al usuario
# operador sin permiso para leer la key que él mismo había creado.
variable "operator_user_name" {
  description = "Nombre del usuario IAM operador (Fase L/N) -- administrador estable de la KMS key de EKS"
  type        = string
  default     = "dlinear-mlops-admin"
}

# Sin default a propósito (como git_repo_token en el proyecto 611): un
# email de alerta wrong-but-present es peor que uno explícitamente
# requerido -- nadie debería heredar un destinatario de alertas por accidente.
# Ver terraform/alarms.tf.
variable "alert_email" {
  description = "Email que recibe las alertas de CloudWatch (dead man's switch del CronJob de batch inference). Suscripción SNS pendiente de confirmación manual (AWS envía un email de verificación al aplicar)."
  type        = string
}
