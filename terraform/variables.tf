variable "aws_region" {
  description = "Región de AWS"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nombre del proyecto para etiquetado"
  type        = string
  default     = "dlinear-forecast-cluster"
}

variable "db_username" {
  description = "Usuario administrador de la base de datos RDS (MLflow backend store)"
  type        = string
  default     = "mlopsadmin"
}

# Sin default a propósito: una contraseña por defecto en el repo sería un
# secreto comiteado a Git. Pasala con -var="db_password=..." o la variable
# de entorno TF_VAR_db_password, nunca hardcodeada acá.
variable "db_password" {
  description = "Contraseña del usuario administrador de RDS (MLflow backend store)"
  type        = string
  sensitive   = true
}

variable "gitlab_project_path" {
  description = "Ruta del proyecto en GitLab (namespace/proyecto), usada en el trust policy OIDC de GitLabCI_OIDC_Role"
  type        = string
  default     = "personal-group7745334/612-forecasting-oil-temperature-mlops"
}
