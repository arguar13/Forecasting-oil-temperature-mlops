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
