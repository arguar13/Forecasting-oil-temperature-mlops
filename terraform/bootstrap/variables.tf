variable "aws_region" {
  description = "Región de AWS"
  default     = "us-east-1"
}

variable "state_bucket_name" {
  description = "Nombre del bucket S3 para el state remoto de Terraform (debe coincidir con backend \"s3\" en ../provider.tf)"
  type        = string
  default     = "dlinear-terraform-state-prod"
}

variable "lock_table_name" {
  description = "Nombre de la tabla DynamoDB para locking del state (debe coincidir con dynamodb_table en ../provider.tf)"
  type        = string
  default     = "terraform-state-locks"
}

variable "db_secret_name" {
  description = "Nombre del secreto de Secrets Manager con la password de RDS (debe coincidir con ../rds.tf)"
  type        = string
  default     = "dlinear-mlops/db-password"
}

variable "db_username" {
  description = "Usuario administrador de RDS (debe coincidir con db_username en ../variables.tf)"
  type        = string
  default     = "mlopsadmin"
}

variable "operator_user_name" {
  description = "Nombre del usuario IAM operador que ejecuta el primer terraform apply (Fase L/N de la guía) y las operaciones manuales posteriores"
  type        = string
  default     = "dlinear-mlops-admin"
}
