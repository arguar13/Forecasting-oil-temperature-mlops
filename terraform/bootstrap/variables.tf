variable "aws_region" {
  description = "Región de AWS"
  type        = string
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

variable "db_username" {
  description = "Usuario administrador de RDS (debe coincidir con db_username en ../variables.tf)"
  type        = string
  default     = "mlopsadmin"
}

variable "operator_user_name" {
  description = "Nombre del usuario IAM operador que ejecuta el primer terraform apply y las operaciones manuales posteriores"
  type        = string
  default     = "dlinear-mlops-admin"
}
