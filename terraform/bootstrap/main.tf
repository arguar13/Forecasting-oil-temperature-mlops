# ==============================================================================
# Bootstrap (Fase L/N de Guia_Completa_Ejecucion_Proyecto 612.docx)
#
# Crea, con credenciales administrativas de un solo uso, los recursos que el
# resto del proyecto (../*.tf) necesita para existir *antes* de su propio
# `terraform init`: el backend remoto (S3 + DynamoDB), el secreto de RDS, y
# un usuario IAM operador -- reemplaza los pasos manuales de AWS CLI/consola
# que la guía documentaba, por Infraestructura como Código.
# ==============================================================================

# ------------------------------------------------------------------------------
# Backend remoto del state principal (../provider.tf::backend "s3")
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "tf_state" {
  bucket = var.state_bucket_name

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tf_locks" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# ------------------------------------------------------------------------------
# Secreto de la password de RDS (../rds.tf lo lee por nombre exacto)
# ------------------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 32
  special = false # Evita caracteres especiales incompatibles con la URI de conexión de PostgreSQL
}

resource "aws_secretsmanager_secret" "db_password" {
  name = var.db_secret_name
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    db_password = random_password.db_password.result
  })
}

# ------------------------------------------------------------------------------
# Usuario IAM operador -- ejecuta el primer `terraform apply` del stack
# principal (antes de que exista el rol OIDC de GitLab) y las operaciones
# manuales de kubectl/AWS CLI posteriores. Creado vía Terraform, no consola.
# ------------------------------------------------------------------------------
resource "aws_iam_user" "operator" {
  name = var.operator_user_name
  tags = {
    Purpose = "dlinear-mlops-bootstrap-operator"
  }
}

resource "aws_iam_access_key" "operator" {
  user = aws_iam_user.operator.name
}

resource "aws_iam_user_policy_attachment" "operator_admin" {
  user       = aws_iam_user.operator.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

output "state_bucket_name" {
  value = aws_s3_bucket.tf_state.bucket
}

output "lock_table_name" {
  value = aws_dynamodb_table.tf_locks.name
}

output "db_secret_arn" {
  value = aws_secretsmanager_secret.db_password.arn
}

output "operator_access_key_id" {
  value = aws_iam_access_key.operator.id
}

output "operator_secret_access_key" {
  value     = aws_iam_access_key.operator.secret
  sensitive = true
}

output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}
