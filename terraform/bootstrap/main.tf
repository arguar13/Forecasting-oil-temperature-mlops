# ==============================================================================
# Bootstrap
#
# Crea, con credenciales administrativas de un solo uso, los recursos que el
# resto del proyecto (../*.tf) necesita para existir *antes* de su propio
# `terraform init`: el backend remoto (S3 + DynamoDB), la password de RDS, y
# un usuario IAM operador -- todo como Infraestructura como Código, sin
# pasos manuales de AWS CLI/consola.
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
# Password de RDS -- se genera acá una única vez y se pasa a mano como
# var.db_password al stack principal (../rds.tf) y como valor del Secret de
# Kubernetes (kubernetes/base/secret.yaml). Sin Secrets Manager de por medio:
# un valor sensible, generado una vez, copiado a los dos lugares que lo usan.
# ------------------------------------------------------------------------------
resource "random_password" "db_password" {
  length  = 32
  special = false # Evita caracteres especiales incompatibles con la URI de conexión de PostgreSQL
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

output "db_password" {
  value     = random_password.db_password.result
  sensitive = true
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
