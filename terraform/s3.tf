# Nombre determinístico (Account ID, no un sufijo aleatorio): un
# `random_id` obligaría a un paso manual ("copiar el nombre real desde
# `terraform output` a los manifiestos de K8s") cada vez que se recree el
# bucket -- exactamente el tipo de mutación frágil que esta fase elimina.
# Con el Account ID, el nombre es conocido de antemano por Terraform *y*
# por el ConfigMap de Kustomize (kubernetes/base/configmap.yaml) sin
# necesidad de pasarse valores en runtime.
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-model-artifacts-${data.aws_caller_identity.current.account_id}"

  # Sin esto, "terraform destroy" falla con "BucketNotEmpty" en cuanto el
  # bucket tiene un solo objeto real (dvc-store/, mlflow/, batch/ -- este
  # bucket SIEMPRE los tiene una vez usado). Versionado está activo, así
  # que force_destroy purga todas las versiones, no solo la actual.
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

output "model_artifacts_bucket_name" {
  description = "Nombre determinístico del bucket -- referenciado por kubernetes/base/configmap.yaml"
  value       = aws_s3_bucket.model_artifacts.bucket
}
