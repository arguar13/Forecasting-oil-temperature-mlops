# ecr.tf
#
# Un solo repositorio: la imagen de la API (sirve tanto el endpoint HTTP
# como el CronJob de batch inference, ver Dockerfile). MLflow corre con la
# imagen oficial ghcr.io/mlflow/mlflow directamente (ver
# kubernetes/base/mlflow.yaml) -- no se publica en ECR.

resource "aws_ecr_repository" "dlinear_api" {
  name                 = "dlinear-forecast-api"
  image_tag_mutability = "IMMUTABLE"

  # Sin esto, "terraform destroy" falla con RepositoryNotEmptyException en
  # cuanto el repo tiene una sola imagen publicada.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Conserva las últimas 10 imágenes (permite rollback) y expira el resto.
resource "aws_ecr_lifecycle_policy" "cleanup_policy" {
  repository = aws_ecr_repository.dlinear_api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 10 images (rollback); expire the rest"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

output "ecr_repository_url" {
  value = aws_ecr_repository.dlinear_api.repository_url
}
