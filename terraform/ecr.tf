# ecr.tf

# 1. Creación del repositorio ECR
# NOTA: el nombre debe coincidir exactamente con $ECR_REPOSITORY en
# .gitlab-ci.yml -- antes decía "dlinear-etdataset-api" aquí pero
# "dlinear-forecast-api" en el pipeline, así que `docker push` apuntaba a
# un repositorio que Terraform nunca creaba.
resource "aws_ecr_repository" "dlinear_api" {
  name                 = "dlinear-forecast-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# 2. Política de ciclo de vida: conserva las últimas 10 imágenes (permite
# rollback) y expira el resto para controlar costos -- la versión anterior
# ("mantener solo 1") eliminaba la imagen previa en cuanto se publicaba la
# siguiente, sin nada a lo que hacer rollback.
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

# ============================================================
# Repositorio ECR para la imagen del MLflow Tracking Server en
# producción (kubernetes/base/mlflow.yaml) -- reutiliza mlflow.Dockerfile,
# la misma imagen que docker-compose.yml levanta en local, pero publicada
# en ECR para que el Deployment de EKS pueda descargarla.
# ============================================================
resource "aws_ecr_repository" "dlinear_mlflow" {
  name                 = "dlinear-mlflow-server"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "mlflow_cleanup_policy" {
  repository = aws_ecr_repository.dlinear_mlflow.name

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

output "mlflow_ecr_repository_url" {
  value = aws_ecr_repository.dlinear_mlflow.repository_url
}
