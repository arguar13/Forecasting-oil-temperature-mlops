# IAM de este proyecto, organizado en dos roles:
#
#   - dlinear_api_s3_read: policy de S3 adjuntada al rol del node group de
#     EKS (ver eks.tf) -- todos los Pods del cluster heredan el permiso del
#     nodo donde corren, en vez de un rol de IAM por ServiceAccount (IRSA).
#   - gitlab_ci_role: el rol que GitLab CI asume vía OIDC (sin credenciales
#     de larga duración) para hacer build/push de la imagen a ECR, leer/
#     escribir el bucket de artefactos (S3, incluido el remoto de DVC) y
#     desplegar en el cluster.

# Least-privilege: solo el bucket de artefactos de este proyecto (no
# AmazonS3ReadOnlyAccess a nivel de cuenta), salvo el prefijo batch/, donde
# el CronJob de batch inference (core_ml/src/batch_inference.py) sí necesita
# escribir su resultado. Acotado a ese prefijo -- no al bucket entero -- para
# no poder pisar por error el artefacto del modelo (mlflow/) ni el store de
# DVC (dvc-store/).
resource "aws_iam_policy" "dlinear_api_s3_read" {
  name = "dlinear-api-s3-read-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.model_artifacts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/batch/*"
      }
    ]
  })
}

# ============================================================
# OIDC federation con GitLab.com -- permite que GitLab CI asuma un rol de
# AWS con credenciales temporales (sts:AssumeRoleWithWebIdentity), sin
# guardar un access key de larga duración como variable de CI.
# ============================================================
data "tls_certificate" "gitlab" {
  url = "https://gitlab.com"
}

resource "aws_iam_openid_connect_provider" "gitlab" {
  url             = "https://gitlab.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.gitlab.certificates[0].sha1_fingerprint]
}

# Nombre exacto que .gitlab-ci.yml espera en `sts assume-role-with-web-identity`.
resource "aws_iam_role" "gitlab_ci_role" {
  name = "GitLabCI_OIDC_Role"

  # `sub` se restringe al proyecto y rama exactos que despliegan -- ningún
  # otro proyecto/rama de GitLab.com puede asumir este rol.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.gitlab.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "gitlab.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "gitlab.com:sub" = "project_path:${var.gitlab_project_path}:ref_type:branch:ref:main"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "gitlab_ci_ecr" {
  name = "ecr-push-pull-dlinear-repo"
  role = aws_iam_role.gitlab_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = aws_ecr_repository.dlinear_api.arn
      }
    ]
  })
}

# Least-privilege: S3 limitado al bucket de este proyecto (no
# AmazonS3FullAccess). Cubre tanto los artefactos de modelo/MLflow como el
# remoto de DVC (mismo bucket, prefijo dvc-store/, ver .dvc/config).
resource "aws_iam_role_policy" "gitlab_ci_s3" {
  name = "s3-rw-model-artifacts-bucket"
  role = aws_iam_role.gitlab_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.model_artifacts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = "${aws_s3_bucket.model_artifacts.arn}/*"
      }
    ]
  })
}

# El job de deploy (.gitlab-ci.yml::deploy) necesita poder describir el
# cluster para generar el kubeconfig (`aws eks update-kubeconfig`). El
# permiso para de verdad operar sobre los recursos de Kubernetes (RBAC) se
# concede aparte, mapeando este rol en el ConfigMap aws-auth del cluster --
# un paso manual, una sola vez por cluster (ver README.md, sección Desplegar).
resource "aws_iam_role_policy" "gitlab_ci_eks_describe" {
  name = "eks-describe-cluster"
  role = aws_iam_role.gitlab_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["eks:DescribeCluster"]
        Resource = module.eks.cluster_arn
      }
    ]
  })
}
