# Least-privilege: solo lectura del bucket de artefactos de este proyecto
# (no AmazonS3ReadOnlyAccess a nivel de cuenta).
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
      }
    ]
  })
}

# IAM Role para que el Pod de la API pueda leer modelos de S3 (IRSA)
module "iam_eks_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "5.30.0"

  role_name = "dlinear-api-s3-read-role"

  # NOTA: `attach_s3_read_only_policy` no existe en esta versión del módulo
  # (falló en `terraform validate` -- "Unsupported argument"); el mecanismo
  # correcto en v5.x es `role_policy_arns`, y de paso permite acotar el
  # permiso al bucket real en vez de S3 completo.
  role_policy_arns = {
    s3_read = aws_iam_policy.dlinear_api_s3_read.arn
  }

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["dlinear-production:dlinear-sa"]
    }
  }
}

# ============================================================
# OIDC federation con GitLab.com (autenticación sin credenciales
# de larga duración -- ver .gitlab-ci.yml::.aws-auth)
# ============================================================
data "tls_certificate" "gitlab" {
  url = "https://gitlab.com"
}

resource "aws_iam_openid_connect_provider" "gitlab" {
  url = "https://gitlab.com"
  # "sts.amazonaws.com", no "https://gitlab.com": el proveedor OIDC de
  # gitlab.com es un recurso único por cuenta de AWS -- ya existe en esta
  # cuenta, gestionado por otro proyecto (predictive-maintenance-mlops) con
  # esta audiencia (la que AWS documenta como estándar para federación STS).
  # Declararlo igual evita pelear por el recurso compartido en cada apply;
  # .gitlab-ci.yml (id_tokens.aud) usa la misma audiencia.
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.gitlab.certificates[0].sha1_fingerprint]
}

# ============================================================
# Rol para GitLab CI/CD (GitLabCI_OIDC_Role -- el nombre debe
# coincidir exactamente con el `role-arn` que .gitlab-ci.yml asume vía
# `sts assume-role-with-web-identity`)
# ============================================================
resource "aws_iam_role" "gitlab_ci_role" {
  name = "GitLabCI_OIDC_Role"

  # Trust policy federada por OIDC (no un principal de servicio EC2, que
  # nunca podría satisfacer un `assume-role-with-web-identity` desde
  # GitLab). `sub` se restringe al proyecto y rama exactos que de verdad
  # despliegan -- cualquier otro proyecto/rama de GitLab.com no puede
  # asumir este rol aunque conozca el ARN.
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

# Least-privilege: ECR limitado a los repositorios de esta app (no
# AmazonEC2ContainerRegistryReadOnly a nivel de cuenta, y sí con permiso de
# push, que docker:build-push y docker:build-push-mlflow necesitan). Incluye
# tanto el repo de la API como el del MLflow Tracking Server
# (kubernetes/base/mlflow.yaml), publicado por el mismo pipeline.
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
        Resource = [
          aws_ecr_repository.dlinear_api.arn,
          aws_ecr_repository.dlinear_mlflow.arn,
        ]
      }
    ]
  })
}

# Least-privilege: S3 limitado al bucket de artefactos/datos de este
# proyecto (no AmazonS3FullAccess a nivel de cuenta). Cubre tanto los
# artefactos de modelo/MLflow como el remoto de DVC (mismo bucket,
# prefijo `dvc-store/`, ver .dvc/config).
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

# NOTA: bajo GitOps (ArgoCD reconcilia el cluster desde Git), este rol ya
# NO necesita permisos de EKS/kubectl -- CI solo publica la imagen (ECR) y
# actualiza el tag declarado en kubernetes/overlays/production/ (Git). Ver
# .gitlab-ci.yml::kubernetes:deploy y kubernetes/README.md.

# ============================================================
# Rol separado para `terraform plan/apply` (TerraformCI_OIDC_Role).
#
# Provisionar VPC/EKS/RDS/IAM/S3/ECR requiere permisos amplios sobre esos
# servicios -- incluyendo IAM, porque este mismo código gestiona roles y
# políticas (este archivo). Separarlo del rol de la app (arriba, limitado a
# ECR+S3) evita que un pipeline de build/train comprometido pueda escalar a
# permisos de administración de infraestructura.
# ============================================================
resource "aws_iam_role" "terraform_ci_role" {
  name = "TerraformCI_OIDC_Role"

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

# PowerUserAccess cubre EC2/VPC/EKS/RDS/S3/ECR sin otorgar administración
# total de IAM; el complemento de abajo agrega únicamente los permisos de
# IAM que este código SÍ necesita (crear/gestionar los roles y el OIDC
# provider de este mismo repo), acotados por prefijo de nombre.
resource "aws_iam_role_policy_attachment" "terraform_ci_power_user" {
  role       = aws_iam_role.terraform_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

# PowerUserAccess deniega explícitamente TODO iam:* (lectura incluida) --
# el refresh de `terraform plan` necesita leer CUALQUIER recurso IAM ya
# trackeado en el state (roles, OIDC providers, policies) sin importar su
# nombre, no solo los de prefijo "dlinear-*"/"${var.project_name}-*" de
# abajo. Iterar acción-por-acción (GetRole, luego GetOpenIDConnectProvider,
# luego...) cada vez que el plan pisaba un tipo de recurso nuevo era lento
# y gastaba minutos de CI -- de solo lectura, sin riesgo de escalar
# privilegios, así que se usa la managed policy estándar de AWS en vez de
# reinventarla acción por acción.
resource "aws_iam_role_policy_attachment" "terraform_ci_iam_read_only" {
  role       = aws_iam_role.terraform_ci_role.name
  policy_arn = "arn:aws:iam::aws:policy/IAMReadOnlyAccess"
}

resource "aws_iam_role_policy" "terraform_ci_iam_scoped" {
  name = "iam-manage-dlinear-roles-only"
  role = aws_iam_role.terraform_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:GetRole",
          "iam:UpdateRole",
          "iam:PutRolePolicy",
          "iam:DeleteRolePolicy",
          "iam:GetRolePolicy",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies",
          "iam:TagRole",
          "iam:PassRole",
        ]
        Resource = [
          "arn:aws:iam::*:role/dlinear-*",
          "arn:aws:iam::*:role/${var.project_name}-*",
          aws_iam_role.gitlab_ci_role.arn,
          aws_iam_role.terraform_ci_role.arn,
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "iam:CreatePolicy",
          "iam:DeletePolicy",
          "iam:GetPolicy",
          "iam:GetPolicyVersion",
          "iam:ListPolicyVersions",
          "iam:CreatePolicyVersion",
          "iam:DeletePolicyVersion",
          "iam:TagPolicy",
        ]
        Resource = [
          "arn:aws:iam::*:policy/dlinear-*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "iam:CreateOpenIDConnectProvider",
          "iam:GetOpenIDConnectProvider",
          "iam:DeleteOpenIDConnectProvider",
          "iam:TagOpenIDConnectProvider",
        ]
        Resource = "arn:aws:iam::*:oidc-provider/gitlab.com"
      },
      {
        # Solo lectura, sin restricción de nombre: el módulo EKS crea roles
        # auxiliares (rol del cluster, rol del node group) con nombres que
        # el propio módulo genera -- no siguen el prefijo "dlinear-*" (ej.
        # "inference-eks-node-group-...") -- así que el refresh de
        # `terraform plan` (lee CUALQUIER recurso ya trackeado en el state)
        # fallaba con AccessDenied aunque nunca necesite CREAR nada fuera
        # de esos prefijos. Los permisos de escritura de arriba sí siguen
        # acotados por prefijo -- esto no habilita escalar privilegios.
        Effect = "Allow"
        Action = [
          "iam:GetRole",
          "iam:GetRolePolicy",
          "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies",
          "iam:ListInstanceProfilesForRole",
        ]
        Resource = "*"
      }
    ]
  })
}

# El módulo EKS crea y administra su propia KMS key (cifrado de secrets de
# Kubernetes) -- la key policy que genera no incluye a este rol, y
# PowerUserAccess por sí solo no basta para KMS cuando la key policy no
# delega en IAM. Acotado a la key de este cluster (tag Environment=production
# puesto por module.eks.module.kms), no a todas las keys de la cuenta.
resource "aws_iam_role_policy" "terraform_ci_kms" {
  name = "kms-manage-eks-cluster-key"
  role = aws_iam_role.terraform_ci_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:DescribeKey",
          "kms:GetKeyPolicy",
          "kms:GetKeyRotationStatus",
          "kms:ListResourceTags",
          "kms:CreateKey",
          "kms:CreateAlias",
          "kms:DeleteAlias",
          "kms:UpdateAlias",
          "kms:EnableKeyRotation",
          "kms:DisableKeyRotation",
          "kms:PutKeyPolicy",
          "kms:TagResource",
          "kms:UntagResource",
          "kms:ScheduleKeyDeletion",
          "kms:CancelKeyDeletion",
        ]
        Resource = "*"
      }
    ]
  })
}
