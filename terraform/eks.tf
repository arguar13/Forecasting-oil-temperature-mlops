module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.16.0"

  cluster_name = "${var.project_name}-eks"
  # 1.30 salió de soporte extendido en AWS (el AMI administrado dejó de
  # publicarse para esa versión, "terraform apply" fallaba con
  # "Requested AMI for this version 1.30 is not supported") -- 1.36 es la
  # versión estándar actual (mayor ventana de soporte).
  cluster_version = "1.36"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # AGREGADO: Habilitar OIDC para IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Sin esto, el módulo pone como único "KeyAdministrator" de la KMS key de
  # secrets al caller ACTUAL del apply -- terraform:plan en CI (asumiendo
  # TerraformCI_OIDC_Role) fallaba con "AccessDeniedException: ... because
  # no resource-based policy allows the kms:DescribeKey action", porque la
  # key policy nunca delega en IAM (no incluye al account root). Se listan
  # ambos: el operador local (Fase L/N) y el rol de CI, para que cualquiera
  # de los dos pueda leer/administrar la key en applies subsecuentes.
  kms_key_administrators = [
    data.aws_caller_identity.current.arn,
    aws_iam_role.terraform_ci_role.arn,
  ]

  # Nombre corto deliberado: el módulo EKS genera un IAM role name_prefix como
  # "<clave>-eks-node-group-", limitado a 38 caracteres por la API de IAM --
  # "dlinear_inference_nodes" lo excedía ("terraform plan" fallaba con
  # "expected length of name_prefix to be in the range (1 - 38)").
  eks_managed_node_groups = {
    inference = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "ON_DEMAND"
    }
  }

  tags = {
    Environment = "production"
  }
}
