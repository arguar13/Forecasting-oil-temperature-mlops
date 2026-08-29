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
  #
  # ARN del operador construido explícitamente, NO
  # data.aws_caller_identity.current.arn: ese data source es dinámico --
  # refleja a quien esté corriendo ESTE apply en particular. Usarlo
  # directamente hacía que cada apply sobreescribiera la política de la key
  # con la identidad del momento (el operador local en un apply, luego
  # TerraformCI_OIDC_Role en el siguiente apply del pipeline), dejando al
  # otro sin acceso -- verificado en la cuenta real: un apply del pipeline
  # dejó al operador sin poder leer la key que él mismo había creado.
  kms_key_administrators = [
    "arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/${var.operator_user_name}",
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

      # t3.medium (primer intento) resultó insuficiente en producción real:
      # los 3 nodos entraron en MemoryPressure permanente apenas con
      # ArgoCD + el runner self-hosted (kubernetes/gitlab-runner/) + los
      # pods efímeros de sus jobs + MLflow + 3 réplicas de la API (con
      # PyTorch cargado) -- el rolling update de la API nunca lograba
      # estabilizar un solo pod Ready (kubelet perdía el estado de los
      # pods bajo presión, "ContainerStatusUnknown" en bucle). Verificado
      # en el cluster real (`kubectl get nodes` mostraba
      # MemoryPressure=True en los 3 nodos de forma sostenida).
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"

      # Default del módulo (20GB, ~18GB allocatable) resultó insuficiente:
      # train_model falló con "No space left on device" instalando PyTorch
      # -- entre las imágenes ya pulleadas en cada nodo (dlinear-api y
      # dlinear-mlflow-server, ambas con PyTorch/CUDA, varios GB cada una)
      # y el propio pip install de este job necesitando espacio de sobra
      # para las mismas dependencias, 18GB no alcanzaba. Verificado en el
      # cluster real (`kubectl describe node` -- ephemeral-storage
      # allocatable ~18181869946 bytes).
      disk_size = 50
    }
  }

  tags = {
    Environment = "production"
  }
}
