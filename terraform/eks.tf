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

  # metrics-server: EKS no lo instala por defecto (a diferencia de
  # coredns/kube-proxy/vpc-cni, que el propio control plane gestiona sin
  # necesidad de declararlos aquí). Sin él, la API metrics.k8s.io no existe
  # -- kubernetes/base/api.yaml define un HorizontalPodAutoscaler
  # (dlinear-hpa) que, sin esta API, queda permanentemente Degraded
  # ("unable to fetch metrics from resource metrics API: the server could
  # not find the requested resource (get pods.metrics.k8s.io)"),
  # verificado en el cluster real vía `kubectl get application ... -o json`
  # (ArgoCD reporta la Application entera como Degraded por este único
  # recurso, aunque Deployments/Services estén sanos). Se instala como
  # addon administrado por AWS (no un manifest aparte vía kubectl) para
  # mantener todo el cluster reproducible desde Terraform.
  cluster_addons = {
    metrics-server = {
      most_recent = true
    }
  }

  # Sin esto, el add-on de arriba queda instalado (`Deployment 2/2 Running`)
  # pero inalcanzable: el módulo ya abre el security group de los nodos al
  # control plane para los puertos "webhook" comunes (443/4443/6443/8443/9443)
  # y kubelet (10250) -- pero NO para el 10251 que usa metrics-server,
  # porque no todo cluster lo instala. El APIService v1beta1.metrics.k8s.io
  # quedaba "FailedDiscoveryCheck: context deadline exceeded" (verificado:
  # `kubectl get apiservice v1beta1.metrics.k8s.io -o json` -- timeout
  # intentando https://<pod-ip>:10251/..., mientras `kubectl logs`/`exec`
  # -- que sí usan el 10250 ya abierto -- funcionaban con normalidad). Sin
  # metrics.k8s.io, el HorizontalPodAutoscaler (kubernetes/base/api.yaml)
  # queda permanentemente Degraded.
  node_security_group_additional_rules = {
    ingress_cluster_to_node_metrics_server = {
      description                   = "Cluster API to node metrics-server webhook"
      protocol                      = "tcp"
      from_port                     = 10251
      to_port                       = 10251
      type                          = "ingress"
      source_cluster_security_group = true
    }
  }

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

      # NO usar "disk_size" aquí: el submódulo eks-managed-node-group lo
      # ignora en silencio cuando usa un custom launch template (nuestro
      # caso, forzado por otras opciones) -- ver
      # .terraform/modules/eks/modules/eks-managed-node-group/main.tf:332
      # ("disk_size = var.use_custom_launch_template ? null : var.disk_size").
      # Un primer intento con "disk_size = 50" quedó como no-op total: el
      # nodo real seguía con el volumen raíz de 20GB del AMI por defecto
      # (verificado con `aws ec2 describe-volumes` sobre una instancia real
      # -- Size: 20, Device: /dev/xvda -- y una sola versión del launch
      # template sin BlockDeviceMappings). DiskPressure=True en producción
      # y jobs de CI fallando con "no space left on device" incluso para
      # pullear la imagen base python:3.10, no solo instalando PyTorch.
      # block_device_mappings sí se respeta con custom launch template.
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 50
            volume_type           = "gp3"
            delete_on_termination = true
          }
        }
      }
    }
  }

  tags = {
    Environment = "production"
  }
}
