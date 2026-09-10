module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.16.0"

  cluster_name = "${var.project_name}-eks"
  # 1.30 salió de soporte extendido en AWS (el AMI administrado dejó de
  # publicarse para esa versión) -- 1.36 es la versión estándar actual.
  cluster_version = "1.36"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # metrics-server no viene instalado por defecto en EKS (a diferencia de
  # coredns/kube-proxy/vpc-cni). Sin él, el HorizontalPodAutoscaler
  # (kubernetes/base/hpa.yaml) no puede leer métricas de CPU y queda
  # "Degraded" -- se instala como addon administrado por AWS para mantener
  # todo el cluster reproducible desde Terraform.
  cluster_addons = {
    metrics-server = {
      most_recent = true
    }
  }

  # El módulo ya abre el security group de los nodos al control plane para
  # los puertos de webhook comunes y kubelet, pero no para el 10251 que usa
  # metrics-server -- sin esto, la API metrics.k8s.io queda inalcanzable.
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

  # Un solo node group -- suficiente para un proyecto de aprendizaje. El HPA
  # (kubernetes/base/hpa.yaml) escala los Pods dentro de este rango de nodos,
  # sin necesidad de un segundo node group especializado.
  eks_managed_node_groups = {
    inference = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      # t3.medium resultó insuficiente en la práctica: con MLflow + varias
      # réplicas de la API (PyTorch cargado) los nodos entraban en
      # MemoryPressure. t3.large da margen suficiente.
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"

      # "disk_size" no alcanza con un custom launch template (nuestro caso) --
      # el submódulo lo ignora en silencio. block_device_mappings sí se
      # respeta, y evita quedarse sin espacio (ej. al pullear la imagen con
      # PyTorch) con el volumen raíz de 20GB por defecto del AMI.
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

      # El permiso de S3 se adjunta directamente al rol del node group --
      # cualquier Pod del cluster hereda el permiso del nodo donde corre
      # (ver kubernetes/base/serviceaccount.yaml, que no lleva anotaciones
      # IRSA por este motivo).
      iam_role_additional_policies = {
        s3_access = aws_iam_policy.dlinear_api_s3_read.arn
      }
    }
  }

  tags = {
    Environment = "production"
  }
}
