# VPC de 2 AZ: suficiente para un cluster de aprendizaje (RDS y el bucket de
# artefactos no necesitan una tercera zona de disponibilidad) y más barata
# que 3 AZ -- un solo NAT Gateway (single_nat_gateway) alcanza para que los
# nodos privados salgan a internet a pullear imágenes/paquetes.
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.2"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  # Subredes exclusivas para RDS -- EKS y RDS quedan en subredes distintas
  # aunque compartan VPC.
  database_subnets             = ["10.0.201.0/24", "10.0.202.0/24"]
  create_database_subnet_group = true

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  tags = {
    Environment = "production"
    Project     = var.project_name
  }
}
