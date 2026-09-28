# Tests de Terraform 100% offline: todos los providers están mockeados
# (mock_provider), así que no hacen falta credenciales ni se llama a AWS.
#
#   make tf-test
#
# Verifican el contrato de cifrado del RDS de MLflow (rds.tf): una instancia
# nueva nace cifrada, y activar el cifrado sobre una instancia existente sin
# cifrar NO la reemplaza (lo que borraría el backend store de MLflow).

# Solo se fija lo que el stack raíz lee de verdad: el account_id arma el
# nombre del bucket (s3.tf). Los módulos de la comunidad, que consumirían
# muchos más datos del provider, no se ejecutan (override_module, abajo).
mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
      arn        = "arn:aws:iam::123456789012:user/mock"
    }
  }
}

# Los módulos de la comunidad (VPC, EKS) no se ejecutan: se reemplazan por
# las salidas que el resto del stack consume. Estos tests son sobre el RDS,
# no sobre los internals de terraform-aws-modules.
override_module {
  target = module.vpc
  outputs = {
    vpc_id                     = "vpc-0123456789abcdef0"
    private_subnets            = ["subnet-0aaaaaaaaaaaaaaa1", "subnet-0aaaaaaaaaaaaaaa2"]
    database_subnet_group_name = "mock-db-subnet-group"
  }
}

override_module {
  target = module.eks
  outputs = {
    node_security_group_id = "sg-0123456789abcdef0"
    cluster_arn            = "arn:aws:eks:us-east-1:123456789012:cluster/mock"
  }
}

mock_provider "tls" {
  mock_data "tls_certificate" {
    defaults = {
      certificates = [{ sha1_fingerprint = "0000000000000000000000000000000000000000" }]
    }
  }
}

mock_provider "kubernetes" {}
mock_provider "cloudinit" {}
mock_provider "time" {}

variables {
  db_password = "mock-password-not-a-secret" # pragma: allowlist secret
}

run "new_instance_is_created_encrypted" {
  command = plan

  assert {
    condition     = aws_db_instance.mlflow_db.storage_encrypted == true
    error_message = "Una instancia RDS nueva debe crearse con storage_encrypted = true."
  }
}

# Simula un RDS legado, creado antes de que existiera el cifrado.
run "legacy_unencrypted_instance_exists" {
  command = apply

  variables {
    db_storage_encrypted = false
  }

  assert {
    condition     = aws_db_instance.mlflow_db.storage_encrypted == false
    error_message = "El escenario de partida debe ser una instancia sin cifrar."
  }
}

# Sobre ese estado, el default (true) no debe cambiar storage_encrypted: un
# cambio ahí forzaría destruir y recrear la instancia.
run "enabling_encryption_does_not_replace_legacy_instance" {
  command = plan

  assert {
    condition     = aws_db_instance.mlflow_db.storage_encrypted == false
    error_message = "storage_encrypted cambió sobre una instancia existente: eso fuerza su reemplazo."
  }
}
