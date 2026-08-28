terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  # Estado local deliberado: este módulo crea el backend remoto (bucket S3
  # + tabla DynamoDB) que usará el resto del proyecto (../provider.tf), así
  # que no puede depender de sí mismo. Se ejecuta una única vez, con
  # credenciales administrativas, y su .tfstate (ignorado por Git) queda en
  # esta máquina como registro de qué recursos de bootstrap existen.
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
