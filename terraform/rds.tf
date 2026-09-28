# Security Group para RDS (Zero-Trust con EKS)
resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Allow PostgreSQL traffic exclusively from EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# Instancia única de RDS para el backend store de MLflow -- sin Multi-AZ ni
# réplicas de lectura, suficiente para un proyecto de aprendizaje (Multi-AZ
# duplica el costo por una disponibilidad que este proyecto no necesita).
#
# La contraseña se pasa por variable de Terraform (var.db_password, sin
# default -- ver variables.tf), no se lee de AWS Secrets Manager: menos
# piezas que gestionar. Los pods de Kubernetes nunca leen esta contraseña
# desde Terraform ni desde AWS -- se crea a mano, una sola vez, como un
# Secret plano de Kubernetes (ver kubernetes/base/mlflow.yaml).
resource "aws_db_instance" "mlflow_db" {
  identifier        = "${var.project_name}-backend-db"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  engine            = "postgres"
  engine_version    = "18.3"
  # Sin `db_name`, RDS Postgres no crea ninguna base de datos inicial --
  # `mlflow server --backend-store-uri postgresql://.../mlflow_db` fallaría
  # con "database does not exist". Mismo nombre que usa Postgres en
  # docker-compose.yml (POSTGRES_DB: mlflow_db) para que el mismo
  # `--backend-store-uri` sirva en local y en producción.
  db_name                = "mlflow_db"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = true
  multi_az               = false
  # Cifrado en reposo (KMS administrado por AWS, sin costo extra): el
  # backend store de MLflow guarda parámetros, métricas y metadata de runs.
  storage_encrypted = true
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_db.endpoint
}
