# Security Group para RDS (Zero-Trust con EKS)
resource "aws_security_group" "rds_sg" {
  name = "${var.project_name}-rds-sg"
  # GroupDescription de EC2 solo admite ASCII ("terraform apply" fallaba con
  # "Character sets beyond ASCII are not supported" por la tilde en "trafico").
  description = "Allow PostgreSQL traffic exclusively from EKS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
}

# 1. Lectura del secreto en Secrets Manager
data "aws_secretsmanager_secret" "db_password" {
  # Asegúrate de que este nombre coincida exactamente con el que creaste en el Paso 1
  name = "dlinear-mlops/db-password"
}

data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = data.aws_secretsmanager_secret.db_password.id
}

# 2. Inyección del secreto en un local
locals {
  db_password = jsondecode(data.aws_secretsmanager_secret_version.db_password.secret_string)["db_password"]
}

# 3. Instancia RDS para MLflow
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
  password               = local.db_password # <-- Usando el secreto extraído dinámicamente
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = var.environment == "dev" ? true : false
  multi_az               = var.environment == "prod" ? true : false
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_db.endpoint
}
