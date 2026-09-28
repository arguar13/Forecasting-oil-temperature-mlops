# ==============================================================================
# Makefile — interfaz única de ejecución (local == CI)
#
# Todos los comandos de calidad, dependencias y pruebas pasan por aquí.
# El pipeline de GitLab CI (.gitlab-ci.yml) invoca estos mismos targets.
# ==============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

API_DIR    := api
CORE_DIR   := core_ml
PROJECTS   := $(API_DIR) $(CORE_DIR)

# Variables usadas por docker-build/docker-push/deploy -- sobreescribibles,
# ej. `make docker-push AWS_ACCOUNT_ID=123456789012`.
AWS_REGION       ?= us-east-1
AWS_ACCOUNT_ID   ?=
ECR_REPOSITORY   ?= dlinear-forecast-api
IMAGE_TAG        ?= $(shell git rev-parse HEAD)
ECR_REGISTRY     := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
# SHA del host inyectado al contenedor trainer (que no ve .git): queda como
# tag git_commit_hash de cada run de MLflow (ver mlflow_utils.get_git_commit_hash).
GIT_SHA          := $(shell git rev-parse HEAD 2>/dev/null)
TRAINER_RUN      := docker compose run --rm -e GIT_COMMIT_SHA=$(GIT_SHA) trainer
# Bucket que crea localstack/init/01-bootstrap.sh (solo stack local).
LOCAL_BUCKET     := mlops-portafolio-proj3-models

.PHONY: help install install-api install-core-ml install-core-ml-ci lock \
	format format-check lint type-check test \
	hooks pre-commit-run ci clean \
	dvc-pull dvc-push dvc-status data-download data-toy data-raw train train-toy quality-gate reload-api batch-local mlflow-ui \
	up down restart reset-local logs ps \
	docker-build docker-push deploy \
	tf-fmt tf-validate tf-test tf-plan tf-apply k8s-build

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------
# Dependencias deterministas (Poetry + lockfile)
# ------------------------------------------------------------------------------
install: install-api install-core-ml ## Instala dependencias exactas (poetry.lock) en ambos subproyectos

install-api: ## Instala dependencias del servicio api/ (respetando poetry.lock)
	cd $(API_DIR) && poetry install --sync --no-interaction

install-core-ml: ## Instala dependencias del pipeline core_ml/ (respetando poetry.lock)
	cd $(CORE_DIR) && poetry install --sync --no-interaction

# "poetry install" del lockfile de core_ml a veces falla de forma
# intermitente en algunos entornos de contenedor Linux -- "poetry export"
# evita esa ruta de código: genera un requirements.txt pineado desde el
# lockfile, y pip (siempre estable) instala desde ahí.
install-core-ml-ci: ## Instala core_ml/ vía pip (lockfile exportado) -- para contenedores Linux (CI/entrenamiento)
	cd $(CORE_DIR) && poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt --with dev \
		&& pip install --no-cache-dir -q -r /tmp/requirements.txt

lock: ## Regenera poetry.lock en ambos subproyectos (usar tras editar pyproject.toml)
	@for d in $(PROJECTS); do \
		echo "==> poetry lock ($$d)"; \
		(cd $$d && poetry lock) || exit 1; \
	done

# ------------------------------------------------------------------------------
# Validación: formato, lint, tipos, tests
# ------------------------------------------------------------------------------
format: ## Formatea el código (ruff) en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> Formateando $$d"; \
		(cd $$d && poetry run ruff check --fix . && poetry run ruff format .) || exit 1; \
	done

format-check: ## Verifica formateo sin modificar archivos (usado en CI)
	@for d in $(PROJECTS); do \
		echo "==> Verificando formato $$d"; \
		(cd $$d && poetry run ruff format --check .) || exit 1; \
	done

lint: ## Ejecuta Ruff (lint + import order) en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> Lint $$d"; \
		(cd $$d && poetry run ruff check .) || exit 1; \
	done

type-check: ## Ejecuta mypy en ambos subproyectos (opcional, no bloquea `make ci`)
	@for d in $(PROJECTS); do \
		echo "==> mypy $$d"; \
		(cd $$d && poetry run mypy .) || exit 1; \
	done

test: ## Ejecuta pytest con cobertura en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> pytest $$d"; \
		(cd $$d && poetry run pytest) || exit 1; \
	done

# ------------------------------------------------------------------------------
# Datos y modelos versionados (DVC + MLflow)
# ------------------------------------------------------------------------------
dvc-pull: ## Descarga los datasets versionados (toy + raw) desde el remoto S3/LocalStack
	cd $(CORE_DIR) && poetry run dvc pull

dvc-push: ## Sube los datasets versionados al remoto S3/LocalStack
	cd $(CORE_DIR) && poetry run dvc push

dvc-status: ## Muestra si los datasets locales están al día con el remoto
	cd $(CORE_DIR) && poetry run dvc status -c

data-download: ## Descarga ETTh1 del repo público y genera el toy (alternativa a dvc-pull sin remoto)
	cd $(CORE_DIR) && poetry run python -m scripts.download_data

data-toy: ## Procesa el dataset toy (~1000 filas) para un ciclo E2E rápido
	cd $(CORE_DIR) && poetry run python -m src.data_processing --dataset toy --output_dir artifacts_toy

data-raw: ## Procesa el dataset completo de producción
	cd $(CORE_DIR) && poetry run python -m src.data_processing --dataset raw --output_dir artifacts

# train.py/quality_gate.py corren dentro del contenedor "trainer" (Linux), NO
# con "poetry run" contra el host -- mlflow.pyfunc.log_model() graba la ruta
# de cada artefacto con el separador del SO que entrena; entrenar en Windows
# graba rutas que el contenedor de la API (Linux) no puede resolver.
train-toy: data-toy ## Entrena end-to-end sobre el dataset toy en segundos (smoke test, sin GPU)
	$(TRAINER_RUN) python -m src.train --dataset toy --artifact_dir artifacts_toy --epochs 2 --n_trials 1

train: data-raw ## Entrena end-to-end sobre el dataset completo de producción
	$(TRAINER_RUN) python -m src.train --dataset raw

quality-gate: ## Evalúa el quality gate y promueve a producción si corresponde
	$(TRAINER_RUN) python -m src.quality_gate

# La API carga el modelo una sola vez, al arrancar (lifespan): tras una
# promoción hay que reiniciarla para que sirva la nueva versión de production.
reload-api: ## Reinicia la API local para que cargue el modelo "production" actual
	docker compose restart api
	docker compose up -d --wait api

# Mismo entrypoint y misma imagen que el CronJob de Kubernetes
# (kubernetes/base/cronjob.yaml), contra el bucket de LocalStack.
batch-local: ## Corre el batch scoring (src.batch_inference) en local contra LocalStack
	cd $(CORE_DIR) && poetry run python -m scripts.make_batch_input --output artifacts_batch/input_data.csv
	docker compose cp $(CORE_DIR)/artifacts_batch/input_data.csv localstack:/tmp/input_data.csv
	docker compose exec -T localstack awslocal s3 cp /tmp/input_data.csv s3://$(LOCAL_BUCKET)/batch/input_data.csv
	docker compose run --rm --no-deps -e AWS_ENDPOINT_URL=http://localstack:4566 \
		-e MODEL_BUCKET_NAME=$(LOCAL_BUCKET) api python -m src.batch_inference
	docker compose exec -T localstack awslocal s3 cp s3://$(LOCAL_BUCKET)/batch/predictions_output.csv - | head -n 3

mlflow-ui: ## Levanta la UI local de MLflow sobre ./core_ml/mlruns (http://localhost:5000)
	cd $(CORE_DIR) && poetry run mlflow ui --backend-store-uri ./mlruns

# ------------------------------------------------------------------------------
# Entorno local (docker-compose): Postgres, LocalStack, MLflow, API
# ------------------------------------------------------------------------------
up: ## Levanta el stack local completo (Postgres, LocalStack, MLflow, API)
	docker compose up -d --build
	docker compose up -d --wait postgres localstack mlflow api

down: ## Detiene y elimina el stack local (conserva los volúmenes de datos)
	docker compose down

restart: down up ## Reinicia el stack local completo

# LocalStack Community no persiste S3 pero Postgres sí: tras reiniciar
# LocalStack, MLflow listaría runs cuyos artefactos ya no existen. Esto
# borra ambos volúmenes para volver a un estado coherente (luego reentrenar).
reset-local: ## Borra el stack local Y sus volúmenes (MLflow + S3 simulado) y lo levanta limpio
	docker compose down -v
	$(MAKE) up

logs: ## Sigue los logs de todos los servicios del stack local
	docker compose logs -f

ps: ## Muestra el estado de los servicios del stack local
	docker compose ps

# ------------------------------------------------------------------------------
# Build/push de la imagen de la API y despliegue a Kubernetes -- el mismo
# procedimiento que corre .gitlab-ci.yml, para usar a mano si hace falta.
# ------------------------------------------------------------------------------
docker-build: ## Construye la imagen de la API localmente
	docker build -t $(ECR_REPOSITORY):$(IMAGE_TAG) -f Dockerfile .

docker-push: ## Publica la imagen en ECR (requiere `aws ecr get-login-password` ya autenticado y AWS_ACCOUNT_ID)
	@if [ -z "$(AWS_ACCOUNT_ID)" ]; then echo "Uso: make docker-push AWS_ACCOUNT_ID=123456789012"; exit 1; fi
	docker tag $(ECR_REPOSITORY):$(IMAGE_TAG) $(ECR_REGISTRY)/$(ECR_REPOSITORY):$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/$(ECR_REPOSITORY):$(IMAGE_TAG)

deploy: ## Aplica los manifiestos de Kubernetes (kustomize) al cluster configurado en kubectl
	kubectl apply -k kubernetes/overlays/production

# ------------------------------------------------------------------------------
# Terraform / Kubernetes -- validación y operación de la infraestructura
# ------------------------------------------------------------------------------
tf-fmt: ## Verifica el formato de Terraform (sin credenciales AWS)
	terraform -chdir=terraform fmt -check -diff -recursive

tf-validate: ## Valida sintaxis y consistencia interna de Terraform (init local, sin backend remoto ni AWS)
	terraform -chdir=terraform init -backend=false -input=false
	terraform -chdir=terraform validate

tf-test: tf-validate ## Tests de Terraform con providers mockeados (offline, sin credenciales AWS)
	terraform -chdir=terraform test

tf-plan: ## Muestra los cambios que aplicaría Terraform (requiere credenciales AWS)
	terraform -chdir=terraform plan

tf-apply: ## Aplica los cambios de Terraform al entorno real (requiere credenciales AWS)
	terraform -chdir=terraform apply

k8s-build: ## Renderiza el overlay de producción (kustomize, sin cluster real)
	kubectl kustomize kubernetes/overlays/production

# ------------------------------------------------------------------------------
# Git hooks
# ------------------------------------------------------------------------------
hooks: ## Instala los git hooks de pre-commit
	pre-commit install --hook-type pre-commit

pre-commit-run: ## Ejecuta todos los hooks de pre-commit sobre todo el repo
	pre-commit run --all-files

# ------------------------------------------------------------------------------
# Agregado: el mismo target que corre en CI (.gitlab-ci.yml::lint_test)
# ------------------------------------------------------------------------------
ci: format-check lint test ## Ejecuta la validación completa (idéntico a CI)

clean: ## Elimina cachés y artefactos locales de herramientas
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(API_DIR)/.coverage $(CORE_DIR)/.coverage
