# ==============================================================================
# Makefile — interfaz única de ejecución (local == CI)
#
# Todos los comandos de calidad, dependencias y pruebas pasan por aquí.
# El pipeline de GitLab CI invoca estos mismos targets, garantizando que lo
# que pasa localmente es exactamente lo que se valida en CI.
# ==============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

API_DIR    := api
CORE_DIR   := core_ml
PROJECTS   := $(API_DIR) $(CORE_DIR)

.PHONY: help install install-api install-core-ml install-core-ml-ci lock \
	format format-check lint type-check test test-integration security \
	yaml-lint secrets-scan secrets-baseline trivy \
	hooks pre-commit-run ci clean \
	dvc-pull dvc-push dvc-status data-toy data-raw train-toy quality-gate mlflow-ui \
	up down restart logs ps \
	stream-up stream-replay stream-logs stream-local \
	tf-fmt tf-validate k8s-build \
	ci-local ci-local-list \
	runner-register runner-up runner-down runner-logs runner-status

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

# "poetry install" del lockfile de core_ml (DVC trae ~70 dependencias
# transitivas de C-extensions: pygit2, orjson, aiohttp...) muere de forma
# silenciosa e intermitente (exit 1, sin traza ni con -vvv) corriendo dentro
# de un contenedor Linux en este entorno (Docker Desktop/WSL2 con varios
# proyectos corriendo en paralelo) -- reproducido tanto en gitlab-ci-local
# (integration_tests, imagen docker:24.0.5) como en el contenedor de
# entrenamiento local (core_ml/train.Dockerfile). "poetry export" no toca la
# ruta de código que falla (no resuelve/crea ningún entorno) -- se usa para
# generar un requirements.txt pineado desde el lockfile, y pip (siempre
# estable) instala desde ahí. Únicamente para contenedores Linux (CI,
# core_ml/train-entrypoint.sh) -- el host sigue usando install-core-ml de
# arriba, que nunca mostró este problema.
install-core-ml-ci: ## Instala core_ml/ vía pip (lockfile exportado) -- para contenedores Linux (CI/entrenamiento), no para el host
	cd $(CORE_DIR) && poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt --with dev \
		&& pip install --no-cache-dir -q -r /tmp/requirements.txt

lock: ## Regenera poetry.lock en ambos subproyectos (usar tras editar pyproject.toml)
	@for d in $(PROJECTS); do \
		echo "==> poetry lock ($$d)"; \
		(cd $$d && poetry lock) || exit 1; \
	done

# ------------------------------------------------------------------------------
# Validación shift-left: formato, lint, tipos, tests, seguridad
# ------------------------------------------------------------------------------
format: ## Formatea el código (isort + black) en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> Formateando $$d"; \
		(cd $$d && poetry run isort . && poetry run black .) || exit 1; \
	done

format-check: ## Verifica formateo sin modificar archivos (usado en CI)
	@for d in $(PROJECTS); do \
		echo "==> Verificando formato $$d"; \
		(cd $$d && poetry run isort --check-only . && poetry run black --check .) || exit 1; \
	done

lint: ## Ejecuta Ruff (linting estático) en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> Lint $$d"; \
		(cd $$d && poetry run ruff check .) || exit 1; \
	done

type-check: ## Ejecuta mypy en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> mypy $$d"; \
		(cd $$d && poetry run mypy .) || exit 1; \
	done

test: ## Ejecuta pytest con cobertura en ambos subproyectos (rápido; excluye integration)
	@for d in $(PROJECTS); do \
		echo "==> pytest $$d"; \
		(cd $$d && poetry run pytest) || exit 1; \
	done

test-integration: ## Tests de integración con Testcontainers (Postgres/Kafka/LocalStack + docker-compose.yml). Requiere Docker corriendo.
	cd $(CORE_DIR) && poetry run pytest -m integration --no-cov tests/integration

security: ## Ejecuta Bandit (SAST) en ambos subproyectos
	@for d in $(PROJECTS); do \
		echo "==> bandit $$d"; \
		(cd $$d && poetry run bandit -c pyproject.toml -r .) || exit 1; \
	done

yaml-lint: ## Valida sintaxis y estilo de todos los YAML del repo
	pre-commit run yamllint --all-files

secrets-scan: ## Escanea el repo en busca de secretos expuestos (detect-secrets)
	pre-commit run detect-secrets --all-files

secrets-baseline: ## Regenera .secrets.baseline tras auditar nuevos hallazgos
	detect-secrets scan --baseline .secrets.baseline

# Versión pineada (no ":latest"), igual que en .gitlab-ci.yml::security:trivy
# -- mantener ambas sincronizadas si se actualiza una.
trivy: ## Escanea dependencias, IaC y secretos con Trivy (requiere Docker)
	MSYS_NO_PATHCONV=1 docker run --rm -v "$$(pwd)":/repo aquasec/trivy:0.74.0 fs \
		--exit-code 1 --severity HIGH,CRITICAL --ignorefile /repo/.trivyignore \
		--skip-dirs "**/.terraform,**/mlruns,**/mlartifacts,**/.claude" \
		--skip-files "**/*.tfstate,**/*.tfstate.*" /repo

# ------------------------------------------------------------------------------
# Datos y modelos versionados (DVC + MLflow) -- Fase 2
# ------------------------------------------------------------------------------
dvc-pull: ## Descarga los datasets versionados (toy + raw) desde el remoto S3/LocalStack
	cd $(CORE_DIR) && poetry run dvc pull

dvc-push: ## Sube los datasets versionados al remoto S3/LocalStack
	cd $(CORE_DIR) && poetry run dvc push

dvc-status: ## Muestra si los datasets locales están al día con el remoto
	cd $(CORE_DIR) && poetry run dvc status -c

data-toy: ## Procesa el dataset toy (~1000 filas) para un ciclo E2E rápido
	cd $(CORE_DIR) && poetry run python -m src.data_processing --dataset toy --output_dir artifacts_toy

data-raw: ## Procesa el dataset completo de producción
	cd $(CORE_DIR) && poetry run python -m src.data_processing --dataset raw --output_dir artifacts

# train.py/quality_gate.py corren dentro del contenedor "trainer" (Linux),
# NO con "poetry run" contra el host -- mlflow.pyfunc.log_model() graba la
# ruta relativa de cada artefacto con el separador nativo del SO que entrena
# (ver core_ml/train.Dockerfile); entrenar en el host Windows deja
# "artifacts\dlinear_model.pth" grabado en el modelo, y el contenedor de la
# API (Linux) no lo encuentra ("/" != "\\") -- FileNotFoundError al cargar.
# El contenedor ya resuelve mlflow/localstack por DNS interno de Docker
# (docker-compose.yml::trainer.environment), sin depender de qué puerto de
# host tenga publicado cada uno en esta máquina.
train-toy: data-toy ## Entrena end-to-end sobre el dataset toy en segundos (smoke test, sin GPU)
	docker compose run --rm trainer python -m src.train --dataset toy --artifact_dir artifacts_toy --epochs 2 --n_trials 1

quality-gate: ## Evalúa el quality gate y promueve a producción si corresponde
	docker compose run --rm trainer python -m src.quality_gate

mlflow-ui: ## Levanta la UI local de MLflow sobre ./core_ml/mlruns (http://localhost:5000)
	cd $(CORE_DIR) && poetry run mlflow ui --backend-store-uri ./mlruns

# ------------------------------------------------------------------------------
# Entorno local completo (Fase 3): Postgres, LocalStack, Kafka, MLflow, API
# ------------------------------------------------------------------------------
up: ## Levanta el stack local completo (Postgres, LocalStack, Kafka, MLflow, API)
	# kafka-init es un job de una sola corrida (crea el topic y termina con
	# exit 0) -- `--wait` sobre TODOS los servicios lo trata como fallo
	# porque deja de estar "running". Se construye/levanta todo primero
	# (kafka-init corre y termina como dependencia de kafka) y el --wait
	# se acota a los servicios de larga vida.
	docker compose up -d --build
	docker compose up -d --wait postgres localstack kafka mlflow api

down: ## Detiene y elimina el stack local (conserva los volúmenes de datos)
	docker compose down

# ------------------------------------------------------------------------------
# Streaming local (Kinesis vía LocalStack): la única forma de ejercitar
# scripts/sensor_simulator.py y core_ml/src/monitoring/stream_consumer.py sin
# tocar el Kinesis real de terraform/kinesis.tf. Requiere `make up` +
# `make train-toy` + `make quality-gate` primero -- stream-consumer necesita
# una versión "production" real para cargar al arrancar.
# ------------------------------------------------------------------------------
stream-up: ## Levanta el consumidor de Kinesis (perfil "tools", no arranca con `make up`)
	docker compose --profile tools up -d --build stream-consumer

stream-replay: ## Reproduce el dataset toy hacia Kinesis a paso lento (ejercita el trigger del consumidor)
	docker compose --profile tools run --rm trainer python -m scripts.sensor_simulator --dataset toy --delay-seconds 0.2

stream-logs: ## Sigue los logs del consumidor (verdictos CUSUM, predicciones registradas, mitigación)
	docker compose logs -f stream-consumer

stream-local: up train-toy quality-gate stream-up stream-replay ## Todo el ciclo end-to-end de streaming, desde cero
	@echo "stream-consumer corriendo. Ver 'make stream-logs' para el veredicto de drift."

restart: down up ## Reinicia el stack local completo

logs: ## Sigue los logs de todos los servicios del stack local
	docker compose logs -f

ps: ## Muestra el estado de los servicios del stack local
	docker compose ps

# ------------------------------------------------------------------------------
# Fase 4: validar el pipeline de GitLab CI localmente (gitlab-ci-local),
# antes de que el runner real lo ejecute. `gitlab-runner exec` fue removido
# de GitLab Runner; `gitlab-ci-local` (npm) es el reemplazo estándar de
# facto -- corre los jobs con Docker igual que un runner real.
#   npm install -g gitlab-ci-local
#
# MSYS_NO_PATHCONV/MSYS2_ARG_CONV_EXCL: en Git Bash (Windows), el runtime
# MSYS reescribe cualquier argumento que empiece con "/" (ej. el
# `--workdir /builds/...` que gitlab-ci-local pasa a `docker create`) como
# si fuera una ruta de archivo relativa a la instalación de Git, y Docker
# lo rechaza con "the working directory 'C:/Program Files/Git/builds/...'
# is invalid". No-op en Linux/Mac (MSYS no existe ahí).
#
# --privileged: los jobs con `services: [docker:*-dind]` (security:trivy,
# integration_tests, docker:build-push*) necesitan que su propio contenedor
# corra en modo privileged para que el daemon Docker-in-Docker arranque --
# sin esto el healthcheck del servicio nunca pasa ("Cannot connect to the
# Docker daemon"). Verificado corriendo security:trivy localmente. Inocuo
# para el resto de los jobs (terraform:fmt, python:quality, etc.), que no
# usan servicios y no necesitan el flag.
# ------------------------------------------------------------------------------
GCL_ENV := MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL=*

ci-local-list: ## Valida .gitlab-ci.yml (sintaxis, stages, needs) sin ejecutar nada
	$(GCL_ENV) gitlab-ci-local --list

ci-local: ## Ejecuta un job de .gitlab-ci.yml localmente con Docker (uso: make ci-local JOB=python:quality)
	$(GCL_ENV) gitlab-ci-local --privileged $(JOB)

# ------------------------------------------------------------------------------
# Fase 5: Runner self-hosted en TU hardware -- ejecuta el pipeline REAL de
# GitLab (no una simulación) sin consumir un solo minuto de los runners
# compartidos de GitLab.com. Complementa a gitlab-ci-local (Fase 4): ese
# simula localmente ANTES de hacer push; este es el runner que GitLab
# realmente despacha DESPUÉS del push, corriendo en esta misma máquina.
#
# Bootstrap (una sola vez):
#   1. GitLab.com -> este proyecto (o el grupo) -> Settings -> CI/CD ->
#      Runners -> "New runner" -> marcar Linux -> tags: "local-hardware"
#      (debe coincidir EXACTO con el tag usado en .gitlab-ci.yml) ->
#      "Run untagged jobs": NO -> copiar el authentication token (glrt-...).
#   2. make runner-register TOKEN=glrt-xxxxx   (una sola vez; el token
#      queda guardado dentro del volumen Docker "gitlab-runner-config",
#      nunca en Git, nunca en texto plano en este repo)
#   3. make runner-up                          (lo deja corriendo 24/7;
#      --restart always -> sobrevive a un reinicio de Docker Desktop/PC)
#
# --docker-privileged: obligatorio para que `services: [docker:*-dind]`
# (security:trivy, integration_tests, docker:build-push*) puedan levantar
# su propio daemon Docker-in-Docker dentro del job -- sin esto esos jobs
# fallan con "Cannot connect to the Docker daemon".
#
# A propósito SIN --docker-volumes "/var/run/docker.sock:...": ningún job
# de .gitlab-ci.yml necesita el socket del host montado en el contenedor del
# job (todos usan el patrón docker:*-dind de arriba). Se probó incluirlo y,
# en Git Bash (Windows), MSYS reescribe ese argumento como una ruta de
# Windows ("C:\Program Files\Git\var\run\docker.sock"), corrompiendo
# `[runners.docker] volumes` en config.toml y rompiendo TODOS los jobs de
# este runner. Si en el futuro un job realmente lo necesita, agregar el
# flag con `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL=*` por delante en Windows.
# ------------------------------------------------------------------------------
runner-register: ## Registra este PC como runner de GitLab (uso: make runner-register TOKEN=glrt-xxxxx)
	@if [ -z "$(TOKEN)" ]; then echo "Uso: make runner-register TOKEN=glrt-xxxxx  (Settings > CI/CD > Runners > New runner)"; exit 1; fi
	docker run --rm \
		-v gitlab-runner-config:/etc/gitlab-runner \
		gitlab/gitlab-runner:latest register \
		--non-interactive \
		--url "https://gitlab.com" \
		--token "$(TOKEN)" \
		--executor "docker" \
		--docker-image "docker:24.0.5" \
		--docker-privileged \
		--description "local-hardware ($$(hostname))"

runner-up: ## Levanta el runner self-hosted local de forma permanente (restart always)
	docker run -d --name gitlab-runner-local --restart always \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v gitlab-runner-config:/etc/gitlab-runner \
		gitlab/gitlab-runner:latest

runner-down: ## Detiene y elimina el contenedor del runner local (conserva el registro/config)
	docker rm -f gitlab-runner-local

runner-logs: ## Sigue los logs del runner local (para ver qué job está ejecutando)
	docker logs -f gitlab-runner-local

runner-status: ## Verifica que el runner local está corriendo y conectado a GitLab
	docker ps --filter name=gitlab-runner-local
	docker run --rm -v gitlab-runner-config:/etc/gitlab-runner gitlab/gitlab-runner:latest verify

# ------------------------------------------------------------------------------
# Terraform / Kubernetes -- validación local, sin credenciales de AWS ni
# cluster real. `terraform:fmt` (.gitlab-ci.yml) corre exactamente
# `terraform fmt -check`; estos targets dan la misma señal antes de hacer
# push, no solo cuando CI ya la reporta.
# ------------------------------------------------------------------------------
tf-fmt: ## Verifica el formato de Terraform (sin credenciales AWS)
	terraform -chdir=terraform fmt -check -diff -recursive

tf-validate: ## Valida sintaxis y consistencia interna de Terraform (init local, sin backend remoto ni AWS)
	terraform -chdir=terraform init -backend=false -input=false
	terraform -chdir=terraform validate

k8s-build: ## Renderiza el overlay de producción (kustomize, sin cluster real)
	kubectl kustomize kubernetes/overlays/production

# ------------------------------------------------------------------------------
# Git hooks
# ------------------------------------------------------------------------------
hooks: ## Instala los git hooks de pre-commit (commit + push)
	pre-commit install --hook-type pre-commit --hook-type pre-push
	pre-commit install --hook-type commit-msg 2>/dev/null || true

pre-commit-run: ## Ejecuta todos los hooks de pre-commit sobre todo el repo
	pre-commit run --all-files

# ------------------------------------------------------------------------------
# Agregado: el mismo target que corre en CI
# ------------------------------------------------------------------------------
ci: format-check lint type-check test security yaml-lint secrets-scan ## Ejecuta toda la validación (idéntico a CI)

clean: ## Elimina cachés y artefactos locales de herramientas
	find . -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(API_DIR)/.coverage $(CORE_DIR)/.coverage
