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

.PHONY: help install install-api install-core-ml lock \
	format format-check lint type-check test test-integration security \
	yaml-lint secrets-scan secrets-baseline trivy \
	hooks pre-commit-run ci clean \
	dvc-pull dvc-push dvc-status data-toy data-raw train-toy quality-gate mlflow-ui \
	up down restart logs ps \
	ci-local ci-local-list

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

trivy: ## Escanea dependencias, IaC y secretos con Trivy (requiere Docker)
	docker run --rm -v "$$(pwd)":/repo aquasec/trivy:latest fs \
		--exit-code 1 --severity HIGH,CRITICAL --ignorefile /repo/.trivyignore /repo

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

train-toy: data-toy ## Entrena end-to-end sobre el dataset toy en segundos (smoke test, sin GPU)
	cd $(CORE_DIR) && poetry run python -m src.train --dataset toy --artifact_dir artifacts_toy --epochs 2 --n_trials 1

quality-gate: ## Evalúa el quality gate y promueve a producción si corresponde
	cd $(CORE_DIR) && poetry run python -m src.quality_gate

mlflow-ui: ## Levanta la UI local de MLflow sobre ./core_ml/mlruns (http://localhost:5000)
	cd $(CORE_DIR) && poetry run mlflow ui --backend-store-uri ./mlruns

# ------------------------------------------------------------------------------
# Entorno local completo (Fase 3): Postgres, LocalStack, Kafka, MLflow, API
# ------------------------------------------------------------------------------
up: ## Levanta el stack local completo (Postgres, LocalStack, Kafka, MLflow, API)
	docker compose up -d --build --wait

down: ## Detiene y elimina el stack local (conserva los volúmenes de datos)
	docker compose down

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
# ------------------------------------------------------------------------------
ci-local-list: ## Valida .gitlab-ci.yml (sintaxis, stages, needs) sin ejecutar nada
	gitlab-ci-local --list

ci-local: ## Ejecuta un job de .gitlab-ci.yml localmente con Docker (uso: make ci-local JOB=python:quality)
	gitlab-ci-local $(JOB)

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
