# Entrenamiento local -- ver docker-compose.yml::trainer.
#
# mlflow.pyfunc.log_model() persiste la ruta relativa de cada artefacto
# ("model_state_dict", "scaler_X", ...) usando el separador nativo del SO
# donde corre `src.train` en el momento de guardar (mlflow/pyfunc/model.py,
# _save_model_with_class_artifacts_params: os.path.join(...)) -- ese literal
# queda grabado en el manifiesto del modelo y NO se recalcula al servirlo.
# Entrenar directo en el host Windows (`poetry run python -m src.train`)
# graba "artifacts\dlinear_model.pth"; el contenedor de la API (Linux)
# busca "artifacts/dlinear_model.pth" y falla con FileNotFoundError.
# CI ya entrena dentro de `image: python:3.10` (.gitlab-ci.yml::train_model)
# -- esta imagen replica ese mismo entorno para desarrollo local, sea cual
# sea el SO del host.
FROM python:3.10-slim

WORKDIR /workspace

ENV PYTHONUTF8=1

# Solo para exportar el lockfile a requirements.txt (ver train-entrypoint.sh)
# -- NO se usa "poetry install" en esta imagen. En este entorno concreto
# (Docker Desktop/WSL2, python:3.10-slim), `poetry install` con el lockfile
# de core_ml muere de forma silenciosa y 100 % reproducible (exit 1, cero
# traza incluso con -vvv) justo al entrar a la fase de resolución de entorno
# de "virtualenv" ("[virtualenv:virtualenv.app_data] created app data
# folder..." es la última línea que llega a imprimir) -- se probó con/sin
# instalador paralelo, con/sin BuildKit, con PYTHON_KEYRING_BACKEND=null;
# nada lo evita. "poetry export" (no toca esa ruta de código, no resuelve
# ni crea ningún entorno) sí funciona limpio -- se usa para generar un
# requirements.txt pineado desde el lockfile, y las dependencias reales se
# instalan con pip, no con el instalador de poetry.
RUN pip install --no-cache-dir "poetry==1.8.3" "poetry-plugin-export==1.8.0"

COPY core_ml/train-entrypoint.sh /usr/local/bin/train-entrypoint.sh
RUN chmod +x /usr/local/bin/train-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/train-entrypoint.sh"]
