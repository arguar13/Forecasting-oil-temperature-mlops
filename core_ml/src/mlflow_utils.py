"""Trazabilidad y registro de modelos vía MLflow.

Cada corrida de entrenamiento debe poder reconstruirse de forma inequívoca a
partir de la tupla:

    Git Commit Hash + DVC Data Hash + Hyperparameters + MLflow Run ID + Container Image Tag

Este módulo centraliza cómo se captura esa tupla (como tags/params de MLflow)
y cómo se empaqueta el modelo entrenado: NUNCA como un `.pth`/`.pkl` suelto en
un directorio, sino como un único artefacto `mlflow.pyfunc` versionado que
bundlea la red DLinear junto con los scalers de preprocesamiento necesarios
para servirlo.
"""

from __future__ import annotations

import hashlib
import os

# Solo se usa para `git rev-parse HEAD` más abajo, sin shell ni input externo.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.pyfunc import log_model as mlflow_log_model

from src.logging_config import get_logger

logger = get_logger(__name__)

UNKNOWN = "unknown"


def get_git_commit_hash() -> str:
    """Devuelve el SHA completo del commit actual.

    Prioriza `CI_COMMIT_SHA` (inyectado por GitLab CI) para que el hash
    registrado sea el del commit que efectivamente disparó el pipeline,
    incluso si el working tree de CI está en un estado detached/shallow.
    """
    ci_sha = os.getenv("CI_COMMIT_SHA")
    if ci_sha:
        return ci_sha
    try:
        # Lista de argumentos fija (sin shell=True, sin input de usuario
        # interpolado); "git" se resuelve vía PATH a propósito para
        # funcionar igual en devcontainer, CI y bare metal.
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        logger.warning("git_commit_hash_unavailable", fallback=UNKNOWN)
        return UNKNOWN


def get_dvc_data_hash(dvc_file_path: str | Path) -> str:
    """Lee el hash MD5 de contenido del dataset desde su archivo `.dvc`.

    Ese hash identifica de forma unívoca los bytes exactos del dataset usado
    en la corrida (DVC lo recalcula cada vez que el contenido cambia), sin
    depender de que el archivo de datos en sí esté trackeado por Git.
    """
    path = Path(dvc_file_path)
    if not path.exists():
        logger.warning("dvc_file_missing", path=str(path), fallback=UNKNOWN)
        return UNKNOWN
    with path.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    try:
        return str(spec["outs"][0]["md5"])
    except (KeyError, IndexError, TypeError):
        logger.warning("dvc_file_unexpected_format", path=str(path), fallback=UNKNOWN)
        return UNKNOWN


def data_matches_dvc_pointer(dvc_file_path: str | Path) -> bool:
    """True si el archivo de datos local tiene exactamente el hash MD5 que
    declara su puntero `.dvc`.

    `dvc_data_hash` se lee del puntero, no de los bytes usados: sin esta
    verificación, entrenar con un CSV modificado (o descargado de otra
    fuente) registraría en MLflow un hash que no corresponde a los datos
    reales de la corrida.
    """
    dvc_path = Path(dvc_file_path)
    data_path = dvc_path.with_suffix("")  # "ETTh1.csv.dvc" -> "ETTh1.csv"
    expected = get_dvc_data_hash(dvc_path)
    if expected == UNKNOWN or not data_path.exists():
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with data_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    matches = digest.hexdigest() == expected
    if not matches:
        logger.warning(
            "data_does_not_match_dvc_pointer",
            data_path=str(data_path),
            expected_md5=expected,
            actual_md5=digest.hexdigest(),
        )
    return matches


def get_container_image_tag() -> str:
    """Tag de la imagen de contenedor que ejecuta esta corrida.

    En CI coincide exactamente con el tag que `docker:build-push` publica en
    ECR (`$CI_COMMIT_SHA`), cerrando el círculo entre el código que entrenó
    el modelo y la imagen que lo sirve.
    """
    return os.getenv("CI_COMMIT_SHA") or os.getenv("CONTAINER_IMAGE_TAG") or "local-dev"


def build_reproducibility_tags(dvc_file_path: str | Path) -> dict[str, str]:
    """Construye la tupla de reproducibilidad como tags listos para MLflow."""
    return {
        "git_commit_hash": get_git_commit_hash(),
        "dvc_data_hash": get_dvc_data_hash(dvc_file_path),
        "dvc_data_file": str(dvc_file_path),
        "data_matches_dvc_pointer": str(data_matches_dvc_pointer(dvc_file_path)).lower(),
        "container_image_tag": get_container_image_tag(),
    }


class DLinearForecaster(PythonModel):
    """Empaqueta la red DLinear + scalers de pre/post-procesamiento como un
    único artefacto de MLflow, servible con `mlflow.pyfunc.load_model(...)`.

    Esto es lo que reemplaza a los `.pth`/`.pkl` sueltos: una referencia
    inmutable y versionada (`models:/<nombre>/<stage o versión>`) que ya
    incluye todo lo necesario para producir una predicción correcta.
    """

    def load_context(self, context: PythonModelContext) -> None:
        # Import local: al deserializar este modelo en OTRO proceso/contenedor
        # (p. ej. la API), solo hace falta que `src.model_architecture` sea
        # importable -- no todo `src` (que incluiría argparse/Optuna/DVC de
        # train.py, innecesarios para servir inferencia).
        from src.model_architecture import DLinear

        seq_len = int(context.model_config["seq_len"])
        n_features = int(context.model_config["n_features"])
        pred_len = int(context.model_config.get("pred_len", 1))

        self.model = DLinear(seq_len=seq_len, n_features=n_features, pred_len=pred_len)
        state_dict = torch.load(
            context.artifacts["model_state_dict"],
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.scaler_X = joblib.load(context.artifacts["scaler_X"])
        self.scaler_y = joblib.load(context.artifacts["scaler_y"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: Any,
        params: dict | None = None,
    ):
        """Acepta una única secuencia (forma [seq_len, n_features], p. ej. un
        DataFrame o array 2D -- caso de la API online) o un lote de
        secuencias (forma [n_windows, seq_len, n_features] -- caso del batch
        scoring). Siempre hace UNA sola pasada vectorizada por el modelo,
        independientemente de cuántas secuencias traiga el lote.
        """
        array = np.asarray(model_input, dtype=np.float32)
        if array.ndim == 2:
            array = array[np.newaxis, ...]  # [seq_len, n_features] -> [1, seq_len, n_features]

        n_windows, seq_len, n_features = array.shape
        scaled_flat = self.scaler_X.transform(array.reshape(-1, n_features))
        scaled = scaled_flat.reshape(n_windows, seq_len, n_features)

        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            output_scaled = self.model(tensor)  # [n_windows, pred_len]

        # scaler_y se ajustó sobre una única columna (el target, ver
        # data_processing.py) -- espera entradas de forma (N, 1).
        # output_scaled trae (n_windows, pred_len): aplanar a (n_windows *
        # pred_len, 1), des-escalar, y volver a la forma original. Con
        # pred_len=1 esto es un no-op (misma forma antes y después).
        n_windows, pred_len = output_scaled.shape
        flat = output_scaled.numpy().reshape(-1, 1)
        return self.scaler_y.inverse_transform(flat).reshape(n_windows, pred_len)


def get_model_config(model: Any) -> dict[str, Any]:
    """Devuelve el `model_config` (seq_len, n_features, pred_len) con el que
    se registró un modelo pyfunc cargado.

    Permite que los consumidores (API, batch) validen la forma de entrada
    contra el modelo realmente servido en vez de asumir un seq_len fijo.
    Devuelve {} si el modelo no trae metadata (p. ej. dobles de test).
    """
    metadata = getattr(model, "metadata", None)
    flavors = getattr(metadata, "flavors", None) or {}
    config = flavors.get("python_function", {}).get("model_config") or {}
    return dict(config)


def log_and_register_model(
    *,
    model_state_dict_path: str,
    scaler_x_path: str,
    scaler_y_path: str,
    seq_len: int,
    n_features: int,
    registered_model_name: str,
    pred_len: int = 1,
) -> ModelVersion:
    """Loguea el modelo bundleado en el run activo y registra una nueva
    versión inmutable en el Model Registry de MLflow.
    """
    # .as_posix(): en Windows, os.path.join produce "artifacts_toy\dlinear_model.pth".
    # mlflow copia ese separador literal dentro del manifiesto portable del
    # modelo (MLmodel), así que al servirlo luego en un contenedor Linux
    # (la API) "\\" no separa nada y falla con FileNotFoundError buscando
    # ".../artifacts\\dlinear_model.pth" en vez de ".../artifacts/dlinear_model.pth".
    model_info = mlflow_log_model(
        artifact_path="model",
        python_model=DLinearForecaster(),
        artifacts={
            "model_state_dict": Path(model_state_dict_path).as_posix(),
            "scaler_X": Path(scaler_x_path).as_posix(),
            "scaler_y": Path(scaler_y_path).as_posix(),
        },
        model_config={"seq_len": seq_len, "n_features": n_features, "pred_len": pred_len},
        registered_model_name=registered_model_name,
    )
    logger.info(
        "model_registered",
        model_name=registered_model_name,
        model_version=model_info.registered_model_version,
        mlflow_run_id=model_info.run_id,
    )
    client = MlflowClient()
    return client.get_model_version(registered_model_name, model_info.registered_model_version)
