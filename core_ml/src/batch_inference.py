import io
import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel, load_model
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_contracts import (
    ETT_CONTRACT,
    FEATURE_COLUMNS,
    DataContractError,
    validate_feature_dataframe,
)
from src.logging_config import configure_logging, get_logger
from src.mlflow_utils import get_model_config
from src.monitoring.drift_check import ReferenceProfile, check_batch_for_drift, log_drift_report

configure_logging()
logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "dlinear-ett-forecaster"
DEFAULT_MODEL_ALIAS = "production"
# Solo como respaldo si el modelo cargado no declara su seq_len.
DEFAULT_SEQ_LEN = 48

# Retry con backoff acotado (máx. 3 intentos): cubre blips transitorios de
# red contra S3/MLflow sin reintentar para siempre. El propio `backoffLimit`
# de Kubernetes (ver kubernetes/base/cronjob.yaml) ya acota los reintentos
# entre corridas completas del CronJob.
_RESILIENT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)


class BatchInferenceService:
    def __init__(self):
        aws_endpoint = os.getenv("AWS_ENDPOINT_URL")
        self.s3_client = boto3.client("s3", endpoint_url=aws_endpoint)
        self.bucket = os.getenv("MODEL_BUCKET_NAME", "mlops-portafolio-proj3-models")
        self.input_key = os.getenv("BATCH_INPUT_KEY", "batch/input_data.csv")
        self.output_key = os.getenv("BATCH_OUTPUT_KEY", "batch/predictions_output.csv")
        self.model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model_alias = os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
        self.seq_len = DEFAULT_SEQ_LEN
        self.batch_size = int(os.getenv("BATCH_SIZE", "512"))
        self.model: PyFuncModel | None = None
        self.reference_profile: ReferenceProfile | None = None

    @_RESILIENT_RETRY
    def load_model(self):
        """Carga la versión inmutable en `{model_name}@{model_alias}` desde
        el MLflow Model Registry. Nunca se lee un `.pth`/`.pkl` suelto: el
        modelo y sus scalers viajan juntos como un único artefacto versionado.
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        # El alias se resuelve UNA vez a un número de versión, y tanto el
        # modelo como su perfil de referencia se cargan de esa versión: si
        # quality_gate moviera el alias entre dos resoluciones, el drift se
        # compararía contra el perfil de otro modelo.
        version = MlflowClient().get_model_version_by_alias(self.model_name, self.model_alias)
        model_uri = f"models:/{self.model_name}/{version.version}"
        logger.info("model_load_started", model_uri=model_uri, alias=self.model_alias)
        self.model = load_model(model_uri)
        # seq_len sale del modelo servido, no de una constante: si se
        # reentrena con otra ventana, el batch se adapta sin tocar código.
        self.seq_len = int(get_model_config(self.model).get("seq_len", DEFAULT_SEQ_LEN))

        try:
            profile_path = mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path="monitoring/reference_profile.json"
            )
            self.reference_profile = ReferenceProfile.read(profile_path)
        except (MlflowException, OSError, KeyError, TypeError, ValueError):
            # El drift check es informativo: una versión sin perfil, o con un
            # perfil que este código no sabe leer (JSON corrupto, o escrito
            # con otro formato de ReferenceProfile), no debe bloquear el
            # scoring.
            logger.warning("reference_profile_unavailable", run_id=version.run_id, exc_info=True)
            self.reference_profile = None
        logger.info(
            "model_load_succeeded",
            model_uri=model_uri,
            version=version.version,
            seq_len=self.seq_len,
        )

    @_RESILIENT_RETRY
    def _download_input(self) -> pd.DataFrame:
        logger.info("batch_input_download_started", bucket=self.bucket, key=self.input_key)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=self.input_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    @_RESILIENT_RETRY
    def _upload_output(self, body: str) -> None:
        logger.info("batch_output_upload_started", bucket=self.bucket, key=self.output_key)
        self.s3_client.put_object(Bucket=self.bucket, Key=self.output_key, Body=body)

    def process_batch(self):
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load_model() primero.")

        df = self._download_input()

        # FAIL FAST: valida el contrato de datos antes de gastar cómputo en inferencia.
        if len(df) < self.seq_len:
            raise DataContractError(f"El dataset debe tener al menos {self.seq_len} filas.")
        validate_feature_dataframe(df, contract=ETT_CONTRACT)

        # Ventanas deslizantes superpuestas: cada fila i..i+seq_len-1 es una secuencia.
        feature_values = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        n_windows = len(feature_values) - self.seq_len + 1
        windows = np.stack([feature_values[i : i + self.seq_len] for i in range(n_windows)])

        logger.info("batch_scoring_started", n_windows=n_windows, batch_size=self.batch_size)
        predictions: list[np.ndarray] = []
        for start in range(0, n_windows, self.batch_size):
            batch = windows[start : start + self.batch_size]
            preds = self.model.predict(batch)  # una sola pasada vectorizada por lote
            predictions.append(np.asarray(preds))

        stacked = np.concatenate(predictions, axis=0)
        # El modelo puede predecir 1 paso (columna unica "Prediction",
        # comportamiento historico) o varios pasos por ventana ("Prediction_h1"
        # .. "Prediction_hN") -- el nombre de columnas se deriva de la forma
        # real de salida del modelo en vez de asumir un horizonte fijo.
        pred_len = stacked.shape[1] if stacked.ndim > 1 else 1
        if pred_len == 1:
            columns = ["Prediction"]
        else:
            columns = [f"Prediction_h{h}" for h in range(1, pred_len + 1)]
        output_df = pd.DataFrame(stacked.reshape(len(stacked), pred_len), columns=columns)
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)

        self._upload_output(csv_buffer.getvalue())
        logger.info("batch_inference_completed", n_predictions=len(output_df))

        # Chequeo de drift simple sobre el batch recién inferido -- solo
        # loguea el resultado, no dispara ninguna acción automática (ver
        # core_ml/src/monitoring/drift_check.py).
        if self.reference_profile is not None:
            report = check_batch_for_drift(df, self.reference_profile)
            log_drift_report(report)


if __name__ == "__main__":
    service = BatchInferenceService()
    service.load_model()
    service.process_batch()
