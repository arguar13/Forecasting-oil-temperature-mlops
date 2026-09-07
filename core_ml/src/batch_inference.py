import io
import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel, load_model
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data_contracts import (
    ETT_CONTRACT,
    FEATURE_COLUMNS,
    DataContractError,
    validate_feature_dataframe,
)
from src.events import publish_batch_inference_completed
from src.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "dlinear-ett-forecaster"
DEFAULT_MODEL_ALIAS = "production"

# Retry con backoff acotado (máx. 3 intentos): cubre blips transitorios de
# red contra S3/MLflow sin reintentar para siempre. No lleva circuit
# breaker -- a diferencia de la API (que corre indefinidamente y sí se
# beneficia de "dejar de golpear" una dependencia caída), el CronJob es un
# proceso nuevo en cada corrida, así que el propio `backoffLimit` de
# Kubernetes (ver kubernetes/base/cronjob.yaml) ya acota los reintentos
# entre corridas completas.
_RESILIENT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)


CLOUDWATCH_NAMESPACE = "DLinearBatchInference"
CLOUDWATCH_HEARTBEAT_METRIC = "InferenceSuccess"


class BatchInferenceService:
    def __init__(self):
        aws_endpoint = os.getenv("AWS_ENDPOINT_URL")
        self.s3_client = boto3.client("s3", endpoint_url=aws_endpoint)
        self.cloudwatch_client = boto3.client("cloudwatch", endpoint_url=aws_endpoint)
        self.bucket = os.getenv("MODEL_BUCKET_NAME", "mlops-portafolio-proj3-models")
        self.input_key = os.getenv("BATCH_INPUT_KEY", "batch/input_data.csv")
        self.output_key = os.getenv("BATCH_OUTPUT_KEY", "batch/predictions_output.csv")
        self.model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model_alias = os.getenv("MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
        self.seq_len = 48
        self.batch_size = 512
        self.model: PyFuncModel | None = None

    @_RESILIENT_RETRY
    def load_model(self):
        """Carga la versión inmutable en `{model_name}@{model_alias}` desde
        el MLflow Model Registry. Nunca se lee un `.pth`/`.pkl` suelto: el
        modelo y sus scalers viajan juntos como un único artefacto versionado.
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        model_uri = f"models:/{self.model_name}@{self.model_alias}"
        logger.info("model_load_started", model_uri=model_uri)
        self.model = load_model(model_uri)
        logger.info("model_load_succeeded", model_uri=model_uri)

    @_RESILIENT_RETRY
    def _download_input(self) -> pd.DataFrame:
        logger.info("batch_input_download_started", bucket=self.bucket, key=self.input_key)
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=self.input_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    @_RESILIENT_RETRY
    def _upload_output(self, body: str) -> None:
        logger.info("batch_output_upload_started", bucket=self.bucket, key=self.output_key)
        self.s3_client.put_object(Bucket=self.bucket, Key=self.output_key, Body=body)

    def _publish_heartbeat(self) -> None:
        """Dead man's switch: kubernetes/base/cronjob.yaml corre este proceso
        una vez al día y termina, así que "alertar cuando falla" no puede
        depender de que el propio proceso que falló siga vivo lo suficiente
        para reportarlo -- eso deja sin cubrir justo los modos de falla más
        duros (OOMKilled, imagen que nunca arranca, backoffLimit agotado
        antes de correr una sola vez). En vez de eso, cada corrida EXITOSA
        publica este metric; el CloudWatch Alarm de terraform/alarms.tf se
        dispara si no llega ninguno dentro de la ventana esperada -- una
        ausencia de éxito, no una presencia de fallo, que cubre todo lo
        anterior sin excepción.

        Best-effort, igual que publish_batch_inference_completed (events.py):
        un fallo al publicar el heartbeat no debe hacer fallar un batch que
        sí terminó bien.
        """
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace=CLOUDWATCH_NAMESPACE,
                MetricData=[
                    {"MetricName": CLOUDWATCH_HEARTBEAT_METRIC, "Value": 1.0, "Unit": "Count"}
                ],
            )
        except Exception as exc:
            logger.warning("heartbeat_publish_failed", error=str(exc))

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
        predictions = []
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

        # Evento de observabilidad (best-effort, no bloqueante -- ver events.py).
        publish_batch_inference_completed(
            bucket=self.bucket,
            output_key=self.output_key,
            n_predictions=len(output_df),
            model_name=self.model_name,
            model_alias=self.model_alias,
        )
        self._publish_heartbeat()


if __name__ == "__main__":
    service = BatchInferenceService()
    service.load_model()
    service.process_batch()
