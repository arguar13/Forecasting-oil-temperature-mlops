import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.data_contracts import (
    ETT_CONTRACT,
    FEATURE_COLUMNS,
    TOY_ETT_CONTRACT,
    DataContractError,
    ETTDatasetContract,
    validate_ett_csv,
)
from src.logging_config import configure_logging, get_logger
from src.monitoring.drift_check import build_feature_baselines

configure_logging()
logger = get_logger(__name__)

# Datasets versionados por DVC (ver core_ml/data/*.dvc). `dvc pull` los
# materializa localmente antes de correr, para que cada corrida sea
# reproducible a partir de un hash de datos conocido.
TOY_DATA_PATH = "data/toy/ETTh1_toy.csv"
RAW_DATA_PATH = "data/raw/ETTh1.csv"


class DataProcessor:
    def __init__(
        self,
        data_path: str,
        seq_length: int = 48,
        output_dir: str = "artifacts",
        contract: ETTDatasetContract = ETT_CONTRACT,
        pred_length: int = 1,
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.output_dir = output_dir
        self.contract = contract
        # Horizonte de predicción: cuántos pasos futuros predice cada
        # ventana. 1 = un solo paso adelante (comportamiento histórico);
        # >1 genera una secuencia de `pred_length` valores futuros por
        # ventana, no un único escalar.
        self.pred_length = pred_length

        os.makedirs(self.output_dir, exist_ok=True)

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Genera secuencias de ventanas deslizantes.

        Cada ventana de entrada X[i:i+seq_length] se empareja con los
        `pred_length` valores futuros y[i+seq_length : i+seq_length+pred_length]
        -- no solo el siguiente valor. `y` ya llega escalado (ver
        process_and_save) y con una sola columna (el target), así que el
        `.squeeze(-1)` deja cada ventana de salida en forma (pred_length,),
        no (pred_length, 1).
        """
        Xs, ys = [], []
        last_start = len(X) - self.seq_length - self.pred_length + 1
        for i in range(last_start):
            Xs.append(X[i : i + self.seq_length])
            ys.append(y[i + self.seq_length : i + self.seq_length + self.pred_length].squeeze(-1))
        return np.array(Xs), np.array(ys)

    def process_and_save(self):
        """Pipeline principal de procesamiento y exportación de artefactos."""
        # FAIL FAST: valida el contrato de datos antes de gastar cómputo en
        # escalado, ventaneo o entrenamiento.
        logger.info(f"Validando contrato de datos para {self.data_path}...")
        data = validate_ett_csv(self.data_path, contract=self.contract)
        logger.info("Contrato de datos OK.")

        data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")
        # Las ventanas deslizantes asumen orden cronológico.
        data.sort_values("date", inplace=True)
        data.set_index("date", inplace=True)
        date_index = pd.DatetimeIndex(data.index)

        # Extracción de componentes temporales
        data["month"] = date_index.month
        data["day"] = date_index.day
        data["hour"] = date_index.hour

        target_col = self.contract.target_column
        # Columnas explícitas del contrato (mismo orden que esperan la API y
        # el batch), no `data.columns`: una columna extra en el CSV cambiaría
        # en silencio n_features y el orden de entrada del modelo.
        feature_cols = list(FEATURE_COLUMNS)

        # 2. Partición Train/Val/Test (70% / 10% / 20%)
        n = len(data)
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size : train_size + val_size]
        test_data = data.iloc[train_size + val_size :]

        # FAIL FAST: cada split necesita al menos seq_length + pred_length
        # filas para producir una sola ventana. Sin este chequeo, un split
        # corto genera arrays vacíos y el error aparece mucho después, en
        # train.py, como un ZeroDivisionError sin contexto.
        min_split_rows = self.seq_length + self.pred_length
        for split_name, split_df in (("train", train_data), ("val", val_data), ("test", test_data)):
            if len(split_df) < min_split_rows:
                raise DataContractError(
                    f"El split '{split_name}' tiene {len(split_df)} filas; se necesitan al "
                    f"menos {min_split_rows} (seq_length={self.seq_length} + "
                    f"pred_length={self.pred_length}) para generar una ventana."
                )

        # Perfil de referencia para el chequeo de drift (core_ml/src/monitoring/
        # drift_check.py): media/std por sensor del split de train, ANTES de
        # escalar -- para comparar contra lecturas crudas (grados C, MW) del
        # batch de inferencia, no la escala normalizada que produce
        # StandardScaler mas abajo. Solo las estadisticas de features; train.py
        # completa baseline_test_mse/mae despues de evaluar el holdout, algo
        # que este metodo no puede hacer (aqui todavia no existe ningun modelo).
        feature_baselines = build_feature_baselines(train_data)
        feature_baselines.write(os.path.join(self.output_dir, "reference_profile.json"))

        # 3. Escalado
        logger.info("Escalando variables y guardando scalers...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        train_X = scaler_X.fit_transform(train_data[feature_cols])
        val_X = scaler_X.transform(val_data[feature_cols])
        test_X = scaler_X.transform(test_data[feature_cols])

        train_y = scaler_y.fit_transform(train_data[[target_col]])
        val_y = scaler_y.transform(val_data[[target_col]])
        test_y = scaler_y.transform(test_data[[target_col]])

        # Guardar scalers para el contenedor de inferencia (FastAPI)
        joblib.dump(scaler_X, os.path.join(self.output_dir, "scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(self.output_dir, "scaler_y.pkl"))

        # 4. Generación de secuencias
        logger.info("Generando secuencias (Sliding Windows)...")
        X_train_seq, y_train_seq = self._create_sequences(train_X, train_y)
        X_val_seq, y_val_seq = self._create_sequences(val_X, val_y)
        X_test_seq, y_test_seq = self._create_sequences(test_X, test_y)

        # 5. Exportar tensores listos para train.py
        logger.info("Exportando tensores a disco...")
        torch.save(
            (
                torch.tensor(X_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32),
            ),
            os.path.join(self.output_dir, "train_tensors.pt"),
        )
        torch.save(
            (
                torch.tensor(X_val_seq, dtype=torch.float32),
                torch.tensor(y_val_seq, dtype=torch.float32),
            ),
            os.path.join(self.output_dir, "val_tensors.pt"),
        )
        torch.save(
            (
                torch.tensor(X_test_seq, dtype=torch.float32),
                torch.tensor(y_test_seq, dtype=torch.float32),
            ),
            os.path.join(self.output_dir, "test_tensors.pt"),
        )

        logger.info(f"Procesamiento finalizado exitosamente. Artefactos en {self.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesamiento de datos ETTh1")
    parser.add_argument(
        "--dataset",
        choices=["toy", "raw"],
        default="raw",
        help="'toy' usa el dataset de ~1000 filas para pruebas E2E rápidas; 'raw' usa el dataset completo.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Sobreescribe la ruta del CSV de entrada (por defecto se deriva de --dataset).",
    )
    parser.add_argument("--seq_length", type=int, default=48)
    parser.add_argument(
        "--pred_length",
        type=int,
        default=48,
        help="Horizonte de predicción (pasos futuros por ventana). 48 = predice las "
        "próximas 48 horas a partir de las 48 anteriores (mismo orden que seq_length, "
        "uno de los horizontes estándar del paper de DLinear en ETTh1).",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    if args.dataset == "toy":
        data_path = args.data_path or TOY_DATA_PATH
        contract = TOY_ETT_CONTRACT
    else:
        data_path = args.data_path or RAW_DATA_PATH
        contract = ETT_CONTRACT

    processor = DataProcessor(
        data_path,
        args.seq_length,
        args.output_dir,
        contract=contract,
        pred_length=args.pred_length,
    )
    processor.process_and_save()
