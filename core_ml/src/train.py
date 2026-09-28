import argparse
import copy
import os
import random

import joblib
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data_contracts import ETT_CONTRACT, FEATURE_COLUMNS
from src.logging_config import configure_logging, get_logger
from src.mlflow_utils import build_reproducibility_tags, log_and_register_model
from src.model_architecture import DLinear
from src.monitoring.drift_check import ReferenceProfile

configure_logging()
logger = get_logger(__name__)

# Configuración del dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# benchmark=False: cudnn.benchmark elige kernels por timing en cada corrida,
# lo que rompe la reproducibilidad que busca la semilla fija de abajo.
torch.backends.cudnn.benchmark = False

# Sin esto, dos corridas con el MISMO lr (init de pesos vía torch, shuffling
# del DataLoader de train, y la propia búsqueda de Optuna) producían
# métricas finales distintas -- verificado en el registro real de MLflow:
# la v2 (production, best_lr=0.00173, final_val_mse=0.00468) y la v4
# (candidata, best_lr=0.00153, final_val_mse=0.00593) usaron learning
# rates casi idénticos pero terminaron ~26% distintas, y quality_gate
# rechazó v4 por eso. Fijar la semilla no garantiza que un candidato
# futuro le gane a producción, pero hace que la MISMA corrida (mismo
# commit, mismos datos) sea reproducible en vez de variar por azar en
# cada ejecución del pipeline -- una corrida del job `train`, dos veces,
# debe dar el mismo resultado.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_REGISTERED_MODEL_NAME = "dlinear-ett-forecaster"
# Posición del target (OT) dentro de las features de entrada: la línea base
# de persistencia lee su última lectura de la ventana.
TARGET_FEATURE_INDEX = FEATURE_COLUMNS.index(ETT_CONTRACT.target_column)
DVC_FILE_BY_DATASET = {
    "toy": "data/toy/ETTh1_toy.csv.dvc",
    "raw": "data/raw/ETTh1.csv.dvc",
}


# ---------------------------------------------------------
# MLOPS TRAINING PIPELINE
# ---------------------------------------------------------
class ModelTrainer:
    def __init__(self, artifact_dir: str, batch_size: int = 256):
        self.artifact_dir = artifact_dir
        self.batch_size = batch_size

        # Cargar tensores preprocesados
        logger.info(f"Cargando tensores desde {artifact_dir}...")
        X_train, y_train = torch.load(
            os.path.join(artifact_dir, "train_tensors.pt"), weights_only=True
        )
        X_val, y_val = torch.load(os.path.join(artifact_dir, "val_tensors.pt"), weights_only=True)
        # test_tensors.pt existia desde data_processing.py pero nada lo
        # cargaba: el modelo nunca se evaluaba contra un holdout de verdad,
        # solo contra val (usado tambien para early stopping/seleccion de
        # checkpoint, asi que no es un holdout limpio). Ver evaluate_test_set
        # mas abajo, que es lo que finalmente lo usa.
        X_test, y_test = torch.load(
            os.path.join(artifact_dir, "test_tensors.pt"), weights_only=True
        )

        self.n_features = X_train.shape[2]
        self.seq_len = X_train.shape[1]
        # y_train ya viene en forma (N, pred_len) desde data_processing.py
        # (pred_len=1 -> (N, 1), comportamiento histórico).
        self.pred_len = y_train.shape[1] if y_train.dim() > 1 else 1

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
        )

    def optimize_hyperparameters(self, n_trials=3) -> float:
        """Busca el Learning Rate óptimo usando Optuna."""
        logger.info("Iniciando optimización de Optuna...")

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = DLinear(
                seq_len=self.seq_len, n_features=self.n_features, pred_len=self.pred_len
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Entrenamiento rápido de 2 épocas
            for _epoch in range(2):
                model.train()
                for x_b, y_b in self.train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x_b.to(device)), y_b.to(device))
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_b, y_b in self.val_loader:
                    loss = criterion(model(x_b.to(device)), y_b.to(device))
                    val_loss += loss.item()

            return val_loss / len(self.val_loader)

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
        )
        study.optimize(objective, n_trials=n_trials)
        best_lr = float(study.best_params["lr"])
        logger.info(f"Mejor Learning Rate encontrado: {best_lr:.5f}")
        return best_lr

    def train(
        self, best_lr: float, epochs: int = 25, patience: int = 5
    ) -> tuple[str, float, nn.Module]:
        """Bucle de entrenamiento principal con Early Stopping.

        Loguea métricas por época en el run de MLflow activo y devuelve la
        ruta del state_dict exportado (artefacto transitorio, no el producto
        final -- el producto final es la versión registrada en MLflow), la
        mejor pérdida de validación alcanzada, y el modelo restaurado a ese
        mejor checkpoint (para evaluate_test_set, sin releerlo de disco).
        """
        logger.info(f"Entrenando modelo DLinear final en {device}...")
        model = DLinear(
            seq_len=self.seq_len, n_features=self.n_features, pred_len=self.pred_len
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr)

        best_val_loss = float("inf")
        no_improve = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x_b, y_b in self.train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                preds = model(x_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_mean = train_loss / len(self.train_loader)

            # Validación
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_b, y_b in self.val_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    preds = model(x_b)
                    loss = criterion(preds, y_b)
                    val_loss += loss.item()

            val_mean = val_loss / len(self.val_loader)
            mlflow.log_metrics({"train_mse": train_mean, "val_mse": val_mean}, step=epoch)

            # Early Stopping
            if val_mean < best_val_loss:
                best_val_loss, no_improve = val_mean, 0
                # deepcopy: state_dict() devuelve REFERENCIAS a los tensores
                # vivos del modelo, no una copia -- sin esto, best_state
                # seguiría mutando con cada optimizer.step() y el "mejor
                # checkpoint" restaurado sería en realidad el de la última
                # época (early stopping sin efecto, y final_val_mse
                # describiendo pesos distintos a los que se registran).
                best_state = copy.deepcopy(model.state_dict())
                estado = "[Guardado]"
            else:
                no_improve += 1
                estado = ""

            logger.info(
                f"Epoch {epoch:2d} | Train MSE: {train_mean:.4f} | Val MSE: {val_mean:.4f} {estado}"
            )

            if no_improve >= patience:
                logger.info(f"Early stopping activado en la época {epoch}")
                break

        # Cargar los mejores pesos y exportar el state_dict (artefacto
        # transitorio que luego se empaqueta y registra en MLflow).
        if best_state is None:
            raise RuntimeError(
                "El entrenamiento nunca produjo una época con mejora en validación; "
                "no hay pesos que guardar."
            )
        model.load_state_dict(best_state)
        model_export_path = os.path.join(self.artifact_dir, "dlinear_model.pth")
        torch.save(model.state_dict(), model_export_path)
        logger.info(f"Entrenamiento completado. State dict exportado a {model_export_path}")
        return model_export_path, best_val_loss, model

    def evaluate_test_set(
        self,
        model: nn.Module,
        scaler_x_path: str,
        scaler_y_path: str,
        target_feature_index: int = TARGET_FEATURE_INDEX,
    ) -> dict[str, float]:
        """Evalúa el modelo final contra el holdout de test, junto a una línea
        base ingenua sobre exactamente las mismas ventanas.

        Es la única evaluación de este pipeline que no influyó de ninguna
        forma en qué modelo se entrenó ni en qué checkpoint se eligió: train
        ajusta pesos, val decide el learning rate (Optuna) y selecciona el
        mejor checkpoint (early stopping). test nunca se toca hasta este
        punto -- es lo que hace que estos números sean una estimación
        honesta de error, no una que el propio proceso de selección ya
        optimizó indirectamente.

        La línea base es la persistencia: repetir la última lectura observada
        del target durante todo el horizonte. En series con mucha inercia
        (como la temperatura de aceite) es difícil de batir a pocas horas
        vista, así que un error bajo en términos absolutos no dice nada por
        sí solo -- lo que justifica servir un modelo es cuánto mejora a la
        persistencia (`test_mae_skill`), y quality_gate.py lo exige.

        Las métricas se devuelven en la unidad real del target (°C de
        temperatura de aceite), no en la escala normalizada del
        StandardScaler.
        """
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_last_observed: list[np.ndarray] = []
        with torch.no_grad():
            for x_b, y_b in self.test_loader:
                all_preds.append(model(x_b.to(device)).cpu().numpy())
                all_targets.append(y_b.numpy())
                # Última lectura del target dentro de la ventana de entrada
                # (escalada con scaler_X, no con scaler_y).
                all_last_observed.append(x_b[:, -1, target_feature_index].numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        last_observed = np.concatenate(all_last_observed, axis=0)

        # scaler_y se ajustó sobre una sola columna (el target); inverse_transform
        # espera (N, 1), así que cada paso del horizonte se desescala por
        # separado y se reensambla en la forma (N, pred_len) original.
        original_shape = preds.shape
        preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(original_shape)
        targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).reshape(
            original_shape
        )

        # Persistencia: la última lectura, desescalada con los parámetros de
        # su propia columna en scaler_X, repetida en los pred_len pasos.
        last_observed_original = (
            last_observed * scaler_x.scale_[target_feature_index]
            + scaler_x.mean_[target_feature_index]
        )
        persistence_original = np.repeat(
            last_observed_original[:, np.newaxis], original_shape[1], axis=1
        )

        model_metrics = _forecast_metrics(preds_original, targets_original)
        persistence_metrics = _forecast_metrics(persistence_original, targets_original)
        return {
            "test_mse": model_metrics["mse"],
            "test_mae": model_metrics["mae"],
            "test_rmse": model_metrics["rmse"],
            "test_wape": model_metrics["wape"],
            "test_mse_persistence": persistence_metrics["mse"],
            "test_mae_persistence": persistence_metrics["mae"],
            # Fracción del error de la persistencia que el modelo elimina:
            # 0 = igual que repetir el último valor, >0 = mejor, <0 = peor.
            "test_mae_skill": 1.0 - model_metrics["mae"] / persistence_metrics["mae"],
        }


def _forecast_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """MSE, MAE, RMSE y WAPE en las unidades del target."""
    errors = predicted - actual
    mse = float(np.mean(errors**2))
    # WAPE (sum|error| / sum|real|), no MAPE: la temperatura de aceite del
    # split de test de ETTh1 cruza cero, y MAPE divide punto a punto por el
    # valor real, así que unas pocas lecturas cercanas a 0 °C lo dominan.
    # WAPE agrega antes de dividir. El epsilon solo cubre el caso degenerado
    # de un holdout cuyos targets suman exactamente cero.
    epsilon = 1e-8
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(mse)),
        "wape": float(np.sum(np.abs(errors)) / max(np.sum(np.abs(actual)), epsilon) * 100),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Modelo DLinear")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    # 5, no 3: el proxy de Optuna (optimize_hyperparameters) solo entrena 2
    # épocas por trial para evaluar un lr, pero el entrenamiento final corre
    # hasta `epochs` (25) con early stopping -- un lr que luce razonable en
    # ese proxy de 2 épocas puede necesitar más de 25 épocas para converger
    # de verdad. Con solo 3 trials, quality_gate rechazó un candidato real
    # porque Optuna eligió un lr demasiado bajo para ese presupuesto de
    # épocas. La semilla fija (SEED, arriba) hace que la búsqueda sea
    # reproducible, pero no mejor: más trials bajan la probabilidad de
    # quedarse con un lr malo del espacio de búsqueda.
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument(
        "--dataset",
        choices=["toy", "raw"],
        default="raw",
        help="Debe coincidir con el dataset usado en data_processing.py para que el "
        "DVC data hash registrado sea el correcto.",
    )
    parser.add_argument("--registered_model_name", type=str, default=DEFAULT_REGISTERED_MODEL_NAME)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "dlinear-ett-forecasting"))

    trainer = ModelTrainer(args.artifact_dir)

    with mlflow.start_run() as run:
        # Tupla de reproducibilidad: Git Commit + DVC Data Hash + Hyperparameters
        # + MLflow Run ID (este run) + Container Image Tag.
        repro_tags = build_reproducibility_tags(DVC_FILE_BY_DATASET[args.dataset])
        mlflow.set_tags(repro_tags)
        mlflow.log_params(
            {
                "seq_len": trainer.seq_len,
                "pred_len": trainer.pred_len,
                "n_features": trainer.n_features,
                "batch_size": trainer.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "n_trials": args.n_trials,
                "dataset": args.dataset,
            }
        )

        best_lr = trainer.optimize_hyperparameters(n_trials=args.n_trials)
        mlflow.log_param("best_lr", best_lr)

        model_export_path, best_val_loss, best_model = trainer.train(
            best_lr=best_lr, epochs=args.epochs, patience=args.patience
        )
        mlflow.log_metric("final_val_mse", best_val_loss)

        test_metrics = trainer.evaluate_test_set(
            best_model,
            scaler_x_path=os.path.join(args.artifact_dir, "scaler_X.pkl"),
            scaler_y_path=os.path.join(args.artifact_dir, "scaler_y.pkl"),
        )
        # final_test_mse_persistence es la vara que quality_gate.py exige
        # superar antes de promover cualquier candidato.
        mlflow.log_metrics({f"final_{key}": value for key, value in test_metrics.items()})
        logger.info("test_set_evaluated", **test_metrics)

        # Complete the reference profile data_processing.py started (raw
        # per-sensor stats and weekly-mean band, no performance figures yet - it trained no
        # model) with the held-out metrics just computed, and log it as an
        # artifact of THIS run. batch_inference.py downloads it via the
        # served model's run_id and uses it for the post-inference drift
        # check (core_ml/src/monitoring/drift_check.py).
        profile_path = os.path.join(args.artifact_dir, "reference_profile.json")
        reference_profile = ReferenceProfile.read(profile_path)
        reference_profile.baseline_test_mse = test_metrics["test_mse"]
        reference_profile.baseline_test_mae = test_metrics["test_mae"]
        reference_profile.write(profile_path)
        mlflow.log_artifact(profile_path, artifact_path="monitoring")

        model_version = log_and_register_model(
            model_state_dict_path=model_export_path,
            scaler_x_path=os.path.join(args.artifact_dir, "scaler_X.pkl"),
            scaler_y_path=os.path.join(args.artifact_dir, "scaler_y.pkl"),
            seq_len=trainer.seq_len,
            n_features=trainer.n_features,
            pred_len=trainer.pred_len,
            registered_model_name=args.registered_model_name,
        )

        logger.info(
            "training_run_completed",
            mlflow_run_id=run.info.run_id,
            model_name=args.registered_model_name,
            model_version=model_version.version,
            git_commit_hash=repro_tags["git_commit_hash"],
            dvc_data_hash=repro_tags["dvc_data_hash"],
            container_image_tag=repro_tags["container_image_tag"],
        )
