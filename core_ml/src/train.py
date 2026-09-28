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

    def evaluate_test_set(self, model: nn.Module, scaler_y_path: str) -> dict[str, float]:
        """Evalúa el modelo final contra el holdout de test.

        Es la única evaluación de este pipeline que no influyó de ninguna
        forma en qué modelo se entrenó ni en qué checkpoint se eligió: train
        ajusta pesos, val decide el learning rate (Optuna) y selecciona el
        mejor checkpoint (early stopping). test nunca se toca hasta este
        punto -- es lo que hace que estos números sean una estimación
        honesta de error, no una que el propio proceso de selección ya
        optimizó indirectamente.

        Las métricas se devuelven en la unidad real del target (°C de
        temperatura de aceite), no en la escala normalizada del
        StandardScaler: un MSE=0.005 en unidades escaladas no le dice nada a
        un humano ni a un dashboard sobre cuántos grados de error tiene el
        modelo en la práctica.
        """
        scaler_y = joblib.load(scaler_y_path)

        model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        with torch.no_grad():
            for x_b, y_b in self.test_loader:
                preds = model(x_b.to(device)).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_b.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # scaler_y se ajustó sobre una sola columna (el target); inverse_transform
        # espera (N, 1), así que cada paso del horizonte se desescala por
        # separado y se reensambla en la forma (N, pred_len) original.
        original_shape = preds.shape
        preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(original_shape)
        targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).reshape(
            original_shape
        )

        errors = preds_original - targets_original
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        # WAPE (sum|error| / sum|real|), no MAPE: la temperatura de aceite del
        # split de test de ETTh1 cruza cero, y MAPE divide punto a punto por
        # el valor real -- en una corrida real dio ~2860%, un número sin
        # sentido dominado por unas pocas lecturas cercanas a 0 °C. WAPE
        # agrega antes de dividir, así que un target puntual cercano a cero
        # no lo hace explotar. El epsilon solo cubre el caso degenerado de
        # un holdout cuyos targets suman exactamente cero.
        epsilon = 1e-8
        wape = float(np.sum(np.abs(errors)) / max(np.sum(np.abs(targets_original)), epsilon) * 100)

        return {"test_mse": mse, "test_mae": mae, "test_rmse": rmse, "test_wape": wape}


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
            best_model, scaler_y_path=os.path.join(args.artifact_dir, "scaler_y.pkl")
        )
        mlflow.log_metrics(
            {
                "final_test_mse": test_metrics["test_mse"],
                "final_test_mae": test_metrics["test_mae"],
                "final_test_rmse": test_metrics["test_rmse"],
                "final_test_wape": test_metrics["test_wape"],
            }
        )
        logger.info(
            "test_set_evaluated",
            test_mse=test_metrics["test_mse"],
            test_mae=test_metrics["test_mae"],
            test_rmse=test_metrics["test_rmse"],
            test_wape=test_metrics["test_wape"],
        )

        # Complete the reference profile data_processing.py started (raw
        # per-sensor mean/std, no performance figures yet - it trained no
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
