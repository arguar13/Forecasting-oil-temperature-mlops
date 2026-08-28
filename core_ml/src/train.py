import argparse
import os

import mlflow
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.logging_config import configure_logging, get_logger
from src.mlflow_utils import build_reproducibility_tags, log_and_register_model
from src.model_architecture import DLinear

configure_logging()
logger = get_logger(__name__)

# Configuración del dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

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

        self.n_features = X_train.shape[2]
        self.seq_len = X_train.shape[1]

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
        )

    def optimize_hyperparameters(self, n_trials=3) -> float:
        """Busca el Learning Rate óptimo usando Optuna."""
        logger.info("Iniciando optimización de Optuna...")

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = DLinear(seq_len=self.seq_len, n_features=self.n_features).to(device)
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

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_lr = float(study.best_params["lr"])
        logger.info(f"Mejor Learning Rate encontrado: {best_lr:.5f}")
        return best_lr

    def train(self, best_lr: float, epochs: int = 25, patience: int = 5) -> tuple[str, float]:
        """Bucle de entrenamiento principal con Early Stopping.

        Loguea métricas por época en el run de MLflow activo y devuelve la
        ruta del state_dict exportado (artefacto transitorio, no el producto
        final -- el producto final es la versión registrada en MLflow) junto
        con la mejor pérdida de validación alcanzada.
        """
        logger.info(f"Entrenando modelo DLinear final en {device}...")
        model = DLinear(seq_len=self.seq_len, n_features=self.n_features).to(device)
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
                best_state = model.state_dict()
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
        return model_export_path, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Modelo DLinear")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=3)
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

        model_export_path, best_val_loss = trainer.train(
            best_lr=best_lr, epochs=args.epochs, patience=args.patience
        )
        mlflow.log_metric("final_val_mse", best_val_loss)

        model_version = log_and_register_model(
            model_state_dict_path=model_export_path,
            scaler_x_path=os.path.join(args.artifact_dir, "scaler_X.pkl"),
            scaler_y_path=os.path.join(args.artifact_dir, "scaler_y.pkl"),
            seq_len=trainer.seq_len,
            n_features=trainer.n_features,
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
