import joblib
import numpy as np
import torch
import torch.nn as nn
from pytest import approx as pytest_approx
from sklearn.preprocessing import StandardScaler

from src.train import ModelTrainer


class _FixedOutputModel(nn.Module):
    """Ignora la entrada y siempre devuelve el mismo tensor - permite
    verificar la matemática de evaluate_test_set (inverse_transform,
    MSE/MAE/RMSE/WAPE) contra valores calculados a mano, sin depender de si
    un entrenamiento real convergió a algo predecible."""

    def __init__(self, fixed_output: torch.Tensor):
        super().__init__()
        self.fixed_output = fixed_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed_output


def _write_trainer_artifacts(
    tmp_path, scaled_test_targets: torch.Tensor, scaled_test_inputs: torch.Tensor | None = None
) -> None:
    """Escribe train/val/test_tensors.pt sintéticos: train y val son
    contenido arbitrario (solo necesitan existir con la forma correcta para
    que ModelTrainer.__init__ no falle), test lleva los targets exactos que
    el test verifica (y, opcionalmente, las ventanas de entrada exactas)."""
    n_train, n_val, n_test = 6, 4, len(scaled_test_targets)
    seq_len, n_features, pred_len = 3, 2, 1

    def _rand_x(n):
        return torch.randn(n, seq_len, n_features)

    def _rand_y(n):
        return torch.randn(n, pred_len)

    torch.save((_rand_x(n_train), _rand_y(n_train)), tmp_path / "train_tensors.pt")
    torch.save((_rand_x(n_val), _rand_y(n_val)), tmp_path / "val_tensors.pt")
    test_inputs = scaled_test_inputs if scaled_test_inputs is not None else _rand_x(n_test)
    torch.save((test_inputs, scaled_test_targets), tmp_path / "test_tensors.pt")


def _dump_scaler_x(tmp_path) -> str:
    """scaler_X de 2 features, ambas con mean=5 y scale=5 (ajustado sobre
    [0, 10]); la feature 1 hace de target para la línea base."""
    path = tmp_path / "scaler_X.pkl"
    joblib.dump(StandardScaler().fit(np.array([[0.0, 0.0], [10.0, 10.0]])), path)
    return str(path)


def test_evaluate_test_set_computes_metrics_in_original_units(tmp_path):
    # scaler_y: mean=5, scale=5 (ajustado sobre [0, 10]) -> inverse_transform(z) = z*5 + 5.
    scaler_y = StandardScaler().fit(np.array([[0.0], [10.0]]))
    scaler_y_path = tmp_path / "scaler_y.pkl"
    joblib.dump(scaler_y, scaler_y_path)

    # Targets escalados -> originales: [0, 5, 10, 15] (via el scaler de arriba).
    scaled_targets = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    # Última lectura del target (feature 1) en cada ventana: escalada -1 ->
    # original 0, así que la persistencia predice 0 grados en las 4 ventanas.
    scaled_inputs = torch.zeros(4, 3, 2)
    scaled_inputs[:, -1, 1] = -1.0
    _write_trainer_artifacts(tmp_path, scaled_targets, scaled_inputs)

    trainer = ModelTrainer(artifact_dir=str(tmp_path), batch_size=256)

    # Predicciones escaladas -> originales: [0, 5, 10, 20]. Solo la última
    # ventana tiene error (5 grados).
    scaled_preds = torch.tensor([[-1.0], [0.0], [1.0], [3.0]])
    model = _FixedOutputModel(scaled_preds)

    metrics = trainer.evaluate_test_set(
        model,
        scaler_x_path=_dump_scaler_x(tmp_path),
        scaler_y_path=str(scaler_y_path),
        target_feature_index=1,
    )

    # errors originales: [0, 0, 0, 5] -> mse=6.25, mae=1.25, rmse=2.5
    assert metrics["test_mse"] == pytest_approx(6.25)
    assert metrics["test_mae"] == pytest_approx(1.25)
    assert metrics["test_rmse"] == pytest_approx(2.5)
    # rmse debe ser literalmente sqrt(mse), no un número inventado aparte.
    assert metrics["test_rmse"] == pytest_approx(metrics["test_mse"] ** 0.5)
    # wape: sum|errores| / sum|reales| = 5 / (0 + 5 + 10 + 15)
    assert metrics["test_wape"] == pytest_approx(5 / 30 * 100)
    # persistencia (0 en todas): errores [0, 5, 10, 15] -> mae=7.5, mse=87.5
    assert metrics["test_mae_persistence"] == pytest_approx(7.5)
    assert metrics["test_mse_persistence"] == pytest_approx(87.5)
    # skill: el modelo elimina 1 - 1.25/7.5 del error de la persistencia
    assert metrics["test_mae_skill"] == pytest_approx(1 - 1.25 / 7.5)


def test_evaluate_test_set_wape_is_not_blown_up_by_a_near_zero_target(tmp_path):
    scaler_y = StandardScaler().fit(np.array([[0.0], [10.0]]))
    scaler_y_path = tmp_path / "scaler_y.pkl"
    joblib.dump(scaler_y, scaler_y_path)

    # Targets originales [0.0, 10.0]: el primero es exactamente el caso que
    # hace explotar un MAPE punto a punto (division por ~0).
    scaled_targets = torch.tensor([[-1.0], [1.0]])
    _write_trainer_artifacts(tmp_path, scaled_targets)

    trainer = ModelTrainer(artifact_dir=str(tmp_path), batch_size=256)

    # Predicciones originales [1.0, 10.0]: 1 grado de error justo sobre el
    # target 0. WAPE = 1 / (0 + 10) = 10%, un valor acotado e interpretable.
    model = _FixedOutputModel(torch.tensor([[-0.8], [1.0]]))

    metrics = trainer.evaluate_test_set(
        model,
        scaler_x_path=_dump_scaler_x(tmp_path),
        scaler_y_path=str(scaler_y_path),
        target_feature_index=1,
    )

    assert metrics["test_wape"] == pytest_approx(10.0)


def _mean_val_loss(trainer: ModelTrainer, model: nn.Module) -> float:
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        losses = [criterion(model(x_b), y_b).item() for x_b, y_b in trainer.val_loader]
    return sum(losses) / len(losses)


def test_train_restores_the_best_validation_checkpoint(tmp_path, monkeypatch):
    """Regresión: best_state guardaba model.state_dict() sin copiar (son
    referencias a los tensores vivos), así que el modelo "restaurado" era
    en realidad el de la última época. Un lr enorme hace que val empeore
    tras las primeras épocas: el modelo devuelto debe reproducir
    exactamente el mejor val MSE reportado, no el de la última época."""
    monkeypatch.setattr("src.train.mlflow.log_metrics", lambda *a, **k: None)
    torch.manual_seed(0)
    seq_len, n_features = 8, 3
    for name, n in (("train", 64), ("val", 32), ("test", 8)):
        torch.save(
            (torch.randn(n, seq_len, n_features), torch.randn(n, 1)),
            tmp_path / f"{name}_tensors.pt",
        )
    trainer = ModelTrainer(artifact_dir=str(tmp_path), batch_size=16)

    _, best_val_loss, model = trainer.train(best_lr=5.0, epochs=6, patience=10)

    assert _mean_val_loss(trainer, model) == pytest_approx(best_val_loss, rel=1e-5)
