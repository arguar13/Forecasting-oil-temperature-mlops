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
    MSE/MAE/RMSE/MAPE) contra valores calculados a mano, sin depender de si
    un entrenamiento real convergió a algo predecible."""

    def __init__(self, fixed_output: torch.Tensor):
        super().__init__()
        self.fixed_output = fixed_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed_output


def _write_trainer_artifacts(tmp_path, scaled_test_targets: torch.Tensor) -> None:
    """Escribe train/val/test_tensors.pt sintéticos: train y val son
    contenido arbitrario (solo necesitan existir con la forma correcta para
    que ModelTrainer.__init__ no falle), test lleva los targets exactos que
    el test verifica."""
    n_train, n_val, n_test = 6, 4, len(scaled_test_targets)
    seq_len, n_features, pred_len = 3, 2, 1

    def _rand_x(n):
        return torch.randn(n, seq_len, n_features)

    def _rand_y(n):
        return torch.randn(n, pred_len)

    torch.save((_rand_x(n_train), _rand_y(n_train)), tmp_path / "train_tensors.pt")
    torch.save((_rand_x(n_val), _rand_y(n_val)), tmp_path / "val_tensors.pt")
    torch.save((_rand_x(n_test), scaled_test_targets), tmp_path / "test_tensors.pt")


def test_evaluate_test_set_computes_metrics_in_original_units(tmp_path):
    # scaler_y: mean=5, scale=5 (ajustado sobre [0, 10]) -> inverse_transform(z) = z*5 + 5.
    scaler_y = StandardScaler().fit(np.array([[0.0], [10.0]]))
    scaler_y_path = tmp_path / "scaler_y.pkl"
    joblib.dump(scaler_y, scaler_y_path)

    # Targets escalados -> originales: [0, 5, 10, 15] (via el scaler de arriba).
    scaled_targets = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    _write_trainer_artifacts(tmp_path, scaled_targets)

    trainer = ModelTrainer(artifact_dir=str(tmp_path), batch_size=256)

    # Predicciones escaladas -> originales: [0, 5, 10, 20]. Solo la última
    # ventana tiene error (5 grados), y su target real (15) no está cerca de
    # cero -- deja fuera, a propósito, el caso limite de MAPE, que se cubre
    # en el test de abajo.
    scaled_preds = torch.tensor([[-1.0], [0.0], [1.0], [3.0]])
    model = _FixedOutputModel(scaled_preds)

    metrics = trainer.evaluate_test_set(model, scaler_y_path=str(scaler_y_path))

    # errors originales: [0, 0, 0, 5] -> mse=6.25, mae=1.25, rmse=2.5
    assert metrics["test_mse"] == pytest_approx(6.25)
    assert metrics["test_mae"] == pytest_approx(1.25)
    assert metrics["test_rmse"] == pytest_approx(2.5)
    # rmse debe ser literalmente sqrt(mse), no un número inventado aparte.
    assert metrics["test_rmse"] == pytest_approx(metrics["test_mse"] ** 0.5)
    # mape: solo la última ventana aporta error, target real=15 -> 5/15*100/4
    assert metrics["test_mape"] == pytest_approx((5 / 15) * 100 / 4)


def test_evaluate_test_set_guards_mape_against_a_near_zero_target(tmp_path):
    scaler_y = StandardScaler().fit(np.array([[0.0], [10.0]]))
    scaler_y_path = tmp_path / "scaler_y.pkl"
    joblib.dump(scaler_y, scaler_y_path)

    # Target escalado -1.0 -> original 0.0: exactamente el caso limite que
    # rompería un MAPE sin guardia (division por cero).
    scaled_targets = torch.tensor([[-1.0]])
    _write_trainer_artifacts(tmp_path, scaled_targets)

    trainer = ModelTrainer(artifact_dir=str(tmp_path), batch_size=256)

    # Predicción igual al target -> error cero, así que el valor exacto del
    # epsilon de la guardia no importa para este assert: lo que se verifica
    # es que no lanza ZeroDivisionError/produce inf o NaN.
    model = _FixedOutputModel(scaled_targets)

    metrics = trainer.evaluate_test_set(model, scaler_y_path=str(scaler_y_path))

    assert metrics["test_mape"] == pytest_approx(0.0)
    assert np.isfinite(metrics["test_mape"])
