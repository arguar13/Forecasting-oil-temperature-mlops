"""Arquitectura DLinear -- única fuente de verdad dentro de `core_ml`.

`train.py`, `batch_inference.py` y `mlflow_utils.py` importan la red desde
aquí. Este módulo también se copia tal cual a la imagen de la API
(ver `Dockerfile`) porque `mlflow.pyfunc` necesita poder reconstruir esta
misma clase por referencia de módulo (`src.model_architecture.DLinear`) al
deserializar el modelo registrado, sin importar en qué contenedor se cargue.
"""

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, seq_len=48, n_features=10):
        super().__init__()
        self.seq_len = seq_len
        self.decompsition = series_decomp(25)
        self.Linear_Seasonal = nn.Linear(seq_len, 1)
        self.Linear_Trend = nn.Linear(seq_len, 1)
        self.combine = nn.Linear(n_features, 1)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_out = self.Linear_Seasonal(seasonal_init).squeeze(-1)
        trend_out = self.Linear_Trend(trend_init).squeeze(-1)

        x = seasonal_out + trend_out
        return self.combine(x)
