"""Contratos de datos para el pipeline ETT (Electricity Transformer Temperature).

Principio FAIL FAST: este módulo se ejecuta ANTES de cualquier cómputo costoso
(preprocesamiento, entrenamiento, inferencia por lotes). Si el dataset no
cumple el contrato, se lanza `DataContractError` inmediatamente en vez de
dejar que un dato corrupto se propague silenciosamente hasta el modelo o,
peor, hasta producción.

Los rangos por columna están calibrados sobre ETTh1.csv completo (17,420
filas horarias, sin nulos, jul-2016 a jun-2018) con margen para variación
natural de sensores, sin aceptar valores absurdos o corruptos.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class DataContractError(Exception):
    """El dataset viola el contrato de datos. FAIL FAST antes de entrenar/inferir."""


class ColumnContract(BaseModel):
    """Contrato esperado para una columna numérica del dataset ETT."""

    name: str
    min_value: float
    max_value: float
    allow_null: bool = False


class ETTDatasetContract(BaseModel):
    """Contrato de datos declarativo para el dataset ETT."""

    required_columns: tuple[str, ...] = (
        "date",
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",
        "OT",
    )
    target_column: str = "OT"
    # Con seq_len=48 y split 70/10/20, 600 filas garantizan >=10 ventanas
    # deslizantes utilizables incluso en el split de validación (el más chico).
    min_rows: int = Field(default=600)
    columns: tuple[ColumnContract, ...] = (
        ColumnContract(name="HUFL", min_value=-40.0, max_value=40.0),
        ColumnContract(name="HULL", min_value=-15.0, max_value=20.0),
        ColumnContract(name="MUFL", min_value=-40.0, max_value=30.0),
        ColumnContract(name="MULL", min_value=-15.0, max_value=15.0),
        ColumnContract(name="LUFL", min_value=-10.0, max_value=15.0),
        ColumnContract(name="LULL", min_value=-5.0, max_value=8.0),
        ColumnContract(name="OT", min_value=-15.0, max_value=60.0),
    )


ETT_CONTRACT = ETTDatasetContract()

# Contrato relajado para datasets de humo/desarrollo (toy dataset, ~1000 filas).
TOY_ETT_CONTRACT = ETTDatasetContract(min_rows=200)

# Columnas que espera el modelo en inferencia: las 7 lecturas de sensores más
# los componentes temporales derivados (sin 'date', ya no es una columna en
# ese punto del pipeline -- ver data_processing.py).
FEATURE_COLUMNS: tuple[str, ...] = (
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT",
    "month",
    "day",
    "hour",
)


def _check_column_ranges(df: pd.DataFrame, columns: tuple[ColumnContract, ...]) -> list[str]:
    violations: list[str] = []
    for col_contract in columns:
        col = col_contract.name
        series = df[col]

        n_nulls = int(series.isnull().sum())
        if n_nulls and not col_contract.allow_null:
            violations.append(f"Columna '{col}' tiene {n_nulls} valores nulos (no permitido).")

        if not pd.api.types.is_numeric_dtype(series):
            violations.append(f"Columna '{col}' no es numérica (dtype={series.dtype}).")
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        out_of_range = (non_null < col_contract.min_value) | (non_null > col_contract.max_value)
        n_out = int(out_of_range.sum())
        if n_out:
            violations.append(
                f"Columna '{col}': {n_out} valor(es) fuera del rango "
                f"[{col_contract.min_value}, {col_contract.max_value}] "
                f"(min observado={non_null.min():.3f}, max observado={non_null.max():.3f})."
            )
    return violations


def validate_ett_dataframe(df: pd.DataFrame, contract: ETTDatasetContract = ETT_CONTRACT) -> None:
    """Valida un dataset ETT crudo (con columna 'date') contra `contract`.

    Lanza `DataContractError` con TODAS las violaciones encontradas (no solo
    la primera) para que el diagnóstico sea accionable en un único intento.
    """
    missing = [c for c in contract.required_columns if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Columnas faltantes en el dataset: {missing}. "
            f"Se requieren: {list(contract.required_columns)}."
        )

    violations: list[str] = []

    if len(df) < contract.min_rows:
        violations.append(
            f"El dataset tiene {len(df)} filas; se requieren al menos {contract.min_rows}."
        )

    try:
        pd.to_datetime(df["date"])
    except (ValueError, TypeError) as exc:
        violations.append(f"La columna 'date' contiene valores no parseables como fecha: {exc}")
    else:
        duplicated_dates = int(df["date"].duplicated().sum())
        if duplicated_dates:
            violations.append(f"Hay {duplicated_dates} timestamps duplicados en 'date'.")

    violations.extend(_check_column_ranges(df, contract.columns))

    if violations:
        report = "\n  - ".join(violations)
        raise DataContractError(
            f"El dataset no cumple el contrato de datos ({len(violations)} violación(es)):\n  - {report}"
        )


def validate_feature_dataframe(
    df: pd.DataFrame, contract: ETTDatasetContract = ETT_CONTRACT
) -> None:
    """Valida un DataFrame ya "feature-engineered" (sin 'date', con
    month/day/hour) como el que consume el modelo en inferencia -- tanto la
    API síncrona como el batch scoring. FAIL FAST antes de invocar al modelo.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Columnas faltantes para inferencia: {missing}. Se requieren: {list(FEATURE_COLUMNS)}."
        )

    violations = _check_column_ranges(df, contract.columns)
    if violations:
        report = "\n  - ".join(violations)
        raise DataContractError(
            f"El batch de inferencia no cumple el contrato de datos "
            f"({len(violations)} violación(es)):\n  - {report}"
        )


def validate_ett_csv(path: str, contract: ETTDatasetContract = ETT_CONTRACT) -> pd.DataFrame:
    """Carga `path` y lo valida contra `contract`. Devuelve el DataFrame ya
    validado, listo para procesar. FAIL FAST: si el archivo no existe o no
    cumple el contrato, lanza antes de gastar cómputo en escalado/entrenamiento.
    """
    import os

    if not os.path.exists(path):
        raise DataContractError(
            f"No existe el archivo de datos '{path}'. " "¿Olvidaste `dvc pull` o `make data-toy`?"
        )
    df = pd.read_csv(path)
    validate_ett_dataframe(df, contract=contract)
    return df
