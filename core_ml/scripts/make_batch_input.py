"""Genera el CSV de entrada del batch scoring a partir de un CSV ETT crudo.

Toma las últimas `--rows` lecturas horarias, deriva month/day/hour (igual
que data_processing.py) y escribe en `--output` las columnas FEATURE_COLUMNS
que espera src.batch_inference. Pensado para `make batch-local`, que sube el
resultado al bucket de LocalStack.

Uso (desde core_ml/):
    poetry run python -m scripts.make_batch_input --rows 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_contracts import FEATURE_COLUMNS


def build_batch_input(raw_csv: str, rows: int) -> pd.DataFrame:
    df = pd.read_csv(raw_csv).tail(rows)
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["month"], df["day"], df["hour"] = dates.dt.month, dates.dt.day, dates.dt.hour
    return df[list(FEATURE_COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_csv", default="data/toy/ETTh1_toy.csv")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--output", default="artifacts_batch/input_data.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_batch_input(args.raw_csv, args.rows).to_csv(output, index=False, lineterminator="\n")
    print(f"Batch de entrada escrito en {output}")


if __name__ == "__main__":
    main()
