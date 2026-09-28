"""Descarga ETTh1 desde el repositorio público de ETDataset y genera el toy.

Alternativa local a `dvc pull` cuando el remoto DVC no está disponible (un
clon nuevo con LocalStack recién levantado arranca con el bucket vacío, y
el remoto de producción vive en una cuenta AWS que no hace falta tocar).

Los bytes publicados upstream no son idénticos a la versión trackeada por
DVC (esa se reescribió con pandas: mismo contenido, distinto formateo de
floats), así que `dvc status` los reportará como modificados y train.py
registrará `data_matches_dvc_pointer=false` en MLflow -- es lo esperado.
Para versionar esta copia en tu remoto local: `dvc add` + `dvc push`.

Uso (desde core_ml/):
    poetry run python -m scripts.download_data
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
RAW_PATH = Path("data/raw/ETTh1.csv")
TOY_PATH = Path("data/toy/ETTh1_toy.csv")
TOY_ROWS = 1000


def download(url: str, target: Path, timeout: int = 60) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # URL fija de GitHub (https), no input de usuario.
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
        target.write_bytes(response.read())


def write_toy(raw: Path, toy: Path, n_rows: int = TOY_ROWS) -> None:
    """Cabecera + las primeras `n_rows` filas horarias (orden cronológico).

    Bytes, no texto: en Windows, write_text traduce cada "\\n" a "\\r\\n", y
    el md5 del toy dependería del sistema operativo que lo generó.
    """
    toy.parent.mkdir(parents=True, exist_ok=True)
    lines = raw.read_bytes().splitlines(keepends=True)
    toy.write_bytes(b"".join(lines[: n_rows + 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Sobrescribe archivos existentes.")
    args = parser.parse_args()

    if RAW_PATH.exists() and not args.force:
        print(f"{RAW_PATH} ya existe (usa --force para reemplazarlo).")
    else:
        print(f"Descargando {ETTH1_URL} -> {RAW_PATH}")
        download(ETTH1_URL, RAW_PATH)

    if TOY_PATH.exists() and not args.force:
        print(f"{TOY_PATH} ya existe (usa --force para reemplazarlo).")
    else:
        write_toy(RAW_PATH, TOY_PATH)
        print(f"Generado {TOY_PATH} ({TOY_ROWS} filas).")


if __name__ == "__main__":
    main()
