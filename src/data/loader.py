"""Lectura del CSV NLBSE'23.

El fichero pesa 1.5 GB; en Colab free conviene leer por chunks y filtrar
`documentation` al vuelo para no cargar 54k filas que se descartan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


COLUMNS = ["id", "labels", "title", "body", "author_association"]
DTYPES = {
    "id": "int64",
    "labels": "category",
    "title": "string",
    "body": "string",
    "author_association": "category",
}


def load_raw(path: str | Path, chunksize: int | None = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Si `chunksize` es None, devuelve DataFrame completo. Si no, iterador."""
    return pd.read_csv(
        path,
        usecols=COLUMNS,
        dtype=DTYPES,
        chunksize=chunksize,
        low_memory=False,
    )


def iter_filtered(path: str | Path, keep_labels: list[str], chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Streaming + filtrado por etiqueta. Ahorra RAM en Colab free."""
    for chunk in load_raw(path, chunksize=chunksize):
        yield chunk[chunk["labels"].isin(keep_labels)]
