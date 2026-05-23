"""Pipeline de datos: filtrar → anonimizar → limpiar → muestreo balanceado → splits.

Output: 3 parquets en `data/splits/` con columnas (`text`, `label`).
La clase `documentation` se filtra (decisión iteración 1; ver TDT revisión E3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .anonymize import anonymize
from .clean import clean_text
from .loader import iter_filtered


def _build_text(title: str | float, body: str | float) -> str:
    t = title if isinstance(title, str) else ""
    b = body if isinstance(body, str) else ""
    return f"{t} {b}".strip()


def collect_filtered(raw_path: str | Path, classes: list[str], chunksize: int = 200_000) -> pd.DataFrame:
    """Lee CSV por chunks, filtra clases objetivo, devuelve DataFrame consolidado."""
    parts: list[pd.DataFrame] = []
    for chunk in iter_filtered(raw_path, keep_labels=classes, chunksize=chunksize):
        parts.append(chunk[["id", "labels", "title", "body"]].copy())
    df = pd.concat(parts, ignore_index=True)
    df["labels"] = df["labels"].astype(str)
    return df


def preprocess(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    anon = cfg.get("anonymize", {})
    cln = cfg.get("clean", {})
    max_chars = cfg.get("max_text_chars")
    text = df.apply(lambda r: _build_text(r["title"], r["body"]), axis=1)
    text = text.map(lambda s: anonymize(
        s,
        emails=anon.get("emails", True),
        urls=anon.get("urls", True),
        handles=anon.get("handles", True),
    ))
    text = text.map(lambda s: clean_text(
        s,
        max_chars=max_chars,
        do_code=cln.get("strip_code_blocks", True),
        do_md=cln.get("strip_markdown", True),
        do_ws=cln.get("normalize_whitespace", True),
    ))
    out = pd.DataFrame({"text": text, "label": df["labels"].values})
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def stratified_balanced_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Cap por clase. Si una clase tiene menos, se queda con todas las suyas."""
    parts = []
    for cls, sub in df.groupby("label"):
        n = min(per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_splits(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> dict[str, pd.DataFrame]:
    """Hold-out estratificado test, luego train/val sobre el resto."""
    train_val, test = train_test_split(
        df, test_size=test_fraction, stratify=df["label"], random_state=seed
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed
    )
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def prepare_splits(cfg: dict[str, Any], project_root: str | Path, force: bool = False) -> dict[str, Path]:
    """Pipeline completo. Idempotente: si los parquets existen y `force=False`, no rehace."""
    root = Path(project_root)
    splits_dir = root / cfg["paths"]["splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {name: splits_dir / f"{name}.parquet" for name in ("train", "val", "test")}
    if not force and all(p.exists() for p in out_paths.values()):
        return out_paths

    raw_path = root / cfg["paths"]["raw"]
    df = collect_filtered(raw_path, classes=cfg["classes"])
    df = preprocess(df, cfg)
    df = stratified_balanced_sample(df, per_class=cfg["sample_per_class"], seed=cfg["seed"])
    splits = make_splits(df, val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"], seed=cfg["seed"])
    for name, sub in splits.items():
        sub.to_parquet(out_paths[name], index=False)
    return out_paths


def load_splits(cfg: dict[str, Any], project_root: str | Path) -> dict[str, pd.DataFrame]:
    splits_dir = Path(project_root) / cfg["paths"]["splits"]
    return {name: pd.read_parquet(splits_dir / f"{name}.parquet") for name in ("train", "val", "test")}
