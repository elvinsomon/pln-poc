"""Métricas comunes (TDT §8.2): accuracy, F1-macro, F1 por clase, matriz confusión."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: Iterable, y_pred: Iterable, labels: list[str]) -> dict:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_per_class": dict(zip(
            labels,
            [float(x) for x in f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)],
        )),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "report": classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0),
    }


def save_metrics(metrics: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return p


def plot_confusion(cm: list[list[int]] | np.ndarray, labels: list[str], ax=None, normalize: bool = False):
    """Heatmap. Devuelve `ax`. seaborn/matplotlib se importan aquí para evitar overhead."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    arr = np.asarray(cm, dtype=float)
    if normalize:
        arr = arr / arr.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    return ax
