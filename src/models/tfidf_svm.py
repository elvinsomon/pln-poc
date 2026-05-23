"""TF-IDF + LinearSVC — línea base vectorial (TDT §7.2 Candidato A).

Pipeline interpretable: los coeficientes del SVM lineal permiten extraer
los tokens más influyentes por clase (gancho para análisis de errores).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    v = cfg["vectorizer"]
    c = cfg["classifier"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(v["ngram_range"]),
            min_df=v["min_df"],
            max_df=v["max_df"],
            max_features=v["max_features"],
            sublinear_tf=v["sublinear_tf"],
            lowercase=v.get("lowercase", True),
        )),
        ("svm", LinearSVC(
            C=c["C"],
            loss=c.get("loss", "squared_hinge"),
            class_weight=c.get("class_weight", "balanced"),
            max_iter=c.get("max_iter", 5000),
            random_state=cfg.get("seed", 42),
        )),
    ])


def extract_top_features(pipeline: Pipeline, k: int = 20) -> pd.DataFrame:
    """Tabla (clase, rank, token, peso) con los k tokens de mayor `coef_` por clase.

    Para 3 clases con `LinearSVC`, sklearn usa One-vs-Rest: `coef_` tiene shape
    (n_classes, n_features); para 2 clases, shape (1, n_features).
    """
    vec: TfidfVectorizer = pipeline.named_steps["tfidf"]
    svm: LinearSVC = pipeline.named_steps["svm"]
    vocab = np.array(vec.get_feature_names_out())
    classes = list(svm.classes_)
    coef = svm.coef_
    rows: list[dict[str, Any]] = []
    if coef.shape[0] == 1:                                # binario → tratar como ±clases
        idx = np.argsort(coef[0])[-k:][::-1]
        for rank, i in enumerate(idx, start=1):
            rows.append({"class": classes[1], "rank": rank, "token": vocab[i], "weight": float(coef[0, i])})
        idx = np.argsort(coef[0])[:k]
        for rank, i in enumerate(idx, start=1):
            rows.append({"class": classes[0], "rank": rank, "token": vocab[i], "weight": float(coef[0, i])})
    else:
        for ci, cls in enumerate(classes):
            idx = np.argsort(coef[ci])[-k:][::-1]
            for rank, i in enumerate(idx, start=1):
                rows.append({"class": cls, "rank": rank, "token": vocab[i], "weight": float(coef[ci, i])})
    return pd.DataFrame(rows)
