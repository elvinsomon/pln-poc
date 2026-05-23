"""Línea base trivial — referencia mínima obligatoria (TDT §8.2)."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier


def train_majority(X_train, y_train, seed: int = 42) -> DummyClassifier:
    clf = DummyClassifier(strategy="most_frequent", random_state=seed)
    clf.fit(X_train, y_train)
    return clf
