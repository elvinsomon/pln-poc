"""DistilBERT fine-tuning — sistema objetivo vectorial-contextual (TDT §7.2 Candidato B).

Wrapper sklearn-like sobre HuggingFace para que el código de la notebook 04 sea
un calco de 03 (`build_model` ↔ `build_pipeline`; `.fit/.predict/.predict_proba`).

`.predict` devuelve SIEMPRE etiquetas-string en el orden de `cfg['classes']`, de
modo que `compute_metrics(labels=cfg['classes'])` alinea `bert.json` con
`majority.json` / `tfidf_svm.json`. El mapeo etiqueta↔id se deriva una sola vez de
`cfg['classes']` y se hornea en la config del modelo + `label_map.json`.

Las libs pesadas (torch/transformers/datasets) se importan dentro de los métodos:
así este módulo se importa sin error en entornos sin GPU (p. ej. notebook 06 en local),
y solo exige el stack de deep learning cuando realmente se entrena o infiere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def detect_device() -> str:
    """Mejor device disponible: cuda (Colab) > mps (Mac) > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BertClassifier:
    """Clasificador de secuencias por fine-tuning, con interfaz tipo sklearn."""

    LABELMAP_FILE = "label_map.json"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.classes: list[str] = list(cfg["classes"])
        self.label2id: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.id2label: dict[int, str] = {i: c for c, i in self.label2id.items()}
        b = cfg["bert"]
        self.model_name: str = b["model_name"]
        self.max_length: int = b["max_length"]
        self.batch_size: int = b["batch_size"]
        self.seed: int = cfg.get("seed", 42)
        self._device: str | None = b.get("device")   # None => autodetect perezoso (no importa torch en __init__)
        self.tokenizer = None   # lazy hasta fit/load
        self.model = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = detect_device()
        return self._device

    # ------------------------------------------------------------------ helpers
    def _training_args(self, output_dir: str):
        from transformers import TrainingArguments
        b = self.cfg["bert"]
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=b["epochs"],
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=float(b["learning_rate"]),
            weight_decay=b.get("weight_decay", 0.0),
            warmup_ratio=b.get("warmup_ratio", 0.0),
            fp16=(self.device == "cuda" and b.get("fp16", True)),
            eval_strategy="no",      # la eval la hace la notebook vía compute_metrics (como 03)
            save_strategy="no",      # guardamos una sola vez con .save(); no llenar disco de Colab
            logging_steps=50,
            report_to="none",        # sin W&B/TensorBoard en Colab
            seed=self.seed,
            dataloader_num_workers=2,
        )

    # ---------------------------------------------------------------- sklearn-ish
    def fit(self, texts, labels) -> "BertClassifier":
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            set_seed,
        )
        from datasets import Dataset

        set_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.classes),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        y = [self.label2id[str(v)] for v in list(labels)]
        ds = Dataset.from_dict({"text": [str(t) for t in list(texts)], "labels": y})
        ds = ds.map(
            lambda batch: self.tokenizer(batch["text"], truncation=True, max_length=self.max_length),
            batched=True,
            remove_columns=["text"],
        )

        trainer = Trainer(
            model=self.model,
            args=self._training_args(output_dir="/content/_bert_trainer"),
            train_dataset=ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )
        trainer.train()
        self.model.to(self.device).eval()
        return self

    def predict_proba(self, texts) -> np.ndarray:
        """Softmax (n, n_classes); columnas en orden de `self.classes`."""
        import torch

        texts = [str(t) for t in list(texts)]
        self.model.to(self.device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, texts) -> np.ndarray:
        ids = self.predict_proba(texts).argmax(axis=1)
        return np.array([self.id2label[int(i)] for i in ids])

    # ----------------------------------------------------------------- persistencia
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        (d / self.LABELMAP_FILE).write_text(
            json.dumps({"classes": self.classes, "label2id": self.label2id},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return d

    @classmethod
    def load(cls, directory: str | Path, cfg: dict[str, Any] | None = None) -> "BertClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = Path(directory)
        lm = json.loads((d / cls.LABELMAP_FILE).read_text(encoding="utf-8"))
        if cfg is None:
            cfg = {"classes": lm["classes"],
                   "bert": {"model_name": str(d), "max_length": 256, "batch_size": 64}}
        cfg.setdefault("classes", lm["classes"])
        obj = cls(cfg)
        obj.classes = lm["classes"]
        obj.label2id = lm["label2id"]
        obj.id2label = {int(i): c for c, i in obj.label2id.items()}
        obj.tokenizer = AutoTokenizer.from_pretrained(d)
        obj.model = AutoModelForSequenceClassification.from_pretrained(d).to(obj.device).eval()
        return obj


def build_model(cfg: dict[str, Any]) -> BertClassifier:
    """Factory que espeja `build_pipeline(cfg)` de tfidf_svm.py."""
    return BertClassifier(cfg)
