# PoC C2 — Categorizador Temático de Tickets de Soporte

Prueba de Concepto para la **Entrega 3** del Proyecto Longitudinal de Diseño de Sistemas PLN (MULCIA, curso 2025–26).

Implementa el componente **C2 (categorización temática)** del sistema de triaje automático descrito en el TDT, sobre el dataset **NLBSE'23** (3 clases: `bug` / `feature` / `question`).

> **Estado actual — iteración 2:** EDA + majority baseline + TF-IDF+SVM (interpretabilidad) + **DistilBERT fine-tuning** + comparativa de modelos + análisis de errores. Cubre los tres criterios de éxito del E2 §7.2.

**Equipo:** Izaskun Peña Arranz · Miguel Ángel Rodríguez Ortega · Elvin Somón Sánchez.

---

## 1. Modelo de despliegue

Separación entre **código** y **datos**:

| Recurso     | Dónde vive                                       | Cómo se accede                                                |
| ----------- | ------------------------------------------------ | -------------------------------------------------------------- |
| **Código**  | GitHub: `https://github.com/elvinsomon/pln-poc`  | `git clone` en local o Colab (`/content/pln-poc`)              |
| **Dataset** | Google Drive (carpeta compartida, 1.5 GB)        | Drive mount + symlink al path `data/raw/` que esperan configs  |
| **Salidas** | Locales al entorno (splits, modelos, métricas)   | Excluidas de git via `.gitignore`; se regeneran reproducibles  |

Drive **no aloja el proyecto**, solo el CSV. Esto evita ramas distintas por entorno y simplifica el control de versiones.

---

## 2. Estructura del repositorio

```
PoC-C2/
├── configs/                  # Hiperparámetros por experimento (YAML con `extends:`)
│   ├── base.yaml             # seed, clases, paths relativos, ruta Drive
│   ├── data.yaml             # muestreo balanceado, anonimización, limpieza
│   ├── tfidf_svm.yaml        # TF-IDF + LinearSVC
│   └── bert.yaml             # DistilBERT fine-tuning (extends base.yaml)
├── data/                     # (.gitignore — se regenera)
│   ├── raw/                  # CSV NLBSE'23 (symlink desde Drive en Colab)
│   ├── processed/            # cachés del EDA (eda_stats.pkl, eda_sample_50k.parquet)
│   └── splits/               # train/val/test.parquet
├── models/                   # (.gitignore) checkpoints joblib
├── notebooks/
│   ├── 00_bootstrap_test.ipynb     # smoke test entorno (1ª vez por persona)
│   ├── 01_eda.ipynb                # EDA del corpus + decisiones de preprocesado
│   ├── 02_baseline_majority.ipynb  # DummyClassifier — referencia trivial
│   ├── 03_tfidf_svm.ipynb          # baseline vectorial + interpretabilidad
│   ├── 04_bert_finetune.ipynb      # DistilBERT fine-tuning (requiere GPU/Colab)
│   ├── 05_evaluation_compare.ipynb # tabla + barras: majority vs SVM vs BERT
│   └── 06_error_analysis.ipynb     # FP/FN + fenómenos lingüísticos (E1)
├── reports/
│   └── metrics/              # JSONs por experimento (output reproducible)
├── src/
│   ├── utils/
│   │   ├── colab.py          # bootstrap Colab/local, mount Drive, link_dataset
│   │   ├── config.py         # loader YAML con herencia `extends:`
│   │   └── seeds.py          # set_seed (numpy + random + PYTHONHASHSEED)
│   ├── data/
│   │   ├── loader.py         # lectura CSV (chunked, dtypes)
│   │   ├── clean.py          # strip code/markdown/whitespace, truncate
│   │   ├── anonymize.py      # emails/URLs/handles → <EMAIL>/<URL>/<USER>
│   │   └── splits.py         # filtrar documentation, balancear, train/val/test
│   ├── models/
│   │   ├── majority.py       # DummyClassifier wrapper
│   │   ├── tfidf_svm.py      # build_pipeline + extract_top_features
│   │   └── bert.py           # BertClassifier (fit/predict) + build_model
│   └── evaluation/
│       └── metrics.py        # accuracy, F1-macro, matriz confusión, persistencia JSON
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Puesta en marcha

### 3.1 Colab (recomendado para el equipo)

#### Paso 1 — acceso al Drive compartido

Convención de equipo: la carpeta compartida se llama **`MULCIA-PLN`** y todos la añaden a su Mi unidad con ese mismo nombre. Así la ruta del CSV es idéntica para todos y la config va versionada (`configs/base.yaml` § `drive`).

1. Pide al owner que comparta contigo la carpeta `MULCIA-PLN` de Drive.
2. En Drive: **Compartido conmigo** → click derecho sobre `MULCIA-PLN` → **Añadir acceso directo a Mi unidad** → confirmar en la raíz de Mi unidad.
3. Tras este paso, la ruta `/content/drive/MyDrive/MULCIA-PLN/data/nlbse23-issue-classification-train.csv` debe existir cuando montes Drive.

> **Importante:** si renombras el acceso directo o lo colocas en una subcarpeta, la ruta dejará de coincidir con `configs/base.yaml::drive.csv_path` y el bootstrap fallará. Si necesitas otra ruta, cambia el config en el repo (no individualmente).

#### Paso 2 — ejecutar notebooks

1. Abre cualquier notebook de `notebooks/` desde GitHub con *Open in Colab* (o `https://colab.research.google.com/github/elvinsomon/pln-poc/blob/main/notebooks/00_bootstrap_test.ipynb`).
2. La primera celda de **cada notebook** ejecuta automáticamente:
   - `git clone https://github.com/elvinsomon/pln-poc` en `/content/pln-poc`.
   - `pip install -r requirements.txt`.
   - `bootstrap_dataset(cfg)` → monta Drive, copia el CSV a `/content/_dataset_cache/` (si `drive.copy_to_local: true`), y symlinka al path esperado por los configs.
3. Empieza por `00_bootstrap_test.ipynb`. Si pasa sin errores, los demás funcionan.

> **Por qué `copy_to_local: true`:** lectura por symlink desde Drive es 10-20× más lenta que disco local de Colab (≈ 5-15 MB/s vs 200 MB/s). La copia inicial tarda ~2-3 min y se hace una sola vez por sesión de VM. Recomendado para el EDA, que streamea 1.5 GB.

### 3.2 Local

```bash
git clone https://github.com/elvinsomon/pln-poc.git
cd pln-poc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Coloca el CSV manualmente (o crea un symlink) en:
#   data/raw/nlbse23-issue-classification-train.csv

# `configs/local.yaml` no es necesario en local
# (bootstrap_dataset detecta que no estás en Colab y omite Drive).

jupyter lab
```

Los notebooks detectan ausencia de Colab y saltan el bloque de clone/mount/symlink.

### 3.3 Orden de ejecución (iteración 2)

| Orden | Notebook                          | Hace                                                  | Tiempo aprox. |
| ----- | --------------------------------- | ----------------------------------------------------- | ------------- |
| 1     | `00_bootstrap_test.ipynb`         | Verifica entorno, imports, configs, dataset           | <1 min        |
| 2     | `01_eda.ipynb`                    | EDA del corpus (clases, longitudes, fenómenos PLN)    | 5–15 min      |
| 3     | `02_baseline_majority.ipynb`      | DummyClassifier → establece el suelo                  | 1–2 min       |
| 4     | `03_tfidf_svm.ipynb`              | TF-IDF + LinearSVC + top tokens + persistencia        | 2–5 min       |
| 5     | `04_bert_finetune.ipynb`          | DistilBERT fine-tuning + métricas + predicciones      | 6–8 min (T4)  |
| 6     | `05_evaluation_compare.ipynb`     | Tabla + barras + margen BERT vs SVM (criterio E2 §2)  | <1 min        |
| 7     | `06_error_analysis.ipynb`         | FP/FN + fenómenos lingüísticos E1 (criterio E2 §3)    | 1–2 min       |

> **GPU obligatoria para `04`.** Ábrelo en Colab con runtime **T4**; la celda de bootstrap imprime el `device` (debe ser `cuda`). `05` y `06` **no** necesitan GPU.
>
> **Dependencia de artefactos:** `05` requiere que `04` haya generado `reports/metrics/bert.json`; `06` requiere `models/tfidf_svm.joblib` (de `03`) y `reports/preds_test_bert.parquet` (de `04`). Como `models/` y los splits están en `.gitignore`, tras un `git clone` fresco hay que correr `03` y `04` antes que `06`. El parquet de predicciones vive en `reports/` (versionable), de modo que `06` no recarga el modelo en GPU.

`prepare_splits()` es **idempotente**: la primera ejecución lee el CSV (1.5 GB), filtra `documentation`, aplica anonimización + limpieza, muestrea 10k/clase y persiste 3 parquets. Las siguientes ejecuciones cargan los parquets existentes sin rehacer trabajo.

---

## 4. Decisiones de diseño

- **3 clases (sin `documentation`).** El TDT E2 §8.1 fija el problema en `bug`/`feature`/`question`. El CSV real incluye una cuarta clase (`documentation`, 4.3 %) que filtramos. Decisión documentada como revisión respecto a E2 en el informe final.
- **Muestreo balanceado** (10k/clase = 30k train) en lugar del parquet desbalanceado original. Reduce el sesgo hacia clases mayoritarias y permite leer interpretaciones del SVM sin que dominen los términos de `bug`.
- **Anonimización mínima** (C0 del sistema: emails, URLs, handles → tokens placeholder) como paso previo obligatorio para alinearse con el requisito RGPD del TDT §4.
- **Test set como hold-out estratificado** del CSV `train` (10 %) en lugar del split test oficial del paper. Razón: el hold-out interno garantiza arrancar la PoC sin esperar a la descarga del archivo separado de test.

---

## 5. Reproducibilidad

- Seed fija (`42`) en `configs/base.yaml`, aplicada vía `setup_environment()`.
- Versiones de librerías pinned en `requirements.txt`.
- Splits persistidos en `data/splits/` tras la primera generación: cualquier miembro que regenere obtiene los mismos datos.
- Métricas y modelos persistidos en `reports/metrics/` y `models/`.

---

## 6. Convenciones de equipo

- **Limpiar outputs antes de hacer commit** (`Kernel → Restart & Clear Output` o `nbstripout`). Notebooks con outputs voluminosos rompen el diff visual y crecen el repo.
- **No editar la misma notebook simultáneamente.** Coordinar por canal antes de tocar `01_eda` o `03_tfidf_svm`.
- **Toda métrica se guarda** en `reports/metrics/<experimento>.json` vía `save_metrics()`. Centraliza la comparación final.
- **No comitear** `data/raw/`, `data/processed/`, `data/splits/`, `models/`: ya excluidos en `.gitignore`. El CSV se sube manualmente a la carpeta de Drive compartida.

---

## 7. Próximas iteraciones (fuera de scope aquí)

- Informe E3 (3–4 pp) en `reports/`: redacción final apoyada en la tabla comparativa (`reports/metrics/comparison.csv`) y la lectura lingüística de errores (`reports/error_analysis_sampled.csv`).
- (Opcional) Grid/ajuste de hiperparámetros de DistilBERT; probar checkpoints alternativos (RoBERTa, CodeBERT) si el margen sobre el SVM se queda corto.

---

## 8. Solución de problemas frecuentes

| Síntoma                                                       | Causa probable                                                      | Solución                                                              |
| -------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `FileNotFoundError: ...MULCIA-PLN/...csv`                      | El acceso directo a Drive no existe o tiene otro nombre              | Repite el paso 1 de §3.1: el shortcut debe llamarse exactamente `MULCIA-PLN` en la raíz de Mi unidad. |
| `ModuleNotFoundError: src.utils.config`                        | `PROJECT_ROOT` no está en `sys.path`                                | Vuelve a ejecutar la celda 1 (bootstrap) sin saltarla.                |
| `pyarrow` u otras libs ausentes en Colab tras *Restart runtime* | Colab limpia el venv al reiniciar                                  | Re-ejecuta la celda 1; reinstala con `pip install -r requirements.txt`. |
| `prepare_splits` reusa parquets viejos tras cambiar el config  | Idempotencia por defecto                                            | Llama `prepare_splits(cfg, project_root, force=True)`.                |
