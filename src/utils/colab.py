"""Bootstrap unificado Colab/local.

Modelo de despliegue:
- El **código** vive en GitHub. Cada miembro hace `git clone` en su entorno
  (Colab `/content/pln-poc` o local).
- El **dataset** (1.5 GB) vive en Google Drive, en una carpeta compartida.
  Cada miembro indica su ruta personal en `configs/local.yaml` (gitignored;
  plantilla en `configs/local.example.yaml`).
- `bootstrap_dataset()` se encarga de leer esa config, montar Drive si hace
  falta, copiar el CSV a un cache local de Colab (acelera I/O 10-20×) y
  enlazarlo al path que esperan los configs principales (`data/raw/...`).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from .seeds import set_seed


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> str:
    """Monta Drive en Colab. No-op fuera de Colab."""
    if not in_colab():
        return ""
    from google.colab import drive
    drive.mount(mount_point)
    return mount_point


def load_local_config() -> dict[str, Any]:
    """Lee `configs/local.yaml`. Falla con instrucciones si no existe."""
    path = CONFIGS_DIR / "local.yaml"
    if not path.exists():
        example = CONFIGS_DIR / "local.example.yaml"
        raise FileNotFoundError(
            f"Falta {path}.\n"
            f"Copia la plantilla y ajusta `dataset_on_drive` a tu ruta de Drive:\n"
            f"    cp {example} {path}\n"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def link_dataset(source: str | Path, target: str | Path, *, overwrite: bool = False) -> Path:
    """Symlink `target → source`. Idempotente.

    Si el symlink existe pero apunta a otra fuente: error explícito, salvo
    que `overwrite=True`. Esto evita pisar sin avisar a otro miembro.
    """
    src = Path(source).expanduser().resolve()
    tgt = Path(target).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"No existe el dataset en origen: {src}")
    tgt.parent.mkdir(parents=True, exist_ok=True)
    if tgt.is_symlink() or tgt.exists():
        try:
            already_ok = tgt.resolve() == src
        except OSError:
            already_ok = False
        if already_ok:
            return tgt
        if not overwrite:
            raise FileExistsError(
                f"{tgt} ya existe y apunta a {tgt.resolve()}. "
                f"Pasa overwrite=True si quieres reemplazarlo por {src}."
            )
        tgt.unlink()
    os.symlink(src, tgt)
    return tgt


def cache_to_local(source: str | Path, cache_dir: str | Path) -> Path:
    """Copia el CSV de Drive al disco local de Colab la primera vez.

    Lecturas posteriores van por disco local (~200 MB/s) en vez de por
    Drive (~5-15 MB/s). El cache se pierde al reciclar la VM, lo cual no
    es problema porque la copia es idempotente.
    """
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"No existe el dataset en origen: {src}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    print(f"Copiando {src.name} ({src.stat().st_size / 1e9:.2f} GB) a {cache_dir} …")
    shutil.copy2(src, dst)
    return dst


def bootstrap_dataset(cfg: dict[str, Any], *, verbose: bool = True) -> Path:
    """Pone el CSV al path que los configs esperan (`cfg['paths']['raw']`).

    - **Local:** no toca nada; asume que ya está en `data/raw/`.
    - **Colab:** monta Drive, lee `configs/local.yaml`, opcionalmente copia
      el CSV a `local_cache_dir` para acelerar I/O, y symlinka al path
      esperado.

    Devuelve la ruta resoluta del CSV.
    """
    target = Path(cfg["paths"]["raw"])
    if not in_colab():
        if not target.exists():
            raise FileNotFoundError(
                f"{target} no existe. En local debes colocar el CSV manualmente "
                f"(o symlinkarlo) en `data/raw/`."
            )
        return target.resolve()

    mount_drive()
    local = load_local_config()
    drive_csv = local["dataset_on_drive"]
    if local.get("copy_to_local", False):
        cache_dir = local.get("local_cache_dir", "/content/_dataset_cache")
        actual = cache_to_local(drive_csv, cache_dir)
    else:
        actual = Path(drive_csv).expanduser().resolve()

    link_dataset(actual, target, overwrite=True)
    if verbose:
        print(f"Dataset enlazado: {target} → {actual}")
    return target.resolve()


def setup_environment(seed: int = 42, project_root: str | Path | None = None) -> Path:
    """Fija seed, registra `project_root` en sys.path y lo devuelve resuelto.

    Llamar tras `os.chdir(project_root)` (Colab) o desde la raíz del repo
    (local).
    """
    set_seed(seed)
    root = Path(project_root) if project_root else Path.cwd()
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_paths(cfg: dict[str, Any], project_root: Path | str) -> dict[str, Path]:
    """Convierte rutas relativas del config a absolutas bajo `project_root`."""
    root = Path(project_root)
    return {key: root / rel for key, rel in cfg.get("paths", {}).items()}
