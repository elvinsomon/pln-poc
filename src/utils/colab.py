"""Bootstrap unificado Colab/local.

Modelo de despliegue:
- El **código** vive en GitHub. Cada miembro hace `git clone` en su entorno
  (Colab `/content/pln-poc` o local).
- El **dataset** (1.5 GB) vive en Google Drive en una carpeta compartida con
  nombre fijo `MULCIA-PLN`. Cada miembro añade "Añadir acceso directo a Mi
  unidad" sobre esa carpeta — así la ruta del CSV (`cfg['drive']['csv_path']`)
  es la misma para todos y la config va versionada.
- `bootstrap_dataset()` monta Drive si hace falta, copia el CSV a un cache
  local de Colab (acelera I/O 10-20×) y lo enlaza al path local
  (`cfg['paths']['raw']`) que esperan los demás módulos.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .seeds import set_seed


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


def link_dataset(source: str | Path, target: str | Path, *, overwrite: bool = False) -> Path:
    """Symlink `target → source`. Idempotente.

    Si el symlink existe pero apunta a otra fuente: error explícito, salvo
    que `overwrite=True`. Evita pisar sin avisar.
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

    Lecturas posteriores van por disco local (~200 MB/s) en vez de Drive
    (~5-15 MB/s). El cache se pierde al reciclar la VM; la copia es idempotente.
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
    """Pone el CSV en el path que esperan los configs (`cfg['paths']['raw']`).

    - **Local:** no toca nada; espera que el CSV ya esté en `data/raw/`.
    - **Colab:** monta Drive, lee `cfg['drive']`, opcionalmente copia a cache
      local para acelerar I/O, y symlinka al path esperado.

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

    drive_cfg = cfg.get("drive", {})
    drive_csv = drive_cfg.get("csv_path")
    if not drive_csv:
        raise KeyError("Falta `drive.csv_path` en el config base. Ver configs/base.yaml.")

    mount_drive()

    drive_path = Path(drive_csv)
    if not drive_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV en {drive_csv}.\n"
            f"Verifica que has añadido la carpeta compartida `MULCIA-PLN` a tu Mi unidad "
            f"(Drive → Compartido conmigo → click derecho → Añadir acceso directo a Mi unidad)."
        )

    if drive_cfg.get("copy_to_local", False):
        actual = cache_to_local(drive_path, drive_cfg.get("local_cache_dir", "/content/_dataset_cache"))
    else:
        actual = drive_path.resolve()

    link_dataset(actual, target, overwrite=True)
    if verbose:
        print(f"Dataset enlazado: {target} → {actual}")
    return target.resolve()


def setup_environment(seed: int = 42, project_root: str | Path | None = None) -> Path:
    """Fija seed, registra `project_root` en sys.path y lo devuelve resuelto."""
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
