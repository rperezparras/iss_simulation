#!/usr/bin/env python3
"""
Pipeline completa timelapse ISS (pipelinentl) con recuperación robusta.

Pasos base:
1) Descargar imágenes originales ISS.
2) Opcional: buscar yaw/pitch/roll con angle_search.
3) Generar timelapse simulado en Blender.
4) Matching entre simuladas y reales.
5) Proyección de píxeles -> .points.
6) Filtrado + renombrado de puntos.
7) Primera georreferenciación.

Si use_optical_flow = True:
8) Recorte/alineado VIIRS.
9) Flujo óptico ISS-VIIRS.
10) Corrección de puntos.
11) Segunda georreferenciación completa, de muestra, o ninguna.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import json
import math
import re
from pathlib import Path
from datetime import timedelta
import argparse
from argparse import Namespace
import subprocess

# Imports pesados (cv2, numpy, Blender/bpy via generate_timelapse y
# angle_search) se cargan dentro de main(), después de parsear argumentos.
# Así `python -m pipelinentl.timelapse_pipeline --help` no necesita inicializar
# OpenCV, NumPy ni Blender.
cv2 = None
np = None

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent

# Raiz por defecto para datos externos grandes. Puede sobrescribirse con
# ISS_SIMULATION_DATA_ROOT o con el argumento --data-root.
DATA_ROOT = Path(
    os.environ.get("ISS_SIMULATION_DATA_ROOT", PACKAGE_DIR.parents[1])
).expanduser().resolve()


def required_path(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {label}: {path}")
    return path

# ============================================================
# HELPERS GENERALES
# ============================================================

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def configs_equal(a, b) -> bool:
    try:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    except Exception:
        return False


def nonempty_files(folder: Path, pattern: str):
    if not folder.exists():
        return []
    return [
        p for p in folder.glob(pattern)
        if p.is_file() and p.stat().st_size > 0
    ]


def count_nonempty(folder: Path, pattern: str) -> int:
    return len(nonempty_files(folder, pattern))


def is_complete(folder: Path, pattern: str, expected: int, label: str) -> bool:
    """
    Considera completo un paso si hay al menos expected archivos no vacíos.

    Si hay más de los esperados, no se detiene la pipeline: se avisa y continúa.
    """
    files = nonempty_files(folder, pattern)
    n = len(files)

    if n == expected:
        print(f"   OK {label}: completo ({n}/{expected}).")
        return True

    if n > expected:
        print(
            f"   AVISO {label}: hay más archivos de los esperados "
            f"({n}/{expected}). Se continuará usando los necesarios."
        )
        return True

    if n == 0:
        print(f"   INFO {label}: no existe aún (0/{expected}).")
    else:
        print(f"   AVISO {label}: incompleto ({n}/{expected}).")

    return False


def remove_files(folder: Path, pattern: str, label: str):
    if not folder.exists():
        return

    files = list(folder.glob(pattern))
    if not files:
        return

    print(f"   Limpiando {len(files)} archivos de {label}...")
    for p in files:
        try:
            p.unlink()
        except Exception as e:
            print(f"      AVISO: no se pudo borrar {p}: {e}")


def step_should_run(
    folder: Path,
    pattern: str,
    expected: int,
    label: str,
    config_file: Path | None = None,
    current_config: dict | None = None,
    force_downstream: bool = False,
    force_this_step: bool = False,
) -> bool:
    """
    Decide si un paso debe ejecutarse.

    Ejecuta si:
    - force_this_step es True;
    - force_downstream es True;
    - no hay suficientes archivos;
    - la configuración guardada no coincide con la actual.

    Si los outputs están completos pero falta el JSON de configuración,
    se asume válido y se escribe la configuración actual.
    """
    if force_this_step:
        print(f"   AVISO {label}: forzado por configuración.")
        return True

    if force_downstream:
        print(f"   AVISO {label}: se recalculará porque un paso anterior cambió.")
        return True

    complete = is_complete(folder, pattern, expected, label)
    if not complete:
        return True

    if config_file is not None and current_config is not None:
        old_config = read_json(config_file)

        if old_config is None:
            print(
                f"   AVISO {label}: outputs completos pero falta el archivo de configuración. "
                "Se asumirá válido y se escribirá la configuración actual."
            )
            write_json(config_file, current_config)
            return False

        if not configs_equal(old_config, current_config):
            print(f"   AVISO {label}: configuración distinta. Se recalculará.")
            return True

    return False


def extract_id_from_filename(name: str) -> int | None:
    """
    Extrae el ID numérico de nombres como:

      ISS067-E-327041.points
      ISS067-E-327041_real.points
      ISS067-E-327041_corrected.points
      ISS067-E-327041_real_corrected.points

    Devuelve 327041.
    """
    m = re.search(r"ISS\d+-E-(\d+)", name)
    if m:
        return int(m.group(1))

    # Fallback: buscar un número largo al final o antes de sufijos.
    m = re.search(r"(\d{5,8})", name)
    if m:
        return int(m.group(1))

    return None


def ids_from_points_dir(folder: Path, start_id: int, end_id: int):
    if not folder.exists():
        return []

    ids = []
    for f in folder.glob("*.points"):
        sid = extract_id_from_filename(f.name)
        if sid is not None and start_id <= sid <= end_id and f.stat().st_size > 0:
            ids.append(sid)

    return sorted(set(ids))


def count_points_in_range(folder: Path, start_id: int, end_id: int) -> int:
    return len(ids_from_points_dir(folder, start_id, end_id))


def remove_points_in_range(folder: Path, start_id: int, end_id: int, label: str):
    if not folder.exists():
        return

    files = []
    for f in folder.glob("*.points"):
        sid = extract_id_from_filename(f.name)
        if sid is not None and start_id <= sid <= end_id:
            files.append(f)

    if files:
        print(f"   Limpiando {len(files)} archivos de {label}...")
        for p in files:
            try:
                p.unlink()
            except Exception as e:
                print(f"      AVISO: no se pudo borrar {p}: {e}")


def sample_ids_from_available(available_ids, n_samples: int = 10):
    available_ids = sorted(set(available_ids))
    if not available_ids:
        return []

    if len(available_ids) <= n_samples:
        return available_ids

    idxs = [
        round(i * (len(available_ids) - 1) / (n_samples - 1))
        for i in range(n_samples)
    ]
    return [available_ids[i] for i in idxs]


def geo_file_for_id_exists(folder: Path, sid: int) -> bool:
    if not folder.exists():
        return False
    matches = list(folder.glob(f"*{sid}*_rect.tiff"))
    return any(p.is_file() and p.stat().st_size > 0 for p in matches)



# ============================================================
# HELPERS PARA ELEGIR FRAME DE REFERENCIA DE ANGLE_SEARCH
# ============================================================

def _unique_candidate_indices(n_images: int, fractions) -> list[int]:
    """
    Devuelve índices únicos dentro del timelapse a partir de fracciones.

    Ejemplo con las fracciones por defecto:
      0.0 -> primera imagen
      1/3 -> imagen aproximadamente al primer tercio
      0.5 -> imagen central
      2/3 -> imagen aproximadamente al segundo tercio
      1.0 -> última imagen
    """
    if n_images <= 0:
        return []

    indices = []
    for frac in fractions:
        frac = max(0.0, min(1.0, float(frac)))
        idx = int(round(frac * (n_images - 1)))
        idx = max(0, min(n_images - 1, idx))
        indices.append(idx)

    out = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)

    return out


def _score_reference_image_for_angle_search(
    image_path: Path,
    coverage_bins: int = 6,
    work_max_side: int = 1600,
) -> dict:
    """
    Puntúa una imagen real para decidir si es buena referencia para angle_search.

    Es una evaluación barata: no renderiza Blender ni hace matching contra una
    simulación. Solo mira si la imagen real tiene señal suficiente y estructura
    espacial para que luego el matching del angle_search tenga más probabilidad
    de funcionar.
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {
            "score": -1.0e9,
            "n_keypoints": 0,
            "coverage": 0.0,
            "contrast": 0.0,
            "texture": 0.0,
            "non_dark_fraction": 0.0,
        }

    h, w = img_bgr.shape[:2]
    max_side = max(h, w)

    if max_side > work_max_side:
        scale = float(work_max_side) / float(max_side)
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Señal luminosa mínima. Penaliza frames casi negros o con muy poca tierra visible.
    non_dark_mask = gray > 10
    non_dark_fraction = float(np.mean(non_dark_mask))

    # Contraste global y textura local.
    contrast = float(np.std(gray))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = float(np.var(lap))

    # Keypoints ORB: proxy barato de si habrá estructura útil para matching.
    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=7)
    keypoints = orb.detect(gray, None)
    n_keypoints = len(keypoints)

    if n_keypoints > 0:
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        xs = np.clip((pts[:, 0] / max(w, 1) * coverage_bins).astype(int), 0, coverage_bins - 1)
        ys = np.clip((pts[:, 1] / max(h, 1) * coverage_bins).astype(int), 0, coverage_bins - 1)
        occupied = {(int(x), int(y)) for x, y in zip(xs, ys)}
        coverage = float(len(occupied)) / float(coverage_bins * coverage_bins)
    else:
        coverage = 0.0

    keypoint_score = float(np.tanh(n_keypoints / 1000.0))
    contrast_score = float(np.clip(contrast / 64.0, 0.0, 1.0))
    texture_score = float(np.clip(np.log1p(texture) / np.log1p(2000.0), 0.0, 1.0))
    non_dark_score = float(np.clip(non_dark_fraction / 0.35, 0.0, 1.0))

    penalty = 0.0
    if non_dark_fraction < 0.01:
        penalty += 1.0
    if n_keypoints < 100:
        penalty += 1.0
    if coverage < 0.10:
        penalty += 0.5

    score = (
        2.00 * coverage
        + 1.50 * keypoint_score
        + 0.50 * contrast_score
        + 0.50 * texture_score
        + 0.35 * non_dark_score
        - penalty
    )

    return {
        "score": float(score),
        "n_keypoints": int(n_keypoints),
        "coverage": float(coverage),
        "contrast": float(contrast),
        "texture": float(texture),
        "non_dark_fraction": float(non_dark_fraction),
    }


def select_angle_search_reference_frame(
    pics_dir: Path,
    image_files,
    start_dt,
    delta: float,
    reference_fractions=(0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0),
):
    """
    Elige automáticamente qué frame del timelapse usar como referencia para
    angle_search.

    Devuelve:
      real_image_path, obs_time, reference_info
    """
    pics_dir = Path(pics_dir)
    image_files = list(image_files)

    if not image_files:
        raise RuntimeError("No hay imágenes para elegir referencia de angle_search.")

    candidate_indices = _unique_candidate_indices(
        n_images=len(image_files),
        fractions=reference_fractions,
    )

    print("Evaluando frames candidatos para angle_search:")
    print("   índice | imagen | score | keypoints | coverage | contrast | non_dark")

    candidates = []

    for idx in candidate_indices:
        image_name = image_files[idx]
        image_path = pics_dir / image_name
        metrics = _score_reference_image_for_angle_search(image_path)
        obs_time = start_dt + timedelta(seconds=float(delta) * float(idx))

        info = {
            "reference_index": int(idx),
            "reference_image_name": str(image_name),
            "reference_image_path": str(image_path),
            "reference_obs_time": obs_time.isoformat(),
            "reference_score": float(metrics["score"]),
            **metrics,
        }
        candidates.append(info)

        print(
            f"   {idx:5d} | {image_name} | "
            f"{metrics['score']:.4f} | "
            f"{metrics['n_keypoints']:6d} | "
            f"{metrics['coverage']:.3f} | "
            f"{metrics['contrast']:.2f} | "
            f"{metrics['non_dark_fraction']:.3f}"
        )

    best = max(candidates, key=lambda d: float(d["reference_score"]))

    print("Frame elegido para angle_search:")
    print(f"   index = {best['reference_index']}")
    print(f"   image = {best['reference_image_name']}")
    print(f"   obs_time = {best['reference_obs_time']}")
    print(f"   reference_score = {best['reference_score']:.4f}")

    real_image_path = Path(best["reference_image_path"])
    obs_time = start_dt + timedelta(seconds=float(delta) * float(best["reference_index"]))

    return real_image_path, obs_time, best


# ============================================================
# CONFIGURACION CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline completa de georreferenciacion de timelapses ISS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Experimento y datos")
    g.add_argument("--mission", default="ISS067", help="Mision ISS, por ejemplo ISS067.")
    g.add_argument("--start-id", dest="start_id", type=int, default=362508, help="ID inicial de imagen.")
    g.add_argument("--end-id", dest="end_id", type=int, default=363421, help="ID final de imagen.")
    g.add_argument(
        "--data-root",
        dest="data_root",
        default=str(DATA_ROOT),
        help=(
            "Raiz donde estan los datos grandes: ISS_tle, texturas VIIRS y "
            "carpetas ISSxxx-E-start-end. Tambien puede definirse con "
            "ISS_SIMULATION_DATA_ROOT."
        ),
    )
    g.add_argument(
        "--base-dir",
        dest="base_dir",
        default=None,
        help="Carpeta concreta del experimento. Si se omite: <data-root>/<mission>-E-<start>-<end>.",
    )
    g.add_argument("--tle-dir", dest="tle_dir", default=None, help="Directorio con TLEs. Si se omite: <data-root>/ISS_tle.")
    g.add_argument(
        "--texture-path",
        dest="texture_path",
        default=None,
        help="Textura nocturna para Blender. Si se omite se busca en <data-root>.",
    )
    g.add_argument(
        "--viirs-tiff-path",
        dest="viirs_tiff_path",
        default=None,
        help="Mosaico VIIRS GeoTIFF. Si se omite se busca en <data-root>.",
    )

    g = parser.add_argument_group("Control de ejecucion")
    g.add_argument("--angle-search", dest="use_angle_search", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--reuse-cached-angles", dest="reuse_cached_angles", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--optical-flow", dest="use_optical_flow", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--second-georef-mode", choices=["none", "full", "sample"], default="sample")
    g.add_argument("--second-georef-samples", type=int, default=10, help="Numero de muestras si second_georef_mode=sample.")

    g.add_argument("--rerun-simulation", dest="rerun_simulation_if_exists", action="store_true")
    g.add_argument("--rerun-matching", dest="rerun_matching_if_exists", action="store_true")
    g.add_argument("--rerun-projection", dest="rerun_projection_if_exists", action="store_true")
    g.add_argument("--rerun-filtering", dest="rerun_filtering_if_exists", action="store_true")
    g.add_argument("--rerun-first-georef", dest="rerun_first_georef_if_exists", action="store_true")
    g.add_argument("--rerun-viirs", dest="rerun_viirs_if_exists", action="store_true")
    g.add_argument("--rerun-optical-flow", dest="rerun_optical_flow_if_exists", action="store_true")
    g.add_argument("--rerun-correct-points", dest="rerun_correct_points_if_exists", action="store_true")
    g.add_argument("--rerun-second-georef", dest="rerun_second_georef_if_exists", action="store_true")

    g = parser.add_argument_group("Camara, tiempo y orientacion")
    g.add_argument("--yaw", type=float, default=12.5, help="Yaw fallback si angle_search esta desactivado o falla.")
    g.add_argument("--pitch", type=float, default=63.5, help="Pitch fallback si angle_search esta desactivado o falla.")
    g.add_argument("--roll", type=float, default=-1.0, help="Roll fallback si angle_search esta desactivado o falla.")
    g.add_argument("--orientation-mode", choices=["north", "forward"], default="forward")
    g.add_argument("--time-offset-seconds", type=float, default=0.0)
    g.add_argument("--time-offset-minutes", type=float, default=0.0)
    g.add_argument("--time-offset-hours", type=float, default=0.0)
    g.add_argument("--sensor-width", type=float, default=36.0)
    g.add_argument("--sensor-height", type=float, default=28.0)
    g.add_argument("--earth-radius", type=float, default=10.0)

    g = parser.add_argument_group("Busqueda de angulos")
    g.add_argument("--yaw-range", nargs=2, type=float, default=(-180.0, 180.0), metavar=("MIN", "MAX"))
    g.add_argument("--pitch-range", nargs=2, type=float, default=(30.0, 90.0), metavar=("MIN", "MAX"))
    g.add_argument("--roll-range", nargs=2, type=float, default=(-45.0, 45.0), metavar=("MIN", "MAX"))
    g.add_argument("--coarse-steps", nargs=3, type=float, default=(10.0, 5.0, 5.0), metavar=("YAW", "PITCH", "ROLL"))
    g.add_argument("--fine-steps", nargs=3, type=float, default=(2.5, 1.5, 1.0), metavar=("YAW", "PITCH", "ROLL"))
    g.add_argument("--fine-windows", nargs=3, type=float, default=(5.0, 4.0, 3.0), metavar=("YAW", "PITCH", "ROLL"))
    g.add_argument("--refine-steps", nargs=3, type=float, default=(1.0, 0.5, 0.5), metavar=("YAW", "PITCH", "ROLL"))
    g.add_argument("--refine-windows", nargs=3, type=float, default=(2.0, 1.0, 1.0), metavar=("YAW", "PITCH", "ROLL"))
    g.add_argument("--angle-reference-fractions", nargs="+", type=float, default=(0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0))
    g.add_argument("--simplex-max-iter", type=int, default=45)
    g.add_argument("--simplex-max-evals", type=int, default=140)
    g.add_argument("--simplex-score-tol", type=float, default=1e-3)

    g = parser.add_argument_group("Backend de matching")
    g.add_argument("--matcher-backend", choices=["auto", "vismatch", "matching", "none"], default="auto")
    g.add_argument("--matcher-model", default=None, help="Modelo de matching. Por defecto depende del backend.")
    g.add_argument("--matching-repo", default=None, help="Repo local opcional de vismatch o image-matching-models legacy.")
    g.add_argument("--matcher-device", choices=["cpu", "cuda"], default=None)
    g.add_argument("--no-orb-fallback", action="store_true", help="Desactiva ORB fallback si falla el matcher neural.")

    g = parser.add_argument_group("Matching real-simulada")
    g.add_argument("--matching-grid-step", type=int, default=185)
    g.add_argument("--matching-show-every", type=int, default=50)
    g.add_argument("--matching-min-grid-points", type=int, default=30)
    g.add_argument("--matching-plot-max-matches", type=int, default=150)
    g.add_argument("--matching-min-success-ratio-to-continue", type=float, default=0.20)
    g.add_argument("--matching-min-success-count-to-continue", type=int, default=50)
    g.add_argument("--matching-warning-success-ratio", type=float, default=0.95)

    g = parser.add_argument_group("Filtrado de puntos")
    g.add_argument("--filter-radius-km", type=float, default=80.0)
    g.add_argument("--temporal-filter", dest="filter_use_temporal", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--filter-temporal-order", type=int, default=3)
    g.add_argument("--filter-temporal-threshold-mode", default="sigma", choices=["sigma", "absolute"])
    g.add_argument("--filter-temporal-sigma", type=float, default=3.0)
    g.add_argument("--filter-min-track-points", type=int, default=6)
    g.add_argument("--filter-min-track-coverage", type=float, default=0.20)
    g.add_argument("--filter-max-gap-frames", type=int, default=8)
    g.add_argument(
        "--pre-spatial-filter",
        dest="filter_use_pre_spatial_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activa/desactiva el prefiltro espacial antes del filtrado temporal.",
    )
    g.add_argument("--filter-plot-dir", default=None, help="Carpeta de plots QC. Si se omite: <base-dir>/temporal_qc_plots.")
    g.add_argument("--filter-plot-reference-frames", default="first,mid,last")
    g.add_argument("--filter-diagnostic-inset", type=float, default=0.25)

    g = parser.add_argument_group("Georreferenciacion")
    g.add_argument("--georef-plot-every", type=int, default=50)

    g = parser.add_argument_group("VIIRS")
    g.add_argument("--viirs-nproc", type=int, default=8)
    g.add_argument("--viirs-mode", default="fast", choices=["fast", "safe"])
    g.add_argument("--viirs-roi-mode", default="gcp")
    g.add_argument("--viirs-roi-margin-px", type=int, default=10)
    g.add_argument("--viirs-align", default="roi_exact")
    g.add_argument("--viirs-resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"])
    g.add_argument("--viirs-threads", default="auto")

    g = parser.add_argument_group("Flujo optico")
    g.add_argument("--optical-flow-plot-every", type=int, default=50)
    g.add_argument("--flow-crop-x-start", type=float, default=0.0)
    g.add_argument("--flow-crop-x-end", type=float, default=1.0)
    g.add_argument("--flow-crop-y-start", type=float, default=0.0)
    g.add_argument("--flow-crop-y-end", type=float, default=1.0)

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)

# ============================================================
# PIPELINE
# ============================================================

def main(argv=None):
    args = parse_args(argv)

    # Imports diferidos: si el usuario solo pide --help, argparse sale antes
    # de llegar aquí y no se carga Blender/bpy.
    global cv2, np
    import cv2 as _cv2
    import numpy as _np

    from pipelinentl.get_pics import download_all_images
    from pipelinentl.generate_timelapse import (
        extract_exif_data,
        get_image_files,
    )
    from pipelinentl import generate_timelapse
    from pipelinentl import angle_search

    cv2 = _cv2
    np = _np

    # ============================================================
    # 1. CONFIGURACION GENERAL DEL EXPERIMENTO
    # ============================================================

    data_root = Path(args.data_root).expanduser().resolve()

    mission = args.mission
    start_id = int(args.start_id)
    end_id = int(args.end_id)

    if end_id < start_id:
        raise ValueError(f"end_id ({end_id}) debe ser >= start_id ({start_id}).")

    if args.base_dir:
        base_dir = Path(args.base_dir).expanduser().resolve()
    else:
        base_dir = data_root / f"{mission}-E-{start_id}-{end_id}"

    pics_dir = base_dir / "pics"
    output_dir = base_dir / "output"
    matches_output_dir = output_dir / "matches"

    search_output_dir = base_dir / "search_angles"

    filtered_points_dir = base_dir / "filtered_points"
    geo_dir = base_dir / "geo"

    viirs_output_dir = base_dir / "viirs_cropped_aligned"
    flow_dir = base_dir / "flow"
    corrected_points_dir = base_dir / "corrected_points"
    geo_corrected_dir = base_dir / "geo_corrected"

    tle_dir = required_path(
        Path(args.tle_dir) if args.tle_dir else data_root / "ISS_tle",
        "directorio de TLE",
    )

    default_texture_name = "VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.sqrt.full.40k_20k.png"
    texture_path = str(required_path(
        Path(args.texture_path) if args.texture_path else data_root / default_texture_name,
        "textura nocturna para Blender",
    ))

    default_viirs_name = "VNL_v2_npp_2021_global_vcmslcfg_c202203152300.median_masked.tif"
    viirs_tiff_path = str(required_path(
        Path(args.viirs_tiff_path) if args.viirs_tiff_path else data_root / default_viirs_name,
        "mosaico VIIRS",
    ))

    earth_radius = float(args.earth_radius)

    # ------------------------------------------------------------
    # Controles principales
    # ------------------------------------------------------------

    use_angle_search = bool(args.use_angle_search)
    reuse_cached_angles = bool(args.reuse_cached_angles)

    rerun_simulation_if_exists = bool(args.rerun_simulation_if_exists)
    rerun_matching_if_exists = bool(args.rerun_matching_if_exists)
    rerun_projection_if_exists = bool(args.rerun_projection_if_exists)
    rerun_filtering_if_exists = bool(args.rerun_filtering_if_exists)
    rerun_first_georef_if_exists = bool(args.rerun_first_georef_if_exists)
    rerun_viirs_if_exists = bool(args.rerun_viirs_if_exists)
    rerun_optical_flow_if_exists = bool(args.rerun_optical_flow_if_exists)
    rerun_correct_points_if_exists = bool(args.rerun_correct_points_if_exists)
    rerun_second_georef_if_exists = bool(args.rerun_second_georef_if_exists)

    use_optical_flow = bool(args.use_optical_flow)
    second_georef_mode = args.second_georef_mode
    second_georef_samples = int(args.second_georef_samples)

    # Angulos fallback
    yaw = float(args.yaw)
    pitch = float(args.pitch)
    roll = float(args.roll)

    orientation_mode = args.orientation_mode

    # Offsets temporales
    time_offset_seconds = float(args.time_offset_seconds)
    time_offset_minutes = float(args.time_offset_minutes)
    time_offset_hours = float(args.time_offset_hours)

    # ------------------------------------------------------------
    # Parametros de busqueda de angulos
    # ------------------------------------------------------------

    yaw_range = tuple(map(float, args.yaw_range))
    pitch_range = tuple(map(float, args.pitch_range))
    roll_range = tuple(map(float, args.roll_range))

    coarse_steps = tuple(map(float, args.coarse_steps))
    fine_steps = tuple(map(float, args.fine_steps))
    fine_windows = tuple(map(float, args.fine_windows))
    refine_steps = tuple(map(float, args.refine_steps))
    refine_windows = tuple(map(float, args.refine_windows))

    angle_reference_fractions = tuple(map(float, args.angle_reference_fractions))
    simplex_max_iter = int(args.simplex_max_iter)
    simplex_max_evals = int(args.simplex_max_evals)
    simplex_score_tol = float(args.simplex_score_tol)

    # Sensor fisico
    sensor_width = float(args.sensor_width)
    sensor_height = float(args.sensor_height)

    # Parametros del backend de matching. Usuarios nuevos usan vismatch por pip.
    # El backend legacy matching solo se activa si se indica explicitamente o si auto lo encuentra.
    matcher_backend = args.matcher_backend
    matcher_model = args.matcher_model
    matching_repo = args.matching_repo
    matcher_device = args.matcher_device
    no_orb_fallback = bool(args.no_orb_fallback)

    # Parametros matching
    matching_grid_step = int(args.matching_grid_step)
    matching_show_every = int(args.matching_show_every)
    matching_min_grid_points = int(args.matching_min_grid_points)
    matching_plot_max_matches = int(args.matching_plot_max_matches)

    matching_min_success_ratio_to_continue = float(args.matching_min_success_ratio_to_continue)
    matching_min_success_count_to_continue = int(args.matching_min_success_count_to_continue)
    matching_warning_success_ratio = float(args.matching_warning_success_ratio)

    # Parametros filter_points
    filter_radius_km = float(args.filter_radius_km)
    filter_use_temporal = bool(args.filter_use_temporal)
    filter_temporal_order = int(args.filter_temporal_order)
    filter_temporal_threshold_mode = args.filter_temporal_threshold_mode
    filter_temporal_sigma = float(args.filter_temporal_sigma)
    filter_min_track_points = int(args.filter_min_track_points)
    filter_min_track_coverage = float(args.filter_min_track_coverage)
    filter_max_gap_frames = int(args.filter_max_gap_frames)
    filter_use_pre_spatial_filter = bool(args.filter_use_pre_spatial_filter)
    filter_plot_dir = Path(args.filter_plot_dir).expanduser().resolve() if args.filter_plot_dir else base_dir / "temporal_qc_plots"
    filter_plot_reference_frames = args.filter_plot_reference_frames
    filter_diagnostic_inset = float(args.filter_diagnostic_inset)

    # Parametros georef
    georef_plot_every = int(args.georef_plot_every)

    # Parametros VIIRS
    viirs_nproc = int(args.viirs_nproc)
    viirs_mode = args.viirs_mode
    viirs_roi_mode = args.viirs_roi_mode
    viirs_roi_margin_px = int(args.viirs_roi_margin_px)
    viirs_align = args.viirs_align
    viirs_resampling = args.viirs_resampling
    viirs_threads = args.viirs_threads

    # Parametros optical flow
    optical_flow_plot_every = int(args.optical_flow_plot_every)
    flow_crop_x_start = float(args.flow_crop_x_start)
    flow_crop_x_end = float(args.flow_crop_x_end)
    flow_crop_y_start = float(args.flow_crop_y_start)
    flow_crop_y_end = float(args.flow_crop_y_end)

    print("Configuracion principal:")
    print(f"  mission = {mission}")
    print(f"  start_id = {start_id}")
    print(f"  end_id = {end_id}")
    print(f"  data_root = {data_root}")
    print(f"  base_dir = {base_dir}")
    print(f"  use_angle_search = {use_angle_search}")
    print(f"  use_optical_flow = {use_optical_flow}")
    print(f"  second_georef_mode = {second_georef_mode}")
    print(f"  matcher_backend = {matcher_backend}")

    # ============================================================
    # CONTADOR DE PASOS
    # ============================================================

    total_steps = 7
    if use_optical_flow:
        total_steps += 3
        if second_georef_mode != "none":
            total_steps += 1

    step = 1
    force_downstream = False

    # ============================================================
    # 2. DESCARGA DE IMÁGENES
    # ============================================================

    pics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{step}/{total_steps}] Descargando imágenes ISS...")
    step += 1

    download_all_images(mission, start_id, end_id, pics_dir)

    # ============================================================
    # 3. EXTRAER RANGO TEMPORAL
    # ============================================================

    img_files = get_image_files(pics_dir)
    if not img_files:
        raise RuntimeError("No se encontraron imágenes en la carpeta de fotos descargadas.")

    first_img = img_files[0]
    last_img = img_files[-1]

    print(f"Primera imagen: {first_img}")
    print(f"Última imagen:  {last_img}")

    start_dt, focal_length, pixel_width, pixel_height = extract_exif_data(pics_dir / first_img)
    end_dt, _, _, _ = extract_exif_data(pics_dir / last_img)

    if start_dt is None or end_dt is None:
        raise RuntimeError("No se pudo extraer fecha EXIF de la primera o última imagen.")

    if focal_length is None:
        raise RuntimeError("No se pudo extraer focal_length del EXIF.")

    total_offset = timedelta(
        seconds=time_offset_seconds,
        minutes=time_offset_minutes,
        hours=time_offset_hours,
    )

    start_dt = start_dt + total_offset
    end_dt = end_dt + total_offset

    print("Rango temporal tras offset:")
    print(f"  start_dt = {start_dt}")
    print(f"  end_dt   = {end_dt}")

    n_images = len(img_files)
    if n_images < 2:
        raise RuntimeError("Se necesitan al menos dos imágenes para calcular delta temporal.")

    delta = (end_dt - start_dt).total_seconds() / (n_images - 1)

    if delta <= 0:
        raise RuntimeError(
            f"Delta temporal inválido: {delta} s. "
            "Revisa el orden de las imágenes y los EXIF."
        )

    expected_count_by_range = end_id - start_id + 1
    if n_images != expected_count_by_range:
        print(
            f"AVISO: se esperaban {expected_count_by_range} imágenes por rango "
            f"[{start_id}, {end_id}], pero hay {n_images} en {pics_dir}."
        )
        print(
            "El delta se calculará con las imágenes realmente presentes. "
            "Si faltan imágenes intermedias, puede haber desfase temporal."
        )

    print("Paso temporal automático:")
    print(f"  n_images = {n_images}")
    print(f"  delta    = {delta:.6f} s")

    angle_reference_info = None

    # ============================================================
    # 4. BÚSQUEDA DE YAW/PITCH/ROLL
    # ============================================================

    if use_angle_search:
        cache_file = base_dir / "angle_search_results.txt"
        reuse_from_cache = False

        if reuse_cached_angles and cache_file.exists():
            print(f"[{step}/{total_steps}] Cargando yaw/pitch/roll desde cache...")
            with cache_file.open("r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            cached = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    cached[k.strip()] = v.strip()

            try:
                yaw = float(cached.get("yaw"))
                pitch = float(cached.get("pitch"))
                roll = float(cached.get("roll"))

                # Si el cache es antiguo y no guarda la imagen de referencia,
                # recalculamos para aplicar el nuevo criterio de selección.
                cache_has_reference = (
                    "reference_image" in cached
                    and "reference_index" in cached
                    and "reference_obs_time" in cached
                )

                if not cache_has_reference:
                    print(
                        "AVISO: el cache de angle_search no tiene información "
                        "del frame de referencia. Se recalculará."
                    )
                    cache_file.unlink(missing_ok=True)
                    reuse_from_cache = False
                else:
                    print(f"   yaw   = {yaw}")
                    print(f"   pitch = {pitch}")
                    print(f"   roll  = {roll}")

                    if "score" in cached:
                        print(f"   score = {cached['score']}")

                    print(f"   reference_image = {cached['reference_image']}")
                    print(f"   reference_index = {cached['reference_index']}")
                    print(f"   reference_obs_time = {cached['reference_obs_time']}")

                    if "reference_score" in cached:
                        print(f"   reference_score = {cached['reference_score']}")

                    angle_reference_info = {
                        "reference_image": cached.get("reference_image"),
                        "reference_index": cached.get("reference_index"),
                        "reference_obs_time": cached.get("reference_obs_time"),
                        "reference_score": cached.get("reference_score"),
                        "reference_keypoints": cached.get("reference_keypoints"),
                        "reference_coverage": cached.get("reference_coverage"),
                    }
                    reuse_from_cache = True

            except Exception as e:
                print(f"AVISO: error leyendo cache de ángulos ({e}). Se recalculará.")
                cache_file.unlink(missing_ok=True)
                reuse_from_cache = False

        if not reuse_from_cache:
            print(f"[{step}/{total_steps}] Buscando yaw/pitch/roll óptimos con angle_search...")

            search_output_dir.mkdir(parents=True, exist_ok=True)

            # En vez de usar siempre la primera imagen, se evalúan varias
            # candidatas del timelapse: primera, 1/3, 1/2, 2/3 y última.
            # Solo cambia la imagen de referencia; la búsqueda de yaw/pitch/roll
            # sigue haciéndola angle_search.search_best_yaw_pitch_roll.
            real_image_path, obs_time, reference_info = select_angle_search_reference_frame(
                pics_dir=pics_dir,
                image_files=img_files,
                start_dt=start_dt,
                delta=delta,
                reference_fractions=angle_reference_fractions,
            )

            reference_search_output_dir = (
                search_output_dir / f"ref_{reference_info['reference_index']:05d}"
            )
            reference_search_output_dir.mkdir(parents=True, exist_ok=True)

            best_angles, search_details = angle_search.search_best_yaw_pitch_roll(
                real_image_path=str(real_image_path),
                obs_time=obs_time,
                search_output_dir=str(reference_search_output_dir),
                tle_dir=str(tle_dir),
                texture_path=texture_path,

                focal_length=focal_length,
                sensor_width=sensor_width,
                sensor_height=sensor_height,
                pixel_width=pixel_width,
                pixel_height=pixel_height,

                yaw_range=yaw_range,
                pitch_range=pitch_range,
                roll_range=roll_range,

                coarse_steps=coarse_steps,
                fine_steps=fine_steps,
                fine_windows=fine_windows,
                refine_steps=refine_steps,
                refine_windows=refine_windows,

                orientation_mode=orientation_mode,
                earth_radius=earth_radius,
                matcher_backend=matcher_backend,
                matcher_model=matcher_model,
                matching_repo=matching_repo,
                device=matcher_device,
                no_orb_fallback=no_orb_fallback,
                simplex_max_iter=simplex_max_iter,
                simplex_max_evals=simplex_max_evals,
                simplex_score_tol=simplex_score_tol,
            )

            if best_angles is not None:
                yaw = float(best_angles["yaw"])
                pitch = float(best_angles["pitch"])
                roll = float(best_angles["roll"])
                score = best_angles.get("score", None)

                # Guardamos en el resultado qué frame se usó para la búsqueda.
                best_angles.update(reference_info)
                angle_reference_info = {
                    "reference_image": reference_info["reference_image_name"],
                    "reference_index": reference_info["reference_index"],
                    "reference_obs_time": reference_info["reference_obs_time"],
                    "reference_score": reference_info["reference_score"],
                    "reference_keypoints": reference_info["n_keypoints"],
                    "reference_coverage": reference_info["coverage"],
                }

                print("Mejores ángulos encontrados:")
                print(f"   yaw   = {yaw}")
                print(f"   pitch = {pitch}")
                print(f"   roll  = {roll}")
                print(f"   score = {score}")
                print(f"   reference_image = {reference_info['reference_image_name']}")
                print(f"   reference_index = {reference_info['reference_index']}")
                print(f"   reference_obs_time = {reference_info['reference_obs_time']}")
                print(f"   reference_score = {reference_info['reference_score']:.4f}")
                print(f"   reference_keypoints = {reference_info['n_keypoints']}")
                print(f"   reference_coverage = {reference_info['coverage']:.4f}")

                with cache_file.open("w", encoding="utf-8") as f:
                    f.write(f"yaw={yaw}\n")
                    f.write(f"pitch={pitch}\n")
                    f.write(f"roll={roll}\n")
                    if score is not None:
                        f.write(f"score={score}\n")

                    f.write(f"reference_image={reference_info['reference_image_name']}\n")
                    f.write(f"reference_index={reference_info['reference_index']}\n")
                    f.write(f"reference_obs_time={reference_info['reference_obs_time']}\n")
                    f.write(f"reference_score={reference_info['reference_score']}\n")
                    f.write(f"reference_keypoints={reference_info['n_keypoints']}\n")
                    f.write(f"reference_coverage={reference_info['coverage']}\n")

                summary_json = base_dir / "angle_search_best_result.json"
                try:
                    with summary_json.open("w", encoding="utf-8") as f:
                        json.dump(best_angles, f, indent=2)
                except Exception as e:
                    print(f"AVISO: no se pudo guardar {summary_json.name}: {e}")

                force_downstream = True

            else:
                print("AVISO: angle_search no devolvió resultado válido. Se usan ángulos fallback.")

    else:
        print(
            f"[{step}/{total_steps}] Búsqueda de ángulos desactivada. "
            f"Usando yaw={yaw}, pitch={pitch}, roll={roll}."
        )

    step += 1

    # ============================================================
    # CONFIGURACIONES DE REPRODUCIBILIDAD
    # ============================================================

    common_config = {
        "mission": mission,
        "start_id": start_id,
        "end_id": end_id,
        "n_images": n_images,
        "first_img": first_img,
        "last_img": last_img,
        "start_dt": start_dt.isoformat(),
        "end_dt": end_dt.isoformat(),
        "delta": delta,
        "time_offset_seconds": time_offset_seconds,
        "time_offset_minutes": time_offset_minutes,
        "time_offset_hours": time_offset_hours,
        "focal_length": focal_length,
        "sensor_width": sensor_width,
        "sensor_height": sensor_height,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "earth_radius": earth_radius,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "orientation_mode": orientation_mode,
        "angle_reference_info": angle_reference_info,
        "texture_path": texture_path,
        "tle_dir": str(tle_dir),
    }

    # ============================================================
    # 5. GENERAR TIMELAPSE SIMULADO
    # ============================================================

    print(f"[{step}/{total_steps}] Generando simulación timelapse en Blender...")
    step += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    sim_config = {
        **common_config,
        "step": "generate_timelapse",
    }
    sim_config_file = output_dir / "_render_config.json"

    # Para la simulación no usamos configs_equal() a pelo, porque es demasiado
    # estricta: puede forzar rerender por cambios irrelevantes en el JSON.
    # Solo rerenderizamos si cambian parámetros que realmente modifican los renders.
    existing_render_count = count_nonempty(output_dir, "render_output_*.png")

    if existing_render_count >= n_images and not rerun_simulation_if_exists:
        old_sim_config = read_json(sim_config_file)

        keys_that_require_rerender = [
            "mission",
            "start_id",
            "end_id",
            "n_images",
            "start_dt",
            "end_dt",
            "delta",
            "focal_length",
            "sensor_width",
            "sensor_height",
            "pixel_width",
            "pixel_height",
            "earth_radius",
            "yaw",
            "pitch",
            "roll",
            "orientation_mode",
            "texture_path",
            "tle_dir",
        ]

        changed_keys = []

        if old_sim_config is None:
            print(
                f"   OK renders simulados: ya existen {existing_render_count}/{n_images}. "
                "Falta _render_config.json, se reutilizan y se escribe la config actual."
            )
            run_simulation = False
            write_json(sim_config_file, sim_config)

        else:
            for k in keys_that_require_rerender:
                old_v = old_sim_config.get(k)
                new_v = sim_config.get(k)

                # Comparación tolerante para floats.
                if isinstance(old_v, (float, int)) or isinstance(new_v, (float, int)):
                    try:
                        same = abs(float(old_v) - float(new_v)) < 1e-6
                    except Exception:
                        same = old_v == new_v
                else:
                    same = old_v == new_v

                if not same:
                    changed_keys.append((k, old_v, new_v))

            if changed_keys:
                print("   AVISO renders simulados: cambió configuración geométrica real.")
                print("   Claves que fuerzan rerender:")
                for k, old_v, new_v in changed_keys:
                    print(f"      {k}: {old_v!r} -> {new_v!r}")

                run_simulation = True

            else:
                print(
                    f"   OK renders simulados: completos ({existing_render_count}/{n_images}) "
                    "y configuración geométrica compatible. Se reutilizan."
                )
                run_simulation = False

                # Actualizamos el JSON para evitar futuros falsos positivos.
                write_json(sim_config_file, sim_config)

    else:
        run_simulation = step_should_run(
            folder=output_dir,
            pattern="render_output_*.png",
            expected=n_images,
            label="renders simulados",
            config_file=sim_config_file,
            current_config=sim_config,
            force_downstream=force_downstream,
            force_this_step=rerun_simulation_if_exists,
        )

    if run_simulation:
        existing_render_count = count_nonempty(output_dir, "render_output_*.png")

        if existing_render_count > 0 and not rerun_simulation_if_exists:
            print(
                f"   AVISO: hay {existing_render_count}/{n_images} renders existentes. "
                "No se borran; se intentará continuar/regenerar sobre la misma carpeta."
            )
        else:
            remove_files(output_dir, "render_output_*.png", "renders simulados")

        sim_config_file.unlink(missing_ok=True)

        Args_gen = Namespace(
            pics=str(pics_dir),
            output=str(output_dir),
            tle=str(tle_dir),
            texture=texture_path,
            earth_radius=earth_radius,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            delta=delta,
            test=False,
            time_offset_seconds=time_offset_seconds,
            time_offset_minutes=time_offset_minutes,
            time_offset_hours=time_offset_hours,
            orientation_mode=orientation_mode,
        )

        generate_timelapse.main(Args_gen)

        if not is_complete(output_dir, "render_output_*.png", n_images, "renders simulados"):
            print(
                "   AVISO: la simulación terminó con menos renders de los esperados. "
                "La pipeline continuará, pero el matching puede fallar si faltan demasiados pares."
            )

        write_json(sim_config_file, sim_config)
        force_downstream = True
    else:
        print("   Se omite la generación: renders completos y configuración válida.")

    # ============================================================
    # 6. MATCHING REAL-SIMULADA
    # ============================================================

    print(f"[{step}/{total_steps}] Comparando imágenes reales y simuladas...")
    step += 1

    matches_output_dir.mkdir(parents=True, exist_ok=True)

    # Configuración que SÍ afecta al resultado del matching.
    # Importante: no metemos aquí los umbrales de aceptación de la pipeline,
    # porque cambiar un umbral de "continuar/no continuar" no debería obligar
    # a recalcular todos los matches.
    match_config = {
        **common_config,
        "step": "match_timelapse",
        "matching_grid_step": matching_grid_step,
        "matching_show_every": matching_show_every,
        "matching_min_grid_points": matching_min_grid_points,
        "matching_plot_max_matches": matching_plot_max_matches,
        "matcher_backend": matcher_backend,
        "matcher_model": matcher_model,
        "matching_repo": matching_repo,
        "matcher_device": matcher_device,
        "no_orb_fallback": no_orb_fallback,
        "match_timelapse_version": "portable_backend_ransac_normalized_poly_grid_visualization_v2",
    }
    match_config_file = matches_output_dir / "_match_config.json"

    n_existing_csv = count_nonempty(matches_output_dir, "transformed_coordinates_*.csv")

    # Umbral duro: por debajo de esto sí paramos, porque no hay suficiente material.
    min_csv_hard = min(
        n_images,
        max(
            1,
            int(matching_min_success_count_to_continue),
            math.ceil(float(matching_min_success_ratio_to_continue) * n_images),
        ),
    )

    # Umbral recomendado: solo avisa.
    min_csv_warning = min(
        n_images,
        max(
            1,
            math.ceil(float(matching_warning_success_ratio) * n_images),
        ),
    )

    old_match_config = read_json(match_config_file)

    # Compatibilidad con configs antiguas que tenían min_matching_success_ratio.
    # Ese parámetro no afecta a los CSV generados, solo a si la pipeline paraba.
    if old_match_config is not None:
        old_match_config_comparable = dict(old_match_config)
        old_match_config_comparable.pop("min_matching_success_ratio", None)
        old_match_config_comparable.pop("matching_min_success_ratio_to_continue", None)
        old_match_config_comparable.pop("matching_min_success_count_to_continue", None)
        old_match_config_comparable.pop("matching_warning_success_ratio", None)
    else:
        old_match_config_comparable = None

    match_outputs_ok = n_existing_csv >= min_csv_hard

    print(
        f"   CSVs de matching existentes: {n_existing_csv}/{n_images} "
        f"({100 * n_existing_csv / max(n_images, 1):.1f}%)."
    )
    print(
        f"   Mínimo duro para continuar: {min_csv_hard}/{n_images} "
        f"({100 * min_csv_hard / max(n_images, 1):.1f}%)."
    )
    print(
        f"   Umbral recomendado: {min_csv_warning}/{n_images} "
        f"({100 * min_csv_warning / max(n_images, 1):.1f}%)."
    )

    if rerun_matching_if_exists:
        print("   AVISO matching: forzado por configuración.")
        run_matching = True

    elif force_downstream:
        print("   AVISO matching: se recalculará porque un paso anterior cambió.")
        run_matching = True

    elif not match_outputs_ok:
        print(
            f"   AVISO CSVs de matching: insuficientes para continuar "
            f"({n_existing_csv}/{n_images}). "
            f"Mínimo duro requerido: {min_csv_hard}."
        )
        run_matching = True

    elif old_match_config_comparable is None:
        print(
            f"   AVISO CSVs de matching: suficientes "
            f"({n_existing_csv}/{n_images}), pero falta config. "
            "Se asumirá válido y se escribirá la configuración actual."
        )
        write_json(match_config_file, match_config)
        run_matching = False

    elif not configs_equal(old_match_config_comparable, match_config):
        print("   AVISO matching: configuración que afecta al matching distinta. Se recalculará.")
        run_matching = True

    else:
        print(
            f"   OK CSVs de matching suficientes para continuar: "
            f"{n_existing_csv}/{n_images} "
            f"({100 * n_existing_csv / n_images:.1f}%)."
        )
        run_matching = False

    if run_matching:
        remove_files(matches_output_dir, "transformed_coordinates_*.csv", "CSVs de matching")
        remove_files(matches_output_dir, "matches_color_*.png", "plots de matches")
        remove_files(matches_output_dir, "grid_real_*.png", "plots de rejilla real")
        remove_files(matches_output_dir, "grid_sim_deformed_*.png", "plots de rejilla simulada")
        match_config_file.unlink(missing_ok=True)

        subprocess.run(
            [
                sys.executable, "-m", "pipelinentl.match_timelapse",
                "--output_dir", str(output_dir),
                "--pictures_dir", str(pics_dir),
                "--matches_output_dir", str(matches_output_dir),
                "--grid_step", str(matching_grid_step),
                "--show_every", str(matching_show_every),
                "--min_grid_points", str(matching_min_grid_points),
                "--plot_max_matches", str(matching_plot_max_matches),
                "--matcher_backend", matcher_backend,
            ]
            + (["--matcher_model", str(matcher_model)] if matcher_model else [])
            + (["--matching_repo", str(matching_repo)] if matching_repo else [])
            + (["--device", str(matcher_device)] if matcher_device else [])
            + (["--no_orb_fallback"] if no_orb_fallback else []),
            check=True,
        )

        n_csv = count_nonempty(matches_output_dir, "transformed_coordinates_*.csv")

        if n_csv < min_csv_hard:
            raise RuntimeError(
                f"match_timelapse terminó, pero hay demasiado pocos CSVs válidos: "
                f"{n_csv}/{n_images}. "
                f"Mínimo duro requerido: {min_csv_hard} "
                f"({100 * min_csv_hard / max(n_images, 1):.1f}%)."
            )

        if n_csv < min_csv_warning:
            print(
                f"   AVISO: hay menos CSVs válidos que el umbral recomendado "
                f"({n_csv}/{n_images}, {100 * n_csv / n_images:.1f}%). "
                f"Recomendado: {min_csv_warning}/{n_images} "
                f"({100 * matching_warning_success_ratio:.1f}%). "
                "La pipeline continuará con los frames válidos."
            )
        elif n_csv < n_images:
            print(
                f"   AVISO: faltan algunos CSVs de matching "
                f"({n_csv}/{n_images}, {100 * n_csv / n_images:.1f}%). "
                "La pipeline continuará con los frames válidos."
            )
        else:
            print(f"   OK CSVs de matching: completo ({n_csv}/{n_images}).")

        write_json(match_config_file, match_config)
        force_downstream = True

    n_valid_csv = count_nonempty(matches_output_dir, "transformed_coordinates_*.csv")
    print(f"   Frames con CSV de matching válido: {n_valid_csv}/{n_images}")

    if n_valid_csv < min_csv_hard:
        raise RuntimeError(
            f"No hay suficientes CSVs válidos para continuar: "
            f"{n_valid_csv}/{n_images}. "
            f"Mínimo duro requerido: {min_csv_hard}."
        )

    if n_valid_csv < min_csv_warning:
        print(
            f"   AVISO: se continuará aunque el porcentaje de matching válido sea bajo "
            f"({100 * n_valid_csv / n_images:.1f}%). "
            "Esto es esperable en timelapses con mucho mar, nubes o frames pobres."
        )
    # ============================================================
    # 7. PROYECCIÓN DE PÍXELES -> .points
    # ============================================================

    print(f"[{step}/{total_steps}] Proyectando píxeles y generando .points...")
    step += 1

    points_mode = "real"

    project_config = {
        **common_config,
        "step": "project_timelapse",
        "points_mode": points_mode,
        "n_valid_csv": n_valid_csv,
    }
    project_config_file = output_dir / "_project_config.json"

    run_projection = step_should_run(
        folder=output_dir,
        pattern="*_real.points",
        expected=n_valid_csv,
        label="puntos proyectados *_real.points",
        config_file=project_config_file,
        current_config=project_config,
        force_downstream=force_downstream,
        force_this_step=rerun_projection_if_exists,
    )

    if run_projection:
        remove_files(output_dir, "*_real.points", "puntos proyectados reales")
        remove_files(output_dir, "*_simulated.points", "puntos proyectados simulados")
        project_config_file.unlink(missing_ok=True)

        subprocess.run(
            [
                sys.executable, "-m", "pipelinentl.project_timelapse",
                "--output_directory", str(output_dir),
                "--texture_path", texture_path,
                "--csv_dir", str(matches_output_dir),
                "--image_dir", str(pics_dir),
                "--tle_directory", str(tle_dir),

                "--yaw", str(yaw),
                "--pitch", str(pitch),
                "--roll", str(roll),

                "--focal_length", str(focal_length),
                "--sensor_width", str(sensor_width),
                "--sensor_height", str(sensor_height),
                "--pixel_width", str(pixel_width),
                "--pixel_height", str(pixel_height),

                "--start_date", start_dt.isoformat(),
                "--end_date", end_dt.isoformat(),
                "--time_step", str(delta),
                "--points_mode", points_mode,
                "--orientation_mode", orientation_mode,
            ],
            check=True,
        )

        if not is_complete(output_dir, "*_real.points", n_valid_csv, "puntos proyectados *_real.points"):
            raise RuntimeError("project_timelapse terminó, pero faltan *_real.points.")

        write_json(project_config_file, project_config)
        force_downstream = True
    else:
        print("   Se omite project_timelapse: .points completos y configuración válida.")

    n_projected_points = count_nonempty(output_dir, "*_real.points")
    print(f"   Frames con puntos proyectados: {n_projected_points}/{n_images}")

    if n_projected_points == 0:
        raise RuntimeError("No hay ningún archivo *_real.points. No se puede continuar.")

    # ============================================================
    # 8. FILTRADO + RENOMBRADO DE .points
    # ============================================================

    print(f"[{step}/{total_steps}] Filtrando y renombrando puntos...")
    step += 1

    filtered_points_dir.mkdir(parents=True, exist_ok=True)

    filter_config = {
        **common_config,
        "step": "filter_points",
        "filter_radius_km": filter_radius_km,
        "filter_use_temporal": filter_use_temporal,
        "filter_temporal_order": filter_temporal_order,
        "filter_temporal_threshold_mode": filter_temporal_threshold_mode,
        "filter_temporal_sigma": filter_temporal_sigma,
        "filter_min_track_points": filter_min_track_points,
        "filter_min_track_coverage": filter_min_track_coverage,
        "filter_max_gap_frames": filter_max_gap_frames,
        "filter_use_pre_spatial_filter": filter_use_pre_spatial_filter,
        "filter_plot_dir": str(filter_plot_dir),
        "filter_plot_reference_frames": filter_plot_reference_frames,
        "filter_diagnostic_inset": filter_diagnostic_inset,
        "n_projected_points": n_projected_points,
    }
    filter_config_file = filtered_points_dir / "_filter_config.json"

    existing_filtered_count = count_points_in_range(filtered_points_dir, start_id, end_id)
    filtered_complete = existing_filtered_count >= n_projected_points
    filter_config_ok = configs_equal(read_json(filter_config_file), filter_config)

    if rerun_filtering_if_exists or force_downstream or not filtered_complete or not filter_config_ok:
        print(
            f"   AVISO puntos filtrados: {existing_filtered_count}/{n_projected_points}. "
            "Se recalculará."
        )

        remove_points_in_range(filtered_points_dir, start_id, end_id, "puntos filtrados")
        filter_config_file.unlink(missing_ok=True)

        filter_cmd = [
            sys.executable, "-m", "pipelinentl.filter_points",
            "--input_folder", str(output_dir),
            "--output_folder", str(filtered_points_dir),
            "--radius_km", str(filter_radius_km),
            "--start_id", str(start_id),
            "--end_id", str(end_id),
            "--mission", mission,
            "--temporal_order", str(filter_temporal_order),
            "--temporal_threshold_mode", filter_temporal_threshold_mode,
            "--temporal_sigma", str(filter_temporal_sigma),
            "--min_track_points", str(filter_min_track_points),
            "--min_track_coverage", str(filter_min_track_coverage),
            "--max_gap_frames", str(filter_max_gap_frames),
            "--plot_dir", str(filter_plot_dir),
            "--image_dir", str(pics_dir),
            "--plot_reference_frames", filter_plot_reference_frames,
            "--diagnostic_inset", str(filter_diagnostic_inset),
        ]

        if not filter_use_temporal:
            filter_cmd.append("--disable_temporal")

        if not filter_use_pre_spatial_filter:
            filter_cmd.append("--no_pre_spatial_filter")

        subprocess.run(filter_cmd, check=True)

        existing_filtered_count = count_points_in_range(filtered_points_dir, start_id, end_id)

        if existing_filtered_count < n_projected_points:
            raise RuntimeError(
                f"filter_points terminó, pero hay "
                f"{existing_filtered_count}/{n_projected_points} .points filtrados."
            )

        write_json(filter_config_file, filter_config)
        force_downstream = True
    else:
        print(
            f"   Se omite filter_points: puntos filtrados suficientes "
            f"({existing_filtered_count}/{n_projected_points})."
        )

    n_filtered_points = count_points_in_range(filtered_points_dir, start_id, end_id)
    print(f"   Frames con puntos filtrados: {n_filtered_points}/{n_images}")

    if n_filtered_points == 0:
        raise RuntimeError("No hay puntos filtrados. No se puede georreferenciar.")

    # ============================================================
    # 9. PRIMERA GEORREFERENCIACIÓN
    # ============================================================

    print(f"[{step}/{total_steps}] Georreferenciando imágenes ISS primera pasada...")
    step += 1

    geo_dir.mkdir(parents=True, exist_ok=True)

    geo_config = {
        **common_config,
        "step": "georef_timelapse_first",
        "georef_plot_every": georef_plot_every,
        "points_dir": str(filtered_points_dir),
        "n_filtered_points": n_filtered_points,
    }
    geo_config_file = geo_dir / "_geo_config.json"

    run_geo = step_should_run(
        folder=geo_dir,
        pattern="*_rect.tiff",
        expected=n_filtered_points,
        label="GeoTIFFs primera pasada",
        config_file=geo_config_file,
        current_config=geo_config,
        force_downstream=force_downstream,
        force_this_step=rerun_first_georef_if_exists,
    )

    if run_geo:
        remove_files(geo_dir, "*_rect.tiff", "GeoTIFFs primera pasada")
        remove_files(geo_dir, "*.png", "plots primera georreferenciación")
        geo_config_file.unlink(missing_ok=True)

        subprocess.run(
            [
                sys.executable, "-m", "pipelinentl.georef_timelapse",
                "--input_dir", str(pics_dir),
                "--points_dir", str(filtered_points_dir),
                "--output_dir", str(geo_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--plot_every", str(georef_plot_every),
            ],
            check=True,
        )

        if not is_complete(geo_dir, "*_rect.tiff", n_filtered_points, "GeoTIFFs primera pasada"):
            raise RuntimeError("georef_timelapse terminó, pero faltan GeoTIFFs de primera pasada.")

        write_json(geo_config_file, geo_config)
        force_downstream = True
    else:
        print("   Se omite primera georreferenciación: GeoTIFFs suficientes y configuración válida.")

    n_geo = count_nonempty(geo_dir, "*_rect.tiff")
    print(f"   Frames georreferenciados primera pasada: {n_geo}/{n_images}")

    if n_geo == 0:
        raise RuntimeError("No hay GeoTIFFs de primera pasada. No se puede continuar.")

    # ============================================================
    # SI NO HAY FLUJO ÓPTICO, TERMINAR
    # ============================================================

    if not use_optical_flow:
        print("\nPipeline completada SIN flujo óptico.")
        return

    # ============================================================
    # 10. RECORTE Y ALINEADO VIIRS
    # ============================================================

    print(f"[{step}/{total_steps}] Recortando y alineando VIIRS...")
    step += 1

    viirs_output_dir.mkdir(parents=True, exist_ok=True)

    viirs_config = {
        **common_config,
        "step": "viirs_roi_crop",
        "viirs_tiff_path": viirs_tiff_path,
        "viirs_nproc": viirs_nproc,
        "viirs_mode": viirs_mode,
        "viirs_roi_mode": viirs_roi_mode,
        "viirs_roi_margin_px": viirs_roi_margin_px,
        "viirs_align": viirs_align,
        "viirs_resampling": viirs_resampling,
        "n_geo": n_geo,
    }
    viirs_config_file = viirs_output_dir / "_viirs_config.json"

    run_viirs = step_should_run(
        folder=viirs_output_dir,
        pattern="*_viirs.tiff",
        expected=n_geo,
        label="recortes VIIRS alineados",
        config_file=viirs_config_file,
        current_config=viirs_config,
        force_downstream=force_downstream,
        force_this_step=rerun_viirs_if_exists,
    )

    if run_viirs:
        remove_files(viirs_output_dir, "*_viirs.tiff", "recortes VIIRS")
        remove_files(viirs_output_dir, "*.json", "metadatos VIIRS")
        remove_files(viirs_output_dir, "*.png", "plots VIIRS")
        viirs_config_file.unlink(missing_ok=True)

        subprocess.run(
            [
                sys.executable, "-m", "pipelinentl.viirs_roi_crop",
                "--geo_dir", str(geo_dir),
                "--viirs_tiff", str(viirs_tiff_path),
                "--output_dir", str(viirs_output_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--nproc", str(viirs_nproc),
                "--mode", viirs_mode,

                "--roi_mode", viirs_roi_mode,
                "--roi_margin_px", str(viirs_roi_margin_px),

                "--align", viirs_align,
                "--resampling", viirs_resampling,

                "--threads", str(viirs_threads),
            ],
            check=True,
        )

        if not is_complete(viirs_output_dir, "*_viirs.tiff", n_geo, "recortes VIIRS alineados"):
            raise RuntimeError("viirs_roi_crop terminó, pero faltan recortes VIIRS.")

        write_json(viirs_config_file, viirs_config)
        force_downstream = True
    else:
        print("   Se omite VIIRS: recortes suficientes y configuración válida.")

    n_viirs = count_nonempty(viirs_output_dir, "*_viirs.tiff")
    print(f"   Frames VIIRS alineados: {n_viirs}/{n_images}")

    if n_viirs == 0:
        raise RuntimeError("No hay recortes VIIRS. No se puede calcular flujo óptico.")

    # ============================================================
    # 11. FLUJO ÓPTICO ISS-VIIRS
    # ============================================================

    print(f"[{step}/{total_steps}] Calculando flujo óptico ISS-VIIRS...")
    step += 1

    flow_dir.mkdir(parents=True, exist_ok=True)

    flow_config = {
        **common_config,
        "step": "optical_flow",
        "optical_flow_plot_every": optical_flow_plot_every,
        "flow_crop_x_start": flow_crop_x_start,
        "flow_crop_x_end": flow_crop_x_end,
        "flow_crop_y_start": flow_crop_y_start,
        "flow_crop_y_end": flow_crop_y_end,
        "geo_dir": str(geo_dir),
        "viirs_output_dir": str(viirs_output_dir),
        "n_viirs": n_viirs,
    }
    flow_config_file = flow_dir / "_flow_config.json"

    run_flow = step_should_run(
        folder=flow_dir,
        pattern="*_flow.npy",
        expected=n_viirs,
        label="flujos ópticos",
        config_file=flow_config_file,
        current_config=flow_config,
        force_downstream=force_downstream,
        force_this_step=rerun_optical_flow_if_exists,
    )

    if run_flow:
        remove_files(flow_dir, "*_flow.npy", "flujos ópticos")
        remove_files(flow_dir, "*.png", "plots de flujo óptico")
        flow_config_file.unlink(missing_ok=True)

        subprocess.run(
            [
                sys.executable, "-m", "pipelinentl.optical_flow",
                "--geo_dir", str(geo_dir),
                "--viirs_dir", str(viirs_output_dir),
                "--flow_dir", str(flow_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--plot_every", str(optical_flow_plot_every),

                "--crop_x_start", str(flow_crop_x_start),
                "--crop_x_end", str(flow_crop_x_end),
                "--crop_y_start", str(flow_crop_y_start),
                "--crop_y_end", str(flow_crop_y_end),
            ],
            check=True,
        )

        if not is_complete(flow_dir, "*_flow.npy", n_viirs, "flujos ópticos"):
            raise RuntimeError("optical_flow terminó, pero faltan .npy de flujo.")

        write_json(flow_config_file, flow_config)
        force_downstream = True
    else:
        print("   Se omite optical_flow: flujos suficientes y configuración válida.")

    n_flow = count_nonempty(flow_dir, "*_flow.npy")
    print(f"   Frames con flujo óptico: {n_flow}/{n_images}")

    if n_flow == 0:
        raise RuntimeError("No hay flujos ópticos. No se pueden corregir puntos.")

    # ============================================================
    # 12. CORREGIR PUNTOS CON FLUJO
    # ============================================================

    print(f"[{step}/{total_steps}] Corrigiendo puntos con flujo óptico...")
    step += 1

    corrected_points_dir.mkdir(parents=True, exist_ok=True)

    expected_corrected = min(n_filtered_points, n_flow)

    corrected_config = {
        **common_config,
        "step": "correct_points",
        "filtered_points_dir": str(filtered_points_dir),
        "flow_dir": str(flow_dir),
        "geo_dir": str(geo_dir),
        "expected_corrected": expected_corrected,
        "flow_crop_x_start": flow_crop_x_start,
        "flow_crop_x_end": flow_crop_x_end,
        "flow_crop_y_start": flow_crop_y_start,
        "flow_crop_y_end": flow_crop_y_end,
    }
    corrected_config_file = corrected_points_dir / "_corrected_points_config.json"

    run_correct_points = step_should_run(
        folder=corrected_points_dir,
        pattern="*.points",
        expected=expected_corrected,
        label="puntos corregidos",
        config_file=corrected_config_file,
        current_config=corrected_config,
        force_downstream=force_downstream,
        force_this_step=rerun_correct_points_if_exists,
    )

    if run_correct_points:
        remove_files(corrected_points_dir, "*.points", "puntos corregidos")
        corrected_config_file.unlink(missing_ok=True)

        subprocess.run(
            [
                sys.executable, "-m", "pipelinentl.correct_points",
                "--input_points_dir", str(filtered_points_dir),
                "--flow_dir", str(flow_dir),
                "--geo_dir", str(geo_dir),
                "--output_dir", str(corrected_points_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--crop_x_start", str(flow_crop_x_start),
                "--crop_x_end", str(flow_crop_x_end),
                "--crop_y_start", str(flow_crop_y_start),
                "--crop_y_end", str(flow_crop_y_end),
            ],
            check=True,
        )

        if not is_complete(corrected_points_dir, "*.points", expected_corrected, "puntos corregidos"):
            raise RuntimeError("correct_points terminó, pero faltan puntos corregidos.")

        write_json(corrected_config_file, corrected_config)
        force_downstream = True
    else:
        print("   Se omite correct_points: puntos corregidos suficientes y configuración válida.")

    n_corrected_points = count_nonempty(corrected_points_dir, "*.points")
    print(f"   Frames con puntos corregidos: {n_corrected_points}/{n_images}")

    if n_corrected_points == 0:
        raise RuntimeError("No hay puntos corregidos. No se puede hacer segunda georreferenciación.")

    # ============================================================
    # 13. SEGUNDA GEORREFERENCIACIÓN
    # ============================================================

    if second_georef_mode == "none":
        print("\nPipeline completada con flujo óptico, sin segunda georreferenciación.")
        return

    geo_corrected_dir.mkdir(parents=True, exist_ok=True)

    if second_georef_mode == "full":
        print(f"[{step}/{total_steps}] Segunda georreferenciación completa con puntos corregidos...")

        geo_corrected_config = {
            **common_config,
            "step": "georef_timelapse_corrected_full",
            "points_dir": str(corrected_points_dir),
            "georef_plot_every": georef_plot_every,
            "n_corrected_points": n_corrected_points,
        }
        geo_corrected_config_file = geo_corrected_dir / "_geo_corrected_config.json"

        run_geo_corrected = step_should_run(
            folder=geo_corrected_dir,
            pattern="*_rect.tiff",
            expected=n_corrected_points,
            label="GeoTIFFs segunda georreferenciación",
            config_file=geo_corrected_config_file,
            current_config=geo_corrected_config,
            force_downstream=force_downstream,
            force_this_step=rerun_second_georef_if_exists,
        )

        if run_geo_corrected:
            remove_files(geo_corrected_dir, "*_rect.tiff", "GeoTIFFs segunda georreferenciación")
            remove_files(geo_corrected_dir, "*.png", "plots segunda georreferenciación")
            geo_corrected_config_file.unlink(missing_ok=True)

            subprocess.run(
                [
                    sys.executable, "-m", "pipelinentl.georef_timelapse",
                    "--input_dir", str(pics_dir),
                    "--points_dir", str(corrected_points_dir),
                    "--output_dir", str(geo_corrected_dir),
                    "--start_id", str(start_id),
                    "--end_id", str(end_id),
                    "--plot_every", str(georef_plot_every),
                ],
                check=True,
            )

            if not is_complete(
                geo_corrected_dir,
                "*_rect.tiff",
                n_corrected_points,
                "GeoTIFFs segunda georreferenciación",
            ):
                raise RuntimeError("Segunda georreferenciación terminó, pero faltan GeoTIFFs.")

            write_json(geo_corrected_config_file, geo_corrected_config)

        print("\nPipeline completada con segunda georreferenciación COMPLETA.")
        return

    if second_georef_mode == "sample":
        print(f"[{step}/{total_steps}] Segunda georreferenciación de muestra con puntos corregidos...")

        available_corrected_ids = ids_from_points_dir(corrected_points_dir, start_id, end_id)
        sample_ids = sample_ids_from_available(available_corrected_ids, n_samples=second_georef_samples)

        if not sample_ids:
            raise RuntimeError("No hay IDs corregidos disponibles para la georreferenciación de muestra.")

        print(f"   IDs de muestra ({len(sample_ids)}): {sample_ids}")

        sample_config = {
            **common_config,
            "step": "georef_timelapse_corrected_sample",
            "points_dir": str(corrected_points_dir),
            "sample_ids": sample_ids,
            "n_corrected_points": n_corrected_points,
        }
        sample_config_file = geo_corrected_dir / "_geo_corrected_sample_config.json"

        old_sample_config = read_json(sample_config_file)
        sample_config_ok = configs_equal(old_sample_config, sample_config)

        sample_complete = all(geo_file_for_id_exists(geo_corrected_dir, sid) for sid in sample_ids)

        if (
            rerun_second_georef_if_exists
            or force_downstream
            or not sample_complete
            or not sample_config_ok
        ):
            print("   Se recalculará la segunda georreferenciación de muestra.")

            for sid in sample_ids:
                for p in geo_corrected_dir.glob(f"*{sid}*"):
                    if p.is_file():
                        try:
                            p.unlink()
                        except Exception as e:
                            print(f"      AVISO: no se pudo borrar {p}: {e}")

                print(f"   - Georreferenciando muestra ID {sid}")
                subprocess.run(
                    [
                        sys.executable, "-m", "pipelinentl.georef_timelapse",
                        "--input_dir", str(pics_dir),
                        "--points_dir", str(corrected_points_dir),
                        "--output_dir", str(geo_corrected_dir),
                        "--start_id", str(sid),
                        "--end_id", str(sid),
                        "--plot_every", "1",
                    ],
                    check=True,
                )

            sample_complete = all(geo_file_for_id_exists(geo_corrected_dir, sid) for sid in sample_ids)
            if not sample_complete:
                raise RuntimeError("La georreferenciación de muestra terminó, pero falta algún GeoTIFF.")

            write_json(sample_config_file, sample_config)
        else:
            print("   Se omite segunda georreferenciación de muestra: outputs completos y configuración válida.")

        print("\nPipeline completada con segunda georreferenciación de MUESTRA.")
        return

    raise ValueError(
        f"second_georef_mode inválido: {second_georef_mode}. "
        "Debe ser 'none', 'full' o 'sample'."
    )


if __name__ == "__main__":
    main()