#!/usr/bin/env python3
"""
Pipeline completa timelapse ISS (scripts_v3) con recuperación robusta.

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
from pathlib import Path
from datetime import timedelta
from argparse import Namespace
import subprocess

from scripts_v3.get_pics import download_all_images
from scripts_v3.generate_timelapse import (
    extract_exif_data,
    get_image_files,
)
from scripts_v3 import generate_timelapse
from scripts_v3 import angle_search


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
    try:
        return int(name.split("-")[-1].split(".")[0])
    except Exception:
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
# PIPELINE
# ============================================================

def main():
    # ============================================================
    # 1. CONFIGURACIÓN GENERAL DEL EXPERIMENTO
    # ============================================================

    mission = "ISS067"
    start_id = 327041
    end_id = 328344

    base_dir = Path(f"{mission}-E-{start_id}-{end_id}")

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

    tle_dir = Path("/home/rpz/iss_simulation/ISS_tle")
    texture_path = (
        "/home/rpz/iss_simulation/"
        "VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.sqrt.full.40k_20k.png"
    )
    viirs_tiff_path = (
        "/home/rpz/iss_simulation/"
        "VNL_v2_npp_2021_global_vcmslcfg_c202203152300.median_masked.tif"
    )

    earth_radius = 10.0

    # ------------------------------------------------------------
    # Controles principales
    # ------------------------------------------------------------

    use_angle_search = False
    reuse_cached_angles = True

    rerun_simulation_if_exists = False
    rerun_matching_if_exists = False
    rerun_projection_if_exists = False
    rerun_filtering_if_exists = False
    rerun_first_georef_if_exists = False
    rerun_viirs_if_exists = False
    rerun_optical_flow_if_exists = False
    rerun_correct_points_if_exists = False
    rerun_second_georef_if_exists = False

    use_optical_flow = True

    # "none", "full", "sample"
    second_georef_mode = "sample"

    # Ángulos fallback
    yaw = 12.5
    pitch = 63.5
    roll = -1.0

    orientation_mode = "forward"

    # Offsets temporales
    time_offset_seconds = 0.0
    time_offset_minutes = 0.0
    time_offset_hours = 0.0

    # ------------------------------------------------------------
    # Parámetros de búsqueda de ángulos
    # ------------------------------------------------------------

    yaw_range = (-20.0, 20.0)
    pitch_range = (40.0, 70.0)
    roll_range = (-15.0, 15.0)

    coarse_steps = (10.0, 5.0, 5.0)

    fine_steps = (2.5, 1.5, 1.0)
    fine_windows = (5.0, 4.0, 3.0)

    refine_steps = (1.0, 0.5, 0.5)
    refine_windows = (2.0, 1.0, 1.0)

    # Sensor físico
    sensor_width = 36.0
    sensor_height = 28.0

    # Parámetros matching
    matching_grid_step = 185
    matching_show_every = 50
    matching_min_grid_points = 30
    matching_plot_max_matches = 150
    min_matching_success_ratio = 0.95

    # Parámetros filter_points
    filter_radius_km = 80

    # Parámetros georef
    georef_plot_every = 50

    # Parámetros VIIRS
    viirs_nproc = 8
    viirs_mode = "fast"
    viirs_roi_mode = "gcp"
    viirs_roi_margin_px = 10
    viirs_align = "roi_exact"
    viirs_resampling = "bilinear"

    # Parámetros optical flow
    optical_flow_plot_every = 50

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

                print(f"   yaw   = {yaw}")
                print(f"   pitch = {pitch}")
                print(f"   roll  = {roll}")

                if "score" in cached:
                    print(f"   score = {cached['score']}")

                reuse_from_cache = True

            except Exception as e:
                print(f"AVISO: error leyendo cache de ángulos ({e}). Se recalculará.")
                cache_file.unlink(missing_ok=True)
                reuse_from_cache = False

        if not reuse_from_cache:
            print(f"[{step}/{total_steps}] Buscando yaw/pitch/roll óptimos con angle_search...")
            real_image_path = pics_dir / first_img
            obs_time = start_dt

            search_output_dir.mkdir(parents=True, exist_ok=True)

            best_angles, search_details = angle_search.search_best_yaw_pitch_roll(
                real_image_path=str(real_image_path),
                obs_time=obs_time,
                search_output_dir=str(search_output_dir),
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
            )

            if best_angles is not None:
                yaw = float(best_angles["yaw"])
                pitch = float(best_angles["pitch"])
                roll = float(best_angles["roll"])
                score = best_angles.get("score", None)

                print("Mejores ángulos encontrados:")
                print(f"   yaw   = {yaw}")
                print(f"   pitch = {pitch}")
                print(f"   roll  = {roll}")
                print(f"   score = {score}")

                with cache_file.open("w", encoding="utf-8") as f:
                    f.write(f"yaw={yaw}\n")
                    f.write(f"pitch={pitch}\n")
                    f.write(f"roll={roll}\n")
                    if score is not None:
                        f.write(f"score={score}\n")

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

    match_config = {
        **common_config,
        "step": "match_timelapse",
        "matching_grid_step": matching_grid_step,
        "matching_show_every": matching_show_every,
        "matching_min_grid_points": matching_min_grid_points,
        "matching_plot_max_matches": matching_plot_max_matches,
        "min_matching_success_ratio": min_matching_success_ratio,
        "match_timelapse_version": "ransac_normalized_poly_grid_visualization_v1",
    }
    match_config_file = matches_output_dir / "_match_config.json"

    n_existing_csv = count_nonempty(matches_output_dir, "transformed_coordinates_*.csv")
    min_csv = max(1, math.ceil(min_matching_success_ratio * n_images))

    old_match_config = read_json(match_config_file)
    match_outputs_ok = n_existing_csv >= min_csv

    if rerun_matching_if_exists:
        print("   AVISO matching: forzado por configuración.")
        run_matching = True
    elif force_downstream:
        print("   AVISO matching: se recalculará porque un paso anterior cambió.")
        run_matching = True
    elif not match_outputs_ok:
        print(
            f"   AVISO CSVs de matching: insuficientes "
            f"({n_existing_csv}/{n_images}). Mínimo requerido: {min_csv}."
        )
        run_matching = True
    elif old_match_config is None:
        print(
            f"   AVISO CSVs de matching: suficientes "
            f"({n_existing_csv}/{n_images}), pero falta config. "
            "Se asumirá válido y se escribirá la configuración actual."
        )
        write_json(match_config_file, match_config)
        run_matching = False
    elif not configs_equal(old_match_config, match_config):
        print("   AVISO matching: configuración distinta. Se recalculará.")
        run_matching = True
    else:
        print(
            f"   OK CSVs de matching suficientes: "
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
                sys.executable, "-m", "scripts_v3.match_timelapse",
                "--output_dir", str(output_dir),
                "--pictures_dir", str(pics_dir),
                "--matches_output_dir", str(matches_output_dir),
                "--grid_step", str(matching_grid_step),
                "--show_every", str(matching_show_every),
                "--min_grid_points", str(matching_min_grid_points),
                "--plot_max_matches", str(matching_plot_max_matches),
            ],
            check=True,
        )

        n_csv = count_nonempty(matches_output_dir, "transformed_coordinates_*.csv")

        if n_csv < min_csv:
            raise RuntimeError(
                f"match_timelapse terminó, pero hay pocos CSVs válidos: "
                f"{n_csv}/{n_images}. Mínimo requerido: {min_csv} "
                f"({100 * min_matching_success_ratio:.1f}%)."
            )

        if n_csv < n_images:
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

    if n_valid_csv < min_csv:
        raise RuntimeError(
            f"No hay suficientes CSVs válidos para continuar: "
            f"{n_valid_csv}/{n_images}. Mínimo requerido: {min_csv}."
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
                sys.executable, "-m", "scripts_v3.project_timelapse",
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

        subprocess.run(
            [
                sys.executable, "-m", "scripts_v3.filter_points",
                "--input_folder", str(output_dir),
                "--output_folder", str(filtered_points_dir),
                "--radius_km", str(filter_radius_km),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--mission", mission,
            ],
            check=True,
        )

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
                sys.executable, "-m", "scripts_v3.georef_timelapse",
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
                sys.executable, "-m", "scripts_v3.viirs_roi_crop",
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

                "--threads", "auto",
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
                sys.executable, "-m", "scripts_v3.optical_flow",
                "--geo_dir", str(geo_dir),
                "--viirs_dir", str(viirs_output_dir),
                "--flow_dir", str(flow_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--plot_every", str(optical_flow_plot_every),

                "--crop_x_start", "0.0",
                "--crop_x_end", "1.0",
                "--crop_y_start", "0.0",
                "--crop_y_end", "1.0",
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
                sys.executable, "-m", "scripts_v3.correct_points",
                "--input_points_dir", str(filtered_points_dir),
                "--flow_dir", str(flow_dir),
                "--geo_dir", str(geo_dir),
                "--output_dir", str(corrected_points_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
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
                    sys.executable, "-m", "scripts_v3.georef_timelapse",
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
        sample_ids = sample_ids_from_available(available_corrected_ids, n_samples=10)

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
                        sys.executable, "-m", "scripts_v3.georef_timelapse",
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