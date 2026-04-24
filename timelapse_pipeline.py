#!/usr/bin/env python3
"""
Pipeline completa timelapse ISS (scripts_v3) con opción de:

- Búsqueda automática de ángulos (yaw/pitch) con angle_search.
- Flujo óptico ISS–VIIRS y corrección de puntos.
- Segunda georreferenciación completa o solo de muestra.

Pasos base (sin flujo óptico):
1) Descargar imágenes originales ISS (get_pics.py).
2) (Opcional) Buscar yaw/pitch con angle_search.py.
3) Generar timelapse simulado en Blender (generate_timelapse.py).
4) Matching entre simuladas y reales (match_timelapse.py).
5) Proyección de píxeles → .points (project_timelapse.py).
6) Filtrado + renombrado de puntos (filter_points.py).
7) 1ª georreferenciación del timelapse ISS (georef_timelapse.py).

Si use_optical_flow = True, además:
8) Recorte/alineado VIIRS (viirs_roi_crop.py).
9) Flujo óptico ISS–VIIRS (optical_flow.py).
10) Corrección de puntos con el flujo (correct_points.py).
11) 2ª georreferenciación usando puntos corregidos (georef_timelapse.py),
    que puede ser:
      - completa (second_georef_mode="full")
      - solo unas imágenes de comprobación (second_georef_mode="sample")
      - o no hacerse (second_georef_mode="none").
"""

import os

# 🔧 IMPORTANTE: forzar Qt a modo "offscreen" para que cv2/matplotlib no revienten
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import json
from pathlib import Path
from datetime import timedelta
from argparse import Namespace
import subprocess

# --- IMPORTS de scripts_v3 ---

from scripts_v3.get_pics import download_all_images
from scripts_v3.generate_timelapse import (
    extract_exif_data,
    get_image_files,
)
from scripts_v3 import generate_timelapse
from scripts_v3 import angle_search  # para search_best_yaw_pitch
from scripts_v3.iss_simulation import (
    list_tle_files,
    read_tle_from_files,
)


def main():
    # ============================================================
    # 1. CONFIGURACIÓN GENERAL DEL EXPERIMENTO
    # ============================================================

    # --- IDENTIFICACIÓN DEL TIMELAPSE ---
    mission  = "ISS067"
    start_id = 366800
    end_id   = 366816

    # Carpeta base del experimento, estilo: ISS053-E-462550-462560
    base_dir = Path(f"{mission}-E-{start_id}-{end_id}")
    pics_dir = base_dir / "pics"        # Imágenes ISS originales
    output_dir = base_dir / "output"    # Renders simulados + .points sin filtrar
    matches_output_dir = output_dir / "matches"  # CSVs del matching

    search_output_dir = base_dir / "search_angles" #NUEVO: renders de angle_search

    filtered_points_dir   = base_dir / "filtered_points"
    geo_dir               = base_dir / "geo"               # 1ª georreferenciación
    viirs_output_dir      = base_dir / "viirs_cropped_aligned"
    flow_dir              = base_dir / "flow"
    corrected_points_dir  = base_dir / "corrected_points"
    geo_corrected_dir     = base_dir / "geo_corrected"     # 2ª georreferenciación

    tle_dir = Path("/home/rpz/iss_simulation/ISS_tle")
    texture_path = (
        "/home/rpz/iss_simulation/VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.sqrt.full.40k_20k.png"
    )
    viirs_tiff_path = (
        "/home/rpz/iss_simulation/VNL_v2_npp_2021_global_vcmslcfg_c202203152300.median_masked.tif"
    )

    earth_radius = 10.0

        # --- CONTROL: usar o no angle_search ---
    use_angle_search = True
    reuse_cached_angles = True

    # ¿Recalcular SIEMPRE la simulación aunque ya existan renders?
    rerun_simulation_if_exists = False

    # --- CONTROL: usar o no flujo óptico + segunda georef ---
    use_optical_flow = True

    # "none"  → solo generar corrected_points y terminar sin 2ª georef
    # "full"  → 2ª georreferenciación de TODO el timelapse
    # "sample"→ 2ª georef solo unas imágenes de comprobación
    second_georef_mode = "sample"

    # --- ÁNGULOS DE LA CÁMARA (fallback) ---
    yaw = 0.0
    pitch = 60.0
    roll = 0.0

    # Modo de orientación
    orientation_mode = "forward"

    # Paso temporal del timelapse simulado
    delta = 2.0 / 3.0

    # Offsets temporales
    time_offset_seconds = 0.0
    time_offset_minutes = 0.0
    time_offset_hours = 0.0

    # ============================================================
    # PARÁMETROS DE BÚSQUEDA DE ÁNGULOS (angle_search_v2)
    # ============================================================

    # Rangos
    yaw_range = (-20.0, 20.0)
    pitch_range = (40.0, 70.0)
    roll_range = (-15.0, 15.0)

    # Búsqueda gruesa
    coarse_steps = (10.0, 5.0, 5.0)      # (yaw, pitch, roll)

    # Refinado intermedio alrededor de top-k
    fine_steps = (2.5, 1.5, 1.0)
    fine_windows = (5.0, 4.0, 3.0)

    # Refinado final
    refine_steps = (1.0, 0.5, 0.5)
    refine_windows = (2.0, 1.0, 1.0)

    # Sensor físico
    sensor_width = 36.0
    sensor_height = 28.0
    # ============================================================
    # CALCULAR NÚMERO TOTAL DE PASOS PARA LOS PRINTS
    # ============================================================

    # Pasos base (hasta 1ª georreferenciación)
    total_steps = 7
    # Añadimos pasos extra si hay flujo óptico
    if use_optical_flow:
        total_steps += 3  # viirs_roi_crop, optical_flow, correct_points
        if second_georef_mode != "none":
            total_steps += 1  # 2ª georreferenciación (full o sample)

    step = 1  # contador de pasos

    # ============================================================
    # 2. DESCARGA DE IMÁGENES (get_pics.py)
    # ============================================================
    pics_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶️  [{step}/{total_steps}] Descargando imágenes ISS...")
    step += 1    

    download_all_images(mission, start_id, end_id, pics_dir)

    # ============================================================
    # 3. EXTRAER RANGO TEMPORAL (EXIF) PARA START/END
    # ============================================================
    img_files = get_image_files(pics_dir)
    if not img_files:
        raise RuntimeError("No se encontraron imágenes en la carpeta de fotos descargadas.")

    first_img = img_files[0]
    last_img  = img_files[-1]

    print(f"Primera imagen: {first_img}")
    print(f"Última imagen:  {last_img}")

    start_dt, focal_length, pixel_width, pixel_height = extract_exif_data(pics_dir / first_img)
    end_dt,   _,            _,           _            = extract_exif_data(pics_dir / last_img)

    total_offset = timedelta(
        seconds=time_offset_seconds,
        minutes=time_offset_minutes,
        hours=time_offset_hours,
    )

    start_dt = start_dt + total_offset
    end_dt   = end_dt + total_offset

    print("Rango temporal (tras offset):")
    print(f"  start_dt = {start_dt}")
    print(f"  end_dt   = {end_dt}")

    # ============================================================
    # 4. (OPCIONAL) BÚSQUEDA DE YAW/PITCH CON angle_search.py
    #    + CACHEO DE RESULTADOS
    # ============================================================
        # ============================================================
    # 4. (OPCIONAL) BÚSQUEDA DE YAW/PITCH/ROLL CON angle_search_v2.py
    #    + CACHEO DE RESULTADOS
    # ============================================================
    if use_angle_search:
        cache_file = base_dir / "angle_search_results.txt"

        # 4.1. Si queremos reutilizar y ya existe un resultado guardado → cargarlo
        if reuse_cached_angles and cache_file.exists():
            print(f"▶️  [{step}/{total_steps}] Cargando yaw/pitch/roll desde cache (angle_search_results.txt)...")
            with cache_file.open("r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            cached = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    cached[k.strip()] = v.strip()

            try:
                yaw   = float(cached.get("yaw"))
                pitch = float(cached.get("pitch"))
                roll  = float(cached.get("roll"))

                print(f"   yaw (cache)   = {yaw}")
                print(f"   pitch (cache) = {pitch}")
                print(f"   roll (cache)  = {roll}")

                if "score" in cached:
                    print(f"   score (cache) = {cached['score']}")

                reuse_from_cache = True

            except Exception as e:
                print(f"⚠️ Error leyendo cache de ángulos ({e}), recalculando con angle_search_v2...")
                cache_file.unlink(missing_ok=True)
                reuse_from_cache = False

        else:
            reuse_from_cache = False

        # 4.2. Si no hay cache usable → ejecutar angle_search_v2 y guardar resultado
        if not reuse_from_cache:
            print(f"▶️  [{step}/{total_steps}] Buscando yaw/pitch/roll óptimos con angle_search_v2...")

            real_image_path = pics_dir / first_img
            obs_time = start_dt  # ya contiene offset

            # Asegúrate de que la carpeta existe
            search_output_dir.mkdir(parents=True, exist_ok=True)

            best_angles, search_details = angle_search.search_best_yaw_pitch_roll(
                real_image_path=str(real_image_path),
                obs_time=obs_time,
                search_output_dir=str(search_output_dir),
                tle_dir=str(tle_dir),
                texture_path=texture_path,

                # Parámetros de cámara extraídos del EXIF + sensores físicos
                focal_length=focal_length,
                sensor_width=sensor_width,
                sensor_height=sensor_height,
                pixel_width=pixel_width,
                pixel_height=pixel_height,

                # Rangos de búsqueda
                yaw_range=yaw_range,
                pitch_range=pitch_range,
                roll_range=roll_range,

                # Resoluciones de búsqueda
                coarse_steps=coarse_steps,
                fine_steps=fine_steps,
                fine_windows=fine_windows,
                refine_steps=refine_steps,
                refine_windows=refine_windows,

                # Parámetros de orientación
                orientation_mode=orientation_mode,
                earth_radius=earth_radius,
            )

            if best_angles is not None:
                yaw   = float(best_angles["yaw"])
                pitch = float(best_angles["pitch"])
                roll  = float(best_angles["roll"])
                score = best_angles.get("score", None)

                print("\n✅ Mejores ángulos encontrados por angle_search_v2:")
                print(f"   yaw   = {yaw}")
                print(f"   pitch = {pitch}")
                print(f"   roll  = {roll}")
                print(f"   score = {score}")

                # Guardar a cache
                with cache_file.open("w") as f:
                    f.write(f"yaw={yaw}\n")
                    f.write(f"pitch={pitch}\n")
                    f.write(f"roll={roll}\n")
                    if score is not None:
                        f.write(f"score={score}\n")

                # Opcional: guardar resumen JSON adicional
                summary_json = base_dir / "angle_search_best_result.json"
                try:
                    with summary_json.open("w") as f:
                        json.dump(best_angles, f, indent=2)
                except Exception as e:
                    print(f"⚠️ No se pudo guardar {summary_json.name}: {e}")

            else:
                print("⚠️ angle_search_v2 no devolvió ningún resultado válido. "
                      "Usando yaw/pitch/roll por defecto.")
    else:
        print(f"ℹ️  [{step}/{total_steps}] Búsqueda de ángulos desactivada. "
              f"Usando yaw={yaw}, pitch={pitch}, roll={roll}.")
    
    step += 1

    # ============================================================
    # 5. GENERAR TIMELAPSE SIMULADO (generate_timelapse.py)
    #    (con detección de renders ya existentes)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Generando simulación timelapse en Blender...")
    step += 1

    # Mirar si ya hay renders simulados tipo 'render_output_*.png'
    existing_renders = []
    if output_dir.exists():
        existing_renders = [
            f for f in os.listdir(output_dir)
            if f.startswith("render_output_") and f.lower().endswith(".png")
        ]

    if existing_renders and not rerun_simulation_if_exists:
        print(f"ℹ️ Se han encontrado {len(existing_renders)} renders en {output_dir}.")
        print("ℹ️ Se omite la generación de timelapse (se usan imágenes simuladas ya existentes).")
    else:
        if existing_renders and rerun_simulation_if_exists:
            print(f"⚠️ Hay {len(existing_renders)} renders existentes, pero 'rerun_simulation_if_exists=True',")
            print("⚠️ por lo que se volverá a generar la simulación.")

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

    # ============================================================
    # 6. MATCHING REAL–SIMULADA (match_timelapse.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Comparando imágenes reales y simuladas (matches)...")
    step += 1

    matches_output_dir.mkdir(parents=True, exist_ok=True)

    # Comprobar si ya existen CSV de matches
    existing_matches = list(matches_output_dir.glob("transformed_coordinates_*.csv"))

    if existing_matches:
        print(f"   ➜ Se encontraron {len(existing_matches)} archivos de matches en {matches_output_dir}.")
        print("   ➜ Se asume que el paso de matching ya está hecho, se omite esta etapa.")
    else:
        print("   ➜ No se encontraron CSV de matches, ejecutando scripts_v3.match_timelapse...")
        subprocess.run(
            [
                sys.executable, "-m", "scripts_v3.match_timelapse",
                "--output_dir", str(output_dir),
                "--pictures_dir", str(pics_dir),
                "--matches_output_dir", str(matches_output_dir),
                "--grid_step", "185",   # como en tu script original
                "--show_every", "50",
            ],
            check=True,
    )


    # ============================================================
    # 7. PROYECCIÓN DE PÍXELES → .points (project_timelapse.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Proyectando píxeles y generando archivos .points...")
    step += 1

    # En project_timelapse.py:
    #   --points_mode : 'real' | 'simulated' | 'both'
    # Aquí usamos solo los puntos de las imágenes reales; cambia a "both"
    # cuando quieras tener también los de las simuladas.
    points_mode = "real"

    # Comprobar si ya existen .points de las imágenes reales
    existing_real_points = list(output_dir.glob("*_real.points"))

    if existing_real_points:
        print(f"   ℹ️ Encontrados {len(existing_real_points)} archivos *_real.points en {output_dir}.")
        print("   ℹ️ Se omite project_timelapse.py y se reutilizan los .points existentes.")
    else:
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

                # 🔴 NUEVO: parámetros de cámara desde EXIF / pipeline
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


    # ============================================================
    # 8. FILTRADO + RENOMBRADO DE .points (filter_points.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Filtrando y renombrando puntos...")
    step += 1

    filtered_points_dir.mkdir(parents=True, exist_ok=True)

    # Comprobamos si ya hay .points filtrados para todo el rango
    def extract_id_from_filename(name: str) -> int | None:
        """
        Espera nombres del tipo: ISS067-E-327360.points
        """
        try:
            return int(name.split("-")[-1].split(".")[0])
        except Exception:
            return None

    existing_filtered = [
        f for f in filtered_points_dir.glob("*.points")
        if (extract_id_from_filename(f.name) is not None)
        and (start_id <= extract_id_from_filename(f.name) <= end_id)
    ]

    expected_count = end_id - start_id + 1

    if len(existing_filtered) == expected_count:
        print(f"   ℹ️ Encontrados {len(existing_filtered)}/{expected_count} archivos .points filtrados.")
        print("   ℹ️ Se omite filter_points.py y se reutilizan los archivos existentes.")
    else:
        subprocess.run(
            [
                sys.executable, "-m", "scripts_v3.filter_points",
                "--input_folder", str(output_dir),
                "--output_folder", str(filtered_points_dir),
                "--radius_km", "80",          # tu valor típico
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--mission", mission,          # 🔴 IMPORTANTE: añadir esto
            ],
            check=True,
        )

    # ============================================================
    # 9. 1ª GEORREFERENCIACIÓN IMÁGENES ISS (georef_timelapse.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Georreferenciando imágenes ISS (1ª pasada)...")
    step += 1

    geo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable, "-m", "scripts_v3.georef_timelapse",
            "--input_dir", str(pics_dir),
            "--points_dir", str(filtered_points_dir),
            "--output_dir", str(geo_dir),
            "--start_id", str(start_id),
            "--end_id", str(end_id),
            "--plot_every", "50",
        ],
        check=True,
    )

    # ============================================================
    # SI NO HAY FLUJO ÓPTICO → TERMINAR AQUÍ
    # ============================================================
    if not use_optical_flow:
        print("\n✅ Pipeline completada SIN flujo óptico (termina tras la 1ª georreferenciación).")
        return

    # ============================================================
    # 10. RECORTE Y ALINEADO VIIRS (viirs_roi_crop.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Recortando y alineando VIIRS al timelapse georreferenciado...")
    step += 1

    viirs_output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable, "-m", "scripts_v3.viirs_roi_crop",
            "--geo_dir", str(geo_dir),
            "--viirs_tiff", str(viirs_tiff_path),
            "--output_dir", str(viirs_output_dir),
            "--start_id", str(start_id),
            "--end_id", str(end_id),
            "--nproc", "8",
            "--mode", "fast",

            # 🔥 CLAVE: ROI por GCPs
            "--roi_mode", "gcp",
            "--roi_margin_px", "10",

            # 🔥 CLAVE: que el VIIRS quede EXACTO al grid del ROI
            "--align", "roi_exact",
            "--resampling", "bilinear",

            "--threads", "auto",
        ],
        check=True,
    )

    # ============================================================
    # 11. FLUJO ÓPTICO ISS–VIIRS (optical_flow.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Calculando flujo óptico ISS–VIIRS...")
    step += 1

    flow_dir.mkdir(parents=True, exist_ok=True)

    # Comprobamos si ya existe al menos un archivo de flujo para este timelapse
    first_flow_file = flow_dir / f"{mission}-E-{start_id}_flow.npy"

    if first_flow_file.exists():
        print(f"🟡 Flujo óptico ya existente (ejemplo: {first_flow_file.name}).")
        print("    → Se salta el recálculo de optical_flow.py y se usan los .npy existentes.")
    else:
        print("ℹ️  No se han encontrado archivos de flujo. Lanzando optical_flow.py...")
        subprocess.run(
            [
                sys.executable, "-m", "scripts_v3.optical_flow",
                "--geo_dir", str(geo_dir),
                "--viirs_dir", str(viirs_output_dir),
                "--flow_dir", str(flow_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--plot_every", "50",

                # crop_* SOLO como fallback si faltase roi.json (normalmente no hace falta)
                "--crop_x_start", "0.0",
                "--crop_x_end",   "1.0",
                "--crop_y_start", "0.0",
                "--crop_y_end",   "1.0",
            ],
            check=True,
        )


    # ============================================================
    # 12. CORREGIR PUNTOS CON FLUJO (correct_points.py)
    # ============================================================
    print(f"▶️  [{step}/{total_steps}] Corrigiendo puntos con el flujo óptico...")
    step += 1

    corrected_points_dir.mkdir(parents=True, exist_ok=True)

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


    # ============================================================
    # 13. 2ª GEORREFERENCIACIÓN (OPCIONAL, COMPLETA O DE MUESTRA)
    # ============================================================
    if second_georef_mode == "none":
        print("\n✅ Pipeline completada con flujo óptico. "
              "Se han generado corrected_points, pero NO se ha hecho 2ª georreferenciación.")
        return

    if second_georef_mode == "full":
        print(f"▶️  [{step}/{total_steps}] 2ª georreferenciación COMPLETA con puntos corregidos...")
        subprocess.run(
            [
                sys.executable, "-m", "scripts_v3.georef_timelapse",
                "--input_dir", str(pics_dir),
                "--points_dir", str(corrected_points_dir),
                "--output_dir", str(geo_corrected_dir),
                "--start_id", str(start_id),
                "--end_id", str(end_id),
                "--plot_every", "50",
            ],
            check=True,
        )
        print("\n✅ Pipeline completada con 2ª georreferenciación COMPLETA.")
        return

    if second_georef_mode == "sample":
        print(f"▶️  [{step}/{total_steps}] 2ª georreferenciación de MUESTRA con puntos corregidos...")

        geo_corrected_dir.mkdir(parents=True, exist_ok=True)

        # 10 imágenes equiespaciadas (incluye start y end)
        n_samples = 10
        if end_id <= start_id:
            sample_ids = [start_id]
        else:
            step_id = max(1, (end_id - start_id) // (n_samples - 1))
            sample_ids = [start_id + i * step_id for i in range(n_samples - 1)]
            sample_ids.append(end_id)

            # asegurar rango y unicidad por si el rango es pequeño
            sample_ids = sorted({sid for sid in sample_ids if start_id <= sid <= end_id})

            # si por rango pequeño salen menos de 10, rellenar con los siguientes IDs disponibles
            sid = start_id
            while len(sample_ids) < n_samples and sid <= end_id:
                sample_ids.append(sid)
                sample_ids = sorted(set(sample_ids))
                sid += 1

        print(f"   - IDs de muestra ({len(sample_ids)}): {sample_ids}")

        for sid in sample_ids:
            print(f"   - Georreferenciando solo la imagen con ID {sid} (muestra)")
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

        print("\n✅ Pipeline completada con 2ª georreferenciación de MUESTRA.")



if __name__ == "__main__":
    main()