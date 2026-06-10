#!/usr/bin/env python3
"""
Proyección de píxeles ISS → Tierra usando la simulación de Blender (v3).

Versión pensada para integrarse con la pipeline:

- Ángulos (yaw/pitch/roll) vienen de la pipeline (fijados o de angle_search).
- Parámetros de cámara (focal, sensor_width, sensor_height, pixel_width, pixel_height)
  también vienen de la pipeline (a partir de los EXIF de la primera imagen).
- Rango temporal (start_date / end_date) viene de la pipeline, ya con offsets aplicados.

Para cada instante de tiempo:
1. Busca el CSV correspondiente a ese timestamp.
2. Lee la imagen real correspondiente.
3. Reconstruye la cámara en Blender con los parámetros dados.
4. Traza rayos desde cada píxel simulado hacia la esfera-Tierra.
5. Intersecta con la esfera y genera archivos .points (QGIS) para real y simulada.

Uso típico desde la pipeline:

    python -m pipelinentl.project_timelapse \
        --output_directory ... \
        --texture_path ... \
        --csv_dir ... \
        --image_dir ... \
        --tle_directory ... \
        --yaw ... --pitch ... --roll ... \
        --focal_length ... \
        --sensor_width ... --sensor_height ... \
        --pixel_width ... --pixel_height ... \
        --start_date ... --end_date ... \
        --time_step ... \
        --points_mode real|simulated|both \
        --orientation_mode north|forward
"""

import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2

from .iss_simulation import (
    reset_scene,
    list_tle_files,
    read_tle_from_files,
    find_closest_tle,
    check_tle_validity,
    get_iss_position_and_velocity,
    creaimagen,
    project_pixels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesar timelapse ISS: proyectar píxeles y generar .points"
    )
    parser.add_argument(
        '--output_directory', type=str, required=True,
        help='Directorio para imágenes renderizadas y archivos .points'
    )
    parser.add_argument(
        '--texture_path', type=str, required=True,
        help='Ruta a la textura de la Tierra (para reset_scene)'
    )
    parser.add_argument(
        '--csv_dir', type=str, required=True,
        help='Directorio con archivos CSV de coordenadas transformadas (match_timelapse)'
    )
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Directorio con imágenes reales (pics)'
    )
    parser.add_argument(
        '--tle_directory', type=str, required=True,
        help='Directorio con archivos TLE de la ISS'
    )

    # Ángulos de la cámara (ya ajustados o fijos)
    parser.add_argument('--yaw', type=float, required=True, help='Ángulo yaw en grados')
    parser.add_argument('--pitch', type=float, required=True, help='Ángulo pitch en grados')
    parser.add_argument('--roll', type=float, required=True, help='Ángulo roll en grados')

    # Parámetros de la cámara (desde EXIF / pipeline)
    parser.add_argument(
        '--focal_length', type=float, required=True,
        help='Longitud focal (mm)'
    )
    parser.add_argument(
        '--sensor_width', type=float, required=True,
        help='Ancho de sensor (mm)'
    )
    parser.add_argument(
        '--sensor_height', type=float, required=True,
        help='Alto de sensor (mm)'
    )
    parser.add_argument(
        '--pixel_width', type=int, required=True,
        help='Resolución horizontal del render (px)'
    )
    parser.add_argument(
        '--pixel_height', type=int, required=True,
        help='Resolución vertical del render (px)'
    )

    # Rango temporal (ya calculado en la pipeline a partir de EXIF + offset)
    parser.add_argument(
        '--start_date', type=str, required=True,
        help='Fecha inicio UTC en formato ISO 8601 (ej: 2017-09-13T21:44:33+00:00)'
    )
    parser.add_argument(
        '--end_date', type=str, required=True,
        help='Fecha fin UTC en formato ISO 8601'
    )
    parser.add_argument(
        '--time_step', type=float, default=0.5,
        help='Intervalo temporal entre frames (segundos)'
    )

    parser.add_argument(
        '--points_mode', type=str, default='real',
        choices=['real', 'simulated', 'both'],
        help="Qué .points conservar: solo 'real', solo 'simulated' o 'both'"
    )
    parser.add_argument(
        '--orientation_mode', type=str, default='forward',
        choices=['north', 'forward'],
        help="Modo de orientación de la cámara ('north' o 'forward')"
    )

    return parser.parse_args()

def parse_time_from_render_filename(render_filename: str) -> datetime:
    """
    Extrae el timestamp desde un nombre tipo:

    render_output_2022-09-05T00-20-44-965_L50-25_G22-94_...
    """
    stem = Path(render_filename).stem

    prefix = "render_output_"
    if not stem.startswith(prefix):
        raise ValueError(f"Nombre de render no esperado: {render_filename}")

    rest = stem[len(prefix):]
    timestamp_str = rest.split("_L", 1)[0]

    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S-%f")
    return dt.replace(tzinfo=timezone.utc)


def find_csv_for_render(csv_files: list[str], render_filename: str) -> str | None:
    """
    Busca el CSV exacto asociado a un render.

    Si render_filename es:
      render_output_XXX.png

    busca:
      transformed_coordinates_render_output_XXX.csv
    """
    render_stem = Path(render_filename).stem
    exact_name = f"transformed_coordinates_{render_stem}.csv"

    if exact_name in csv_files:
        return exact_name

    # Fallback por prefijo, por si hay pequeñas variantes.
    expected_prefix = f"transformed_coordinates_{render_stem}"
    return next((f for f in csv_files if f.startswith(expected_prefix)), None)


def image_id_from_filename(image_filename: str) -> str:
    """
    Devuelve ISS067-E-327041 a partir de ISS067-E-327041.JPG.
    """
    return Path(image_filename).stem


def main():
    args = parse_args()

    output_directory = Path(args.output_directory)
    csv_dir = Path(args.csv_dir)
    image_dir = Path(args.image_dir)
    tle_directory = Path(args.tle_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    # Listar archivos
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    ])

    if not image_files:
        print(f"No se encontraron imágenes en {image_dir}")
        sys.exit(1)

    # Parsear fechas (start/end ya son UTC en la pipeline)
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    time_step = timedelta(seconds=args.time_step)

    # Reiniciar escena y leer TLEs
    print("🔁 Reiniciando escena de Blender y cargando TLEs...")
    sphere = reset_scene(earth_radius=10, texture_path=args.texture_path)
    file_paths = list_tle_files(str(tle_directory))
    tle_data = read_tle_from_files(file_paths)

    # Parámetros de cámara (ya vienen de la pipeline)
    focal_length  = float(args.focal_length)
    sensor_width  = float(args.sensor_width)
    sensor_height = float(args.sensor_height)
    pixel_width   = int(args.pixel_width)
    pixel_height  = int(args.pixel_height)

    print(f"Usando cámara:")
    print(f"  focal_length = {focal_length} mm")
    print(f"  sensor       = {sensor_width} x {sensor_height} mm")
    print(f"  resolución   = {pixel_width} x {pixel_height} px")
    print(f"Ángulos:")
    print(f"  yaw   = {args.yaw}")
    print(f"  pitch = {args.pitch}")
    print(f"  roll  = {args.roll}")
    print(f"orientation_mode = {args.orientation_mode}")

    current_time = start_date
    image_index = 0

    # Loop temporal: igual filosofía que generate_timelapse
    # Renders simulados ya existentes. Deben ser los mismos que usó match_timelapse.
    render_files = sorted([
        f for f in os.listdir(output_directory)
        if f.startswith("render_output_") and f.lower().endswith(".png")
    ])

    if not render_files:
        raise RuntimeError(f"No se encontraron renders simulados en {output_directory}")

    n_pairs = min(len(render_files), len(image_files))

    if len(render_files) != len(image_files):
        print(
            f"⚠️ Número distinto de renders e imágenes reales: "
            f"{len(render_files)} renders vs {len(image_files)} imágenes. "
            f"Se usarán {n_pairs} pares por orden, igual que match_timelapse.py."
        )

    csv_set = set(csv_files)

    processed = 0
    skipped_no_csv = 0
    skipped_bad_image = 0

    # Emparejar exactamente igual que match_timelapse.py:
    # render_files ordenados ↔ image_files ordenadas.
    for image_index, (render_file, image_file) in enumerate(
        zip(render_files[:n_pairs], image_files[:n_pairs])
    ):
        csv_file = find_csv_for_render(csv_files, render_file)

        if csv_file is None:
            print(
                f"⚠️ No se encontró CSV para render {render_file}, "
                f"imagen real {image_file}. Saltando frame."
            )
            skipped_no_csv += 1
            continue

        try:
            current_time = parse_time_from_render_filename(render_file)
        except Exception as e:
            print(f"⚠️ No se pudo extraer timestamp de {render_file}: {e}. Saltando frame.")
            skipped_no_csv += 1
            continue

        csv_path = csv_dir / csv_file
        image_path = image_dir / image_file

        real_photo = cv2.imread(str(image_path))
        if real_photo is None:
            print(f"⚠️ No se pudo leer la imagen {image_path}, saltando frame.")
            skipped_bad_image += 1
            continue

        real_photo_height = real_photo.shape[0]

        timestamp_str = current_time.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

        print(f"[Procesando] idx={image_index}, tiempo={timestamp_str}")
        print(f"   Render: {render_file}")
        print(f"   CSV:    {csv_file}")
        print(f"   Imagen: {image_file} (altura real = {real_photo_height}px)")

        closest_tle = find_closest_tle(tle_data, current_time)
        check_tle_validity(closest_tle, current_time)

        latitude, longitude, altitude, v_icrf, v_itrs = get_iss_position_and_velocity(
            closest_tle,
            current_time,
        )
        velocity = v_itrs

        observation_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        before_points = set(
            f for f in os.listdir(output_directory) if f.endswith(".points")
        )

        camera, _ = creaimagen(
            latitude, longitude, altitude,
            args.yaw, args.pitch, args.roll,
            velocity, sphere,
            focal_length, sensor_width, sensor_height,
            pixel_width, pixel_height,
            observation_time_str, str(output_directory), 10,
            render_image=False,
            orientation_mode=args.orientation_mode,
        )

        project_pixels(
            str(csv_path),
            latitude, longitude, altitude,
            args.yaw, args.pitch, args.roll,
            velocity, sphere, camera,
            real_photo_height, observation_time_str,
            str(output_directory), 10,
        )

        after_points = set(
            f for f in os.listdir(output_directory) if f.endswith(".points")
        )
        new_points = after_points - before_points

        # Renombrar los puntos nuevos para incluir el ID real de imagen.
        # Esto evita descuadres posteriores si faltan algunos frames.
        img_id = image_id_from_filename(image_file)

        renamed_points = set()
        for fname in new_points:
            src = output_directory / fname

            if fname.endswith("_real.points"):
                dst = output_directory / f"{img_id}_real.points"
            elif fname.endswith("_simulated.points"):
                dst = output_directory / f"{img_id}_simulated.points"
            else:
                renamed_points.add(fname)
                continue

            try:
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
                renamed_points.add(dst.name)
            except Exception as e:
                print(f"⚠️ No se pudo renombrar {fname} -> {dst.name}: {e}")
                renamed_points.add(fname)

        new_points = renamed_points

        # Filtrar según points_mode
        if args.points_mode == "real":
            for fname in new_points:
                if fname.endswith("_simulated.points"):
                    try:
                        os.remove(output_directory / fname)
                    except Exception as e:
                        print(f"⚠️ No se pudo borrar {fname}: {e}")

        elif args.points_mode == "simulated":
            for fname in new_points:
                if fname.endswith("_real.points"):
                    try:
                        os.remove(output_directory / fname)
                    except Exception as e:
                        print(f"⚠️ No se pudo borrar {fname}: {e}")

        processed += 1
        print(
            f"✔ Frame procesado: idx={image_index}, "
            f"imagen={image_file}, CSV={csv_file}"
        )

    print("✅ Todos los pares render-imagen han sido revisados.")
    print(f"   Frames proyectados: {processed}")
    print(f"   Sin CSV:            {skipped_no_csv}")
    print(f"   Imagen ilegible:    {skipped_bad_image}")
    print("✅ Todos los frames del timelapse han sido procesados.")


if __name__ == "__main__":
    main()
