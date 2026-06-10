#!/usr/bin/env python3
"""
Generación de timelapse simulado de la ISS a partir de imágenes reales (pics).

Uso típico como script:

    python -m pipelinentl.generate_timelapse \
        --pics /ruta/a/pics \
        --output /ruta/a/output \
        --tle /ruta/a/ISS_tle \
        --texture /ruta/a/textura.png \
        --yaw 0 --pitch 60 --roll 0 \
        --orientation_mode north \
        --delta 1.0

También puede usarse en modo test (solo primer fotograma):

    python -m pipelinentl.generate_timelapse --pics ... --output ... --test

Y desde la pipeline, llamando a main(Args) donde Args es un Namespace.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import exifread
from PIL import Image

from .iss_simulation import (
    reset_scene,
    list_tle_files,
    read_tle_from_files,
    generate_image_series,
    plot_iss_trajectory,
    save_timelapse,
)


# -------------------------------------------------------------------
# UTILIDADES PARA FICHEROS E IMÁGENES
# -------------------------------------------------------------------

def get_image_files(directory: Path):
    """
    Devuelve la lista de ficheros .jpg/.jpeg ordenados alfabéticamente
    en el directorio dado.
    """
    files = sorted(
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg"))
    )
    return files


def extract_exif_data(filepath: Path):
    """
    Extrae:
      - DateTimeOriginal (o Image DateTime) -> datetime (UTC)
      - FocalLength -> float en mm
      - Ancho y alto de imagen -> px

    Si el tamaño no está en EXIF, se usa Pillow para obtenerlo.
    """
    with open(filepath, "rb") as f:
        tags = exifread.process_file(f)

    datetime_original = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    focal_length = tags.get("EXIF FocalLength")

    # Fecha y hora
    dt = None
    if datetime_original:
        dt_str = str(datetime_original)
        dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)

    # Longitud focal
    focal = None
    if focal_length:
        try:
            focal = float(focal_length.values[0].num) / float(focal_length.values[0].den)
        except Exception:
            try:
                focal = float(str(focal_length))
            except Exception:
                focal = None

    # Tamaño de imagen
    image_width = tags.get("EXIF ExifImageWidth") or tags.get("Image ImageWidth")
    image_height = tags.get("EXIF ExifImageHeight") or tags.get("Image ImageLength")

    width = int(str(image_width)) if image_width else None
    height = int(str(image_height)) if image_height else None

    # Si falta width/height, abrir con Pillow
    if width is None or height is None:
        with Image.open(filepath) as img:
            width, height = img.size

    return dt, focal, width, height


# -------------------------------------------------------------------
# MODO TEST: SOLO PRIMER FOTOGRAMA
# -------------------------------------------------------------------

def test_mode(args: argparse.Namespace):
    """
    Genera una única imagen simulada usando solo la primera foto real.
    Sirve para comprobar rápidamente que todo está bien configurado.
    """
    img_dir = Path(args.pics).expanduser()
    img_files = get_image_files(img_dir)
    if not img_files:
        print("No se encontraron imágenes JPG en el directorio.")
        return

    first_img = img_files[0]
    print(f"Modo test: Solo se usará la primera imagen: {first_img}")

    # Extraer datos EXIF de la primera imagen
    start_dt, focal_length, pixel_width, pixel_height = extract_exif_data(img_dir / first_img)

    # Offset temporal
    total_offset = timedelta(
        seconds=getattr(args, "time_offset_seconds", 0),
        minutes=getattr(args, "time_offset_minutes", 0),
        hours=getattr(args, "time_offset_hours", 0),
    )
    if start_dt is not None:
        start_dt = start_dt + total_offset

    if not (start_dt and focal_length and pixel_width and pixel_height):
        print("No se pudo extraer toda la información EXIF necesaria. Aborta (modo test).")
        print(
            f"start_dt: {start_dt}, focal_length: {focal_length}, "
            f"pixel_width: {pixel_width}, pixel_height: {pixel_height}"
        )
        return

    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Escena y TLEs
    sphere = reset_scene(args.earth_radius, args.texture)
    tle_dir = Path(args.tle).expanduser()
    file_paths = list_tle_files(str(tle_dir))
    tle_data = read_tle_from_files(file_paths)

    # Un fotograma: start_dt = end_dt
    latitudes, longitudes = generate_image_series(
        start_date=start_dt,
        end_date=start_dt,
        delta=1.0,
        tle_data=tle_data,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        sphere=sphere,
        focal_length=focal_length,
        sensor_width=36,   # Sensor estándar full-frame
        sensor_height=28,  # Aproximado
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        output_directory=str(output_directory),
        earth_radius=args.earth_radius,
        render_image=True,
        orientation_mode=args.orientation_mode,
    )

    # Opcionalmente, trayectoria y guardar .blend
    plot_iss_trajectory(latitudes, longitudes, str(output_directory))
    save_timelapse(str(output_directory))

    print("Simulación en modo test terminada: se ha generado un solo fotograma.")


# -------------------------------------------------------------------
# MAIN: GENERACIÓN COMPLETA DEL TIMELAPSE
# -------------------------------------------------------------------

def main(args: argparse.Namespace | None = None):
    """
    Punto de entrada principal.

    Si args es None, se parsean argumentos desde línea de comandos.
    Si args es un Namespace (por ejemplo desde la pipeline), se usa tal cual.
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description="Simulación ISS usando parámetros extraídos de imágenes reales (pics)."
        )
        parser.add_argument(
            "--pics",
            type=str,
            required=True,
            help="Carpeta con las imágenes reales (pics).",
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Carpeta de salida para los renders simulados.",
        )
        parser.add_argument(
            "--tle",
            type=str,
            default="/home/raul/planb/ISS_Simulation/ISS_tle",
            help="Directorio con los archivos TLE.",
        )
        parser.add_argument(
            "--texture",
            type=str,
            default="/home/raul/planb/ISS_Simulation/"
                    "VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.sqrt.full.40k_20k.png",
            help="Ruta de la textura nocturna de la Tierra.",
        )
        parser.add_argument(
            "--earth_radius",
            type=float,
            default=10.0,
            help="Radio de la esfera Tierra en unidades Blender.",
        )
        parser.add_argument("--yaw", type=float, default=0.0, help="Yaw inicial (azimut local).")
        parser.add_argument("--pitch", type=float, default=60.0, help="Pitch (off-nadir).")
        parser.add_argument("--roll", type=float, default=0.0, help="Roll (giro del horizonte).")
        parser.add_argument(
            "--orientation_mode",
            choices=["north", "forward"],
            default="north",
            help="Convención de orientación de la cámara.",
        )
        parser.add_argument(
            "--delta",
            type=float,
            default=1.0,
            help="Delta de tiempo (segundos entre imágenes simuladas).",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Genera solo el primer fotograma (modo test rápido).",
        )
        parser.add_argument(
            "--time_offset_seconds",
            type=float,
            default=0.0,
            help="Offset en segundos a sumar/restar a las fechas extraídas de las fotos.",
        )
        parser.add_argument(
            "--time_offset_minutes",
            type=float,
            default=0.0,
            help="Offset en minutos a sumar/restar.",
        )
        parser.add_argument(
            "--time_offset_hours",
            type=float,
            default=0.0,
            help="Offset en horas a sumar/restar.",
        )
        parser.add_argument(
            "--no_plot_trajectory",
            action="store_true",
            help="Si se activa, no se genera el mapa de trayectoria de la ISS.",
        )
        parser.add_argument(
            "--no_save_blend",
            action="store_true",
            help="Si se activa, no se guarda el archivo .blend al final.",
        )

        args = parser.parse_args()

    # Si args viene de la pipeline, ya está construído.

    # MODO TEST
    if getattr(args, "test", False):
        test_mode(args)
        return

    img_dir = Path(args.pics).expanduser()
    img_files = get_image_files(img_dir)
    if not img_files:
        print("No se encontraron imágenes JPG en el directorio.")
        return

    first_img = img_files[0]
    last_img = img_files[-1]

    print(f"Primera imagen: {first_img}")
    print(f"Última imagen: {last_img}")

    # Extraer datos EXIF
    start_dt, focal_length, pixel_width, pixel_height = extract_exif_data(img_dir / first_img)
    end_dt, _, _, _ = extract_exif_data(img_dir / last_img)

    # Offset temporal total
    total_offset = timedelta(
        seconds=getattr(args, "time_offset_seconds", 0),
        minutes=getattr(args, "time_offset_minutes", 0),
        hours=getattr(args, "time_offset_hours", 0),
    )
    if start_dt is not None:
        start_dt = start_dt + total_offset
    if end_dt is not None:
        end_dt = end_dt + total_offset

    if not (start_dt and end_dt and focal_length and pixel_width and pixel_height):
        print("No se pudo extraer toda la información EXIF necesaria. Aborta.")
        print(
            f"start_dt: {start_dt}, end_dt: {end_dt}, focal_length: {focal_length}, "
            f"pixel_width: {pixel_width}, pixel_height: {pixel_height}"
        )
        return

    print(f"Fecha/hora inicio: {start_dt}  |  Fecha/hora fin: {end_dt}")
    print(f"Focal length: {focal_length} mm  |  Tamaño: {pixel_width} x {pixel_height} px")

    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Escena y TLEs
    sphere = reset_scene(args.earth_radius, args.texture)
    tle_dir = Path(args.tle).expanduser()
    file_paths = list_tle_files(str(tle_dir))
    tle_data = read_tle_from_files(file_paths)

    # Generar la serie de imágenes simuladas
    latitudes, longitudes = generate_image_series(
        start_date=start_dt,
        end_date=end_dt,
        delta=args.delta,
        tle_data=tle_data,
        yaw=args.yaw,
        pitch=args.pitch,
        roll=args.roll,
        sphere=sphere,
        focal_length=focal_length,
        sensor_width=36,   # sensor "equivalente" estándar
        sensor_height=28,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        output_directory=str(output_directory),
        earth_radius=args.earth_radius,
        render_image=True,
        orientation_mode=getattr(args, "orientation_mode", "north"),
    )

    # Opcionales (según flags)
    if not getattr(args, "no_plot_trajectory", False):
        plot_iss_trajectory(latitudes, longitudes, str(output_directory))

    if not getattr(args, "no_save_blend", False):
        save_timelapse(str(output_directory))


if __name__ == "__main__":
    main()
