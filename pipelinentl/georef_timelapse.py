#!/usr/bin/env python3
import os
import glob
import subprocess
from multiprocessing import Pool, cpu_count

from astropy.io import ascii
from osgeo import gdal

gdal.UseExceptions()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import pandas as pd
import argparse


# ------------------------------------------------------------
# Utilidades GDAL / VRT / GCP
# ------------------------------------------------------------

def is_vrt_geometrically_valid(vrt_path, max_extent_deg=60):
    ds = gdal.Open(vrt_path)
    if ds is None:
        return False

    gcps = ds.GetGCPs()
    if not gcps:
        return False

    lons = [gcp.GCPX for gcp in gcps]
    lats = [gcp.GCPY for gcp in gcps]

    lon_span = max(lons) - min(lons)
    lat_span = max(lats) - min(lats)

    return lon_span < max_extent_deg and lat_span < max_extent_deg


def create_vrt_with_gcps(gcps, vrt_path, image_path):
    gdal_command = [
        "gdal_translate",
        "-of", "VRT",
        "-a_srs", "EPSG:4326",
        image_path,
        vrt_path,
    ]

    for gcp in gcps:
        gdal_command += [
            "-gcp",
            str(gcp["sourceX"]),
            str(-gcp["sourceY"]),
            str(gcp["mapX"]),
            str(gcp["mapY"]),
        ]

    subprocess.run(gdal_command, check=True)


def estimate_warp_size(vrt_path, temp_warp_vrt):
    """
    Estima el tamaño que produciría gdalwarp sin escribir el GeoTIFF final.
    Crea solo un VRT temporal.
    """
    warp_cmd = [
        "gdalwarp",
        "-of", "VRT",
        "-t_srs", "EPSG:4326",
        "-r", "near",
        "-tps",
        vrt_path,
        temp_warp_vrt,
    ]

    subprocess.run(
        warp_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    ds = gdal.Open(temp_warp_vrt)
    if ds is None:
        raise RuntimeError(f"No se pudo abrir el VRT estimado: {temp_warp_vrt}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None

    return width, height


def gdal_warp(
    vrt_path,
    final_output_path,
    max_side_px=50000,
    max_pixels=50000 * 50000,
):
    """
    Ejecuta gdalwarp con TPS, pero antes estima el tamaño de salida.
    Si GDAL intenta crear una imagen absurda, se omite.
    """
    temp_warp_vrt = final_output_path.replace("_rect.tiff", "_warp_estimate.vrt")

    try:
        width, height = estimate_warp_size(vrt_path, temp_warp_vrt)
    finally:
        if os.path.exists(temp_warp_vrt):
            os.remove(temp_warp_vrt)

    total_pixels = width * height

    if width > max_side_px or height > max_side_px or total_pixels > max_pixels:
        return False, (
            f"⛔ Warp omitido por tamaño absurdo: "
            f"{width} x {height} px = {total_pixels:,} píxeles"
        )

    gdal_warp_command = [
        "gdalwarp",
        "-co", "TFW=YES",
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=IF_SAFER",
        "-t_srs", "EPSG:4326",
        "-r", "near",
        "-tps",
        vrt_path,
        final_output_path,
    ]

    subprocess.run(gdal_warp_command, check=True)

    return True, f"✅ Warp correcto: {width} x {height} px"


def table_to_dict_list(table):
    return [
        {
            "mapX": row["mapX"],
            "mapY": row["mapY"],
            "sourceX": row["sourceX"],
            "sourceY": row["sourceY"],
        }
        for row in table
    ]


def save_pixel_mapping(gcps, geo_image_path, mapping_file):
    dataset = gdal.Open(geo_image_path)
    if dataset is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen georreferenciada: {geo_image_path}")

    geotransform = dataset.GetGeoTransform()

    def geo_to_pixel(mapX, mapY, geotransform):
        px = int((mapX - geotransform[0]) / geotransform[1])
        py = int((mapY - geotransform[3]) / geotransform[5])
        return px, py

    with open(mapping_file, "w") as f:
        f.write("sourceX,sourceY,mapX,mapY,geoX,geoY\n")
        for gcp in gcps:
            geoX, geoY = geo_to_pixel(gcp["mapX"], gcp["mapY"], geotransform)
            f.write(
                f"{gcp['sourceX']},{gcp['sourceY']},"
                f"{gcp['mapX']},{gcp['mapY']},"
                f"{geoX},{geoY}\n"
            )


# ------------------------------------------------------------
# Utilidades varias
# ------------------------------------------------------------

def extract_id_from_filename(filename):
    try:
        return int(filename.split("-")[-1])
    except Exception:
        return None


def find_image(input_dir, base_name):
    exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".tiff"]
    for ext in exts:
        files = glob.glob(os.path.join(input_dir, f"{base_name}{ext}"))
        if files:
            return os.path.basename(files[0])
    return None


def plot_georef_comparison(image_path, points_path, geo_image_path, mapping_file, output_plot):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    points = pd.read_csv(points_path, comment="M")

    if "enable" in points.columns:
        points = points[points["enable"].astype(int) == 1].copy()

    h_img, w_img = img.shape[:2]

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].imshow(img, origin="upper")

    x = points["sourceX"].to_numpy(dtype=float)
    sy = points["sourceY"].to_numpy(dtype=float)
    y = -sy

    mask = (x >= 0) & (x < w_img) & (y >= 0) & (y < h_img)
    x_plot = x[mask]
    y_plot = y[mask]

    axs[0].scatter(x_plot, y_plot, c="lime", s=10, label="Source Points")
    axs[0].set_title("ISS Real Image + Ground Control Points")
    axs[0].legend()
    axs[0].set_xlim(0, w_img)
    axs[0].set_ylim(h_img, 0)

    geo_bgr = cv2.imread(geo_image_path)
    if geo_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen georreferenciada: {geo_image_path}")

    geo_img = cv2.cvtColor(geo_bgr, cv2.COLOR_BGR2RGB)
    mapping = pd.read_csv(mapping_file)

    h_geo, w_geo = geo_img.shape[:2]

    axs[1].imshow(geo_img, origin="upper")
    axs[1].scatter(mapping["geoX"], mapping["geoY"], c="red", s=10, label="Mapped Points")
    axs[1].set_title("ISS Georeferenced Image + Ground Control Points")
    axs[1].legend()
    axs[1].set_xlim(0, w_geo)
    axs[1].set_ylim(h_geo, 0)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Procesado de una imagen
# ------------------------------------------------------------

def process_image(
    image_name,
    points_name,
    input_dir,
    points_dir,
    output_dir,
    temp_dir,
    idx,
    plot_every,
):
    image_path = os.path.join(input_dir, image_name)
    points_path = os.path.join(points_dir, points_name)

    base = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_dir, base + "_rect.tiff")
    vrt_path = os.path.join(temp_dir, base + ".vrt")
    mapping_file = os.path.join(output_dir, base + "_pixel_mapping.csv")

    if os.path.exists(output_path) and os.path.exists(mapping_file):
        return f"⏭️ {image_name} ya estaba procesado, se omite"

    for f in (output_path, vrt_path, mapping_file):
        if os.path.exists(f):
            os.remove(f)

    gcps_table = ascii.read(points_path, delimiter=",", comment="M")

    required_cols = ["mapX", "mapY", "sourceX", "sourceY"]
    if not all(col in gcps_table.colnames for col in required_cols):
        raise ValueError(f"Formato de GCP incorrecto en {points_path}")

    gcps_dict = table_to_dict_list(gcps_table)

    if len(gcps_dict) < 6:
        return f"⛔ Muy pocos GCPs ({len(gcps_dict)}): {image_name}"

    create_vrt_with_gcps(gcps_dict, vrt_path, image_path)

    if not is_vrt_geometrically_valid(vrt_path):
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
        return f"⛔ VRT inválido por dispersión excesiva de GCPs: {image_name}"

    ok, msg = gdal_warp(
        vrt_path,
        output_path,
        max_side_px=50000,
        max_pixels=50000 * 50000,
    )

    if not ok:
        for f in (output_path, mapping_file):
            if os.path.exists(f):
                os.remove(f)
        return f"{msg}: {image_name}"

    if not os.path.exists(output_path):
        return f"⚠️ No se generó la imagen georreferenciada: {output_path}"

    save_pixel_mapping(gcps_dict, output_path, mapping_file)

    if idx % plot_every == 0:
        plot_path = os.path.join(output_dir, base + "_comparison.png")
        plot_georef_comparison(image_path, points_path, output_path, mapping_file, plot_path)

    if os.path.exists(vrt_path):
        os.remove(vrt_path)

    return f"✅ {image_name} procesado correctamente | {msg}"


def process_one(base_name, input_dir, points_dir, output_dir, temp_dir, idx, plot_every):
    try:
        image_name = find_image(input_dir, base_name)

        points_name_1 = f"{base_name}.points"
        points_name_2 = f"{base_name}_corrected.points"

        points_path_1 = os.path.join(points_dir, points_name_1)
        points_path_2 = os.path.join(points_dir, points_name_2)

        if os.path.exists(points_path_1):
            points_name = points_name_1
        elif os.path.exists(points_path_2):
            points_name = points_name_2
        else:
            return f"⚠️ No se encontraron puntos (.points) para {base_name}"

        if image_name is None:
            return f"⚠️ No se encontró imagen para {base_name}"

        return process_image(
            image_name=image_name,
            points_name=points_name,
            input_dir=input_dir,
            points_dir=points_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            idx=idx,
            plot_every=plot_every,
        )

    except subprocess.CalledProcessError as e:
        return f"❌ Error GDAL procesando {base_name}: {e}"

    except Exception as e:
        return f"❌ Error procesando {base_name}: {e}"


def process_one_with_idx(args_tuple):
    base, input_dir, points_dir, output_dir, temp_dir, idx, plot_every = args_tuple
    return process_one(base, input_dir, points_dir, output_dir, temp_dir, idx, plot_every)


# ------------------------------------------------------------
# Procesado paralelo de todo el timelapse
# ------------------------------------------------------------

def process_timelapse_parallel(
    input_dir,
    points_dir,
    output_dir,
    temp_dir,
    start_id,
    end_id,
    plot_every,
):
    all_images = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(input_dir)
        if any(f.endswith(ext) for ext in [
            ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".tiff"
        ])
    ])

    filtered_bases = []

    for base in all_images:
        img_id = extract_id_from_filename(base)

        if img_id is None or not (start_id <= img_id <= end_id):
            continue

        tiff_file = os.path.join(output_dir, f"{base}_rect.tiff")
        mapping_file = os.path.join(output_dir, f"{base}_pixel_mapping.csv")

        if os.path.exists(tiff_file) and os.path.exists(mapping_file):
            continue

        filtered_bases.append(base)

    print(f"🔵 Imágenes a georreferenciar (nuevas): {len(filtered_bases)}\n")

    tasks = [
        (base, input_dir, points_dir, output_dir, temp_dir, idx, plot_every)
        for idx, base in enumerate(filtered_bases)
    ]

    if not tasks:
        print("Nada que hacer: todas las imágenes del rango parecen ya procesadas.")
        return

    nproc = min(8, cpu_count())
    print(f"🔵 Usando {nproc} procesos para georreferenciación.\n")

    with Pool(nproc) as pool:
        results = pool.map(process_one_with_idx, tasks)

    for res in results:
        print(res)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Georreferenciación paralela de timelapse ISS con GDAL TPS"
    )

    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--plot_every", type=int, default=50)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--points_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    input_dir = args.input_dir
    points_dir = args.points_dir
    output_dir = args.output_dir
    temp_dir = os.path.join(output_dir, "temp")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    process_timelapse_parallel(
        input_dir=input_dir,
        points_dir=points_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        start_id=args.start_id,
        end_id=args.end_id,
        plot_every=args.plot_every,
    )


if __name__ == "__main__":
    main()