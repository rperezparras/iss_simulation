#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte y alineado de VIIRS a las imágenes ISS georreferenciadas (pipeline v3).

NUEVO (modo ROI por GCPs):
- roi_mode='gcp': ROI = bounding box de los GCPs en píxeles (geoX/geoY desde *_pixel_mapping.csv)
  + margen en píxeles (roi_margin_px).
- Se guarda <base_name>_roi.json en output_dir con x0,x1,y0,y1 (coords píxel en el raster ISS rectificado).
- Se recorta el mosaico VIIRS a esa ROI (en CRS) y se reproyecta EXACTAMENTE al grid del ROI.

Mantiene modos antiguos (valid bbox/polygon) si roi_mode='valid'.

Ejemplo recomendado (para tu caso):
    python -m pipelinentl.viirs_roi_crop \
        --geo_dir .../geo \
        --output_dir .../viirs_cropped_aligned \
        --viirs_tiff .../VNL...tif \
        --start_id ... --end_id ... \
        --mode fast \
        --roi_mode gcp --roi_margin_px 200 \
        --align roi_exact --resampling bilinear \
        --threads auto --nproc 8
"""

import os
import time
import json
import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_geom
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tempfile
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# =========================
# Helpers
# =========================

def extract_id_from_filename(filename: str):
    try:
        # 'ISS053-E-201283_rect.tiff' -> 201283
        return int(filename.split("-")[-1].split("_")[0])
    except Exception:
        return None


def compute_valid_bbox_geom(reference_raster):
    """BBox de píxeles válidos del raster de referencia."""
    with rasterio.open(reference_raster) as src:
        arr = src.read(1)

        if src.nodata is not None:
            valid = arr != src.nodata
        else:
            valid = arr > 0

        if not valid.any():
            raise ValueError("No hay píxeles válidos en la referencia.")

        rows, cols = np.where(valid)
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        win = Window.from_slices((rmin, rmax + 1), (cmin, cmax + 1))
        bminx, bminy, bmaxx, bmaxy = window_bounds(win, src.transform)
        return mapping(box(bminx, bminy, bmaxx, bmaxy))


def extract_roi_polygon(reference_raster, robust=False):
    """Polígono de píxeles válidos del raster de referencia."""
    with rasterio.open(reference_raster) as src:
        image = src.read(1)
        mask_array = (image != src.nodata) if (src.nodata is not None) else (image > 0)

        geoms = []
        if robust:
            from shapely.errors import TopologicalError
            for geom, val in shapes(image, mask=mask_array, transform=src.transform):
                if val != 0:
                    try:
                        geoms.append(shape(geom))
                    except TopologicalError:
                        continue
        else:
            geoms = [
                shape(g)
                for g, v in shapes(image, mask=mask_array, transform=src.transform)
                if v != 0
            ]

        if not geoms:
            raise ValueError("No se pudo extraer ROI: referencia vacía.")

        merged_polygon = unary_union(geoms)
        return mapping(merged_polygon)


def roi_window_from_pixel_mapping(reference_raster: str, margin_px: int = 200):
    """
    ROI en píxeles del raster de referencia a partir de geoX/geoY de *_pixel_mapping.csv.
    Devuelve (x0, x1, y0, y1) con x1/y1 EXCLUSIVOS.
    """
    mapping_csv = reference_raster.replace("_rect.tiff", "_pixel_mapping.csv")
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"No existe pixel_mapping.csv: {mapping_csv}")

    data = np.genfromtxt(mapping_csv, delimiter=",", names=True, dtype=None, encoding=None)

    if "geoX" not in data.dtype.names or "geoY" not in data.dtype.names:
        raise ValueError(f"pixel_mapping.csv sin geoX/geoY: {mapping_csv}")

    xs = np.array(data["geoX"], dtype=float)
    ys = np.array(data["geoY"], dtype=float)

    ok = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[ok]
    ys = ys[ok]
    if xs.size == 0:
        raise ValueError("No hay geoX/geoY válidos en pixel_mapping.csv")

    with rasterio.open(reference_raster) as ref:
        W, H = ref.width, ref.height

    x0 = int(np.floor(xs.min() - margin_px))
    x1 = int(np.ceil(xs.max() + margin_px))
    y0 = int(np.floor(ys.min() - margin_px))
    y1 = int(np.ceil(ys.max() + margin_px))

    x0 = max(0, min(x0, W - 1))
    x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1))
    y1 = max(y0 + 1, min(y1, H))

    return x0, x1, y0, y1


def roi_geom_from_window(reference_raster: str, x0: int, x1: int, y0: int, y1: int):
    """Convierte una ventana de píxeles (en reference) a geometría CRS (bbox)."""
    with rasterio.open(reference_raster) as ref:
        win = Window(col_off=x0, row_off=y0, width=(x1 - x0), height=(y1 - y0))
        bminx, bminy, bmaxx, bmaxy = window_bounds(win, ref.transform)
    return mapping(box(bminx, bminy, bmaxx, bmaxy))


# =========================
# Recorte y normalización VIIRS
# =========================

def clip_and_normalize_viirs(
    input_viirs,
    reference_raster,
    roi_geom_ref=None,
    clip_mode="bbox",
    gamma=2.5,
    min_val=0,
    max_val=200,
):
    """
    Recorta VIIRS al ROI (en CRS del reference) y normaliza.

    Si roi_geom_ref != None, se usa ese ROI directamente.
    Si no, usa clip_mode sobre la zona válida de reference.
    """
    if roi_geom_ref is None:
        roi_geom_ref = (
            compute_valid_bbox_geom(reference_raster)
            if clip_mode == "bbox"
            else extract_roi_polygon(reference_raster, robust=False)
        )

    with rasterio.open(reference_raster) as ref_src:
        ref_crs = ref_src.crs

    with rasterio.open(input_viirs) as src:
        viirs_crs = src.crs

        roi_for_viirs = roi_geom_ref
        if (viirs_crs is not None) and (ref_crs is not None) and (viirs_crs != ref_crs):
            roi_for_viirs = transform_geom(ref_crs, viirs_crs, roi_geom_ref)

        try:
            out_image, out_transform = mask(src, [roi_for_viirs], crop=True)
        except ValueError as e:
            raise RuntimeError(f"El ROI no solapa con el VIIRS ({e}).")

        out_image = out_image[0].astype(np.float32)

        out_image = np.nan_to_num(out_image, nan=0.0, posinf=0.0, neginf=0.0)

        out_image = np.clip(out_image, min_val, max_val)
        denom = (max_val - min_val)
        if denom <= 0:
            raise ValueError("Parámetros min_val/max_val inválidos.")

        out_image = (out_image - min_val) / (denom + 1e-9)
        out_image = np.power(out_image, 1.0 / gamma)
        out_image = np.clip(out_image * 100.0, 0, 100)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[0],
            "width": out_image.shape[1],
            "transform": out_transform,
            "dtype": "float32",
            "count": 1,
            "compress": "lzw",
            "nodata": 0.0,
        })

        return out_image, out_meta, viirs_crs


# =========================
# Alineado / reproyección
# =========================

def align_viirs(
    reference_path,
    viirs_image_array,
    viirs_meta,
    mode="minimal",
    resampling=Resampling.nearest,
    gdal_threads=None,
    output_aligned_path=None,
    roi_window=None,  # (x0,x1,y0,y1) en píxeles del reference
):
    """
    Alinea/reproyecta el recorte VIIRS al CRS/grid del reference.

    mode:
      - 'roi_exact' → misma resolución y alineación que reference PERO solo ventana ROI (output pequeño)
      - 'exact'     → misma rejilla completa que reference (output tamaño completo)
      - 'minimal'   → rejilla mínima (más rápido, no coincide píxel a píxel)

    roi_window requerido para 'roi_exact'.
    """
    assert output_aligned_path is not None

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width, ref_height = ref.width, ref.height

        if mode == "roi_exact":
            if roi_window is None:
                raise ValueError("mode='roi_exact' requiere roi_window=(x0,x1,y0,y1)")
            x0, x1, y0, y1 = roi_window
            win = Window(col_off=x0, row_off=y0, width=(x1 - x0), height=(y1 - y0))
            dst_crs = ref_crs
            dst_transform = window_transform(win, ref_transform)
            dst_width = int(x1 - x0)
            dst_height = int(y1 - y0)
        elif mode == "exact":
            dst_crs = ref_crs
            dst_transform = ref_transform
            dst_width = ref_width
            dst_height = ref_height
        else:
            # minimal
            dst_crs = ref_crs
            # se calculará con calculate_default_transform abajo usando bounds del temporal
            dst_transform = None
            dst_width = None
            dst_height = None

    # temporal del recorte VIIRS en su CRS/transform actual (del recorte)
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmpfile:
        with rasterio.open(tmpfile.name, "w", **viirs_meta) as tmp_ds:
            tmp_ds.write(viirs_image_array[np.newaxis, :, :])

        with rasterio.open(tmpfile.name) as tmp_ds:
            if mode == "minimal":
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    tmp_ds.crs, ref_crs, tmp_ds.width, tmp_ds.height, *tmp_ds.bounds
                )

            profile = tmp_ds.profile.copy()
            profile.update({
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "dtype": "float32",
                "count": 1,
                "compress": "lzw",
                "nodata": 0.0,
            })

            env_kwargs = {}
            if gdal_threads is not None:
                env_kwargs["GDAL_NUM_THREADS"] = str(gdal_threads)

            with rasterio.Env(**env_kwargs):
                with rasterio.open(output_aligned_path, "w", **profile) as dst:
                    reproject(
                        source=rasterio.band(tmp_ds, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=tmp_ds.transform,
                        src_crs=tmp_ds.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                        num_threads=gdal_threads if gdal_threads else 0,
                    )


# =========================
# Procesado por imagen
# =========================

def process_one_image(args_tuple):
    (
        ref_file,
        geo_dir,
        output_dir,
        viirs_tiff,
        clip_mode,
        align_mode,
        resampling,
        gdal_threads,
        roi_mode,
        roi_margin_px,
    ) = args_tuple

    start = time.time()
    ref_path = os.path.join(geo_dir, ref_file)
    base_name = ref_file.replace("_rect.tiff", "")
    output_aligned_path = os.path.join(output_dir, f"{base_name}_viirs.tiff")
    roi_json_path = os.path.join(output_dir, f"{base_name}_roi.json")

    if os.path.exists(output_aligned_path) and os.path.exists(roi_json_path):
        return f"🟡 {ref_file} ya existe, se omite."

    try:
        roi_geom_ref = None
        roi_window = None

        if roi_mode == "gcp":
            x0, x1, y0, y1 = roi_window_from_pixel_mapping(ref_path, margin_px=roi_margin_px)
            roi_window = (x0, x1, y0, y1)
            roi_geom_ref = roi_geom_from_window(ref_path, x0, x1, y0, y1)

            with rasterio.open(ref_path) as ref:
                meta = {
                    "x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1),
                    "roi_width": int(x1 - x0),
                    "roi_height": int(y1 - y0),
                    "ref_width": int(ref.width),
                    "ref_height": int(ref.height),
                    "roi_margin_px": int(roi_margin_px),
                    "roi_mode": "gcp",
                }
            os.makedirs(output_dir, exist_ok=True)
            with open(roi_json_path, "w") as f:
                json.dump(meta, f)

        viirs_image, viirs_meta, _ = clip_and_normalize_viirs(
            viirs_tiff,
            ref_path,
            roi_geom_ref=roi_geom_ref,
            clip_mode=clip_mode,
            gamma=2.5,
            min_val=0,
            max_val=200,
        )

        align_viirs(
            reference_path=ref_path,
            viirs_image_array=viirs_image,
            viirs_meta=viirs_meta,
            mode=align_mode,
            resampling=resampling,
            gdal_threads=gdal_threads,
            output_aligned_path=output_aligned_path,
            roi_window=roi_window,
        )

        # Si no es gcp, igualmente guardamos roi.json “vacío” para consistencia
        if roi_mode != "gcp":
            with rasterio.open(ref_path) as ref:
                meta = {
                    "roi_mode": roi_mode,
                    "ref_width": int(ref.width),
                    "ref_height": int(ref.height),
                }
            with open(roi_json_path, "w") as f:
                json.dump(meta, f)

        elapsed = time.time() - start
        return f"✅ {ref_file} procesado en {elapsed:.2f} s"

    except RuntimeError as e:
        return f"⚪ {ref_file} sin solape: {e}"
    except Exception as e:
        return f"❌ Error en {ref_file}: {e}"


def process_timelapse_parallel(
    geo_dir,
    output_dir,
    viirs_tiff,
    start_id,
    end_id,
    nproc,
    clip_mode,
    align_mode,
    resampling,
    gdal_threads,
    roi_mode,
    roi_margin_px,
):
    reference_files = sorted(
        f
        for f in os.listdir(geo_dir)
        if f.endswith("_rect.tiff")
        and (file_id := extract_id_from_filename(f)) is not None
        and start_id <= file_id <= end_id
    )

    print(f"🔵 Se encontraron {len(reference_files)} imágenes para procesar.")
    max_processes = min(nproc, cpu_count())
    print(f"🔵 Usando {max_processes} procesos. (GDAL threads: {gdal_threads})")
    print(f"🔵 roi_mode={roi_mode}, roi_margin_px={roi_margin_px}, align={align_mode}, clip={clip_mode}, resampling={resampling}\n")

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    resampling_enum = resampling_map[resampling]

    job_args = [
        (
            f,
            geo_dir,
            output_dir,
            viirs_tiff,
            clip_mode,
            align_mode,
            resampling_enum,
            gdal_threads,
            roi_mode,
            roi_margin_px,
        )
        for f in reference_files
    ]

    start_time = time.time()
    with Pool(processes=max_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, job_args), total=len(reference_files)):
            print(result)

    total_time = time.time() - start_time
    print(f"\n⏱️ Tiempo total de ejecución: {total_time:.2f} segundos")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Recorte y alineado de VIIRS a imágenes ISS georreferenciadas."
    )
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--viirs_tiff",
        type=str,
        required=False,
        default="/home/raul/planb/VNL_v2_npp_2017_global_vcmslcfg_c202101211500.median.tif",
    )
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--nproc", type=int, default=8)

    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "safe"])

    parser.add_argument("--clip", type=str, default=None, choices=["bbox", "polygon"])
    parser.add_argument("--align", type=str, default=None, choices=["minimal", "exact", "roi_exact"])
    parser.add_argument("--resampling", type=str, default=None, choices=["nearest", "bilinear", "cubic"])

    parser.add_argument("--threads", type=str, default="auto")

    # ROI por GCPs
    parser.add_argument("--roi_mode", type=str, default="gcp", choices=["gcp", "valid"])
    parser.add_argument("--roi_margin_px", type=int, default=200)

    args = parser.parse_args()

    geo_dir = os.path.abspath(args.geo_dir)
    output_dir = os.path.abspath(args.output_dir)
    viirs_tiff = os.path.abspath(args.viirs_tiff)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(viirs_tiff):
        raise FileNotFoundError(f"No se encontró el TIFF de VIIRS: {viirs_tiff}")

    # Defaults según modo
    if args.mode == "fast":
        clip_mode = args.clip or "bbox"
        # para tu caso queremos ROI exacta:
        align_mode = args.align or ("roi_exact" if args.roi_mode == "gcp" else "minimal")
        resampling = args.resampling or "bilinear"
    else:
        clip_mode = args.clip or "polygon"
        align_mode = args.align or ("roi_exact" if args.roi_mode == "gcp" else "exact")
        resampling = args.resampling or "bilinear"

    # Hilos GDAL
    if args.threads == "auto":
        gdal_threads = cpu_count()
    else:
        try:
            gdal_threads = int(args.threads)
        except Exception:
            gdal_threads = cpu_count()

    process_timelapse_parallel(
        geo_dir=geo_dir,
        output_dir=output_dir,
        viirs_tiff=viirs_tiff,
        start_id=args.start_id,
        end_id=args.end_id,
        nproc=args.nproc,
        clip_mode=clip_mode,
        align_mode=align_mode,
        resampling=resampling,
        gdal_threads=gdal_threads,
        roi_mode=args.roi_mode,
        roi_margin_px=args.roi_margin_px,
    )


if __name__ == "__main__":
    main()