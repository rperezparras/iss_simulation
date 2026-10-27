#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de flujo óptico entre imágenes ISS georreferenciadas y VIIRS.

El flujo se calcula en dirección ISS -> VIIRS, para poder corregir directamente
los GCPs muestreados en la imagen ISS ya georreferenciada.

Soporta dos tipos de VIIRS de entrada:

1) VIIRS alineado a una ROI exacta, generado por viirs_roi_crop.py con
   --roi_mode gcp --align roi_exact. En este caso existe un
   <image_id>_roi.json y el GeoTIFF VIIRS ya tiene exactamente la rejilla de
   esa ROI. NO se debe redimensionar a la imagen ISS completa.

2) VIIRS alineado a la imagen completa o a una rejilla mínima antigua. En ese
   caso se usa el comportamiento de compatibilidad: redimensionar VIIRS a la
   resolución de la ISS y aplicar el recorte normalizado crop_*.

Para cada imagen guarda:
  - *_flow.npy       : campo de flujo en coordenadas de la ROI usada.
  - *_flow_meta.json : metadatos exactos de la ROI dentro del GeoTIFF ISS.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

import cv2
import flow_vis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage import exposure


FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 19,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}


def clamp_crop_fraction(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito, recibido: {value!r}")
    return max(0.0, min(1.0, value))


def normalized_crop_to_pixels(
    width: int,
    height: int,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
) -> tuple[int, int, int, int]:
    """Convierte crop normalizado [0,1] a píxeles. Devuelve x0,y0,x1,y1."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensiones inválidas: width={width}, height={height}")

    crop_x_start = clamp_crop_fraction(crop_x_start, "crop_x_start")
    crop_x_end = clamp_crop_fraction(crop_x_end, "crop_x_end")
    crop_y_start = clamp_crop_fraction(crop_y_start, "crop_y_start")
    crop_y_end = clamp_crop_fraction(crop_y_end, "crop_y_end")

    if crop_x_end <= crop_x_start:
        raise ValueError("crop_x_end debe ser mayor que crop_x_start")
    if crop_y_end <= crop_y_start:
        raise ValueError("crop_y_end debe ser mayor que crop_y_start")

    x0 = int(width * crop_x_start)
    x1 = int(width * crop_x_end)
    y0 = int(height * crop_y_start)
    y1 = int(height * crop_y_end)

    x0 = max(0, min(x0, width - 1))
    x1 = max(x0 + 1, min(x1, width))
    y0 = max(0, min(y0, height - 1))
    y1 = max(y0 + 1, min(y1, height))

    return x0, y0, x1, y1


def roi_json_path_from_viirs_file(viirs_file: str) -> str:
    """Devuelve el path esperado de <image_id>_roi.json junto al *_viirs.tiff."""
    if viirs_file.endswith("_viirs.tiff"):
        return viirs_file.replace("_viirs.tiff", "_roi.json")
    if viirs_file.endswith("_viirs.tif"):
        return viirs_file.replace("_viirs.tif", "_roi.json")
    return os.path.splitext(viirs_file)[0] + "_roi.json"


def load_roi_from_viirs_json(viirs_file: str, full_width: int, full_height: int):
    """
    Lee <image_id>_roi.json generado por viirs_roi_crop.py.

    Devuelve (x0, y0, x1, y1, roi_source) o None si no hay JSON utilizable.
    El JSON de viirs_roi_crop.py guarda x0,x1,y0,y1 con x1/y1 exclusivos.
    """
    roi_json = roi_json_path_from_viirs_file(viirs_file)
    if not os.path.exists(roi_json):
        return None

    with open(roi_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required = {"x0", "x1", "y0", "y1"}
    if not required.issubset(meta):
        return None

    x0 = int(meta["x0"])
    x1 = int(meta["x1"])
    y0 = int(meta["y0"])
    y1 = int(meta["y1"])

    x0 = max(0, min(x0, full_width - 1))
    x1 = max(x0 + 1, min(x1, full_width))
    y0 = max(0, min(y0, full_height - 1))
    y1 = max(y0 + 1, min(y1, full_height))

    return x0, y0, x1, y1, f"viirs_roi_json:{os.path.basename(roi_json)}"


def to_uint8_normalized(arr: np.ndarray, label: str) -> np.ndarray:
    """Normaliza un array a uint8 [0,255] de forma robusta."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = float(np.max(arr))
    vmin = float(np.min(arr))
    if vmax <= vmin:
        raise ValueError(f"{label} no tiene rango dinámico útil: min={vmin}, max={vmax}")

    out = (arr - vmin) / (vmax - vmin + 1e-9) * 255.0
    return np.uint8(np.clip(out, 0, 255))


def write_flow_metadata(
    meta_path: str,
    image_id: str,
    iss_file: str,
    viirs_file: str,
    full_width: int,
    full_height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
    roi_source: str,
    viirs_width: int,
    viirs_height: int,
):
    meta = {
        "metadata_version": 3,
        "image_id": image_id,
        "iss_rect_tiff": os.path.abspath(iss_file),
        "viirs_tiff": os.path.abspath(viirs_file),
        "full_width": int(full_width),
        "full_height": int(full_height),
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "flow_width": int(x1 - x0),
        "flow_height": int(y1 - y0),
        "viirs_input_width": int(viirs_width),
        "viirs_input_height": int(viirs_height),
        "roi_source": str(roi_source),
        "crop_x_start": float(crop_x_start),
        "crop_x_end": float(crop_x_end),
        "crop_y_start": float(crop_y_start),
        "crop_y_end": float(crop_y_end),
        "flow_definition": "cv2.calcOpticalFlowFarneback(reference_iss, distorted_viirs)",
        "correction_to_apply_in_correct_points": "new_pixel = old_pixel + flow[y, x]",
        "farneback_params": FARNEBACK_PARAMS,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def compute_flow(reference_u8: np.ndarray, distorted_u8: np.ndarray) -> np.ndarray:
    """Calcula Farneback tras validar tamaños y tipos."""
    if reference_u8.shape != distorted_u8.shape:
        raise ValueError(
            f"reference y distorted deben tener el mismo tamaño, recibido "
            f"{reference_u8.shape} vs {distorted_u8.shape}"
        )
    if reference_u8.dtype != np.uint8:
        reference_u8 = np.uint8(np.clip(reference_u8, 0, 255))
    if distorted_u8.dtype != np.uint8:
        distorted_u8 = np.uint8(np.clip(distorted_u8, 0, 255))

    return cv2.calcOpticalFlowFarneback(
        reference_u8,
        distorted_u8,
        None,
        FARNEBACK_PARAMS["pyr_scale"],
        FARNEBACK_PARAMS["levels"],
        FARNEBACK_PARAMS["winsize"],
        FARNEBACK_PARAMS["iterations"],
        FARNEBACK_PARAMS["poly_n"],
        FARNEBACK_PARAMS["poly_sigma"],
        FARNEBACK_PARAMS["flags"],
    )



def _source_image_path(source_dir, image_id):
    if not source_dir:
        return None
    for ext in ("JPG", "jpg", "jpeg", "JPEG", "png", "PNG", "tif", "tiff"):
        path = Path(source_dir) / f"{image_id}.{ext}"
        if path.is_file():
            return path
    return None


def _generate_footprint_tps(mask_dir, image_id, iss_file, source_images_dir, points_dir):
    """Warp an all-255 raster with the SAME source GCPs and target grid as first TPS.

    Never detect coverage from dark pixels. Raster creation uses rasterio rather
    than GDAL ReadAsArray (which fails on some Python/GDAL installations).
    """
    source_img = _source_image_path(source_images_dir, image_id)
    points_file = Path(points_dir) / f"{image_id}.points" if points_dir else None
    if source_img is None or points_file is None or not points_file.is_file():
        raise FileNotFoundError(
            f"Missing footprint for {image_id}; provide --coverage_mask_dir with a mask, "
            f"or --source_images_dir and --points_dir (source={source_img}, points={points_file})"
        )
    df = pd.read_csv(points_file, comment="M")
    required = ("sourceX", "sourceY", "mapX", "mapY")
    if not all(col in df.columns for col in required):
        raise ValueError(f"GCP columns missing from {points_file}")
    # Match georef_timelapse exactly: it supplies every row to gdal_translate.
    df = df.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if len(df) < 6 or not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Invalid GCP table: {points_file}")
    with rasterio.open(source_img) as src:
        source_w, source_h = src.width, src.height
    with rasterio.open(iss_file) as dst:
        target_w, target_h = dst.width, dst.height
        if dst.crs is None:
            raise ValueError(f"GeoTIFF without CRS: {iss_file}")
        gt = dst.transform
        if abs(gt.b) > 1e-10 or abs(gt.d) > 1e-10:
            raise ValueError("Cannot rebuild footprint on a rotated raster grid")
        left, bottom, right, top = dst.bounds
        epsg = dst.crs.to_string()

    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    target = mask_dir / f"iss_geometric_footprint_{image_id.split('-')[-1]}.tif"
    with tempfile.TemporaryDirectory(prefix=f"mask_{image_id}_", dir=str(mask_dir)) as tmp:
        tmp = Path(tmp)
        constant = tmp / "constant_255.tif"
        vrt = tmp / "source_gcps.vrt"
        warped = tmp / "warped_mask.tif"
        with rasterio.open(
            constant, "w", driver="GTiff", width=source_w, height=source_h,
            count=1, dtype="uint8", compress="lzw", tiled=True,
        ) as mask_dst:
            for _, window in mask_dst.block_windows(1):
                mask_dst.write(np.full((int(window.height), int(window.width)), 255, dtype=np.uint8),
                               1, window=window)
        cmd = ["gdal_translate", "-q", "-of", "VRT", "-a_srs", epsg,
               str(constant), str(vrt)]
        for row in df.itertuples(index=False):
            cmd.extend(["-gcp", str(row.sourceX), str(-row.sourceY),
                        str(row.mapX), str(row.mapY)])
        subprocess.run(cmd, check=True)
        subprocess.run([
            "gdalwarp", "-q", "-overwrite", "-of", "GTiff",
            "-co", "COMPRESS=LZW", "-co", "TILED=YES",
            "-t_srs", epsg, "-r", "near", "-tps", "-dstnodata", "0",
            "-te", repr(left), repr(bottom), repr(right), repr(top),
            "-ts", str(target_w), str(target_h), str(vrt), str(warped),
        ], check=True)
        with rasterio.open(iss_file) as iss_ds, rasterio.open(warped) as mask_ds:
            if ((iss_ds.width, iss_ds.height) != (mask_ds.width, mask_ds.height)
                or iss_ds.crs != mask_ds.crs
                or not iss_ds.transform.almost_equals(mask_ds.transform)):
                raise ValueError(f"Generated footprint does not match ISS grid: {warped}")
        os.replace(warped, target)
    print(f"Generated geometric ISS footprint: {target}")
    return target


def load_geometric_footprint(mask_dir, image_id, iss_file, x0, y0, x1, y1):
    """Read the geometrically warped ISS footprint, never infer it from RGB=0.

    The mask must have the EXACT same pixel grid as the initial ISS GeoTIFF.
    """
    number = image_id.split("-")[-1]
    candidates = [
        Path(mask_dir) / f"iss_geometric_footprint_{number}.tif",
        Path(mask_dir) / f"{image_id}_footprint.tif",
        Path(mask_dir) / f"{image_id}_coverage.tif",
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            f"No geometric ISS mask for {image_id} in {mask_dir}. "
            f"Expected one of: {[str(p) for p in candidates]}"
        )
    with rasterio.open(iss_file) as iss_src, rasterio.open(found) as mask_src:
        if (iss_src.width, iss_src.height) != (mask_src.width, mask_src.height):
            raise ValueError("ISS footprint mask size differs from initial GeoTIFF")
        if iss_src.crs != mask_src.crs or not iss_src.transform.almost_equals(mask_src.transform):
            raise ValueError("ISS footprint mask CRS/transform differs from initial GeoTIFF")
        window = Window(x0, y0, x1 - x0, y1 - y0)
        valid = mask_src.read(1, window=window) > 0
    if not valid.any():
        raise ValueError(f"Empty ISS coverage mask in ROI for {image_id}")
    return valid, str(found.resolve())


def match_histograms_inside_footprint(source_u8, target_u8, valid):
    """Match ISS histogram to VIIRS using only geometrically covered pixels."""
    source_values, source_counts = np.unique(source_u8[valid], return_counts=True)
    target_values, target_counts = np.unique(target_u8[valid], return_counts=True)
    if len(source_values) < 2 or len(target_values) < 2:
        raise ValueError("Insufficient radiometric variation inside ISS footprint")
    source_quantiles = np.cumsum(source_counts) / float(source_counts.sum())
    target_quantiles = np.cumsum(target_counts) / float(target_counts.sum())
    mapped = np.interp(source_quantiles, target_quantiles, target_values)
    return np.uint8(np.clip(
        np.interp(source_u8.ravel(), source_values, mapped).reshape(source_u8.shape),
        0, 255,
    ))


def masked_flow_with_edge_guard(reference_u8, distorted_u8, footprint, margin_px, feather_px):
    """Suppress spurious border comparisons; accept flow only deep in coverage.

    Farneback has no native validity-mask argument. Invalid regions in BOTH
    images are blended to a common neutral value; a conservative inward
    margin rejects flow influenced by the coverage edge. This is an
    EXPERIMENTAL mitigation, not a proof of optical-flow reliability.
    """
    if reference_u8.shape != distorted_u8.shape or reference_u8.shape != footprint.shape:
        raise ValueError("ISS, VIIRS and coverage mask must share ROI dimensions")
    if margin_px < 1 or feather_px < 1 or feather_px > margin_px:
        raise ValueError("Require 1 <= feather_px <= margin_px")
    padded = cv2.copyMakeBorder(
        footprint.astype(np.uint8), 1, 1, 1, 1,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    valid_flow = distance >= float(margin_px)
    if not valid_flow.any():
        raise ValueError("Mask erosion removed all pixels; reduce --mask_margin_px")
    alpha = np.minimum(distance / float(feather_px), 1.0).astype(np.float32)
    neutral = 128.0
    ref_blend = np.uint8(np.clip(alpha * reference_u8 + (1.0 - alpha) * neutral, 0, 255))
    dst_blend = np.uint8(np.clip(alpha * distorted_u8 + (1.0 - alpha) * neutral, 0, 255))
    flow = compute_flow(ref_blend, dst_blend)
    flow[~valid_flow] = 0.0
    return flow, valid_flow


def compute_and_save_optical_flow(
    image_id: str,
    geo_dir: str,
    viirs_dir: str,
    flow_dir: str,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
    show_plot: bool = False,
    coverage_mask_dir: str = None,
    mask_margin_px: int = 96,
    mask_feather_px: int = 32,
    histogram_mode: str = "masked",
    source_images_dir: str = None,
    points_dir: str = None,
    regenerate_footprints: bool = False,
):
    """Calcula y guarda el flujo óptico y sus metadatos para un ID de imagen."""
    iss_file = os.path.join(geo_dir, f"{image_id}_rect.tiff")
    viirs_file = os.path.join(viirs_dir, f"{image_id}_viirs.tiff")
    flow_outfile = os.path.join(flow_dir, f"{image_id}_flow.npy")
    meta_outfile = os.path.join(flow_dir, f"{image_id}_flow_meta.json")

    try:
        if not os.path.exists(iss_file):
            raise FileNotFoundError(f"No se encontró la imagen ISS: {iss_file}")
        if not os.path.exists(viirs_file):
            raise FileNotFoundError(f"No se encontró la imagen VIIRS: {viirs_file}")

        with rasterio.open(iss_file) as src:
            if src.count >= 3:
                iss = src.read([1, 2, 3])
                iss = np.transpose(iss, (1, 2, 0))
            else:
                band = src.read(1)
                iss = np.dstack([band, band, band])

        with rasterio.open(viirs_file) as src:
            viirs = src.read(1)

        viirs = np.nan_to_num(viirs, nan=0.0, posinf=0.0, neginf=0.0)
        viirs_input_h, viirs_input_w = viirs.shape[:2]
        full_height, full_width = iss.shape[:2]

        iss_gray_full = cv2.cvtColor(iss, cv2.COLOR_RGB2GRAY)

        roi_from_json = load_roi_from_viirs_json(viirs_file, full_width, full_height)

        if roi_from_json is not None:
            # Caso correcto para viirs_roi_crop.py --roi_mode gcp --align roi_exact:
            # el VIIRS ya está en la rejilla exacta de la ROI, así que NO se
            # redimensiona a la imagen completa.
            x0, y0, x1, y1, roi_source = roi_from_json
            roi_w = x1 - x0
            roi_h = y1 - y0

            if viirs_input_w != roi_w or viirs_input_h != roi_h:
                raise ValueError(
                    f"VIIRS ROI size {viirs_input_w}x{viirs_input_h} does not match "
                    f"ROI JSON {roi_w}x{roi_h} for {image_id}; refusing to resize "
                    "because this would change pixel displacement units"
                )
            else:
                viirs_roi = viirs

            iss_roi = iss_gray_full[y0:y1, x0:x1]
            viirs_u8 = to_uint8_normalized(viirs_roi, f"VIIRS {image_id}")
            iss_matched = exposure.match_histograms(iss_roi, viirs_u8)
            iss_u8 = np.uint8(np.clip(iss_matched, 0, 255))

            # Para corregir GCPs de la ISS georreferenciada necesitamos el
            # flujo ISS -> VIIRS. Cada GCP se muestrea en coordenadas de la
            # ISS ya georreferenciada y se desplaza hacia la referencia VIIRS.
            reference_crop = iss_u8
            distorted_crop = viirs_u8

        else:
            # Compatibilidad con flujos antiguos o VIIRS alineado a imagen completa.
            roi_source = "normalized_crop_args_full_frame"
            viirs_full = cv2.resize(viirs, (full_width, full_height), interpolation=cv2.INTER_LINEAR)
            viirs_u8_full = to_uint8_normalized(viirs_full, f"VIIRS {image_id}")
            iss_matched_full = exposure.match_histograms(iss_gray_full, viirs_u8_full)
            iss_u8_full = np.uint8(np.clip(iss_matched_full, 0, 255))

            x0, y0, x1, y1 = normalized_crop_to_pixels(
                full_width,
                full_height,
                crop_x_start,
                crop_x_end,
                crop_y_start,
                crop_y_end,
            )
            # Fallback coherente con la rama ROI-aware: flujo ISS -> VIIRS.
            reference_crop = iss_u8_full[y0:y1, x0:x1]
            distorted_crop = viirs_u8_full[y0:y1, x0:x1]

        if coverage_mask_dir is None:
            coverage_mask_dir = os.path.join(flow_dir, "footprints")
        numeric_id = image_id.split("-")[-1]
        mask_file = Path(coverage_mask_dir) / f"iss_geometric_footprint_{numeric_id}.tif"
        if regenerate_footprints or not mask_file.is_file():
            # For previous RGB-only GeoTIFFs the footprint must be reconstructed.
            # If already available, the audited file is reused without TPS.
            _generate_footprint_tps(
                coverage_mask_dir, image_id, iss_file, source_images_dir, points_dir
            )
        footprint, mask_source = load_geometric_footprint(
            coverage_mask_dir, image_id, iss_file, x0, y0, x1, y1
        )
        if histogram_mode == "masked":
            reference_crop = match_histograms_inside_footprint(
                iss_gray_full[y0:y1, x0:x1], distorted_crop, footprint
            )
        elif histogram_mode != "legacy":
            raise ValueError(f"Unknown histogram mode: {histogram_mode}")
        flow, flow_valid = masked_flow_with_edge_guard(
            reference_crop, distorted_crop, footprint,
            margin_px=mask_margin_px, feather_px=mask_feather_px,
        )

        os.makedirs(flow_dir, exist_ok=True)
        np.save(flow_outfile, flow)
        valid_filename = f"{image_id}_flow_valid.npy"
        np.save(os.path.join(flow_dir, valid_filename), flow_valid.astype(np.uint8))
        write_flow_metadata(
            meta_path=meta_outfile,
            image_id=image_id,
            iss_file=iss_file,
            viirs_file=viirs_file,
            full_width=full_width,
            full_height=full_height,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            crop_x_start=crop_x_start,
            crop_x_end=crop_x_end,
            crop_y_start=crop_y_start,
            crop_y_end=crop_y_end,
            roi_source=roi_source,
            viirs_width=viirs_input_w,
            viirs_height=viirs_input_h,
        )
        with open(meta_outfile, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.update({
            "metadata_version": 5,
            "pipeline_flow_algorithm": "iss_viirs_masked_v2",
            "experimental_masked_flow": True,
            "mask_margin_validated": False,
            "iss_geometric_footprint_file": mask_source,
            "flow_valid_mask_file": valid_filename,
            "mask_margin_px": int(mask_margin_px),
            "mask_feather_px": int(mask_feather_px),
            "histogram_mode": str(histogram_mode),
            "flow_valid_pixel_count": int(flow_valid.sum()),
            "mask_origin": mask_source,
            "flow_roi_pixel_count": int(flow_valid.size),
            "mask_method": "common-neutral feathering + inward support guard",
        })
        with open(meta_outfile, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(
            f"✅ Flujo guardado: {flow_outfile} | "
            f"ROI x={x0}:{x1}, y={y0}:{y1}, "
            f"size={x1 - x0}x{y1 - y0}, source={roi_source}"
        )

        if show_plot:
            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
            flow_color[~flow_valid] = 255  # white outside trusted support
            plt.figure(figsize=(6, 6))
            plt.imshow(flow_color)
            plt.title(f"Flujo óptico: {image_id}")
            plt.axis("off")
            plot_path = os.path.join(flow_dir, f"{image_id}_flow_vis.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"🖼  Visualización guardada: {plot_path}")

    except Exception as e:
        print(f"ERROR {image_id}: {e}")
        raise  # fail closed: do not silently complete the pipeline


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cálculo de flujo óptico entre imágenes ISS georreferenciadas y VIIRS. "
            "El flujo se calcula en dirección ISS -> VIIRS, para poder corregir "
            "directamente los GCPs muestreados en la imagen ISS ya georreferenciada."
        )
    )
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--viirs_dir", type=str, required=True)
    parser.add_argument("--flow_dir", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--plot_every", type=int, default=100)
    parser.add_argument("--coverage_mask_dir", default=None,
                        help="Directory containing/storing geometric ISS footprint GeoTIFFs")
    parser.add_argument("--source_images_dir", default=None,
                        help="Original ISS JPG directory, used to build missing TPS footprints")
    parser.add_argument("--points_dir", default=None,
                        help="First-pass filtered GCP directory, used to build missing TPS footprints")
    parser.add_argument("--regenerate_footprints", action="store_true",
                        help="Rebuild masks even when a footprint already exists")
    parser.add_argument("--mask_margin_px", type=int, default=96,
                        help="Conservative inward margin from footprint/ROI edge")
    parser.add_argument("--mask_feather_px", type=int, default=32,
                        help="Transition to a shared neutral background near edges")
    parser.add_argument("--histogram_mode", choices=["masked", "legacy"], default="masked",
                        help="masked: match histograms only on ISS coverage; legacy: original matching")
    parser.add_argument("--crop_x_start", type=float, default=0.0)
    parser.add_argument("--crop_x_end", type=float, default=1.0)
    parser.add_argument("--crop_y_start", type=float, default=0.0)
    parser.add_argument("--crop_y_end", type=float, default=1.0)

    args = parser.parse_args()

    geo_dir = os.path.abspath(args.geo_dir)
    viirs_dir = os.path.abspath(args.viirs_dir)
    flow_dir = os.path.abspath(args.flow_dir)
    os.makedirs(flow_dir, exist_ok=True)

    all_ids = sorted(
        f.replace("_rect.tiff", "")
        for f in os.listdir(geo_dir)
        if f.endswith("_rect.tiff")
    )

    ids = [
        img_id
        for img_id in all_ids
        if img_id.split("-")[-1].isdigit()
        and args.start_id <= int(img_id.split("-")[-1]) <= args.end_id
    ]

    print(f"🔵 Encontrados {len(ids)} IDs en rango [{args.start_id}, {args.end_id}].")

    for idx, image_id in enumerate(ids):
        show_plot = args.plot_every > 0 and idx % args.plot_every == 0
        compute_and_save_optical_flow(
            image_id=image_id,
            geo_dir=geo_dir,
            viirs_dir=viirs_dir,
            flow_dir=flow_dir,
            crop_x_start=args.crop_x_start,
            crop_x_end=args.crop_x_end,
            crop_y_start=args.crop_y_start,
            crop_y_end=args.crop_y_end,
            show_plot=show_plot,
            coverage_mask_dir=args.coverage_mask_dir,
            mask_margin_px=args.mask_margin_px,
            mask_feather_px=args.mask_feather_px,
            histogram_mode=args.histogram_mode,
            source_images_dir=args.source_images_dir,
            points_dir=args.points_dir,
            regenerate_footprints=args.regenerate_footprints,
        )

    print("✅ Cálculo de flujo óptico completado.")


if __name__ == "__main__":
    main()
