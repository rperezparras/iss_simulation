#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de flujo óptico entre imágenes ISS georreferenciadas y VIIRS.

Para cada imagen georreferenciada:
  1. Lee la imagen ISS *_rect.tiff desde geo_dir.
  2. Lee la imagen VIIRS correspondiente *_viirs.tiff desde viirs_dir.
  3. Redimensiona VIIRS a la resolución de la ISS georreferenciada.
  4. Preprocesa ISS/VIIRS y calcula flujo óptico Farneback sobre una ROI.
  5. Guarda:
       - *_flow.npy       : campo de flujo en coordenadas de la ROI.
       - *_flow_meta.json : metadatos exactos de la ROI dentro del GeoTIFF.

El JSON de metadatos es necesario para que correct_points.py pueda aplicar el
flujo en el sistema de coordenadas correcto sin inferencias ambiguas.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import flow_vis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
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
    """Convierte crop normalizado [0,1] a píxeles usando la misma regla siempre."""
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
):
    meta = {
        "metadata_version": 1,
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
        "crop_x_start": float(crop_x_start),
        "crop_x_end": float(crop_x_end),
        "crop_y_start": float(crop_y_start),
        "crop_y_end": float(crop_y_end),
        "flow_definition": "cv2.calcOpticalFlowFarneback(reference_viirs, distorted_iss)",
        "correction_to_apply_in_correct_points": "new_pixel = old_pixel - flow[y, x]",
        "farneback_params": FARNEBACK_PARAMS,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


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
        full_height, full_width = iss.shape[:2]
        viirs = cv2.resize(viirs, (full_width, full_height), interpolation=cv2.INTER_LINEAR)

        vmax = float(np.max(viirs))
        if vmax <= 0:
            raise ValueError(f"VIIRS para {image_id} tiene máximo <= 0; no se puede normalizar.")
        viirs_u8 = np.uint8(np.clip(viirs / vmax * 255.0, 0, 255))

        iss_gray = cv2.cvtColor(iss, cv2.COLOR_RGB2GRAY)
        iss_gray_matched = exposure.match_histograms(iss_gray, viirs_u8)
        iss_gray_matched = np.asarray(iss_gray_matched)
        if iss_gray_matched.dtype != np.uint8:
            iss_gray_matched = np.uint8(np.clip(iss_gray_matched, 0, 255))

        x0, y0, x1, y1 = normalized_crop_to_pixels(
            full_width,
            full_height,
            crop_x_start,
            crop_x_end,
            crop_y_start,
            crop_y_end,
        )

        reference_crop = viirs_u8[y0:y1, x0:x1]
        distorted_crop = iss_gray_matched[y0:y1, x0:x1]

        flow = cv2.calcOpticalFlowFarneback(
            reference_crop,
            distorted_crop,
            None,
            FARNEBACK_PARAMS["pyr_scale"],
            FARNEBACK_PARAMS["levels"],
            FARNEBACK_PARAMS["winsize"],
            FARNEBACK_PARAMS["iterations"],
            FARNEBACK_PARAMS["poly_n"],
            FARNEBACK_PARAMS["poly_sigma"],
            FARNEBACK_PARAMS["flags"],
        )

        os.makedirs(flow_dir, exist_ok=True)
        np.save(flow_outfile, flow)
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
        )

        print(
            f"✅ Flujo guardado: {flow_outfile} | "
            f"ROI x={x0}:{x1}, y={y0}:{y1}, size={x1 - x0}x{y1 - y0}"
        )

        if show_plot:
            flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
            plt.figure(figsize=(6, 6))
            plt.imshow(flow_color)
            plt.title(f"Flujo óptico: {image_id}")
            plt.axis("off")
            plot_path = os.path.join(flow_dir, f"{image_id}_flow_vis.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"🖼  Visualización guardada: {plot_path}")

    except Exception as e:
        print(f"❌ Error con {image_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Cálculo de flujo óptico entre imágenes ISS georreferenciadas y VIIRS."
    )
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--viirs_dir", type=str, required=True)
    parser.add_argument("--flow_dir", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--plot_every", type=int, default=100)
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
        )

    print("✅ Cálculo de flujo óptico completado.")


if __name__ == "__main__":
    main()
