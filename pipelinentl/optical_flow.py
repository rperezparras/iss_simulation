#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de flujo óptico entre imágenes ISS georreferenciadas y VIIRS (pipeline v3).

NUEVO:
- Usa ROI por GCPs guardada en viirs_dir/<id>_roi.json (creada por viirs_roi_crop.py).
- Calcula flujo en dirección ISS -> VIIRS para que la corrección de puntos sea directa.
- Guarda flow.npy (sobre ROI) y flow_roi.json (x0,x1,y0,y1) en flow_dir.

Si no existe roi.json, cae a recorte manual (crop_*), pero para tu pipeline lo normal
es usar ROI por GCPs.

Salida:
- <id>_flow.npy           (Hroi, Wroi, 2)
- <id>_flow_roi.json      (ROI + dirección)

"""

import os
import json
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import exposure
import flow_vis
import rasterio
from rasterio.windows import Window


def _normalize_to_u8(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(img)) if img.size else 0.0
    if vmax <= 0:
        return np.zeros_like(img, dtype=np.uint8)
    out = np.clip(img / vmax * 255.0, 0, 255).astype(np.uint8)
    return out


def _load_roi_from_json(viirs_dir: str, image_id: str):
    roi_json = os.path.join(viirs_dir, f"{image_id}_roi.json")
    if not os.path.exists(roi_json):
        return None, None
    with open(roi_json, "r") as f:
        roi = json.load(f)

    if not all(k in roi for k in ["x0", "x1", "y0", "y1"]):
        return roi, None

    x0, x1, y0, y1 = int(roi["x0"]), int(roi["x1"]), int(roi["y0"]), int(roi["y1"])
    return roi, (x0, x1, y0, y1)


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
    iss_file = os.path.join(geo_dir, f"{image_id}_rect.tiff")
    viirs_file = os.path.join(viirs_dir, f"{image_id}_viirs.tiff")

    flow_outfile = os.path.join(flow_dir, f"{image_id}_flow.npy")
    flow_roi_json = os.path.join(flow_dir, f"{image_id}_flow_roi.json")

    if not os.path.exists(iss_file):
        raise FileNotFoundError(f"No se encontró la imagen ISS: {iss_file}")
    if not os.path.exists(viirs_file):
        raise FileNotFoundError(f"No se encontró la imagen VIIRS: {viirs_file}")

    # 1) Cargar ROI (preferente desde JSON del VIIRS)
    roi_meta, roi_window = _load_roi_from_json(viirs_dir, image_id)

    with rasterio.open(iss_file) as iss_src:
        H_full, W_full = iss_src.height, iss_src.width

        if roi_window is None:
            # fallback recorte manual por fracciones sobre la imagen completa
            x0 = int(W_full * crop_x_start)
            x1 = int(W_full * crop_x_end)
            y0 = int(H_full * crop_y_start)
            y1 = int(H_full * crop_y_end)

            x0 = max(0, min(x0, W_full - 1))
            x1 = max(x0 + 1, min(x1, W_full))
            y0 = max(0, min(y0, H_full - 1))
            y1 = max(y0 + 1, min(y1, H_full))

            roi_window = (x0, x1, y0, y1)
            roi_meta = {
                "roi_mode": "manual",
                "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                "ref_width": W_full, "ref_height": H_full
            }

        x0, x1, y0, y1 = roi_window
        roi_w = x1 - x0
        roi_h = y1 - y0

        # 2) Leer ISS ROI (RGB) usando window para no cargar todo
        win = Window(col_off=x0, row_off=y0, width=roi_w, height=roi_h)
        ISS_rgb = iss_src.read([1, 2, 3], window=win)
        ISS_rgb = np.transpose(ISS_rgb, (1, 2, 0))  # (Hroi, Wroi, 3)

    # 3) Leer VIIRS (debería ya ser ROI y estar alineado al grid)
    with rasterio.open(viirs_file) as vsrc:
        VIIRS = vsrc.read(1)

    # Si el VIIRS ya es ROI exacta, sus dimensiones deben coincidir
    if VIIRS.shape[0] != roi_h or VIIRS.shape[1] != roi_w:
        # fallback: si VIIRS fuese full size, recortamos
        if VIIRS.shape[0] == roi_meta.get("ref_height", -1) and VIIRS.shape[1] == roi_meta.get("ref_width", -1):
            VIIRS = VIIRS[y0:y1, x0:x1]
        else:
            raise ValueError(
                f"VIIRS dims {VIIRS.shape[1]}x{VIIRS.shape[0]} no coinciden con ROI {roi_w}x{roi_h} "
                f"para {image_id}. Revisa viirs_roi_crop (align roi_exact)."
            )

    # 4) Preprocesado
    viirs_u8 = _normalize_to_u8(VIIRS)

    iss_gray = cv2.cvtColor(ISS_rgb, cv2.COLOR_RGB2GRAY)
    iss_gray = iss_gray.astype(np.uint8)

    # Igualar histograma de ISS a VIIRS para ayudar al flujo
    iss_gray_matched = exposure.match_histograms(iss_gray, viirs_u8)
    iss_gray_matched = np.clip(iss_gray_matched, 0, 255).astype(np.uint8)

    # 5) Flujo óptico: ISS -> VIIRS
    # prev = ISS, next = VIIRS  => flow(p) = desplazamiento para ir de ISS a VIIRS
    flow = cv2.calcOpticalFlowFarneback(
        iss_gray_matched,
        viirs_u8,
        None,
        0.5,   # pyr_scale
        3,     # levels
        19,    # winsize
        3,     # iterations
        7,     # poly_n
        1.5,   # poly_sigma
        0,     # flags
    )

    os.makedirs(flow_dir, exist_ok=True)
    np.save(flow_outfile, flow)

    # Guardar metadatos ROI para correct_points.py
    roi_out = dict(roi_meta)
    roi_out.update({
        "x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1),
        "roi_width": int(roi_w), "roi_height": int(roi_h),
        "flow_direction": "ISS_to_VIIRS",
        "flow_units": "pixels",
    })
    with open(flow_roi_json, "w") as f:
        json.dump(roi_out, f)

    print(f"✅ Flujo guardado: {flow_outfile}")
    print(f"✅ ROI guardada:  {flow_roi_json}")

    if show_plot:
        flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        plt.figure(figsize=(6, 6))
        plt.imshow(flow_color)
        plt.title(f"Flow ISS→VIIRS (ROI): {image_id}")
        plt.axis("off")
        plot_path = os.path.join(flow_dir, f"{image_id}_flow_vis.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"🖼  Visualización guardada: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Flujo óptico entre ISS (geo) y VIIRS (alineado) con ROI por GCPs."
    )
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--viirs_dir", type=str, required=True)
    parser.add_argument("--flow_dir", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--plot_every", type=int, default=100)

    # fallback manual (solo si no hay roi.json)
    parser.add_argument("--crop_x_start", type=float, default=0.0)
    parser.add_argument("--crop_x_end", type=float, default=1.0)
    parser.add_argument("--crop_y_start", type=float, default=0.0)
    parser.add_argument("--crop_y_end", type=float, default=1.0)

    args = parser.parse_args()

    geo_dir = os.path.abspath(args.geo_dir)
    viirs_dir = os.path.abspath(args.viirs_dir)
    flow_dir = os.path.abspath(args.flow_dir)

    os.makedirs(flow_dir, exist_ok=True)

    # IDs a partir de *_rect.tiff
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
        show_plot = (args.plot_every > 0 and idx % args.plot_every == 0)
        try:
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
        except Exception as e:
            print(f"❌ Error con {image_id}: {e}")

    print("✅ Cálculo de flujo óptico completado.")


if __name__ == "__main__":
    main()