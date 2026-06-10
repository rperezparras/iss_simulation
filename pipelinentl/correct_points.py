#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrección de puntos usando flujo óptico (pipeline v3).

NUEVO:
- Usa flow_dir/<id>_flow_roi.json para saber x0,x1,y0,y1 y dirección del flujo.
- Empareja puntos con pixel_mapping.csv por (sourceX, sourceY) (redondeado), NO por índice.
- Muestra el flow con bilinear para suavidad.
- Con flujo ISS->VIIRS, la corrección en píxel es:
      (geoX_corr, geoY_corr) = (geoX + u, geoY + v)

Luego se convierten esos píxeles corregidos a mapX/mapY usando la geotransform del *_rect.tiff.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from osgeo import gdal


def extract_id_from_filename(filename: str):
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        return int(base.split("-")[-1])
    except Exception:
        return None


def pixel_to_geo(px: float, py: float, geotransform):
    mapX = geotransform[0] + px * geotransform[1] + py * geotransform[2]
    mapY = geotransform[3] + px * geotransform[4] + py * geotransform[5]
    return mapX, mapY


def bilinear_flow_sample(flow: np.ndarray, x: float, y: float):
    h, w = flow.shape[:2]
    if x < 0 or y < 0 or x > (w - 1) or y > (h - 1):
        return 0.0, 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    f00 = flow[y0, x0]
    f10 = flow[y0, x1]
    f01 = flow[y1, x0]
    f11 = flow[y1, x1]

    f0 = f00 * (1 - dx) + f10 * dx
    f1 = f01 * (1 - dx) + f11 * dx
    f = f0 * (1 - dy) + f1 * dy

    return float(f[0]), float(f[1])


def correct_points_with_flow(
    points_path: str,
    flow_path: str,
    mapping_file: str,
    geo_image_path: str,
    corrected_points_path: str,
):
    # Leer puntos originales
    points = pd.read_csv(points_path)

    # Leer mapping con geoX/geoY (de georef_timelapse)
    mapping = pd.read_csv(mapping_file)

    required_cols = {"sourceX", "sourceY", "geoX", "geoY"}
    if not required_cols.issubset(set(mapping.columns)):
        raise ValueError(f"pixel_mapping.csv no tiene columnas requeridas {required_cols}: {mapping_file}")

    # Crear clave robusta por píxeles (redondeo)
    points = points.copy()
    mapping = mapping.copy()

    points["sx_i"] = np.rint(points["sourceX"].astype(float)).astype(int)
    points["sy_i"] = np.rint(points["sourceY"].astype(float)).astype(int)

    mapping["sx_i"] = np.rint(mapping["sourceX"].astype(float)).astype(int)
    mapping["sy_i"] = np.rint(mapping["sourceY"].astype(float)).astype(int)

    # Merge: nos traemos geoX/geoY para cada punto
    merged = points.merge(
        mapping[["sx_i", "sy_i", "geoX", "geoY"]],
        on=["sx_i", "sy_i"],
        how="left",
        suffixes=("", "_m"),
    )

    # Cargar flow
    flow = np.load(flow_path)  # (Hroi, Wroi, 2)

    # Leer ROI json asociado al flow
    roi_json = flow_path.replace("_flow.npy", "_flow_roi.json")
    if not os.path.exists(roi_json):
        raise FileNotFoundError(f"No existe ROI json del flow: {roi_json}")

    with open(roi_json, "r") as f:
        roi = json.load(f)

    if not all(k in roi for k in ["x0", "x1", "y0", "y1", "flow_direction"]):
        raise ValueError(f"ROI json incompleto: {roi_json}")

    x0, x1, y0, y1 = int(roi["x0"]), int(roi["x1"]), int(roi["y0"]), int(roi["y1"])
    flow_dir = roi["flow_direction"]

    # Abrir geotiff para geotransform
    dataset = gdal.Open(geo_image_path)
    if dataset is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen georreferenciada: {geo_image_path}")
    geotransform = dataset.GetGeoTransform()

    roi_w = x1 - x0
    roi_h = y1 - y0

    if flow.shape[0] != roi_h or flow.shape[1] != roi_w:
        raise ValueError(
            f"Flow dims {flow.shape[1]}x{flow.shape[0]} != ROI {roi_w}x{roi_h} en {points_path}. "
            f"Revisa viirs_roi_crop (roi_exact) y optical_flow."
        )

    corrected_mapX = []
    corrected_mapY = []
    dX = []
    dY = []

    for _, row in merged.iterrows():
        geoX = row.get("geoX", np.nan)
        geoY = row.get("geoY", np.nan)

        if not (np.isfinite(geoX) and np.isfinite(geoY)):
            # No se pudo mapear este punto a píxel geo -> no corregimos
            corrected_mapX.append(row["mapX"])
            corrected_mapY.append(row["mapY"])
            dX.append(0.0)
            dY.append(0.0)
            continue

        # coords relativas dentro del ROI
        rel_x = float(geoX) - float(x0)
        rel_y = float(geoY) - float(y0)

        if 0 <= rel_x <= (roi_w - 1) and 0 <= rel_y <= (roi_h - 1):
            u, v = bilinear_flow_sample(flow, rel_x, rel_y)

            # Dirección del flujo:
            # flow_direction = "ISS_to_VIIRS": geoX_corr = geoX + u
            # Si algún día cambiases la dirección, lo controlas aquí.
            if flow_dir == "ISS_to_VIIRS":
                dx_pix, dy_pix = u, v
            elif flow_dir == "VIIRS_to_ISS":
                dx_pix, dy_pix = -u, -v
            else:
                dx_pix, dy_pix = u, v

            new_geoX = float(geoX) + float(dx_pix)
            new_geoY = float(geoY) + float(dy_pix)
        else:
            dx_pix, dy_pix = 0.0, 0.0
            new_geoX, new_geoY = float(geoX), float(geoY)

        new_mapX, new_mapY = pixel_to_geo(new_geoX, new_geoY, geotransform)

        corrected_mapX.append(new_mapX)
        corrected_mapY.append(new_mapY)
        dX.append(dx_pix)
        dY.append(dy_pix)

    # actualizar points originales (sin columnas auxiliares)
    points_out = points.copy()
    points_out["mapX"] = corrected_mapX
    points_out["mapY"] = corrected_mapY
    points_out["dX"] = dX
    points_out["dY"] = dY
    points_out["residual"] = 0.0
    points_out["enable"] = 1

    # Formato QGIS
    points_out = points_out[["mapX", "mapY", "sourceX", "sourceY", "enable", "dX", "dY", "residual"]]

    os.makedirs(os.path.dirname(corrected_points_path), exist_ok=True)
    points_out.to_csv(corrected_points_path, index=False)
    print(f"✅ Corregido: {corrected_points_path}")


def main():
    parser = argparse.ArgumentParser(description="Corrección de puntos con flujo óptico (ROI por GCPs).")
    parser.add_argument("--input_points_dir", type=str, required=True)
    parser.add_argument("--flow_dir", type=str, required=True)
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)

    args = parser.parse_args()

    input_points_dir = os.path.abspath(args.input_points_dir)
    flow_dir = os.path.abspath(args.flow_dir)
    geo_dir = os.path.abspath(args.geo_dir)
    output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    all_point_files = sorted(f for f in os.listdir(input_points_dir) if f.endswith(".points"))

    selected_files = []
    for fname in all_point_files:
        file_id = extract_id_from_filename(fname)
        if file_id is not None and args.start_id <= file_id <= args.end_id:
            selected_files.append(fname)

    print(f"🔵 Archivos .points a procesar en rango [{args.start_id}, {args.end_id}]: {len(selected_files)}")

    for fname in selected_files:
        base_name = os.path.splitext(fname)[0]  # 'ISS067-E-201283'

        points_path = os.path.join(input_points_dir, fname)
        flow_path = os.path.join(flow_dir, f"{base_name}_flow.npy")
        mapping_file = os.path.join(geo_dir, f"{base_name}_pixel_mapping.csv")
        geo_image_path = os.path.join(geo_dir, f"{base_name}_rect.tiff")
        corrected_points_path = os.path.join(output_dir, f"{base_name}_corrected.points")

        if not os.path.exists(flow_path):
            print(f"⚠️ Sin flujo para {base_name}, se omite: {flow_path}")
            continue
        if not os.path.exists(mapping_file):
            print(f"⚠️ Sin pixel_mapping.csv para {base_name}, se omite: {mapping_file}")
            continue
        if not os.path.exists(geo_image_path):
            print(f"⚠️ Sin geo tiff para {base_name}, se omite: {geo_image_path}")
            continue

        try:
            correct_points_with_flow(
                points_path=points_path,
                flow_path=flow_path,
                mapping_file=mapping_file,
                geo_image_path=geo_image_path,
                corrected_points_path=corrected_points_path,
            )
        except Exception as e:
            print(f"❌ Error corrigiendo {base_name}: {e}")

    print("✅ Corrección de puntos completada.")


if __name__ == "__main__":
    main()