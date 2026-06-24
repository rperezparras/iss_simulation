#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrección de puntos usando flujo óptico.

Entrada:
  - filtered_points/*.points
  - flow/*_flow.npy
  - flow/*_flow_meta.json, generado por optical_flow.py
  - geo/*_pixel_mapping.csv y geo/*_rect.tiff

El flujo óptico puede estar calculado sobre una ROI del GeoTIFF. Por eso este
script lee el *_flow_meta.json para conocer el origen exacto de esa ROI. Si el
metadato no existe, se conserva un fallback compatible con versiones antiguas.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()


def extract_id_from_filename(filename: str):
    """Extrae el ID numérico final de nombres tipo 'ISS067-E-201283.points'."""
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        return int(base.split("-")[-1])
    except Exception:
        return None


def pixel_to_geo(px: float, py: float, geotransform):
    """Convierte píxel (col=px, fila=py) a coordenadas geográficas."""
    map_x = geotransform[0] + px * geotransform[1] + py * geotransform[2]
    map_y = geotransform[3] + px * geotransform[4] + py * geotransform[5]
    return map_x, map_y


def normalized_crop_to_pixels(
    width: int,
    height: int,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
) -> tuple[int, int, int, int]:
    """Misma conversión que en optical_flow.py. Devuelve x0, y0, x1, y1."""
    crop_x_start = max(0.0, min(1.0, float(crop_x_start)))
    crop_x_end = max(0.0, min(1.0, float(crop_x_end)))
    crop_y_start = max(0.0, min(1.0, float(crop_y_start)))
    crop_y_end = max(0.0, min(1.0, float(crop_y_end)))

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


def flow_meta_path_from_flow_path(flow_path: str) -> str:
    path = Path(flow_path)
    name = path.name
    if name.endswith("_flow.npy"):
        return str(path.with_name(name.replace("_flow.npy", "_flow_meta.json")))
    return str(path.with_suffix(".json"))


def infer_roi_from_mapping(mapping: pd.DataFrame, flow_w: int, flow_h: int, width: int, height: int):
    """
    Fallback para flujos antiguos sin JSON: infiere una ROI centrada en los GCPs.
    Se usa sólo para compatibilidad; la ruta robusta es leer *_flow_meta.json.
    """
    gx = pd.to_numeric(mapping["geoX"], errors="coerce").dropna().to_numpy(dtype=float)
    gy = pd.to_numeric(mapping["geoY"], errors="coerce").dropna().to_numpy(dtype=float)
    if gx.size == 0 or gy.size == 0:
        raise ValueError("No se pudo inferir la ROI: geoX/geoY no contienen valores válidos.")

    cx = 0.5 * (float(gx.min()) + float(gx.max()))
    cy = 0.5 * (float(gy.min()) + float(gy.max()))

    x0 = int(round(cx - 0.5 * flow_w))
    y0 = int(round(cy - 0.5 * flow_h))
    x0 = max(0, min(x0, max(0, width - flow_w)))
    y0 = max(0, min(y0, max(0, height - flow_h)))
    x1 = x0 + flow_w
    y1 = y0 + flow_h
    return x0, y0, x1, y1


def get_flow_roi(
    flow_path: str,
    flow: np.ndarray,
    width: int,
    height: int,
    mapping: pd.DataFrame,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
):
    """
    Devuelve x0, y0, x1, y1 y origen de la información.

    Prioridad:
      1. *_flow_meta.json, robusto y exacto.
      2. crop_* de CLI si coinciden con el tamaño del flujo.
      3. inferencia desde pixel_mapping.csv para compatibilidad con flujos viejos.
    """
    flow_h, flow_w = flow.shape[:2]
    meta_path = flow_meta_path_from_flow_path(flow_path)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        x0 = int(meta["x0"])
        y0 = int(meta["y0"])
        x1 = int(meta["x1"])
        y1 = int(meta["y1"])

        meta_flow_w = int(meta.get("flow_width", x1 - x0))
        meta_flow_h = int(meta.get("flow_height", y1 - y0))
        meta_full_w = int(meta.get("full_width", width))
        meta_full_h = int(meta.get("full_height", height))

        if meta_flow_w != flow_w or meta_flow_h != flow_h:
            raise ValueError(
                f"Metadatos incompatibles con {os.path.basename(flow_path)}: "
                f"meta={meta_flow_w}x{meta_flow_h}, flow={flow_w}x{flow_h}."
            )

        if meta_full_w != width or meta_full_h != height:
            raise ValueError(
                f"El flujo fue calculado sobre un GeoTIFF de tamaño "
                f"{meta_full_w}x{meta_full_h}, pero el GeoTIFF actual mide "
                f"{width}x{height}. Recalcula optical_flow para esta geo/."
            )

        if x1 - x0 != flow_w or y1 - y0 != flow_h:
            raise ValueError(
                f"ROI incompatible con el flujo: ROI={x1 - x0}x{y1 - y0}, "
                f"flow={flow_w}x{flow_h}."
            )

        return x0, y0, x1, y1, "metadata"

    # Fallback antiguo: usar crop_* si da exactamente el tamaño del flujo.
    x0, y0, x1, y1 = normalized_crop_to_pixels(
        width,
        height,
        crop_x_start,
        crop_x_end,
        crop_y_start,
        crop_y_end,
    )
    if (x1 - x0) == flow_w and (y1 - y0) == flow_h:
        return x0, y0, x1, y1, "crop_args"

    # Último recurso para no perder ejecuciones antiguas.
    x0, y0, x1, y1 = infer_roi_from_mapping(mapping, flow_w, flow_h, width, height)
    print(
        f"⚠️ Flujo sin *_flow_meta.json y crop_* no coincide. "
        f"Se infiere ROI x={x0}:{x1}, y={y0}:{y1}. "
        "Para resultados reproducibles, recalcula optical_flow.py con la versión nueva."
    )
    return x0, y0, x1, y1, "inferred_from_mapping"


def correct_points_with_flow(
    points_path: str,
    flow_path: str,
    mapping_file: str,
    geo_image_path: str,
    corrected_points_path: str,
    crop_x_start: float,
    crop_x_end: float,
    crop_y_start: float,
    crop_y_end: float,
):
    """Corrige un .points usando el flujo óptico y el pixel_mapping.csv."""
    points = pd.read_csv(points_path)
    mapping = pd.read_csv(mapping_file)

    required_mapping_cols = {"geoX", "geoY"}
    missing = required_mapping_cols - set(mapping.columns)
    if missing:
        raise ValueError(f"El mapping no contiene columnas {sorted(missing)}: {mapping_file}")

    required_point_cols = {"mapX", "mapY", "sourceX", "sourceY"}
    missing_points = required_point_cols - set(points.columns)
    if missing_points:
        raise ValueError(f"El .points no contiene columnas {sorted(missing_points)}: {points_path}")

    if len(mapping) != len(points):
        n = min(len(mapping), len(points))
        print(
            f"⚠️ Aviso: mapping ({len(mapping)}) y points ({len(points)}) tienen distinta "
            f"longitud. Se usarán los primeros {n} registros por índice."
        )
        points = points.iloc[:n].copy()
        mapping = mapping.iloc[:n].copy()
    else:
        points = points.copy()
        mapping = mapping.copy()

    if len(points) == 0:
        raise ValueError(f"No hay puntos válidos en {points_path}")

    flow = np.load(flow_path)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"El flujo debe tener forma (H, W, 2), recibido {flow.shape}: {flow_path}")

    dataset = gdal.Open(geo_image_path)
    if dataset is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen georreferenciada: {geo_image_path}")
    geotransform = dataset.GetGeoTransform()
    width = int(dataset.RasterXSize)
    height = int(dataset.RasterYSize)

    x0, y0, x1, y1, roi_source = get_flow_roi(
        flow_path=flow_path,
        flow=flow,
        width=width,
        height=height,
        mapping=mapping,
        crop_x_start=crop_x_start,
        crop_x_end=crop_x_end,
        crop_y_start=crop_y_start,
        crop_y_end=crop_y_end,
    )

    geo_x = pd.to_numeric(mapping["geoX"], errors="coerce").to_numpy(dtype=float)
    geo_y = pd.to_numeric(mapping["geoY"], errors="coerce").to_numpy(dtype=float)

    corrected_map_x = []
    corrected_map_y = []
    dx_values = []
    dy_values = []

    inside_flow_count = 0
    nonzero_flow_count = 0

    for geo_x_pix, geo_y_pix in zip(geo_x, geo_y):
        if not np.isfinite(geo_x_pix) or not np.isfinite(geo_y_pix):
            dx_pix, dy_pix = 0.0, 0.0
            new_geo_x_pix, new_geo_y_pix = geo_x_pix, geo_y_pix
        else:
            x_int = int(round(float(geo_x_pix)))
            y_int = int(round(float(geo_y_pix)))
            rel_x = x_int - x0
            rel_y = y_int - y0

            if 0 <= rel_x < flow.shape[1] and 0 <= rel_y < flow.shape[0]:
                inside_flow_count += 1
                dx_pix, dy_pix = -flow[rel_y, rel_x]
                dx_pix = float(dx_pix)
                dy_pix = float(dy_pix)
                if abs(dx_pix) > 1e-6 or abs(dy_pix) > 1e-6:
                    nonzero_flow_count += 1
                new_geo_x_pix = float(geo_x_pix) + dx_pix
                new_geo_y_pix = float(geo_y_pix) + dy_pix
            else:
                dx_pix, dy_pix = 0.0, 0.0
                new_geo_x_pix, new_geo_y_pix = float(geo_x_pix), float(geo_y_pix)

        new_map_x, new_map_y = pixel_to_geo(new_geo_x_pix, new_geo_y_pix, geotransform)
        corrected_map_x.append(new_map_x)
        corrected_map_y.append(new_map_y)
        dx_values.append(dx_pix)
        dy_values.append(dy_pix)

    points["mapX"] = corrected_map_x
    points["mapY"] = corrected_map_y
    points["dX"] = dx_values
    points["dY"] = dy_values
    points["residual"] = 0.0
    points["enable"] = 1

    output_cols = ["mapX", "mapY", "sourceX", "sourceY", "enable", "dX", "dY", "residual"]
    output = points[output_cols]

    os.makedirs(os.path.dirname(corrected_points_path), exist_ok=True)
    output.to_csv(corrected_points_path, index=False)

    stats = {
        "points_file": os.path.abspath(points_path),
        "flow_file": os.path.abspath(flow_path),
        "mapping_file": os.path.abspath(mapping_file),
        "geo_image": os.path.abspath(geo_image_path),
        "output_file": os.path.abspath(corrected_points_path),
        "n_points": int(len(points)),
        "inside_flow_count": int(inside_flow_count),
        "inside_flow_percent": float(100.0 * inside_flow_count / len(points)),
        "nonzero_flow_count": int(nonzero_flow_count),
        "nonzero_flow_percent": float(100.0 * nonzero_flow_count / len(points)),
        "roi_source": roi_source,
        "roi": {
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
        },
        "full_width": int(width),
        "full_height": int(height),
    }

    stats_path = corrected_points_path.replace("_corrected.points", "_correction_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Corregido: {corrected_points_path} | "
        f"inside_flow={inside_flow_count}/{len(points)} "
        f"({100.0 * inside_flow_count / len(points):.1f}%), "
        f"nonzero={nonzero_flow_count}/{len(points)} "
        f"({100.0 * nonzero_flow_count / len(points):.1f}%), "
        f"ROI={roi_source} x={x0}:{x1}, y={y0}:{y1}"
    )


def main():
    parser = argparse.ArgumentParser(description="Corrección de puntos con flujo óptico.")
    parser.add_argument("--input_points_dir", type=str, required=True)
    parser.add_argument("--flow_dir", type=str, required=True)
    parser.add_argument("--geo_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--crop_x_start", type=float, default=0.0)
    parser.add_argument("--crop_x_end", type=float, default=1.0)
    parser.add_argument("--crop_y_start", type=float, default=0.0)
    parser.add_argument("--crop_y_end", type=float, default=1.0)

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

    processed = 0
    failed = 0
    for fname in selected_files:
        base_name = os.path.splitext(fname)[0]
        points_path = os.path.join(input_points_dir, fname)
        flow_path = os.path.join(flow_dir, f"{base_name}_flow.npy")
        mapping_file = os.path.join(geo_dir, f"{base_name}_pixel_mapping.csv")
        geo_image_path = os.path.join(geo_dir, f"{base_name}_rect.tiff")
        corrected_points_path = os.path.join(output_dir, f"{base_name}_corrected.points")

        missing = [
            ("flujo óptico", flow_path),
            ("pixel_mapping.csv", mapping_file),
            ("imagen georreferenciada", geo_image_path),
        ]
        missing = [(label, path) for label, path in missing if not os.path.exists(path)]
        if missing:
            for label, path in missing:
                print(f"⚠️ Sin {label} para {base_name}, se omite: {path}")
            continue

        try:
            correct_points_with_flow(
                points_path=points_path,
                flow_path=flow_path,
                mapping_file=mapping_file,
                geo_image_path=geo_image_path,
                corrected_points_path=corrected_points_path,
                crop_x_start=args.crop_x_start,
                crop_x_end=args.crop_x_end,
                crop_y_start=args.crop_y_start,
                crop_y_end=args.crop_y_end,
            )
            processed += 1
        except Exception as e:
            failed += 1
            print(f"❌ Error corrigiendo {base_name}: {e}")

    print(f"✅ Corrección de puntos completada. Procesados: {processed}. Errores: {failed}.")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
