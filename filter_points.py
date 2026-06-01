#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtrado espacial + coherencia temporal de archivos .points de un timelapse ISS.

Esta versión sustituye/amplía el antiguo filter_points.py:

1) Lee los .points generados por project_timelapse.py, normalmente *_real.points.
2) Identifica cada trayectoria temporal por el punto de la rejilla real:
       track_id = (sourceX, sourceY)
   Es decir, el mismo punto de la imagen real a través de todos los frames.
3) Marca como inválidos los puntos geográficamente aislados dentro de cada frame
   usando BallTree/haversine, igual que el filtro anterior.
4) Para cada track_id ajusta polinomios robustos lat(t), lon(t):
       mapY = latitud
       mapX = longitud
   y detecta observaciones temporalmente incoherentes por residuo geodésico.
5) Imputa los outliers y pequeños huecos mediante el polinomio temporal.
6) Opcionalmente vuelve a aplicar el filtro espacial final.
7) Guarda .points listos para georef_timelapse.py con nombres:
       <mission>-E-<ID>.points
8) Guarda CSVs de control de calidad y plots opcionales.

Uso recomendado en la pipeline, después de project_timelapse.py:

    python -m scripts_v3.filter_points \
        --input_folder ISS030-E-281044-281946/output \
        --output_folder ISS030-E-281044-281946/filtered_points \
        --radius_km 40 \
        --start_id 281044 \
        --end_id 281946 \
        --mission ISS030 \
        --temporal_order 3 \
        --temporal_outlier_km 80 \
        --max_gap_frames 8 \
        --plot_dir ISS030-E-281044-281946/temporal_qc_plots

Si quieres reproducir el comportamiento antiguo casi exactamente:

    python -m scripts_v3.filter_points ... --disable_temporal --no_pre_spatial_filter

Notas importantes:
- No se usa el índice de fila para emparejar puntos, sino sourceX/sourceY.
  Esto evita descuadres cuando un frame pierde algunos puntos.
- La longitud se ajusta con unwrap angular para evitar saltos en ±180 grados.
- Por defecto se imputan outliers y huecos interiores cortos. No se extrapolan
  huecos al principio/final de una trayectoria salvo que uses --allow_extrapolation.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = {"mapX", "mapY", "sourceX", "sourceY"}
DEFAULT_OUTPUT_COLUMNS = ["mapX", "mapY", "sourceX", "sourceY", "enable", "dX", "dY", "residual"]


@dataclass
class FrameInfo:
    frame_index: int
    frame_id: int
    input_path: Path
    input_name: str
    output_name: str
    input_count: int = 0
    pre_spatial_kept: int = 0
    output_before_post_spatial: int = 0
    post_spatial_kept: int = 0
    n_imputed: int = 0
    n_temporal_outliers: int = 0
    n_pre_spatial_rejected_used_as_missing: int = 0


@dataclass
class TrackResult:
    key: str
    sourceX: float
    sourceY: float
    observed: np.ndarray
    finite: np.ndarray
    pre_spatial_ok: np.ndarray
    temporal_good: np.ndarray
    temporal_outlier: np.ndarray
    imputed: np.ndarray
    output_present: np.ndarray
    raw_lon: np.ndarray
    raw_lat: np.ndarray
    pred_lon: np.ndarray
    pred_lat: np.ndarray
    residual_km: np.ndarray
    threshold_km: float
    degree: int
    rmse_km: float
    max_residual_km: float


def extract_id_from_point_filename(name: str) -> Optional[int]:
    """
    Extrae el ID real desde nombres como:
      ISS067-E-327041_real.points
      ISS067-E-327041.points
      coordinates_... -> None
    """
    m = re.search(r"[A-Za-z0-9]+-E-(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def make_source_key(source_x: Any, source_y: Any, decimals: int) -> str:
    """
    Clave estable para identificar el mismo punto de la rejilla real.
    Se redondea para evitar ruido flotante mínimo.
    """
    sx = round(float(source_x), decimals)
    sy = round(float(source_y), decimals)
    return f"{sx:.{decimals}f}|{sy:.{decimals}f}"


def parse_source_key(key: str) -> Tuple[float, float]:
    sx, sy = key.split("|", 1)
    return float(sx), float(sy)


def wrap_lon_deg(lon: np.ndarray | float) -> np.ndarray | float:
    """Envuelve longitud a [-180, 180)."""
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def haversine_km(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """Distancia geodésica aproximada sobre esfera, vectorizada."""
    lat1 = np.radians(lat1_deg.astype(float))
    lat2 = np.radians(lat2_deg.astype(float))
    lon1 = np.radians(lon1_deg.astype(float))
    lon2 = np.radians(lon2_deg.astype(float))

    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + np.pi) % (2.0 * np.pi) - np.pi
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def spatial_neighbor_mask(df: pd.DataFrame, radius_km: float) -> np.ndarray:
    """
    Devuelve True para puntos que tienen al menos otro vecino dentro de radius_km.
    Reproduce el criterio del filter_points.py original, pero tolera NaNs.
    """
    mask = np.zeros(len(df), dtype=bool)
    if len(df) == 0:
        return mask

    required = ["mapY", "mapX"]
    if not set(required).issubset(df.columns):
        return mask

    coords_deg = df[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(coords_deg).all(axis=1)
    finite_idx = np.where(finite)[0]

    if len(finite_idx) < 2:
        return mask

    coords_rad = np.radians(coords_deg[finite_idx])
    radius_rad = float(radius_km) / EARTH_RADIUS_KM
    tree = BallTree(coords_rad, metric="haversine")
    neighbors = tree.query_radius(coords_rad, r=radius_rad)
    keep_finite = np.array([len(n) > 1 for n in neighbors], dtype=bool)
    mask[finite_idx] = keep_finite
    return mask


def list_point_files(input_folder: Path, input_glob: str) -> List[Path]:
    files = sorted(Path(p) for p in glob(str(input_folder / input_glob)))
    return files


def read_points_file(path: Path) -> pd.DataFrame:
    """Lee .points tolerando líneas de metadatos QGIS que empiecen por M."""
    return pd.read_csv(path, comment="M")


def coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def discover_output_columns(input_columns: List[str], add_qc_columns: bool) -> List[str]:
    """
    Mantiene las columnas de entrada si existen y añade las típicas de QGIS si faltan.
    Por defecto no añade columnas QC para no romper lectores externos estrictos.
    """
    cols = list(input_columns)
    for c in DEFAULT_OUTPUT_COLUMNS:
        if c not in cols:
            cols.append(c)
    if add_qc_columns:
        for c in ["temporal_status", "temporal_residual_km"]:
            if c not in cols:
                cols.append(c)
    return cols


def load_timelapse_points(
    input_folder: str,
    start_id: int,
    end_id: int,
    mission: str,
    input_glob: str,
    source_round_decimals: int,
    radius_km: float,
    pre_spatial_filter: bool,
) -> Tuple[List[FrameInfo], pd.DataFrame, List[str]]:
    """
    Carga todos los .points y construye una tabla larga con columnas auxiliares.
    """
    input_dir = Path(input_folder)
    point_files = list_point_files(input_dir, input_glob)

    expected_count = end_id - start_id + 1
    if len(point_files) != expected_count:
        print(
            f"⚠️ Aviso: se esperaban {expected_count} archivos '{input_glob}', "
            f"pero se encontraron {len(point_files)} en '{input_folder}'."
        )
        print("   Se continuará usando los archivos ordenados por nombre y los IDs parseados si existen.\n")

    frames: List[FrameInfo] = []
    all_rows: List[pd.DataFrame] = []
    first_input_columns: List[str] = []

    for sorted_idx, file_path in enumerate(point_files):
        parsed_id = extract_id_from_point_filename(file_path.name)
        current_id = parsed_id if parsed_id is not None else start_id + sorted_idx

        if current_id < start_id or current_id > end_id:
            print(f"⚠️ {file_path.name}: ID {current_id} fuera de rango, se salta.")
            continue

        frame_index = len(frames)
        output_name = f"{mission}-E-{current_id}.points"
        info = FrameInfo(
            frame_index=frame_index,
            frame_id=current_id,
            input_path=file_path,
            input_name=file_path.name,
            output_name=output_name,
        )

        try:
            df = read_points_file(file_path)
        except Exception as e:
            print(f"❌ Error leyendo {file_path}: {e}")
            continue

        if not first_input_columns:
            first_input_columns = list(df.columns)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"⚠️ {file_path.name}: faltan columnas {sorted(missing)}, se salta.")
            continue

        df = coerce_numeric_columns(df, REQUIRED_COLUMNS | {"enable", "dX", "dY", "residual"})
        info.input_count = len(df)

        if len(df) == 0:
            info.pre_spatial_kept = 0
            frames.append(info)
            continue

        df["_source_key"] = [
            make_source_key(x, y, source_round_decimals)
            for x, y in zip(df["sourceX"], df["sourceY"])
        ]

        n_before_dup = len(df)
        df = df.drop_duplicates(subset=["_source_key"], keep="first").copy()
        if len(df) != n_before_dup:
            print(
                f"⚠️ {file_path.name}: {n_before_dup - len(df)} puntos duplicados por sourceX/sourceY; "
                "se conserva el primero."
            )

        if pre_spatial_filter:
            pre_mask = spatial_neighbor_mask(df, radius_km=radius_km)
        else:
            coords = df[["mapY", "mapX"]].to_numpy(dtype=float)
            pre_mask = np.isfinite(coords).all(axis=1)

        df["_pre_spatial_ok"] = pre_mask.astype(bool)
        info.pre_spatial_kept = int(pre_mask.sum())

        df["_frame_index"] = frame_index
        df["_frame_id"] = current_id
        df["_input_name"] = file_path.name
        df["_row_index"] = np.arange(len(df), dtype=int)

        all_rows.append(df)
        frames.append(info)

    if not frames:
        raise RuntimeError(f"No se pudo cargar ningún archivo .points válido desde {input_folder}")

    if all_rows:
        long_df = pd.concat(all_rows, ignore_index=True)
    else:
        long_df = pd.DataFrame()

    return frames, long_df, first_input_columns


def fit_poly(t: np.ndarray, y: np.ndarray, valid: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Ajusta un polinomio y predice en todos los t."""
    n = int(valid.sum())
    if n < 2:
        raise ValueError("No hay suficientes puntos para ajustar polinomio")
    degree = int(min(max(order, 1), n - 1))
    coeffs = np.polyfit(t[valid], y[valid], deg=degree)
    pred = np.polyval(coeffs, t)
    return pred, coeffs, degree


def can_fill_frame(
    i: int,
    good_mask: np.ndarray,
    max_gap_frames: int,
    allow_extrapolation: bool,
) -> bool:
    """
    Decide si se puede imputar el frame i.
    Por defecto solo interpola huecos interiores; no extrapola extremos.
    """
    if allow_extrapolation:
        return bool(good_mask.any())

    good_idx = np.where(good_mask)[0]
    if len(good_idx) < 2:
        return False

    prev = good_idx[good_idx < i]
    nxt = good_idx[good_idx > i]
    if len(prev) == 0 or len(nxt) == 0:
        return False

    p = int(prev[-1])
    n = int(nxt[0])
    gap_len = n - p - 1
    if max_gap_frames < 0:
        return True
    return gap_len <= max_gap_frames


def robust_temporal_fit_track(
    track: pd.DataFrame,
    n_frames: int,
    t_norm: np.ndarray,
    temporal_order: int,
    temporal_outlier_km: float,
    temporal_sigma: float,
    temporal_min_scale_km: float,
    temporal_max_iter: int,
    min_track_points: int,
    max_gap_frames: int,
    allow_extrapolation: bool,
    fill_missing: bool,
) -> TrackResult:
    """
    Ajusta trayectoria lat/lon para un único sourceX/sourceY.
    """
    key = str(track["_source_key"].iloc[0])
    sourceX, sourceY = parse_source_key(key)

    raw_lon = np.full(n_frames, np.nan, dtype=float)
    raw_lat = np.full(n_frames, np.nan, dtype=float)
    observed = np.zeros(n_frames, dtype=bool)
    pre_spatial_ok = np.zeros(n_frames, dtype=bool)

    # Si hay duplicados residuales por seguridad, nos quedamos con el primero por frame.
    track2 = track.sort_values(["_frame_index", "_row_index"]).drop_duplicates("_frame_index", keep="first")

    frame_indices = track2["_frame_index"].to_numpy(dtype=int)
    raw_lon[frame_indices] = track2["mapX"].to_numpy(dtype=float)
    raw_lat[frame_indices] = track2["mapY"].to_numpy(dtype=float)
    observed[frame_indices] = True
    pre_spatial_ok[frame_indices] = track2["_pre_spatial_ok"].to_numpy(dtype=bool)

    finite = observed & np.isfinite(raw_lon) & np.isfinite(raw_lat)
    initial_valid = finite & pre_spatial_ok

    pred_lon = np.full(n_frames, np.nan, dtype=float)
    pred_lat = np.full(n_frames, np.nan, dtype=float)
    residual_km = np.full(n_frames, np.nan, dtype=float)
    temporal_good = initial_valid.copy()
    temporal_outlier = np.zeros(n_frames, dtype=bool)
    imputed = np.zeros(n_frames, dtype=bool)
    output_present = np.zeros(n_frames, dtype=bool)
    threshold_km = float("nan")
    degree = -1
    rmse_km = float("nan")
    max_residual_km = float("nan")

    if int(initial_valid.sum()) < max(2, min_track_points):
        # Sin ajuste fiable. Se conservan solo observaciones válidas por filtro espacial.
        output_present = initial_valid.copy()
        return TrackResult(
            key=key,
            sourceX=sourceX,
            sourceY=sourceY,
            observed=observed,
            finite=finite,
            pre_spatial_ok=pre_spatial_ok,
            temporal_good=temporal_good,
            temporal_outlier=temporal_outlier,
            imputed=imputed,
            output_present=output_present,
            raw_lon=raw_lon,
            raw_lat=raw_lat,
            pred_lon=pred_lon,
            pred_lat=pred_lat,
            residual_km=residual_km,
            threshold_km=threshold_km,
            degree=degree,
            rmse_km=rmse_km,
            max_residual_km=max_residual_km,
        )

    # Preparar longitud sin discontinuidades en los frames observados.
    lon_unwrapped = np.full(n_frames, np.nan, dtype=float)
    obs_idx = np.where(finite)[0]
    if len(obs_idx) > 0:
        lon_unwrapped[obs_idx] = np.degrees(np.unwrap(np.radians(raw_lon[obs_idx])))

    valid = initial_valid.copy()
    last_valid = None

    for _ in range(max(1, temporal_max_iter)):
        if int(valid.sum()) < max(2, min_track_points):
            break

        try:
            pred_lat_i, _, degree_lat = fit_poly(t_norm, raw_lat, valid, temporal_order)
            pred_lon_unw_i, _, degree_lon = fit_poly(t_norm, lon_unwrapped, valid, temporal_order)
        except Exception:
            break

        pred_lon_i = wrap_lon_deg(pred_lon_unw_i).astype(float)
        residual_i = np.full(n_frames, np.nan, dtype=float)
        residual_i[finite] = haversine_km(
            raw_lat[finite], raw_lon[finite], pred_lat_i[finite], pred_lon_i[finite]
        )

        base_res = residual_i[valid]
        base_res = base_res[np.isfinite(base_res)]
        if len(base_res) == 0:
            break

        med = float(np.median(base_res))
        mad = float(np.median(np.abs(base_res - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < temporal_min_scale_km:
            scale = float(temporal_min_scale_km)

        threshold = max(float(temporal_outlier_km), float(temporal_sigma) * scale)
        new_valid = initial_valid & np.isfinite(residual_i) & (residual_i <= threshold)

        if int(new_valid.sum()) < max(2, min_track_points):
            # No aceptamos una iteración que deje la trayectoria sin soporte.
            new_valid = valid.copy()

        pred_lat = pred_lat_i
        pred_lon = pred_lon_i
        residual_km = residual_i
        threshold_km = threshold
        degree = int(max(degree_lat, degree_lon))

        if last_valid is not None and np.array_equal(new_valid, last_valid):
            valid = new_valid
            break
        if np.array_equal(new_valid, valid):
            valid = new_valid
            break

        last_valid = valid.copy()
        valid = new_valid

    temporal_good = valid.copy()
    temporal_outlier = finite & pre_spatial_ok & (~temporal_good)

    if np.isfinite(residual_km[temporal_good]).any():
        rmse_km = float(np.sqrt(np.nanmean(residual_km[temporal_good] ** 2)))
        max_residual_km = float(np.nanmax(residual_km[temporal_good]))

    # Salida: conservar observaciones buenas. Imputar outliers / pre-spatial bad / huecos cortos.
    output_present = temporal_good.copy()

    if degree >= 0 and np.isfinite(pred_lon).all() and np.isfinite(pred_lat).all():
        for i in range(n_frames):
            if temporal_good[i]:
                continue

            missing = not observed[i]
            bad_observation = observed[i] and (not temporal_good[i])

            if missing and not fill_missing:
                continue

            if missing or bad_observation:
                if can_fill_frame(i, temporal_good, max_gap_frames, allow_extrapolation):
                    output_present[i] = True
                    imputed[i] = True

    return TrackResult(
        key=key,
        sourceX=sourceX,
        sourceY=sourceY,
        observed=observed,
        finite=finite,
        pre_spatial_ok=pre_spatial_ok,
        temporal_good=temporal_good,
        temporal_outlier=temporal_outlier,
        imputed=imputed,
        output_present=output_present,
        raw_lon=raw_lon,
        raw_lat=raw_lat,
        pred_lon=pred_lon,
        pred_lat=pred_lat,
        residual_km=residual_km,
        threshold_km=threshold_km,
        degree=degree,
        rmse_km=rmse_km,
        max_residual_km=max_residual_km,
    )


def row_to_output_dict(row: pd.Series, output_columns: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in output_columns:
        if col in row.index and not col.startswith("_"):
            out[col] = row[col]
        elif col == "enable":
            out[col] = 1
        elif col in {"dX", "dY", "residual"}:
            out[col] = 0.0
        elif col in {"temporal_status", "temporal_residual_km"}:
            out[col] = ""
        else:
            out[col] = np.nan
    return out


def create_template_row(track: pd.DataFrame, output_columns: List[str]) -> Dict[str, Any]:
    if len(track) > 0:
        return row_to_output_dict(track.iloc[0], output_columns)
    return {col: np.nan for col in output_columns}


def build_temporally_consistent_outputs(
    frames: List[FrameInfo],
    long_df: pd.DataFrame,
    output_columns: List[str],
    temporal_order: int,
    temporal_outlier_km: float,
    temporal_sigma: float,
    temporal_min_scale_km: float,
    temporal_max_iter: int,
    min_track_points: int,
    max_gap_frames: int,
    allow_extrapolation: bool,
    fill_missing: bool,
    add_qc_columns: bool,
    disable_temporal: bool,
) -> Tuple[Dict[int, pd.DataFrame], List[TrackResult], pd.DataFrame]:
    """
    Ejecuta coherencia temporal para todos los sourceX/sourceY.
    """
    n_frames = len(frames)
    t_norm = np.linspace(-1.0, 1.0, n_frames, dtype=float) if n_frames > 1 else np.array([0.0])

    outputs_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    track_results: List[TrackResult] = []
    summary_rows: List[Dict[str, Any]] = []

    if long_df.empty:
        return {i: pd.DataFrame(columns=output_columns) for i in range(n_frames)}, track_results, pd.DataFrame()

    grouped = long_df.groupby("_source_key", sort=True)

    for key, track in grouped:
        if disable_temporal:
            # Modo antiguo ampliado: conservar únicamente las observaciones pre_spatial_ok.
            sourceX, sourceY = parse_source_key(str(key))
            observed = np.zeros(n_frames, dtype=bool)
            finite = np.zeros(n_frames, dtype=bool)
            pre_ok = np.zeros(n_frames, dtype=bool)
            raw_lon = np.full(n_frames, np.nan)
            raw_lat = np.full(n_frames, np.nan)
            for _, row in track.iterrows():
                i = int(row["_frame_index"])
                observed[i] = True
                raw_lon[i] = float(row["mapX"])
                raw_lat[i] = float(row["mapY"])
                finite[i] = np.isfinite(raw_lon[i]) and np.isfinite(raw_lat[i])
                pre_ok[i] = bool(row["_pre_spatial_ok"])
                if pre_ok[i] and finite[i]:
                    out = row_to_output_dict(row, output_columns)
                    if add_qc_columns:
                        out["temporal_status"] = "kept_no_temporal"
                        out["temporal_residual_km"] = 0.0
                    outputs_by_frame[i].append(out)

            result = TrackResult(
                key=str(key), sourceX=sourceX, sourceY=sourceY,
                observed=observed, finite=finite, pre_spatial_ok=pre_ok,
                temporal_good=pre_ok & finite, temporal_outlier=np.zeros(n_frames, dtype=bool),
                imputed=np.zeros(n_frames, dtype=bool), output_present=pre_ok & finite,
                raw_lon=raw_lon, raw_lat=raw_lat,
                pred_lon=np.full(n_frames, np.nan), pred_lat=np.full(n_frames, np.nan),
                residual_km=np.full(n_frames, np.nan), threshold_km=float("nan"),
                degree=-1, rmse_km=float("nan"), max_residual_km=float("nan"),
            )
        else:
            result = robust_temporal_fit_track(
                track=track,
                n_frames=n_frames,
                t_norm=t_norm,
                temporal_order=temporal_order,
                temporal_outlier_km=temporal_outlier_km,
                temporal_sigma=temporal_sigma,
                temporal_min_scale_km=temporal_min_scale_km,
                temporal_max_iter=temporal_max_iter,
                min_track_points=min_track_points,
                max_gap_frames=max_gap_frames,
                allow_extrapolation=allow_extrapolation,
                fill_missing=fill_missing,
            )

            # Acceso rápido a filas originales por frame.
            track_by_frame: Dict[int, pd.Series] = {}
            for _, row in track.sort_values(["_frame_index", "_row_index"]).iterrows():
                i = int(row["_frame_index"])
                if i not in track_by_frame:
                    track_by_frame[i] = row

            template = create_template_row(track, output_columns)
            template["sourceX"] = result.sourceX
            template["sourceY"] = result.sourceY

            for i in range(n_frames):
                if not result.output_present[i]:
                    continue

                if result.temporal_good[i] and i in track_by_frame:
                    out = row_to_output_dict(track_by_frame[i], output_columns)
                    status = "kept"
                else:
                    # Outlier o hueco imputado.
                    base = track_by_frame.get(i, None)
                    out = row_to_output_dict(base, output_columns) if base is not None else dict(template)
                    out["mapX"] = float(result.pred_lon[i])
                    out["mapY"] = float(result.pred_lat[i])
                    out["sourceX"] = float(result.sourceX)
                    out["sourceY"] = float(result.sourceY)
                    out["enable"] = 1
                    out["dX"] = 0.0
                    out["dY"] = 0.0
                    out["residual"] = 0.0
                    status = "imputed"
                    frames[i].n_imputed += 1

                if add_qc_columns:
                    out["temporal_status"] = status
                    r = result.residual_km[i]
                    out["temporal_residual_km"] = float(r) if np.isfinite(r) else np.nan

                outputs_by_frame[i].append(out)

        for i in range(n_frames):
            if result.temporal_outlier[i]:
                frames[i].n_temporal_outliers += 1
            if result.observed[i] and result.finite[i] and (not result.pre_spatial_ok[i]):
                frames[i].n_pre_spatial_rejected_used_as_missing += 1

        track_results.append(result)
        summary_rows.append({
            "source_key": result.key,
            "sourceX": result.sourceX,
            "sourceY": result.sourceY,
            "n_observed": int(result.observed.sum()),
            "n_finite": int(result.finite.sum()),
            "n_pre_spatial_ok": int(result.pre_spatial_ok.sum()),
            "n_temporal_good": int(result.temporal_good.sum()),
            "n_temporal_outliers": int(result.temporal_outlier.sum()),
            "n_imputed": int(result.imputed.sum()),
            "n_output": int(result.output_present.sum()),
            "fit_degree": int(result.degree),
            "threshold_km_last": result.threshold_km,
            "rmse_km": result.rmse_km,
            "max_residual_km": result.max_residual_km,
        })

    output_dfs: Dict[int, pd.DataFrame] = {}
    for i in range(n_frames):
        rows = outputs_by_frame.get(i, [])
        df = pd.DataFrame(rows, columns=output_columns)
        if not df.empty:
            # Orden estable: de arriba a abajo en coordenadas de imagen y luego izquierda-derecha.
            sort_y = -pd.to_numeric(df["sourceY"], errors="coerce")
            sort_x = pd.to_numeric(df["sourceX"], errors="coerce")
            df = df.assign(_sort_y=sort_y, _sort_x=sort_x).sort_values(["_sort_y", "_sort_x"])
            df = df.drop(columns=["_sort_y", "_sort_x"])
        frames[i].output_before_post_spatial = len(df)
        output_dfs[i] = df

    track_summary = pd.DataFrame(summary_rows)
    return output_dfs, track_results, track_summary


def apply_post_spatial_filter_to_outputs(
    output_dfs: Dict[int, pd.DataFrame],
    frames: List[FrameInfo],
    radius_km: float,
    post_spatial_filter: bool,
) -> Dict[int, pd.DataFrame]:
    filtered: Dict[int, pd.DataFrame] = {}
    for info in frames:
        df = output_dfs.get(info.frame_index, pd.DataFrame())
        if df.empty or not post_spatial_filter:
            filtered[info.frame_index] = df.copy()
            info.post_spatial_kept = len(df)
            continue

        mask = spatial_neighbor_mask(df, radius_km=radius_km)
        filtered_df = df.loc[mask].copy()
        info.post_spatial_kept = len(filtered_df)
        filtered[info.frame_index] = filtered_df
    return filtered


def write_outputs(
    output_dfs: Dict[int, pd.DataFrame],
    frames: List[FrameInfo],
    output_folder: str,
    output_columns: List[str],
) -> None:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for info in frames:
        df = output_dfs.get(info.frame_index, pd.DataFrame(columns=output_columns))
        output_path = out_dir / info.output_name
        df.to_csv(output_path, index=False)
        if df.empty:
            print(f"⚠️ {info.output_name}: se guarda vacío tras filtros.")
        print(
            f"✅ {info.output_name}: input={info.input_count}, "
            f"pre_spatial={info.pre_spatial_kept}, "
            f"temp_outliers={info.n_temporal_outliers}, "
            f"imputed={info.n_imputed}, "
            f"final={len(df)}"
        )


def write_qc_tables(
    frames: List[FrameInfo],
    track_summary: pd.DataFrame,
    output_folder: str,
) -> None:
    out_dir = Path(output_folder)
    frame_rows = []
    for info in frames:
        frame_rows.append({
            "frame_index": info.frame_index,
            "frame_id": info.frame_id,
            "input_name": info.input_name,
            "output_name": info.output_name,
            "input_count": info.input_count,
            "pre_spatial_kept": info.pre_spatial_kept,
            "pre_spatial_rejected": info.input_count - info.pre_spatial_kept,
            "pre_spatial_rejected_used_as_missing": info.n_pre_spatial_rejected_used_as_missing,
            "temporal_outliers": info.n_temporal_outliers,
            "imputed": info.n_imputed,
            "output_before_post_spatial": info.output_before_post_spatial,
            "post_spatial_kept": info.post_spatial_kept,
            "post_spatial_rejected": info.output_before_post_spatial - info.post_spatial_kept,
        })
    frame_summary = pd.DataFrame(frame_rows)
    frame_summary.to_csv(out_dir / "temporal_frame_summary.csv", index=False)
    if track_summary is not None and not track_summary.empty:
        track_summary.to_csv(out_dir / "temporal_track_summary.csv", index=False)


def choose_plot_tracks(track_results: List[TrackResult], requested_keys: Optional[List[str]] = None) -> List[Tuple[str, TrackResult]]:
    """
    Selecciona tracks representativos: esquinas y centro en coordenadas de imagen real.
    sourceY es negativo de y_imagen; usamos image_y = -sourceY.
    """
    valid_tracks = [tr for tr in track_results if int(tr.observed.sum()) > 0]
    if not valid_tracks:
        return []

    if requested_keys:
        by_key = {tr.key: tr for tr in valid_tracks}
        out = []
        for key in requested_keys:
            if key in by_key:
                out.append(("requested", by_key[key]))
        return out

    sx = np.array([tr.sourceX for tr in valid_tracks], dtype=float)
    iy = np.array([-tr.sourceY for tr in valid_tracks], dtype=float)

    xmin, xmax = float(np.nanmin(sx)), float(np.nanmax(sx))
    ymin, ymax = float(np.nanmin(iy)), float(np.nanmax(iy))
    xmid, ymid = float(np.nanmedian(sx)), float(np.nanmedian(iy))

    targets = {
        "top_left": (xmin, ymin),
        "top_right": (xmax, ymin),
        "center": (xmid, ymid),
        "bottom_left": (xmin, ymax),
        "bottom_right": (xmax, ymax),
    }

    chosen: List[Tuple[str, TrackResult]] = []
    used = set()
    for label, (tx, ty) in targets.items():
        d2 = (sx - tx) ** 2 + (iy - ty) ** 2
        order = np.argsort(d2)
        for idx in order:
            tr = valid_tracks[int(idx)]
            if tr.key not in used:
                chosen.append((label, tr))
                used.add(tr.key)
                break
    return chosen


def safe_filename_fragment(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def plot_track_evolution(
    label: str,
    tr: TrackResult,
    frames: List[FrameInfo],
    plot_dir: Path,
) -> None:
    frame_ids = np.array([f.frame_id for f in frames], dtype=int)
    x = np.arange(len(frames), dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    obs = tr.observed & tr.finite
    good = tr.temporal_good
    outl = tr.temporal_outlier
    imp = tr.imputed
    outp = tr.output_present

    # Latitud
    ax = axes[0]
    ax.plot(x[np.isfinite(tr.pred_lat)], tr.pred_lat[np.isfinite(tr.pred_lat)], linewidth=1.5, label="polinomio")
    ax.scatter(x[obs], tr.raw_lat[obs], s=14, alpha=0.55, label="observado")
    ax.scatter(x[good], tr.raw_lat[good], s=18, label="usado en ajuste")
    ax.scatter(x[outl], tr.raw_lat[outl], s=30, marker="x", label="outlier temporal")
    ax.scatter(x[imp], tr.pred_lat[imp], s=28, marker="D", label="imputado")
    ax.set_ylabel("latitud mapY")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # Longitud
    ax = axes[1]
    ax.plot(x[np.isfinite(tr.pred_lon)], tr.pred_lon[np.isfinite(tr.pred_lon)], linewidth=1.5, label="polinomio")
    ax.scatter(x[obs], tr.raw_lon[obs], s=14, alpha=0.55, label="observado")
    ax.scatter(x[good], tr.raw_lon[good], s=18, label="usado en ajuste")
    ax.scatter(x[outl], tr.raw_lon[outl], s=30, marker="x", label="outlier temporal")
    ax.scatter(x[imp], tr.pred_lon[imp], s=28, marker="D", label="imputado")
    ax.set_ylabel("longitud mapX")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # Residuo
    ax = axes[2]
    finite_res = np.isfinite(tr.residual_km)
    ax.scatter(x[finite_res], tr.residual_km[finite_res], s=14, alpha=0.65, label="residuo obs-polí")
    if np.isfinite(tr.threshold_km):
        ax.axhline(tr.threshold_km, linestyle="--", linewidth=1.0, label=f"umbral {tr.threshold_km:.1f} km")
    ax.scatter(x[outl], tr.residual_km[outl], s=30, marker="x", label="outlier")
    ax.set_ylabel("residuo km")
    ax.set_xlabel("frame index")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    title = (
        f"{label}: sourceX={tr.sourceX:.2f}, sourceY={tr.sourceY:.2f} | "
        f"obs={int(tr.observed.sum())}, good={int(tr.temporal_good.sum())}, "
        f"outliers={int(tr.temporal_outlier.sum())}, imputed={int(tr.imputed.sum())}, "
        f"degree={tr.degree}, rmse={tr.rmse_km:.2f} km"
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fname = f"track_{safe_filename_fragment(label)}_sx{tr.sourceX:.1f}_sy{tr.sourceY:.1f}.png"
    fig.savefig(plot_dir / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_frame_qc(frames: List[FrameInfo], plot_dir: Path) -> None:
    x = np.arange(len(frames), dtype=int)
    frame_ids = [f.frame_id for f in frames]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, [f.input_count for f in frames], label="input")
    ax.plot(x, [f.pre_spatial_kept for f in frames], label="pre_spatial_kept")
    ax.plot(x, [f.output_before_post_spatial for f in frames], label="output_before_post")
    ax.plot(x, [f.post_spatial_kept for f in frames], label="final")
    ax.set_title("Número de puntos por frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("n puntos")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "frame_point_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, [f.n_temporal_outliers for f in frames], label="outliers temporales")
    ax.plot(x, [f.n_imputed for f in frames], label="imputados")
    ax.plot(x, [f.output_before_post_spatial - f.post_spatial_kept for f in frames], label="rechazados post-spatial")
    ax.set_title("Control de calidad temporal/espacial por frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("n puntos")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "frame_qc_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    track_results: List[TrackResult],
    frames: List[FrameInfo],
    plot_dir: Optional[str],
    max_plot_tracks: int,
) -> None:
    if not plot_dir:
        return
    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)

    chosen = choose_plot_tracks(track_results)[:max(0, max_plot_tracks)]
    for label, tr in chosen:
        plot_track_evolution(label, tr, frames, out)
    plot_frame_qc(frames, out)
    print(f"🖼️ Plots de coherencia temporal guardados en: {out}")


def filter_and_rename_points(
    input_folder: str,
    output_folder: str,
    radius_km: float,
    start_id: int,
    end_id: int,
    mission: str,
    input_glob: str = "*_real.points",
    source_round_decimals: int = 3,
    disable_temporal: bool = False,
    temporal_order: int = 3,
    temporal_outlier_km: float = 80.0,
    temporal_sigma: float = 3.5,
    temporal_min_scale_km: float = 5.0,
    temporal_max_iter: int = 4,
    min_track_points: int = 6,
    max_gap_frames: int = 8,
    allow_extrapolation: bool = False,
    fill_missing: bool = True,
    pre_spatial_filter: bool = True,
    post_spatial_filter: bool = True,
    add_qc_columns: bool = False,
    plot_dir: Optional[str] = None,
    max_plot_tracks: int = 5,
) -> None:
    """
    Función principal: carga .points, aplica coherencia temporal, filtra y escribe resultados.
    """
    os.makedirs(output_folder, exist_ok=True)

    frames, long_df, first_input_columns = load_timelapse_points(
        input_folder=input_folder,
        start_id=start_id,
        end_id=end_id,
        mission=mission,
        input_glob=input_glob,
        source_round_decimals=source_round_decimals,
        radius_km=radius_km,
        pre_spatial_filter=pre_spatial_filter,
    )

    output_columns = discover_output_columns(first_input_columns, add_qc_columns=add_qc_columns)

    print(f"🔵 Frames cargados: {len(frames)}")
    print(f"🔵 Observaciones de puntos cargadas: {len(long_df)}")
    print(f"🔵 Tracks sourceX/sourceY: {0 if long_df.empty else long_df['_source_key'].nunique()}")
    print(f"🔵 Coherencia temporal: {'desactivada' if disable_temporal else 'activada'}")
    print(f"🔵 Filtro espacial pre/post: {pre_spatial_filter}/{post_spatial_filter}")

    output_dfs, track_results, track_summary = build_temporally_consistent_outputs(
        frames=frames,
        long_df=long_df,
        output_columns=output_columns,
        temporal_order=temporal_order,
        temporal_outlier_km=temporal_outlier_km,
        temporal_sigma=temporal_sigma,
        temporal_min_scale_km=temporal_min_scale_km,
        temporal_max_iter=temporal_max_iter,
        min_track_points=min_track_points,
        max_gap_frames=max_gap_frames,
        allow_extrapolation=allow_extrapolation,
        fill_missing=fill_missing,
        add_qc_columns=add_qc_columns,
        disable_temporal=disable_temporal,
    )

    output_dfs = apply_post_spatial_filter_to_outputs(
        output_dfs=output_dfs,
        frames=frames,
        radius_km=radius_km,
        post_spatial_filter=post_spatial_filter,
    )

    write_outputs(
        output_dfs=output_dfs,
        frames=frames,
        output_folder=output_folder,
        output_columns=output_columns,
    )

    write_qc_tables(frames=frames, track_summary=track_summary, output_folder=output_folder)
    make_plots(track_results=track_results, frames=frames, plot_dir=plot_dir, max_plot_tracks=max_plot_tracks)

    n_outliers = sum(f.n_temporal_outliers for f in frames)
    n_imputed = sum(f.n_imputed for f in frames)
    n_final = sum(f.post_spatial_kept for f in frames)

    print("\n✔️ Filtrado + coherencia temporal completados.")
    print(f"   Outliers temporales detectados: {n_outliers}")
    print(f"   Puntos imputados:               {n_imputed}")
    print(f"   Puntos finales escritos:        {n_final}")
    print(f"   QC frames: {Path(output_folder) / 'temporal_frame_summary.csv'}")
    print(f"   QC tracks: {Path(output_folder) / 'temporal_track_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filtrado espacial + coherencia temporal de .points: BallTree, ajuste polinómico "
            "por sourceX/sourceY, detección de outliers e imputación."
        )
    )

    # Compatibles con el filter_points.py original
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Carpeta con archivos .points originales, normalmente *_real.points.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Carpeta para guardar archivos filtrados/renombrados.")
    parser.add_argument("--radius_km", type=float, default=130.0,
                        help="Radio km para BallTree/haversine en filtro espacial.")
    parser.add_argument("--start_id", type=int, required=True,
                        help="ID inicial esperado.")
    parser.add_argument("--end_id", type=int, required=True,
                        help="ID final esperado.")
    parser.add_argument("--mission", type=str, required=True,
                        help="Nombre de la misión, p.ej. ISS053, ISS067.")

    # Entrada / clave temporal
    parser.add_argument("--input_glob", type=str, default="*_real.points",
                        help="Patrón de entrada dentro de input_folder. Por defecto: *_real.points")
    parser.add_argument("--source_round_decimals", type=int, default=3,
                        help="Decimales para agrupar sourceX/sourceY como el mismo punto de rejilla.")

    # Coherencia temporal
    parser.add_argument("--disable_temporal", action="store_true",
                        help="Desactiva ajuste temporal; deja solo filtrado espacial + renombrado.")
    parser.add_argument("--temporal_order", type=int, default=3,
                        help="Orden del polinomio temporal lat(t), lon(t). Recomendado: 2 o 3.")
    parser.add_argument("--temporal_outlier_km", type=float, default=80.0,
                        help="Umbral absoluto mínimo en km para declarar outlier temporal.")
    parser.add_argument("--temporal_sigma", type=float, default=3.5,
                        help="Umbral robusto: outlier si residuo > max(outlier_km, sigma*MAD).")
    parser.add_argument("--temporal_min_scale_km", type=float, default=5.0,
                        help="Escala robusta mínima en km para evitar umbrales demasiado pequeños.")
    parser.add_argument("--temporal_max_iter", type=int, default=4,
                        help="Iteraciones robustas máximas por track.")
    parser.add_argument("--min_track_points", type=int, default=6,
                        help="Mínimo de observaciones válidas para ajustar una trayectoria.")
    parser.add_argument("--max_gap_frames", type=int, default=8,
                        help="Máximo hueco interior a imputar. Usa -1 para huecos interiores ilimitados.")
    parser.add_argument("--allow_extrapolation", action="store_true",
                        help="Permite imputar fuera del rango observado bueno. Por defecto NO extrapola.")
    parser.add_argument("--no_fill_missing", action="store_true",
                        help="No añade puntos ausentes; solo sustituye observaciones malas si se pueden imputar.")

    # Filtro espacial
    parser.add_argument("--no_pre_spatial_filter", action="store_true",
                        help="No usa BallTree antes del ajuste temporal para marcar puntos aislados como missing.")
    parser.add_argument("--no_post_spatial_filter", action="store_true",
                        help="No aplica BallTree final después de imputar.")

    # QC / plots
    parser.add_argument("--add_qc_columns", action="store_true",
                        help="Añade columnas temporal_status/temporal_residual_km a los .points de salida.")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Carpeta donde guardar plots de evolución temporal de puntos representativos.")
    parser.add_argument("--max_plot_tracks", type=int, default=5,
                        help="Número máximo de tracks representativos a plotear.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_and_rename_points(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        radius_km=args.radius_km,
        start_id=args.start_id,
        end_id=args.end_id,
        mission=args.mission,
        input_glob=args.input_glob,
        source_round_decimals=args.source_round_decimals,
        disable_temporal=args.disable_temporal,
        temporal_order=args.temporal_order,
        temporal_outlier_km=args.temporal_outlier_km,
        temporal_sigma=args.temporal_sigma,
        temporal_min_scale_km=args.temporal_min_scale_km,
        temporal_max_iter=args.temporal_max_iter,
        min_track_points=args.min_track_points,
        max_gap_frames=args.max_gap_frames,
        allow_extrapolation=args.allow_extrapolation,
        fill_missing=not args.no_fill_missing,
        pre_spatial_filter=not args.no_pre_spatial_filter,
        post_spatial_filter=not args.no_post_spatial_filter,
        add_qc_columns=args.add_qc_columns,
        plot_dir=args.plot_dir,
        max_plot_tracks=args.max_plot_tracks,
    )


if __name__ == "__main__":
    main()
