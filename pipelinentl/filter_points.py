#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal/spatial filtering of ISS timelapse Ground Control Points (.points).

Run this after project_timelapse.py and before georef_timelapse.py.

Each control point is identified by its fixed source image grid coordinate:

    track_id = (sourceX, sourceY)

For each track, the geographic coordinates mapX/mapY should evolve smoothly
through the timelapse. This script:

1. loads all *_real.points files;
2. groups observations by sourceX/sourceY;
3. fits smooth polynomial trajectories in lon/lat;
4. detects temporal outliers statistically from the residual distribution;
5. replaces temporal outliers by the fitted trajectory when the track has
   enough support;
6. optionally fills short missing gaps;
7. applies a final spatial BallTree filter;
8. writes filtered .points files named <mission>-E-<ID>.points;
9. writes QC tables and plots.

Overlay plot colors:
    blue   = untouched points kept in final .points
    green  = temporal outlier/imputed points kept in final .points
    orange = temporal outlier/imputed points removed by final spatial filter
    red    = untouched points removed by final spatial filter
    yellow star = diagnostic tracks
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = {"mapX", "mapY", "sourceX", "sourceY"}
BASE_OUTPUT_COLUMNS = [
    "mapX", "mapY", "sourceX", "sourceY", "enable", "dX", "dY", "residual",
]


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
    n_temporal_outliers: int = 0
    n_imputed: int = 0
    n_post_spatial_rejected: int = 0


@dataclass
class TrackResult:
    key: str
    sourceX: float
    sourceY: float
    raw_lon: np.ndarray
    raw_lat: np.ndarray
    pred_lon: np.ndarray
    pred_lat: np.ndarray
    observed: np.ndarray
    finite: np.ndarray
    pre_spatial_ok: np.ndarray
    fit_used: np.ndarray
    temporal_outlier: np.ndarray
    imputed: np.ndarray
    output_present: np.ndarray
    residual_km: np.ndarray
    threshold_km: float
    residual_mean_km: float
    residual_std_km: float
    degree: int
    rmse_km: float
    max_residual_km: float
    can_impute: bool


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def extract_id_from_point_filename(name: str) -> Optional[int]:
    """Extract image ID from names such as ISS067-E-327041_real.points."""
    m = re.search(r"[A-Za-z0-9]+-E-(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def make_source_key(source_x: Any, source_y: Any, decimals: int) -> str:
    sx = round(float(source_x), decimals)
    sy = round(float(source_y), decimals)
    return f"{sx:.{decimals}f}|{sy:.{decimals}f}"


def parse_source_key(key: str) -> Tuple[float, float]:
    sx, sy = key.split("|", 1)
    return float(sx), float(sy)


def wrap_lon_deg(lon: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(lon, dtype=float)
    out = (arr + 180.0) % 360.0 - 180.0
    if np.isscalar(lon):
        return float(out)
    return out


def haversine_km(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1_deg, dtype=float))
    lat2 = np.radians(np.asarray(lat2_deg, dtype=float))
    lon1 = np.radians(np.asarray(lon1_deg, dtype=float))
    lon2 = np.radians(np.asarray(lon2_deg, dtype=float))

    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + np.pi) % (2.0 * np.pi) - np.pi
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return EARTH_RADIUS_KM * (2.0 * np.arcsin(np.sqrt(a)))


def coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_points_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="M")


def spatial_neighbor_mask(df: pd.DataFrame, radius_km: float) -> np.ndarray:
    """True if a point has at least one other neighbour within radius_km."""
    mask = np.zeros(len(df), dtype=bool)
    if len(df) < 2 or not {"mapY", "mapX"}.issubset(df.columns):
        return mask

    coords_deg = df[["mapY", "mapX"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(coords_deg).all(axis=1)
    finite_idx = np.where(finite)[0]
    if len(finite_idx) < 2:
        return mask

    coords_rad = np.radians(coords_deg[finite_idx])
    radius_rad = float(radius_km) / EARTH_RADIUS_KM
    tree = BallTree(coords_rad, metric="haversine")
    neighbours = tree.query_radius(coords_rad, r=radius_rad)
    mask[finite_idx] = np.array([len(n) > 1 for n in neighbours], dtype=bool)
    return mask


def discover_output_columns(first_input_columns: Sequence[str], add_qc_columns: bool) -> List[str]:
    if first_input_columns and REQUIRED_COLUMNS.issubset(set(first_input_columns)):
        cols = [c for c in first_input_columns if not str(c).startswith("__")]
    else:
        cols = BASE_OUTPUT_COLUMNS.copy()

    for col in BASE_OUTPUT_COLUMNS:
        if col not in cols:
            cols.append(col)

    if add_qc_columns:
        for col in ["temporal_status", "temporal_residual_km", "temporal_threshold_km"]:
            if col not in cols:
                cols.append(col)
    return cols


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_timelapse_points(
    input_folder: str,
    start_id: int,
    end_id: int,
    mission: str,
    input_glob: str,
    source_round_decimals: int,
    radius_km: float,
) -> Tuple[List[FrameInfo], pd.DataFrame, List[str]]:
    input_dir = Path(input_folder)
    point_files = sorted(Path(p) for p in glob(str(input_dir / input_glob)))

    expected_count = end_id - start_id + 1
    if len(point_files) != expected_count:
        print(
            f"WARNING: expected {expected_count} files matching '{input_glob}', "
            f"found {len(point_files)} in '{input_folder}'."
        )
        print("Continuing with sorted files and parsed IDs when possible.\n")

    frames: List[FrameInfo] = []
    rows: List[pd.DataFrame] = []
    first_input_columns: List[str] = []

    for sorted_idx, path in enumerate(point_files):
        parsed_id = extract_id_from_point_filename(path.name)
        frame_id = parsed_id if parsed_id is not None else start_id + sorted_idx

        if frame_id < start_id or frame_id > end_id:
            print(f"WARNING: {path.name}: ID {frame_id} out of range, skipped.")
            continue

        frame_index = len(frames)
        info = FrameInfo(
            frame_index=frame_index,
            frame_id=frame_id,
            input_path=path,
            input_name=path.name,
            output_name=f"{mission}-E-{frame_id}.points",
        )

        try:
            df = read_points_file(path)
        except Exception as exc:
            print(f"ERROR reading {path}: {exc}")
            continue

        if not REQUIRED_COLUMNS.issubset(df.columns):
            print(f"WARNING: {path.name} lacks columns {sorted(REQUIRED_COLUMNS)}, skipped.")
            continue

        if not first_input_columns:
            first_input_columns = list(df.columns)

        df = coerce_numeric_columns(df, REQUIRED_COLUMNS | {"enable", "dX", "dY", "residual"})
        info.input_count = len(df)

        finite_geo = np.isfinite(df[["mapX", "mapY", "sourceX", "sourceY"]].to_numpy(dtype=float)).all(axis=1)

        # Diagnostic only unless --pre_spatial_filter is explicitly used.
        pre_spatial = spatial_neighbor_mask(df, radius_km=radius_km)
        info.pre_spatial_kept = int(pre_spatial.sum())

        df = df.copy()
        df["__frame_index"] = frame_index
        df["__frame_id"] = frame_id
        df["__row_index"] = np.arange(len(df), dtype=int)
        df["__finite_geo"] = finite_geo
        df["__pre_spatial_ok"] = pre_spatial
        df["__source_key"] = [
            make_source_key(x, y, source_round_decimals)
            if np.isfinite(x) and np.isfinite(y) else "nan|nan"
            for x, y in zip(df["sourceX"], df["sourceY"])
        ]

        frames.append(info)
        rows.append(df)

    if not rows:
        return frames, pd.DataFrame(), first_input_columns

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df[all_df["__source_key"] != "nan|nan"].copy()
    return frames, all_df, first_input_columns


# -----------------------------------------------------------------------------
# Temporal fitting
# -----------------------------------------------------------------------------


def normalized_time(n_frames: int) -> np.ndarray:
    t = np.arange(n_frames, dtype=float)
    if n_frames <= 1:
        return np.zeros(n_frames, dtype=float)
    center = 0.5 * (n_frames - 1)
    scale = max(center, 1.0)
    return (t - center) / scale


def safe_polyfit_eval(x: np.ndarray, y: np.ndarray, x_all: np.ndarray, degree: int) -> Optional[np.ndarray]:
    if len(x) <= degree:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeff = np.polyfit(x, y, deg=degree)
        return np.polyval(coeff, x_all)
    except Exception:
        return None


def residual_threshold_km(
    residuals: np.ndarray,
    mode: str,
    temporal_sigma: float,
    temporal_outlier_km: float,
    min_threshold_km: float,
) -> Tuple[float, float, float]:
    vals = residuals[np.isfinite(residuals)]
    if len(vals) == 0:
        return float("inf"), float("nan"), float("nan")

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if not np.isfinite(std):
        std = 0.0

    sigma_thr = mean + float(temporal_sigma) * std

    if mode == "sigma":
        thr = sigma_thr
    elif mode == "hybrid":
        thr = max(float(temporal_outlier_km), sigma_thr)
    elif mode == "absolute":
        thr = float(temporal_outlier_km)
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")

    if np.isfinite(min_threshold_km) and min_threshold_km > 0:
        thr = max(float(min_threshold_km), float(thr))

    return float(thr), mean, std


def fit_predict_track(
    raw_lon: np.ndarray,
    raw_lat: np.ndarray,
    initial_fit_mask: np.ndarray,
    n_frames: int,
    order: int,
    min_track_points: int,
    threshold_mode: str,
    temporal_sigma: float,
    temporal_outlier_km: float,
    temporal_min_threshold_km: float,
    temporal_max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, int, float, float]:
    pred_lon = np.full(n_frames, np.nan, dtype=float)
    pred_lat = np.full(n_frames, np.nan, dtype=float)
    residual = np.full(n_frames, np.nan, dtype=float)
    fit_used = np.zeros(n_frames, dtype=bool)

    current_mask = initial_fit_mask.copy()
    if current_mask.sum() < min_track_points:
        return pred_lon, pred_lat, fit_used, residual, float("nan"), float("nan"), float("nan"), -1, np.nan, np.nan

    x_all = normalized_time(n_frames)
    last_mask: Optional[np.ndarray] = None
    threshold = float("nan")
    res_mean = float("nan")
    res_std = float("nan")
    degree = -1

    max_iter = max(1, int(temporal_max_iter))
    for _ in range(max_iter):
        idx = np.where(current_mask)[0]
        if len(idx) < min_track_points:
            break

        degree = int(min(order, len(idx) - 1))
        if degree < 1:
            break

        x_fit = x_all[idx]
        lon_fit_unwrapped = np.degrees(np.unwrap(np.radians(raw_lon[idx].astype(float))))
        lat_fit = raw_lat[idx].astype(float)

        pred_lon_unwrapped = safe_polyfit_eval(x_fit, lon_fit_unwrapped, x_all, degree)
        pred_lat_all = safe_polyfit_eval(x_fit, lat_fit, x_all, degree)
        if pred_lon_unwrapped is None or pred_lat_all is None:
            break

        pred_lon_all = wrap_lon_deg(pred_lon_unwrapped).astype(float)
        pred_lat_all = pred_lat_all.astype(float)

        observed = np.isfinite(raw_lon) & np.isfinite(raw_lat)
        residual[:] = np.nan
        residual[observed] = haversine_km(
            raw_lat[observed], raw_lon[observed], pred_lat_all[observed], pred_lon_all[observed]
        )

        threshold, res_mean, res_std = residual_threshold_km(
            residual[current_mask],
            mode=threshold_mode,
            temporal_sigma=temporal_sigma,
            temporal_outlier_km=temporal_outlier_km,
            min_threshold_km=temporal_min_threshold_km,
        )

        new_mask = initial_fit_mask & np.isfinite(residual) & (residual <= threshold)

        if new_mask.sum() < min_track_points:
            pred_lon = pred_lon_all
            pred_lat = pred_lat_all
            fit_used = current_mask.copy()
            break

        pred_lon = pred_lon_all
        pred_lat = pred_lat_all
        fit_used = new_mask.copy()

        if last_mask is not None and np.array_equal(new_mask, last_mask):
            break
        if np.array_equal(new_mask, current_mask):
            break

        last_mask = current_mask.copy()
        current_mask = new_mask

    if fit_used.any():
        rmse = float(np.sqrt(np.nanmean(residual[fit_used] ** 2)))
    else:
        rmse = np.nan

    max_res = float(np.nanmax(residual)) if np.isfinite(residual).any() else np.nan
    return pred_lon, pred_lat, fit_used, residual, threshold, res_mean, res_std, degree, rmse, max_res


def bounded_gap_fill_mask(
    candidate_missing: np.ndarray,
    valid_anchor: np.ndarray,
    max_gap_frames: int,
    allow_extrapolation: bool,
) -> np.ndarray:
    n = len(candidate_missing)
    fill = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not candidate_missing[i]:
            i += 1
            continue
        j = i
        while j < n and candidate_missing[j]:
            j += 1

        gap_len = j - i
        length_ok = (max_gap_frames < 0) or (gap_len <= max_gap_frames)
        left_ok = i > 0 and valid_anchor[i - 1]
        right_ok = j < n and valid_anchor[j]
        interior = left_ok and right_ok
        edge_allowed = allow_extrapolation and (left_ok or right_ok)

        if length_ok and (interior or edge_allowed):
            fill[i:j] = True
        i = j
    return fill


def build_track_results(
    all_df: pd.DataFrame,
    frames: List[FrameInfo],
    temporal_order: int,
    threshold_mode: str,
    temporal_outlier_km: float,
    temporal_sigma: float,
    temporal_min_threshold_km: float,
    temporal_max_iter: int,
    min_track_points: int,
    min_track_coverage: float,
    max_gap_frames: int,
    fill_missing: bool,
    allow_extrapolation: bool,
    use_pre_spatial_for_fit: bool,
    disable_temporal: bool,
) -> Dict[str, TrackResult]:
    n_frames = len(frames)
    results: Dict[str, TrackResult] = {}

    for key, g in all_df.groupby("__source_key", sort=True):
        sx, sy = parse_source_key(key)
        raw_lon = np.full(n_frames, np.nan, dtype=float)
        raw_lat = np.full(n_frames, np.nan, dtype=float)
        observed = np.zeros(n_frames, dtype=bool)
        finite = np.zeros(n_frames, dtype=bool)
        pre_ok = np.zeros(n_frames, dtype=bool)

        for frame_index, gf in g.sort_values("__row_index").groupby("__frame_index", sort=True):
            fi = int(frame_index)
            if not (0 <= fi < n_frames):
                continue
            gf_valid = gf[gf["__finite_geo"].astype(bool)]
            row = gf_valid.iloc[0] if len(gf_valid) else gf.iloc[0]
            finite[fi] = bool(row["__finite_geo"])
            observed[fi] = finite[fi]
            pre_ok[fi] = bool(row["__pre_spatial_ok"])
            if finite[fi]:
                raw_lon[fi] = float(row["mapX"])
                raw_lat[fi] = float(row["mapY"])

        if disable_temporal:
            pred_lon = raw_lon.copy()
            pred_lat = raw_lat.copy()
            residual = np.full(n_frames, np.nan, dtype=float)
            fit_used = finite.copy()
            temporal_outlier = np.zeros(n_frames, dtype=bool)
            imputed = np.zeros(n_frames, dtype=bool)
            output_present = finite.copy()
            results[key] = TrackResult(
                key=key, sourceX=sx, sourceY=sy,
                raw_lon=raw_lon, raw_lat=raw_lat,
                pred_lon=pred_lon, pred_lat=pred_lat,
                observed=observed, finite=finite, pre_spatial_ok=pre_ok,
                fit_used=fit_used, temporal_outlier=temporal_outlier,
                imputed=imputed, output_present=output_present,
                residual_km=residual, threshold_km=np.nan,
                residual_mean_km=np.nan, residual_std_km=np.nan,
                degree=-1, rmse_km=np.nan, max_residual_km=np.nan,
                can_impute=False,
            )
            continue

        fit_mask = finite.copy()
        if use_pre_spatial_for_fit:
            fit_mask &= pre_ok

        (
            pred_lon, pred_lat, fit_used, residual, threshold, res_mean, res_std,
            degree, rmse, max_res,
        ) = fit_predict_track(
            raw_lon=raw_lon,
            raw_lat=raw_lat,
            initial_fit_mask=fit_mask,
            n_frames=n_frames,
            order=temporal_order,
            min_track_points=min_track_points,
            threshold_mode=threshold_mode,
            temporal_sigma=temporal_sigma,
            temporal_outlier_km=temporal_outlier_km,
            temporal_min_threshold_km=temporal_min_threshold_km,
            temporal_max_iter=temporal_max_iter,
        )

        can_fit = degree >= 1 and np.isfinite(pred_lon).any() and np.isfinite(pred_lat).any()
        coverage = float(finite.sum()) / max(n_frames, 1)
        can_impute = bool(can_fit and fit_used.sum() >= min_track_points and coverage >= min_track_coverage)

        temporal_outlier = finite & np.isfinite(residual) & np.isfinite(threshold) & (residual > threshold)
        temporal_good = fit_used.copy()
        output_present = temporal_good.copy()
        imputed = np.zeros(n_frames, dtype=bool)

        if can_impute:
            # Observed temporal outliers are bad observations. They are always
            # replaced by the fitted trajectory when the track has enough support.
            # max_gap_frames / allow_extrapolation is reserved for truly missing
            # frames, not for observed outliers.
            outlier_fill = temporal_outlier.copy()

            missing_fill = np.zeros(n_frames, dtype=bool)
            if fill_missing:
                missing_fill = bounded_gap_fill_mask(
                    candidate_missing=~finite,
                    valid_anchor=temporal_good,
                    max_gap_frames=max_gap_frames,
                    allow_extrapolation=allow_extrapolation,
                )

            fill_mask = outlier_fill | missing_fill
            output_present[fill_mask] = True
            imputed[fill_mask] = True

        results[key] = TrackResult(
            key=key, sourceX=sx, sourceY=sy,
            raw_lon=raw_lon, raw_lat=raw_lat,
            pred_lon=pred_lon, pred_lat=pred_lat,
            observed=observed, finite=finite, pre_spatial_ok=pre_ok,
            fit_used=fit_used, temporal_outlier=temporal_outlier,
            imputed=imputed, output_present=output_present,
            residual_km=residual, threshold_km=threshold,
            residual_mean_km=res_mean, residual_std_km=res_std,
            degree=degree, rmse_km=rmse, max_residual_km=max_res,
            can_impute=can_impute,
        )

    return results


# -----------------------------------------------------------------------------
# Output construction
# -----------------------------------------------------------------------------


def row_to_output_dict(row: Optional[pd.Series], output_columns: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in output_columns:
        if row is not None and col in row.index and not str(col).startswith("__"):
            out[col] = row[col]
        elif col == "enable":
            out[col] = 1
        elif col in {"dX", "dY", "residual"}:
            out[col] = 0.0
        elif col in {"temporal_status", "temporal_residual_km", "temporal_threshold_km"}:
            out[col] = np.nan
        else:
            out[col] = np.nan
    return out


def build_outputs_by_frame(
    all_df: pd.DataFrame,
    frames: List[FrameInfo],
    track_results: Dict[str, TrackResult],
    output_columns: Sequence[str],
    add_qc_columns: bool,
) -> Dict[int, pd.DataFrame]:
    n_frames = len(frames)
    outputs: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_frames)}
    grouped = {key: g.sort_values(["__frame_index", "__row_index"]) for key, g in all_df.groupby("__source_key")}

    for key, tr in track_results.items():
        track = grouped.get(key)
        rows_by_frame: Dict[int, pd.Series] = {}
        if track is not None:
            for _, row in track.iterrows():
                fi = int(row["__frame_index"])
                if fi not in rows_by_frame:
                    rows_by_frame[fi] = row

        template_row = next(iter(rows_by_frame.values()), None)
        template = row_to_output_dict(template_row, output_columns)
        template["sourceX"] = tr.sourceX
        template["sourceY"] = tr.sourceY

        for i in range(n_frames):
            if not tr.output_present[i]:
                continue

            if tr.fit_used[i] and not tr.imputed[i] and i in rows_by_frame:
                out = row_to_output_dict(rows_by_frame[i], output_columns)
                status = "kept"
            elif tr.imputed[i]:
                out = dict(template)
                if i in rows_by_frame:
                    # Preserve non-geographic ancillary columns from the original row
                    # when an original bad observation existed.
                    original = row_to_output_dict(rows_by_frame[i], output_columns)
                    for c, v in original.items():
                        if c not in {"mapX", "mapY", "enable", "dX", "dY", "residual"}:
                            out[c] = v
                out["mapX"] = float(tr.pred_lon[i])
                out["mapY"] = float(tr.pred_lat[i])
                out["sourceX"] = float(tr.sourceX)
                out["sourceY"] = float(tr.sourceY)
                out["enable"] = 1
                out["dX"] = 0.0
                out["dY"] = 0.0
                out["residual"] = 0.0
                status = "imputed"
                frames[i].n_imputed += 1
            else:
                continue

            if add_qc_columns:
                out["temporal_status"] = status
                out["temporal_residual_km"] = float(tr.residual_km[i]) if np.isfinite(tr.residual_km[i]) else np.nan
                out["temporal_threshold_km"] = float(tr.threshold_km) if np.isfinite(tr.threshold_km) else np.nan

            outputs[i].append(out)

    output_dfs: Dict[int, pd.DataFrame] = {}
    for info in frames:
        rows = outputs.get(info.frame_index, [])
        df = pd.DataFrame(rows, columns=list(output_columns))
        if not df.empty:
            sort_y = -pd.to_numeric(df["sourceY"], errors="coerce")
            sort_x = pd.to_numeric(df["sourceX"], errors="coerce")
            df = df.assign(__sort_y=sort_y, __sort_x=sort_x).sort_values(["__sort_y", "__sort_x"])
            df = df.drop(columns=["__sort_y", "__sort_x"])
        info.output_before_post_spatial = len(df)
        output_dfs[info.frame_index] = df

    return output_dfs


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
            info.n_post_spatial_rejected = 0
            continue

        mask = spatial_neighbor_mask(df, radius_km=radius_km)
        filtered_df = df.loc[mask].copy()
        info.post_spatial_kept = len(filtered_df)
        info.n_post_spatial_rejected = len(df) - len(filtered_df)
        filtered[info.frame_index] = filtered_df
    return filtered


def write_outputs(output_dfs: Dict[int, pd.DataFrame], frames: List[FrameInfo], output_folder: str) -> None:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for info in frames:
        df = output_dfs.get(info.frame_index, pd.DataFrame())
        output_path = out_dir / info.output_name
        df.to_csv(output_path, index=False)
        if df.empty:
            print(f"WARNING: {info.output_name}: empty after filters.")
        print(
            f"OK {info.output_name}: input={info.input_count}, "
            f"pre_spatial_diag={info.pre_spatial_kept}, "
            f"temp_outliers={info.n_temporal_outliers}, "
            f"imputed={info.n_imputed}, final={len(df)}"
        )


def write_qc_tables(frames: List[FrameInfo], track_results: Dict[str, TrackResult], output_folder: str) -> None:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    for info in frames:
        frame_rows.append({
            "frame_index": info.frame_index,
            "frame_id": info.frame_id,
            "input_name": info.input_name,
            "output_name": info.output_name,
            "input_count": info.input_count,
            "pre_spatial_kept_diagnostic": info.pre_spatial_kept,
            "temporal_outliers": info.n_temporal_outliers,
            "imputed": info.n_imputed,
            "output_before_post_spatial": info.output_before_post_spatial,
            "post_spatial_kept": info.post_spatial_kept,
            "post_spatial_rejected": info.n_post_spatial_rejected,
        })
    pd.DataFrame(frame_rows).to_csv(out_dir / "temporal_frame_summary.csv", index=False)

    track_rows = []
    for tr in track_results.values():
        track_rows.append({
            "source_key": tr.key,
            "sourceX": tr.sourceX,
            "sourceY": tr.sourceY,
            "n_observed": int(tr.observed.sum()),
            "n_finite": int(tr.finite.sum()),
            "n_pre_spatial_ok_diagnostic": int(tr.pre_spatial_ok.sum()),
            "n_fit_used": int(tr.fit_used.sum()),
            "n_temporal_outliers": int(tr.temporal_outlier.sum()),
            "n_imputed": int(tr.imputed.sum()),
            "n_output": int(tr.output_present.sum()),
            "can_impute": bool(tr.can_impute),
            "fit_degree": int(tr.degree),
            "threshold_km": tr.threshold_km,
            "residual_mean_km": tr.residual_mean_km,
            "residual_std_km": tr.residual_std_km,
            "rmse_km": tr.rmse_km,
            "max_residual_km": tr.max_residual_km,
        })
    pd.DataFrame(track_rows).to_csv(out_dir / "temporal_track_summary.csv", index=False)


# -----------------------------------------------------------------------------
# Diagnostics and plots
# -----------------------------------------------------------------------------


def safe_filename_fragment(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def choose_diagnostic_tracks(
    track_results: Dict[str, TrackResult],
    diagnostic_inset: float,
    min_diagnostic_observations: int,
) -> List[Tuple[str, TrackResult]]:
    candidates = [tr for tr in track_results.values() if int(tr.finite.sum()) >= int(min_diagnostic_observations)]
    if not candidates:
        candidates = [tr for tr in track_results.values() if int(tr.finite.sum()) > 0]
    if not candidates:
        return []

    sx = np.array([tr.sourceX for tr in candidates], dtype=float)
    iy = np.array([-tr.sourceY for tr in candidates], dtype=float)

    xmin, xmax = float(np.nanmin(sx)), float(np.nanmax(sx))
    ymin, ymax = float(np.nanmin(iy)), float(np.nanmax(iy))
    dx = xmax - xmin
    dy = ymax - ymin

    inset = float(np.clip(diagnostic_inset, 0.0, 0.49))
    targets = {
        "top_left_inner": (xmin + inset * dx, ymin + inset * dy),
        "top_right_inner": (xmax - inset * dx, ymin + inset * dy),
        "center": (0.5 * (xmin + xmax), 0.5 * (ymin + ymax)),
        "bottom_left_inner": (xmin + inset * dx, ymax - inset * dy),
        "bottom_right_inner": (xmax - inset * dx, ymax - inset * dy),
    }

    chosen: List[Tuple[str, TrackResult]] = []
    used: set[str] = set()
    for label, (tx, ty) in targets.items():
        d2 = (sx - tx) ** 2 + (iy - ty) ** 2
        order = np.argsort(d2)
        for idx in order:
            tr = candidates[int(idx)]
            if tr.key not in used:
                chosen.append((label, tr))
                used.add(tr.key)
                break
    return chosen


def plot_track_evolution(label: str, tr: TrackResult, frames: List[FrameInfo], plot_dir: Path) -> None:
    x = np.arange(len(frames), dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    obs = tr.observed & tr.finite
    good = tr.fit_used
    outl = tr.temporal_outlier
    imp = tr.imputed

    ax = axes[0]
    valid_pred = np.isfinite(tr.pred_lat)
    ax.plot(x[valid_pred], tr.pred_lat[valid_pred], linewidth=1.5, label="polynomial")
    ax.scatter(x[obs], tr.raw_lat[obs], s=14, alpha=0.55, label="observed")
    ax.scatter(x[good], tr.raw_lat[good], s=18, label="used in fit")
    ax.scatter(x[outl], tr.raw_lat[outl], s=30, marker="x", label="temporal outlier")
    ax.scatter(x[imp], tr.pred_lat[imp], s=28, marker="D", label="imputed")
    ax.set_ylabel("latitude mapY")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    valid_pred = np.isfinite(tr.pred_lon)
    ax.plot(x[valid_pred], tr.pred_lon[valid_pred], linewidth=1.5, label="polynomial")
    ax.scatter(x[obs], tr.raw_lon[obs], s=14, alpha=0.55, label="observed")
    ax.scatter(x[good], tr.raw_lon[good], s=18, label="used in fit")
    ax.scatter(x[outl], tr.raw_lon[outl], s=30, marker="x", label="temporal outlier")
    ax.scatter(x[imp], tr.pred_lon[imp], s=28, marker="D", label="imputed")
    ax.set_ylabel("longitude mapX")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    finite_res = np.isfinite(tr.residual_km)
    ax.scatter(x[finite_res], tr.residual_km[finite_res], s=14, alpha=0.65, label="obs-poly residual")
    if np.isfinite(tr.threshold_km):
        ax.axhline(tr.threshold_km, linestyle="--", linewidth=1.0, label=f"threshold {tr.threshold_km:.1f} km")
    ax.scatter(x[outl], tr.residual_km[outl], s=30, marker="x", label="outlier")
    ax.set_ylabel("residual km")
    ax.set_xlabel("frame index")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    rmse_txt = f"{tr.rmse_km:.2f}" if np.isfinite(tr.rmse_km) else "nan"
    thr_txt = f"{tr.threshold_km:.2f}" if np.isfinite(tr.threshold_km) else "nan"
    title = (
        f"{label}: sourceX={tr.sourceX:.2f}, sourceY={tr.sourceY:.2f} | "
        f"obs={int(tr.observed.sum())}, good={int(tr.fit_used.sum())}, "
        f"outliers={int(tr.temporal_outlier.sum())}, imputed={int(tr.imputed.sum())}, "
        f"degree={tr.degree}, rmse={rmse_txt} km, threshold={thr_txt} km"
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fname = f"track_{safe_filename_fragment(label)}_sx{tr.sourceX:.1f}_sy{tr.sourceY:.1f}.png"
    fig.savefig(plot_dir / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_frame_qc(frames: List[FrameInfo], plot_dir: Path) -> None:
    x = np.arange(len(frames), dtype=int)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, [f.input_count for f in frames], label="input")
    ax.plot(x, [f.pre_spatial_kept for f in frames], label="pre_spatial_diag")
    ax.plot(x, [f.output_before_post_spatial for f in frames], label="output_before_post")
    ax.plot(x, [f.post_spatial_kept for f in frames], label="final")
    ax.set_title("Number of points per frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("n points")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "frame_point_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, [f.n_temporal_outliers for f in frames], label="temporal outliers")
    ax.plot(x, [f.n_imputed for f in frames], label="imputed")
    ax.plot(x, [f.n_post_spatial_rejected for f in frames], label="post-spatial rejected")
    ax.set_title("Temporal/spatial quality control per frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("n points")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "frame_qc_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def get_image_files(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def parse_reference_frame_indices(spec: str, n_frames: int) -> List[int]:
    if not spec:
        return []
    indices: List[int] = []
    for token in [t.strip().lower() for t in spec.split(",") if t.strip()]:
        if token == "first":
            indices.append(0)
        elif token in {"mid", "middle", "center"}:
            indices.append(n_frames // 2)
        elif token == "last":
            indices.append(n_frames - 1)
        else:
            try:
                val = int(token)
                if 0 <= val < n_frames:
                    indices.append(val)
            except ValueError:
                pass
    out: List[int] = []
    for i in indices:
        if i not in out and 0 <= i < n_frames:
            out.append(i)
    return out


def xy_from_keys(keys: Sequence[str], track_results: Dict[str, TrackResult]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for key in keys:
        tr = track_results.get(key)
        if tr is not None:
            xs.append(float(tr.sourceX))
            ys.append(float(-tr.sourceY))
            continue
        try:
            sx, sy = parse_source_key(key)
            xs.append(float(sx))
            ys.append(float(-sy))
        except Exception:
            continue
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_diagnostic_points_on_images(
    all_df: pd.DataFrame,
    final_dfs: Dict[int, pd.DataFrame],
    track_results: Dict[str, TrackResult],
    frames: List[FrameInfo],
    chosen_tracks: List[Tuple[str, TrackResult]],
    image_dir: Optional[str],
    plot_dir: Path,
    reference_spec: str,
    source_round_decimals: int,
) -> None:
    """
    Overlay QC on real images for selected frames.

    Mutually exclusive visual coding:
      - blue circles: original points that survive and were not touched;
      - red circles: non-imputed/non-outlier points rejected by final filtering;
      - green diamonds: temporal outliers/imputed points that survive final filtering;
      - orange diamonds: temporal outliers/imputed points rejected by final filtering;
      - yellow stars: selected diagnostic tracks.
    """
    if not image_dir or not reference_spec or not chosen_tracks:
        return

    img_dir = Path(image_dir)
    if not img_dir.exists():
        print(f"WARNING: image_dir does not exist, skipping overlays: {img_dir}")
        return

    image_files = get_image_files(img_dir)
    if not image_files:
        print(f"WARNING: no images found in {img_dir}; skipping overlays.")
        return

    frame_indices = parse_reference_frame_indices(reference_spec, len(frames))
    if not frame_indices:
        return

    label_effect = [pe.withStroke(linewidth=3.5, foreground="black")]

    for fi in frame_indices:
        if fi >= len(frames):
            continue
        if fi >= len(image_files):
            print(f"WARNING: no image for frame_index={fi}; skipping overlay.")
            continue

        img_path = image_files[fi]
        try:
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        except Exception as exc:
            print(f"WARNING: could not open {img_path}: {exc}")
            continue

        frame_points = all_df[all_df["__frame_index"] == fi].copy()
        final_df = final_dfs.get(fi, pd.DataFrame())

        final_keys: set[str] = set()
        if final_df is not None and not final_df.empty and {"sourceX", "sourceY"}.issubset(final_df.columns):
            sx_final = pd.to_numeric(final_df["sourceX"], errors="coerce").to_numpy(dtype=float)
            sy_final = pd.to_numeric(final_df["sourceY"], errors="coerce").to_numpy(dtype=float)
            finite_final = np.isfinite(sx_final) & np.isfinite(sy_final)
            final_keys = {
                make_source_key(x, y, source_round_decimals)
                for x, y in zip(sx_final[finite_final], sy_final[finite_final])
            }

        input_keys: set[str] = set()
        if not frame_points.empty:
            input_keys = set(frame_points["__source_key"].astype(str).tolist())

        # Outliers and imputed points are one visual class for this diagnostic.
        corrected_keys: set[str] = set()
        for key, tr in track_results.items():
            if fi >= len(tr.output_present):
                continue
            if bool(tr.temporal_outlier[fi]) or bool(tr.imputed[fi]):
                corrected_keys.add(key)

        all_visible_keys = input_keys | final_keys | corrected_keys

        blue_keys = sorted(k for k in all_visible_keys if k in final_keys and k not in corrected_keys)
        green_keys = sorted(k for k in all_visible_keys if k in final_keys and k in corrected_keys)
        orange_keys = sorted(k for k in all_visible_keys if k not in final_keys and k in corrected_keys)
        red_keys = sorted(k for k in all_visible_keys if k not in final_keys and k not in corrected_keys)

        fig, ax = plt.subplots(figsize=(13, 8.5))
        ax.imshow(img, origin="upper")

        rx, ry = xy_from_keys(red_keys, track_results)
        if len(rx):
            ax.scatter(
                rx, ry,
                s=34, marker="o",
                facecolors="red", edgecolors="black", linewidths=0.25,
                alpha=0.90, zorder=3,
                label=f"removed by filtering ({len(rx)})",
            )

        bx, by = xy_from_keys(blue_keys, track_results)
        if len(bx):
            ax.scatter(
                bx, by,
                s=22, marker="o",
                facecolors="dodgerblue", edgecolors="black", linewidths=0.20,
                alpha=0.90, zorder=4,
                label=f"untouched kept ({len(bx)})",
            )

        ox, oy = xy_from_keys(orange_keys, track_results)
        if len(ox):
            ax.scatter(
                ox, oy,
                s=74, marker="D",
                facecolors="orange", edgecolors="black", linewidths=0.75,
                alpha=0.98, zorder=6,
                label=f"outlier/imputed removed ({len(ox)})",
            )

        gx, gy = xy_from_keys(green_keys, track_results)
        if len(gx):
            ax.scatter(
                gx, gy,
                s=74, marker="D",
                facecolors="lime", edgecolors="black", linewidths=0.75,
                alpha=0.98, zorder=7,
                label=f"outlier/imputed kept ({len(gx)})",
            )

        for label, tr in chosen_tracks:
            x = float(tr.sourceX)
            y = float(-tr.sourceY)
            if tr.key in green_keys:
                status = "imputed kept"
            elif tr.key in orange_keys:
                status = "imputed removed"
            elif tr.key in blue_keys:
                status = "kept"
            elif tr.key in red_keys:
                status = "removed"
            else:
                status = "missing"

            ax.scatter(
                [x], [y],
                s=190, marker="*",
                facecolors="yellow", edgecolors="black", linewidths=1.1,
                label="diagnostic tracks" if label == chosen_tracks[0][0] else None,
                zorder=11,
            )
            ax.text(
                x + 24, y - 24, f"{label}\n{status}",
                fontsize=9, weight="bold", color="white",
                path_effects=label_effect,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="white", linewidth=0.35, pad=2.5),
                zorder=12,
            )

        ax.set_title(
            f"Grid filtering status on real image | frame {fi} | {img_path.name}\n"
            "blue = untouched kept, green = outlier/imputed kept, orange = outlier/imputed removed, red = removed"
        )
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.legend(loc="best", fontsize=8, framealpha=0.88)
        ax.grid(False)
        fig.tight_layout()

        out_name = f"grid_filter_status_frame_{fi:04d}_{safe_filename_fragment(img_path.stem)}.png"
        fig.savefig(plot_dir / out_name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def make_plots(
    all_df: pd.DataFrame,
    final_dfs: Dict[int, pd.DataFrame],
    track_results: Dict[str, TrackResult],
    frames: List[FrameInfo],
    plot_dir: Optional[str],
    image_dir: Optional[str],
    plot_reference_frames: str,
    diagnostic_inset: float,
    min_diagnostic_observations: int,
    source_round_decimals: int,
) -> None:
    if not plot_dir:
        return
    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)

    chosen = choose_diagnostic_tracks(
        track_results=track_results,
        diagnostic_inset=diagnostic_inset,
        min_diagnostic_observations=min_diagnostic_observations,
    )

    for label, tr in chosen:
        plot_track_evolution(label, tr, frames, out)

    plot_frame_qc(frames, out)
    plot_diagnostic_points_on_images(
        all_df=all_df,
        final_dfs=final_dfs,
        track_results=track_results,
        frames=frames,
        chosen_tracks=chosen,
        image_dir=image_dir,
        plot_dir=out,
        reference_spec=plot_reference_frames,
        source_round_decimals=source_round_decimals,
    )
    print(f"Plots saved in: {out}")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


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
    temporal_threshold_mode: str = "sigma",
    temporal_outlier_km: float = 80.0,
    temporal_sigma: float = 2.0,
    temporal_min_threshold_km: float = 0.0,
    temporal_max_iter: int = 4,
    min_track_points: int = 6,
    min_track_coverage: float = 0.20,
    max_gap_frames: int = 8,
    allow_extrapolation: bool = False,
    fill_missing: bool = True,
    pre_spatial_filter: bool = False,
    post_spatial_filter: bool = True,
    add_qc_columns: bool = False,
    plot_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    plot_reference_frames: str = "",
    diagnostic_inset: float = 0.25,
    min_diagnostic_observations: int = 20,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    frames, all_df, first_input_columns = load_timelapse_points(
        input_folder=input_folder,
        start_id=start_id,
        end_id=end_id,
        mission=mission,
        input_glob=input_glob,
        source_round_decimals=source_round_decimals,
        radius_km=radius_km,
    )

    output_columns = discover_output_columns(first_input_columns, add_qc_columns=add_qc_columns)

    print(f"Frames loaded: {len(frames)}")
    print(f"Point observations loaded: {len(all_df)}")
    print(f"Tracks sourceX/sourceY: {0 if all_df.empty else all_df['__source_key'].nunique()}")
    print(f"Temporal consistency: {'disabled' if disable_temporal else 'enabled'}")
    print(f"Spatial pre/post filter: {pre_spatial_filter}/{post_spatial_filter}")
    print(f"Final spatial radius: {radius_km:.1f} km")
    print(f"Temporal threshold: mode={temporal_threshold_mode}, sigma={temporal_sigma}")

    if not frames or all_df.empty:
        print("WARNING: no valid points to process.")
        return

    track_results = build_track_results(
        all_df=all_df,
        frames=frames,
        temporal_order=temporal_order,
        threshold_mode=temporal_threshold_mode,
        temporal_outlier_km=temporal_outlier_km,
        temporal_sigma=temporal_sigma,
        temporal_min_threshold_km=temporal_min_threshold_km,
        temporal_max_iter=temporal_max_iter,
        min_track_points=min_track_points,
        min_track_coverage=min_track_coverage,
        max_gap_frames=max_gap_frames,
        fill_missing=fill_missing,
        allow_extrapolation=allow_extrapolation,
        use_pre_spatial_for_fit=pre_spatial_filter,
        disable_temporal=disable_temporal,
    )

    for tr in track_results.values():
        for i in np.where(tr.temporal_outlier)[0]:
            frames[int(i)].n_temporal_outliers += 1

    output_dfs = build_outputs_by_frame(
        all_df=all_df,
        frames=frames,
        track_results=track_results,
        output_columns=output_columns,
        add_qc_columns=add_qc_columns,
    )

    final_dfs = apply_post_spatial_filter_to_outputs(
        output_dfs=output_dfs,
        frames=frames,
        radius_km=radius_km,
        post_spatial_filter=post_spatial_filter,
    )

    write_outputs(final_dfs, frames, output_folder)
    write_qc_tables(frames, track_results, output_folder)
    make_plots(
        all_df=all_df,
        final_dfs=final_dfs,
        track_results=track_results,
        frames=frames,
        plot_dir=plot_dir,
        image_dir=image_dir,
        plot_reference_frames=plot_reference_frames,
        diagnostic_inset=diagnostic_inset,
        min_diagnostic_observations=min_diagnostic_observations,
        source_round_decimals=source_round_decimals,
    )

    print("\nTemporal/spatial filtering and renaming completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal/spatial filtering of geographic points and renaming to <mission>-E-<ID>.points"
    )
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with original *_real.points files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder for filtered and renamed .points")
    parser.add_argument("--radius_km", type=float, default=80.0, help="Final BallTree radius in km")
    parser.add_argument("--start_id", type=int, required=True, help="Initial image ID")
    parser.add_argument("--end_id", type=int, required=True, help="Final image ID")
    parser.add_argument("--mission", type=str, required=True, help="Mission name, e.g. ISS053 or ISS067")

    parser.add_argument("--input_glob", type=str, default="*_real.points", help="Input glob inside input_folder")
    parser.add_argument("--source_round_decimals", type=int, default=3, help="Decimals to group sourceX/sourceY tracks")

    parser.add_argument("--disable_temporal", action="store_true", help="Disable temporal coherence")
    parser.add_argument("--temporal_order", type=int, default=3, help="Temporal polynomial order")
    parser.add_argument("--temporal_threshold_mode", choices=["sigma", "hybrid", "absolute"], default="sigma")
    parser.add_argument("--temporal_sigma", type=float, default=2.0, help="Number of sigmas for temporal outlier detection")
    parser.add_argument("--temporal_outlier_km", type=float, default=80.0, help="Absolute threshold for absolute/hybrid mode")
    parser.add_argument("--temporal_min_threshold_km", type=float, default=0.0, help="Optional minimum temporal threshold")
    parser.add_argument("--temporal_max_iter", type=int, default=4, help="Temporal fit/rejection iterations")
    parser.add_argument("--min_track_points", type=int, default=6, help="Minimum observations to fit a track")
    parser.add_argument("--min_track_coverage", type=float, default=0.20, help="Minimum track coverage to allow imputation")
    parser.add_argument("--max_gap_frames", type=int, default=8, help="Maximum missing interior gap to fill; -1 allows any length")
    parser.add_argument("--allow_extrapolation", action="store_true", help="Allow filling missing edge gaps")
    parser.add_argument("--no_fill_missing", action="store_true", help="Do not fill absent frames; only correct observed outliers")

    parser.add_argument("--pre_spatial_filter", action="store_true", help="Use BallTree before temporal fit")
    parser.add_argument("--no_pre_spatial_filter", action="store_true", help="Compatibility flag: force no pre BallTree")
    parser.add_argument("--no_post_spatial_filter", action="store_true", help="Disable final BallTree")

    parser.add_argument("--add_qc_columns", action="store_true", help="Add temporal QC columns to output .points")
    parser.add_argument("--plot_dir", type=str, default=None, help="Folder for QC plots")
    parser.add_argument("--image_dir", type=str, default=None, help="Pics folder for overlay plots")
    parser.add_argument("--plot_reference_frames", type=str, default="", help="Overlay frames: first,mid,last or comma-separated indices")
    parser.add_argument("--diagnostic_inset", type=float, default=0.25, help="Relative inset for internal diagnostic tracks")
    parser.add_argument("--min_diagnostic_observations", type=int, default=20, help="Minimum observations for diagnostic tracks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre_spatial = bool(args.pre_spatial_filter) and not bool(args.no_pre_spatial_filter)

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
        temporal_threshold_mode=args.temporal_threshold_mode,
        temporal_outlier_km=args.temporal_outlier_km,
        temporal_sigma=args.temporal_sigma,
        temporal_min_threshold_km=args.temporal_min_threshold_km,
        temporal_max_iter=args.temporal_max_iter,
        min_track_points=args.min_track_points,
        min_track_coverage=args.min_track_coverage,
        max_gap_frames=args.max_gap_frames,
        allow_extrapolation=args.allow_extrapolation,
        fill_missing=not args.no_fill_missing,
        pre_spatial_filter=pre_spatial,
        post_spatial_filter=not args.no_post_spatial_filter,
        add_qc_columns=args.add_qc_columns,
        plot_dir=args.plot_dir,
        image_dir=args.image_dir,
        plot_reference_frames=args.plot_reference_frames,
        diagnostic_inset=args.diagnostic_inset,
        min_diagnostic_observations=args.min_diagnostic_observations,
    )


if __name__ == "__main__":
    main()
