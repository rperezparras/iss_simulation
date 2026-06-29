#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correct .points using optical flow sampled at the original GCP positions.
No dense synthetic points. No magnitude limit by default.
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()


def extract_id(name):
    m = re.search(r"(\d{5,8})", os.path.basename(name))
    return int(m.group(1)) if m else None


def meta_path(flow_path):
    p = Path(flow_path)
    return p.with_name(p.name.replace("_flow.npy", "_flow_meta.json"))


def pixel_to_geo(px, py, gt):
    return gt[0] + px * gt[1] + py * gt[2], gt[3] + px * gt[4] + py * gt[5]


def sign_from_meta(meta):
    corr = str(meta.get("correction_to_apply_in_correct_points", "")).lower()
    fdef = str(meta.get("flow_definition", "")).lower()
    if "old_pixel + flow" in corr:
        return 1.0
    if "old_pixel - flow" in corr:
        return -1.0
    if "reference_iss" in fdef and "distorted_viirs" in fdef:
        return 1.0
    if "reference_viirs" in fdef and "distorted_iss" in fdef:
        return -1.0
    return 1.0


def bilinear(flow, x, y):
    h, w = flow.shape[:2]
    if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return None
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    f00 = flow[y0, x0].astype(float)
    f10 = flow[y0, x1].astype(float)
    f01 = flow[y1, x0].astype(float)
    f11 = flow[y1, x1].astype(float)
    return (1-wy)*((1-wx)*f00 + wx*f10) + wy*((1-wx)*f01 + wx*f11)


def correct_one(points_path, flow_path, mapping_path, geo_path, out_path,
                cli_sign=None, use_cli_sign=False,
                max_correction_px=0.0, enable_max_filter=False):
    pts = pd.read_csv(points_path, comment="M")
    mp = pd.read_csv(mapping_path)

    for c in ["mapX", "mapY", "sourceX", "sourceY"]:
        if c not in pts.columns:
            raise ValueError(f"Missing column {c} in {points_path}")
    for c in ["geoX", "geoY"]:
        if c not in mp.columns:
            raise ValueError(f"Missing column {c} in {mapping_path}")

    if len(pts) != len(mp):
        n = min(len(pts), len(mp))
        print(f"AVISO {Path(points_path).name}: points={len(pts)}, mapping={len(mp)}; using first {n}")
        pts = pts.iloc[:n].copy()
        mp = mp.iloc[:n].copy()
    else:
        pts = pts.copy(); mp = mp.copy()

    flow = np.load(flow_path)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Invalid flow shape {flow.shape}: {flow_path}")

    mpath = meta_path(flow_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Missing flow metadata: {mpath}")
    meta = json.load(open(mpath, encoding="utf-8"))

    ds = gdal.Open(geo_path)
    gt = ds.GetGeoTransform()
    width, height = int(ds.RasterXSize), int(ds.RasterYSize)

    x0, y0, x1, y1 = int(meta["x0"]), int(meta["y0"]), int(meta["x1"]), int(meta["y1"])
    if int(meta.get("full_width", width)) != width or int(meta.get("full_height", height)) != height:
        raise ValueError("Flow metadata full_width/full_height do not match current GeoTIFF")
    if flow.shape[1] != x1 - x0 or flow.shape[0] != y1 - y0:
        raise ValueError(f"Flow shape {flow.shape[:2]} does not match ROI {(y1-y0, x1-x0)}")

    sign = float(cli_sign) if use_cli_sign and cli_sign is not None else sign_from_meta(meta)
    sign_source = "cli" if use_cli_sign and cli_sign is not None else "flow_meta"

    new_mapx, new_mapy = [], []
    dxs, dys, rawdxs, rawdys, insides, applied, mags, reasons = [], [], [], [], [], [], [], []

    for px, py, oldx, oldy in zip(pd.to_numeric(mp["geoX"], errors="coerce"),
                                  pd.to_numeric(mp["geoY"], errors="coerce"),
                                  pts["mapX"], pts["mapY"]):
        dx = dy = rawdx = rawdy = 0.0
        inside = app = 0
        reason = "outside_flow_roi"
        if np.isfinite(px) and np.isfinite(py):
            f = bilinear(flow, float(px)-x0, float(py)-y0)
            if f is not None:
                inside = 1
                rawdx, rawdy = sign*float(f[0]), sign*float(f[1])
                mag = float(np.hypot(rawdx, rawdy))
                if enable_max_filter and max_correction_px > 0 and mag > max_correction_px:
                    reason = "too_large_filtered"
                else:
                    dx, dy = rawdx, rawdy
                    app = int(mag > 1e-9)
                    reason = "applied" if app else "zero_flow"
            mx, my = pixel_to_geo(float(px)+dx, float(py)+dy, gt)
        else:
            mx, my = oldx, oldy
            reason = "non_finite_mapping_pixel"
        new_mapx.append(mx); new_mapy.append(my)
        dxs.append(dx); dys.append(dy); rawdxs.append(rawdx); rawdys.append(rawdy)
        insides.append(inside); applied.append(app); mags.append(float(np.hypot(dx,dy))); reasons.append(reason)

    out = pd.DataFrame({
        "mapX": new_mapx,
        "mapY": new_mapy,
        "sourceX": pts["sourceX"].to_numpy(),
        "sourceY": pts["sourceY"].to_numpy(),
        "enable": pts["enable"].to_numpy() if "enable" in pts.columns else np.ones(len(pts), dtype=int),
        "dX": dxs,
        "dY": dys,
        "residual": np.zeros(len(pts)),
        "dX_raw": rawdxs,
        "dY_raw": rawdys,
        "correction_mag": mags,
        "flow_inside": insides,
        "flow_applied": applied,
        "correction_reason": reasons,
        "geoX_first": pd.to_numeric(mp["geoX"], errors="coerce").to_numpy(),
        "geoY_first": pd.to_numeric(mp["geoY"], errors="coerce").to_numpy(),
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    arr = np.asarray(mags, float)
    stats = {
        "mode": "original_points_only_exact_flow_sample",
        "no_dense_points_added": True,
        "no_max_pixel_limit_applied": not bool(enable_max_filter),
        "flow_sign_used": sign,
        "flow_sign_source": sign_source,
        "flow_sign_cli_received": cli_sign,
        "n_points": int(len(out)),
        "inside_flow_count": int(np.sum(insides)),
        "applied_count": int(np.sum(applied)),
        "reason_counts": {str(k): int(v) for k, v in pd.Series(reasons).value_counts().to_dict().items()},
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": x1-x0, "height": y1-y0},
        "flow_definition": meta.get("flow_definition"),
        "flow_correction": meta.get("correction_to_apply_in_correct_points"),
        "correction_mag_percentiles": {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        },
    }
    sp = out_path.replace("_corrected.points", "_correction_stats.json")
    json.dump(stats, open(sp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"OK {out_path} | simple original points | inside={stats['inside_flow_count']}/{stats['n_points']} | applied={stats['applied_count']}/{stats['n_points']} | sign={sign} ({sign_source})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_points_dir", required=True)
    ap.add_argument("--flow_dir", required=True)
    ap.add_argument("--geo_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--start_id", type=int, required=True)
    ap.add_argument("--end_id", type=int, required=True)
    ap.add_argument("--crop_x_start", type=float, default=0.0)  # legacy ignored when meta exists
    ap.add_argument("--crop_x_end", type=float, default=1.0)
    ap.add_argument("--crop_y_start", type=float, default=0.0)
    ap.add_argument("--crop_y_end", type=float, default=1.0)
    ap.add_argument("--flow_sign", type=float, default=None)  # ignored unless --use_cli_flow_sign
    ap.add_argument("--use_cli_flow_sign", action="store_true")
    ap.add_argument("--max_correction_px", type=float, default=0.0)  # ignored unless --enable_max_correction_filter
    ap.add_argument("--enable_max_correction_filter", action="store_true")
    ap.add_argument("--clip_correction", action="store_true")  # legacy ignored
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = []
    for f in sorted(os.listdir(args.input_points_dir)):
        if not f.endswith(".points"):
            continue
        sid = extract_id(f)
        if sid is not None and args.start_id <= sid <= args.end_id:
            files.append(f)
    print(f"Archivos .points a procesar: {len(files)}")
    print("Modo correct_points: original points only, exact optical-flow displacement, no dense points, no limit by default")

    errors = 0
    for f in files:
        base = os.path.splitext(f)[0]
        try:
            correct_one(
                os.path.join(args.input_points_dir, f),
                os.path.join(args.flow_dir, base + "_flow.npy"),
                os.path.join(args.geo_dir, base + "_pixel_mapping.csv"),
                os.path.join(args.geo_dir, base + "_rect.tiff"),
                os.path.join(args.output_dir, base + "_corrected.points"),
                cli_sign=args.flow_sign,
                use_cli_sign=args.use_cli_flow_sign,
                max_correction_px=float(args.max_correction_px),
                enable_max_filter=args.enable_max_correction_filter,
            )
        except Exception as e:
            errors += 1
            print(f"ERROR {base}: {e}")
    print(f"Correccion terminada. Errores: {errors}")
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
