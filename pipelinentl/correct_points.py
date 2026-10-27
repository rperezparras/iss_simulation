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


def geo_to_pixel(x, y, gt):
    """Map lon/lat to OpenCV center-based pixel indices (continuous)."""
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if abs(det) < 1e-20:
        raise ValueError("Non-invertible raster geotransform")
    dx, dy = x - gt[0], y - gt[3]
    # GDAL transform is corner-based; array index 0 samples the center at 0.5.
    return (gt[5] * dx - gt[2] * dy) / det - 0.5, (-gt[4] * dx + gt[1] * dy) / det - 0.5


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


def bilinear_mask_valid(mask, x, y):
    """All 4 flow pixels used by bilinear interpolation must be valid."""
    h, w = mask.shape
    if not np.isfinite(x) or not np.isfinite(y) or x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return False
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    return bool(mask[y0, x0] and mask[y0, x1] and mask[y1, x0] and mask[y1, x1])


def correct_one(points_path, flow_path, mapping_path, geo_path, out_path,
                cli_sign=None, use_cli_sign=False,
                max_correction_px=0.0, enable_max_filter=False,
                allow_legacy_flow=False):
    pts = pd.read_csv(points_path, comment="M")
    mp = pd.read_csv(mapping_path)

    for c in ["mapX", "mapY", "sourceX", "sourceY"]:
        if c not in pts.columns:
            raise ValueError(f"Missing column {c} in {points_path}")
    for c in ["geoX", "geoY"]:
        if c not in mp.columns:
            raise ValueError(f"Missing column {c} in {mapping_path}")

    if len(pts) != len(mp):
        raise ValueError(
            f"GCP count mismatch: points={len(pts)} mapping={len(mp)} "
            f"for {points_path}; refusing to pair unmatched rows"
        )
    pts, mp = pts.copy(), mp.copy()
    # Row ordering is essential: sampling the wrong GCP corrupts corrections.
    for col in ("sourceX", "sourceY", "mapX", "mapY"):
        if col in mp.columns:
            p = pd.to_numeric(pts[col], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(mp[col], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(p, m, atol=1e-5, rtol=0, equal_nan=True):
                raise ValueError(f"GCP mapping rows are not aligned on {col}: {mapping_path}")

    flow = np.load(flow_path)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Invalid flow shape {flow.shape}: {flow_path}")

    mpath = meta_path(flow_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Missing flow metadata: {mpath}")
    with open(mpath, encoding="utf-8") as f:
        meta = json.load(f)
    mask_file = meta.get("flow_valid_mask_file")
    flow_valid_mask = None
    if mask_file:
        mask_path = Path(flow_path).parent / Path(mask_file).name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing flow validity mask: {mask_path}")
        flow_valid_mask = np.load(mask_path).astype(bool)
        if flow_valid_mask.shape != flow.shape[:2]:
            raise ValueError("Flow validity mask shape differs from optical flow")
    elif not allow_legacy_flow:
        raise ValueError(
            f"Flow {flow_path} has no validity mask. "
            "Recompute it with corrected optical_flow.py. "
            "For a controlled quantization-only comparison use --allow_legacy_flow."
        )

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
    dxs, dys, rawdxs, rawdys, insides, valid_samples, applied, mags, reasons = [], [], [], [], [], [], [], [], []
    sample_pixels_x, sample_pixels_y = [], []

    # Calculate true subpixel GCP location directly from original geographic
    # coordinates, NEVER from truncated legacy pixel_mapping.csv columns.
    for oldx, oldy in zip(pd.to_numeric(pts["mapX"], errors="coerce"),
                          pd.to_numeric(pts["mapY"], errors="coerce")):
        if np.isfinite(oldx) and np.isfinite(oldy):
            px, py = geo_to_pixel(float(oldx), float(oldy), gt)
        else:
            px, py = float("nan"), float("nan")
        sample_pixels_x.append(px); sample_pixels_y.append(py)
        dx = dy = rawdx = rawdy = 0.0
        inside = valid_sample = app = 0
        reason = "outside_flow_roi"
        if np.isfinite(px) and np.isfinite(py):
            rx, ry = float(px)-x0, float(py)-y0
            f = bilinear(flow, rx, ry)
            if f is not None:
                inside = 1
                if flow_valid_mask is not None and not bilinear_mask_valid(flow_valid_mask, rx, ry):
                    reason = "invalid_iss_footprint_or_margin"
                else:
                    valid_sample = 1
                    rawdx, rawdy = sign*float(f[0]), sign*float(f[1])
                    mag = float(np.hypot(rawdx, rawdy))
                    if not np.isfinite(mag):
                        reason = "non_finite_flow"
                        rawdx = rawdy = 0.0
                    elif enable_max_filter and max_correction_px > 0 and mag > max_correction_px:
                        reason = "too_large_filtered"
                    else:
                        dx, dy = rawdx, rawdy
                        app = int(mag > 1e-9)
                        reason = "applied" if app else "zero_flow"
            # IMPORTANT: keep the exact original geographic GCP coordinates.
            # The legacy script converted integer-truncated pixel_mapping values
            # back to lon/lat even when no flow was applied, causing a hidden
            # quantization shift. Apply ONLY the affine displacement vector.
            mx = float(oldx) + gt[1] * dx + gt[2] * dy
            my = float(oldy) + gt[4] * dx + gt[5] * dy
        else:
            mx, my = float(oldx), float(oldy)
            reason = "non_finite_mapping_pixel"
        new_mapx.append(mx); new_mapy.append(my)
        dxs.append(dx); dys.append(dy); rawdxs.append(rawdx); rawdys.append(rawdy)
        insides.append(inside); valid_samples.append(valid_sample)
        applied.append(app); mags.append(float(np.hypot(dx,dy))); reasons.append(reason)

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
        "flow_valid": valid_samples,
        "flow_applied": applied,
        "correction_reason": reasons,
        "geoX_first": sample_pixels_x,
        "geoY_first": sample_pixels_y,
        "geoX_mapping_legacy": pd.to_numeric(mp["geoX"], errors="coerce").to_numpy(),
        "geoY_mapping_legacy": pd.to_numeric(mp["geoY"], errors="coerce").to_numpy(),
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    arr = np.asarray(mags, float)
    stats = {
        "mode": "original_gcps_subpixel_masked_flow_v2",
        "pixel_sample_source": "inverse_geotransform_of_original_mapX_mapY",
        "used_legacy_unmasked_flow": bool(flow_valid_mask is None),
        "no_dense_points_added": True,
        "preserve_original_geographic_coordinates_when_no_flow": True,
        "no_max_pixel_limit_applied": not bool(enable_max_filter),
        "flow_sign_used": sign,
        "flow_sign_source": sign_source,
        "flow_sign_cli_received": cli_sign,
        "n_points": int(len(out)),
        "inside_flow_count": int(np.sum(insides)),
        "valid_flow_count": int(np.sum(valid_samples)),
        "rejected_by_footprint_or_margin": int(np.sum(np.array(insides) & ~np.array(valid_samples, dtype=bool))),
        "mask_margin_px": int(meta.get("mask_margin_px", -1)),
        "applied_count": int(np.sum(applied)),
        "reason_counts": {str(k): int(v) for k, v in pd.Series(reasons).value_counts().to_dict().items()},
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": x1-x0, "height": y1-y0},
        "flow_definition": meta.get("flow_definition"),
        "flow_correction": meta.get("correction_to_apply_in_correct_points"),
        "correction_mag_percentiles": {
            "p50": float(np.percentile(arr, 50)) if arr.size else None,
            "p90": float(np.percentile(arr, 90)) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
        },
    }
    sp = out_path.replace("_corrected.points", "_correction_stats.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"OK {out_path} | inside={stats['inside_flow_count']}/{stats['n_points']} | valid={stats['valid_flow_count']}/{stats['n_points']} | applied={stats['applied_count']}/{stats['n_points']} | sign={sign} ({sign_source})")


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
    ap.add_argument("--allow_legacy_flow", action="store_true",
                    help="Only for comparison: use an old flow without a validity mask")
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
                allow_legacy_flow=args.allow_legacy_flow,
            )
        except Exception as e:
            errors += 1
            print(f"ERROR {base}: {e}")
    print(f"Correccion terminada. Errores: {errors}")
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
