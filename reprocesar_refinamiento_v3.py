#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reprocesa SOLO ISS->VIIRS optical flow, GCP correction y 2a TPS.

No toca flow/, corrected_points/, geo_corrected/ originales.
Procesa solo los IDs con filtered_points, primera TPS, ROI VIIRS y JPG reales.
Resultados reanudables en CASE/paper/reprocessing_v3, con manifest de procedencia.

Ejemplos:
  python3 scripts/reprocesar_refinamiento_v3.py CASE --plan --expect 1252
  python3 scripts/reprocesar_refinamiento_v3.py CASE --start 327470 --end 327470
  python3 scripts/reprocesar_refinamiento_v3.py CASE --expect 1252 --jobs 2
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

FRAME_PATTERN = re.compile(r"^ISS\d+-E-(\d{5,8})\.points$")
PICS_EXT = ("JPG", "jpg", "JPEG", "jpeg", "png", "PNG", "tif", "tiff")
ALGORITHM = "iss_viirs_masked_v2"


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_image(folder: Path, frame: str):
    return next((folder / f"{frame}.{ext}" for ext in PICS_EXT if (folder / f"{frame}.{ext}").is_file()), None)


def find_frames(case: Path, lo: int, hi: int):
    points = case / "filtered_points"
    if not points.is_dir():
        raise FileNotFoundError(f"No existe {points}")
    selected = []
    for p in sorted(points.iterdir()):
        m = FRAME_PATTERN.fullmatch(p.name)
        if m and lo <= int(m.group(1)) <= hi and p.stat().st_size:
            selected.append(p.stem)
    return selected


def required_inputs(case, frame):
    required = [
        case / "filtered_points" / f"{frame}.points",
        case / "geo" / f"{frame}_rect.tiff",
        case / "geo" / f"{frame}_pixel_mapping.csv",
        case / "viirs_cropped_aligned" / f"{frame}_viirs.tiff",
        case / "viirs_cropped_aligned" / f"{frame}_roi.json",
    ]
    pic = source_image(case / "pics", frame)
    if pic is None:
        required.append(case / "pics" / f"{frame}.JPG")
    return [str(p) for p in required if not p.is_file()]


def source_digest():
    from pipelinentl import optical_flow, correct_points, georef_timelapse
    return {name: sha256_file(Path(mod.__file__).resolve()) for name, mod in [
        ("optical_flow.py", optical_flow),
        ("correct_points.py", correct_points),
        ("georef_timelapse.py", georef_timelapse),
    ]}


def validate_flow(flowdir, frame, margin, feather, mode):
    import numpy as np
    data = flowdir / f"{frame}_flow.npy"
    meta_path = flowdir / f"{frame}_flow_meta.json"
    if not data.is_file() or not meta_path.is_file():
        return False
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        mask = flowdir / str(m["flow_valid_mask_file"])
        if (m.get("pipeline_flow_algorithm") != ALGORITHM
                or int(m["mask_margin_px"]) != margin
                or int(m["mask_feather_px"]) != feather
                or m.get("histogram_mode") != mode
                or not mask.is_file()):
            return False
        array = np.load(data, mmap_mode="r", allow_pickle=False)
        valid = np.load(mask, mmap_mode="r", allow_pickle=False)
        return (array.ndim == 3 and array.shape[2] == 2
                and valid.shape == array.shape[:2]
                and array.shape[:2] == (int(m["y1"]) - int(m["y0"]),
                                        int(m["x1"]) - int(m["x0"])))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def validate_corrected(correctdir, frame):
    import pandas as pd
    p = correctdir / f"{frame}_corrected.points"
    stats = correctdir / f"{frame}_correction_stats.json"
    if not (p.is_file() and stats.is_file()):
        return False
    try:
        s = json.loads(stats.read_text(encoding="utf-8"))
        if s.get("mode") != "original_gcps_subpixel_masked_flow_v2" or s.get("used_legacy_unmasked_flow"):
            return False
        df = pd.read_csv(p, comment="#")
        for col in ("mapX", "mapY", "sourceX", "sourceY", "flow_applied", "flow_valid"):
            if col not in df.columns:
                return False
        if len(df) != s.get("n_points") or len(df) < 6:
            return False
        return bool(df[["mapX", "mapY", "sourceX", "sourceY"]].notna().all().all())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def validate_geotiff(geodir, frame):
    import rasterio
    f = geodir / f"{frame}_rect.tiff"
    p = geodir / f"{frame}_pixel_mapping.csv"
    if not f.is_file() or not p.is_file() or not f.stat().st_size or not p.stat().st_size:
        return False
    try:
        with rasterio.open(f) as ds:
            return ds.width >= 100 and ds.height >= 100 and ds.crs is not None and ds.count >= 1
    except (OSError, ValueError):
        return False


def georef_worker(args):
    frame, case_str, points_dir_str, geo_dir_str, idx = args
    from pipelinentl import georef_timelapse
    case = Path(case_str)
    output = Path(geo_dir_str)
    (output / "temp").mkdir(exist_ok=True, parents=True)
    status = georef_timelapse.process_one(
        frame, str(case / "pics"), points_dir_str, str(output),
        str(output / "temp"), idx, 0,
    )
    return frame, status


def estimate_bytes(case, frames, root, existing):
    from statistics import median
    viirs_roi_dir = case / "viirs_cropped_aligned"
    sizes = []
    for frame in frames[: min(40, len(frames))]:
        try:
            m = json.loads((viirs_roi_dir / f"{frame}_roi.json").read_text())
            sizes.append(max(0, int(m["x1"]) - int(m["x0"])) *
                         max(0, int(m["y1"]) - int(m["y0"])))
        except (OSError, ValueError, KeyError):
            continue
    avg_roi = median(sizes) if sizes else 3_000_000
    # float32 flow (2 channels) + uint8 valid + additional overhead estimate.
    bytes_flow = int(avg_roi * 9 * 1.03)
    old_sizes = [p.stat().st_size for p in list((case / "geo_corrected").glob("*_rect.tiff"))[:25]]
    bytes_geo = int(median(old_sizes) if old_sizes else 35_000_000)
    missing_flow = sum(not existing["flow"](frame) for frame in frames)
    missing_geo = sum(not existing["georef"](frame) for frame in frames)
    estimate = int(bytes_flow * missing_flow + bytes_geo * missing_geo)
    return estimate, bytes_flow, bytes_geo


def write_stats(root, frames, flowdir, correctdir, geodir):
    import numpy as np
    import pandas as pd
    rows = []
    for frame in frames:
        stats_path = correctdir / f"{frame}_correction_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        row = {
            "frame": frame,
            "flow_ready": int((flowdir / f"{frame}_flow.npy").exists()),
            "corrected_ready": int((correctdir / f"{frame}_corrected.points").exists()),
            "second_tps_ready": int((geodir / f"{frame}_rect.tiff").exists()),
            "n_gcps": stats.get("n_points"),
            "n_flow_valid": stats.get("valid_flow_count"),
            "n_flow_applied": stats.get("applied_count"),
            "median_correction_all_gcps_px": (stats.get("correction_mag_percentiles") or {}).get("p50"),
            "max_correction_px": (stats.get("correction_mag_percentiles") or {}).get("max"),
        }
        rows.append(row)
    with (root / "resumen_por_frame.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["frame"])
        w.writeheader(); w.writerows(rows)
    valid = [r for r in rows if r["median_correction_all_gcps_px"] is not None]
    nums = np.array([r["median_correction_all_gcps_px"] for r in valid], dtype=float)
    summary = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frame_candidates": len(frames),
        "flow_files": sum(r["flow_ready"] for r in rows),
        "corrected_point_files": sum(r["corrected_ready"] for r in rows),
        "second_tps_files": sum(r["second_tps_ready"] for r in rows),
        "median_of_per_frame_medians_px": float(np.median(nums)) if nums.size else None,
        "iqr_of_per_frame_medians_px": [float(x) for x in np.percentile(nums, [25, 75])] if nums.size else None,
        "note": "Flow magnitude is not independent geolocation accuracy. Counts and masked parameters require scientific validation.",
    }
    atomic_json(root / "resumen_global.json", summary)
    print("\nRESUMEN:", json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--start", type=int, default=327041)
    parser.add_argument("--end", type=int, default=328344)
    parser.add_argument("--stage", choices=("all", "flow", "correct", "georef"), default="all")
    parser.add_argument("--plan", action="store_true", help="Comprobacion sin ejecutar procesamiento")
    parser.add_argument("--expect", type=int, default=0, help="Numero esperado de frames; 0 = no imponer")
    parser.add_argument("--jobs", type=int, default=2, help="Procesos GDAL para 2a TPS")
    parser.add_argument("--margin", type=int, default=96)
    parser.add_argument("--feather", type=int, default=32)
    parser.add_argument("--histogram", choices=("masked", "legacy"), default="masked")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Borra y rehace productos SOLO del output-root seleccionado")
    parser.add_argument("--allow-low-disk", action="store_true", help="Permite continuar aunque la estimacion de espacio libre sea inferior a la requerida")
    parser.add_argument("--keep-going", action="store_true", help="Registra errores y prosigue con otros fotogramas")
    args = parser.parse_args()

    if args.start > args.end or args.jobs < 1 or args.margin < args.feather or args.feather < 1:
        parser.error("Rango/jobs/margin/feather no validos")
    case = args.case.expanduser().resolve()
    if not case.is_dir():
        parser.error(f"No existe CASE: {case}")
    root = (args.output_root or case / "paper" / "reprocessing_v3").expanduser().resolve()
    if root in (case, case / "flow", case / "geo_corrected", case / "corrected_points") or case not in root.parents:
        parser.error("--output-root debe estar DENTRO de CASE, sin apuntar a los resultados originales")
    flowdir, correctdir, geodir = root / "flow", root / "corrected_points", root / "geo_corrected"
    frames = find_frames(case, args.start, args.end)
    print(f"Frames con filtered_points en rango: {len(frames)}")
    if args.expect and args.expect != len(frames):
        raise SystemExit(f"ABORTADO: se esperaban {args.expect} frames, pero hay {len(frames)}")
    if not frames:
        raise SystemExit("No se han encontrado frames con GCPs filtrados")
    invalid = {frame: required_inputs(case, frame) for frame in frames}
    invalid = {k: v for k, v in invalid.items() if v}
    if invalid:
        print("ERROR: faltan inputs (primeros 12):")
        for frame, missing in list(invalid.items())[:12]:
            print(f"  {frame}: {missing}")
        raise SystemExit(f"ABORTADO: faltan inputs en {len(invalid)} de {len(frames)} fotogramas")

    existing = {
        "flow": lambda fr: validate_flow(flowdir, fr, args.margin, args.feather, args.histogram),
        "correct": lambda fr: validate_corrected(correctdir, fr),
        "georef": lambda fr: validate_geotiff(geodir, fr),
    }
    print("Salidas aisladas:", root)
    for stage, check in existing.items():
        n = sum(check(fr) for fr in frames)
        print(f"  {stage}: {n}/{len(frames)} completados")
    estimated, bytes_flow, bytes_geo = estimate_bytes(case, frames, root, existing)
    free_bytes = shutil.disk_usage(root if root.exists() else case).free
    gb = 1024 ** 3
    print(f"Espacio estimado pendiente ~ {estimated/gb:.1f} GiB "
          f"(flow ~{bytes_flow/gb:.3f} GiB/frame, TPS ~{bytes_geo/gb:.3f} GiB/frame); "
          f"libre: {free_bytes/gb:.1f} GiB")
    if args.plan:
        print("PLAN: no se ha escrito ningun producto. Para ejecutar quita --plan.")
        return
    if estimated + 5*gb > free_bytes and not args.allow_low_disk:
        raise SystemExit("Espacio libre insuficiente segun estimacion. Limpia espacio o usa --allow-low-disk bajo tu responsabilidad")

    from pipelinentl import optical_flow, correct_points
    code_hashes = source_digest()
    stamp = {
        "algorithm": ALGORITHM,
        "code_sha256": code_hashes,
        "inputs": str(case),
        "margin": args.margin,
        "feather": args.feather,
        "histogram": args.histogram,
        "flow_sign": "from_flow_metadata",
        "no_max_correction_filter": True,
        "note": "No files in canonical flow/, corrected_points/, geo_corrected/ are overwritten",
    }
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "_processing_config.json"
    if provenance.is_file():
        saved = json.loads(provenance.read_text(encoding="utf-8"))
        if saved != stamp:
            # Code or mask params changed; never mix old and new versions in a partial rerun.
            if not args.force:
                raise SystemExit("La version/configuracion ha cambiado; usa otra --output-root o --force en TODO el rango")
            existing_frames = set()
            for folder, suffix in ((flowdir, "_flow.npy"), (correctdir, "_corrected.points"),
                                   (geodir, "_rect.tiff")):
                if folder.is_dir():
                    existing_frames.update(f.name[:-len(suffix)] for f in folder.glob(f"*{suffix}"))
            if not existing_frames.issubset(set(frames)):
                raise SystemExit("No puedes --force un subconjunto tras cambiar codigo/config: "
                                 "quedarian outputs de versiones mezcladas. Usa un --output-root nuevo.")
    elif any(p.exists() for p in (flowdir, correctdir, geodir)) and not args.force:
        raise SystemExit("Existen productos sin manifiesto de version. Usa una nueva --output-root o --force")
    for p in (flowdir, correctdir, geodir):
        p.mkdir(parents=True, exist_ok=True)
    if args.force:
        print("--force activo: eliminando SOLO los productos staged seleccionados (flow, corrected, geo).")
        for frame in frames:
            for f in (flowdir / f"{frame}_flow.npy", flowdir / f"{frame}_flow_valid.npy",
                      flowdir / f"{frame}_flow_meta.json", flowdir / f"{frame}_flow_vis.png",
                      correctdir / f"{frame}_corrected.points",
                      correctdir / f"{frame}_correction_stats.json",
                      geodir / f"{frame}_rect.tiff", geodir / f"{frame}_pixel_mapping.csv"):
                f.unlink(missing_ok=True)
    atomic_json(provenance, stamp)
    errors = []

    if args.stage in ("all", "flow"):
        for idx, frame in enumerate(frames, 1):
            if existing["flow"](frame):
                continue
            print(f"[FLOW {idx}/{len(frames)}] {frame}", flush=True)
            try:
                optical_flow.compute_and_save_optical_flow(
                    frame, geo_dir=str(case / "geo"), viirs_dir=str(case / "viirs_cropped_aligned"),
                    flow_dir=str(flowdir), crop_x_start=0, crop_x_end=1, crop_y_start=0, crop_y_end=1,
                    coverage_mask_dir=str(root / "footprints"),
                    source_images_dir=str(case / "pics"), points_dir=str(case / "filtered_points"),
                    mask_margin_px=args.margin, mask_feather_px=args.feather,
                    histogram_mode=args.histogram, show_plot=False,
                )
                if not existing["flow"](frame):
                    raise RuntimeError("Flow and/or mask validation failed after calculation")
            except Exception as e:
                errors.append((frame, "flow", str(e)))
                print(f"ERROR FLOW {frame}: {e}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    break

    if not errors or args.keep_going:
        if args.stage in ("all", "correct"):
            for idx, frame in enumerate(frames, 1):
                if existing["correct"](frame):
                    continue
                if not existing["flow"](frame):
                    errors.append((frame, "correct", "flow missing or invalid"))
                    if not args.keep_going:
                        break
                    continue
                print(f"[GCP {idx}/{len(frames)}] {frame}", flush=True)
                try:
                    correct_points.correct_one(
                        points_path=str(case / "filtered_points" / f"{frame}.points"),
                        flow_path=str(flowdir / f"{frame}_flow.npy"),
                        mapping_path=str(case / "geo" / f"{frame}_pixel_mapping.csv"),
                        geo_path=str(case / "geo" / f"{frame}_rect.tiff"),
                        out_path=str(correctdir / f"{frame}_corrected.points"),
                    )
                    if not existing["correct"](frame):
                        raise RuntimeError("Corrected points not valid")
                except Exception as e:
                    errors.append((frame, "correct", str(e)))
                    print(f"ERROR GCP {frame}: {e}", file=sys.stderr, flush=True)
                    if not args.keep_going:
                        break

    if not errors or args.keep_going:
        if args.stage in ("all", "georef"):
            jobs = [(frame, str(case), str(correctdir), str(geodir), idx)
                    for idx, frame in enumerate(frames)
                    if not existing["georef"](frame) and existing["correct"](frame)]
            missing = [frame for frame in frames if not existing["correct"](frame)]
            if missing:
                errors.extend((fr, "georef", "corrected points missing") for fr in missing)
            print(f"[TPS] georreferenciaciones pendientes: {len(jobs)}, jobs={args.jobs}", flush=True)
            if jobs and not missing or (jobs and args.keep_going):
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
                    for idx, (frame, status) in enumerate(pool.map(georef_worker, jobs), 1):
                        print(f"[TPS {idx}/{len(jobs)}] {status}", flush=True)
                        if not status.startswith("✅") and not status.startswith("⏭"):
                            errors.append((frame, "georef", status))
                        if not existing["georef"](frame):
                            errors.append((frame, "georef", "GeoTIFF or mapping invalid/missing"))

    if errors:
        log = root / "errores.json"
        atomic_json(log, {"errors": [{"frame": a, "stage": b, "message": c} for a,b,c in errors]})
        print(f"ABORTADO: {len(errors)} errores, detalles en {log}", file=sys.stderr)
        raise SystemExit(1)
    write_stats(root, frames, flowdir, correctdir, geodir)
    print("Listo. Datos nuevos en:", root)
    print("Las carpetas originales no han sido modificadas.")


if __name__ == "__main__":
    main()
