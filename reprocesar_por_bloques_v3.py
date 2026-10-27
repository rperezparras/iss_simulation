#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reprocesado ISS/VIIRS v3 por bloques con limpieza segura de flujo intermedio.

No borra ningun dato de CASE/flow, CASE/geo_corrected ni CASE/corrected_points.
No modifica el algoritmo de optical_flow.py. Ejecuta el runner v3 instalado.

Uso:
  python3 reprocesar_por_bloques_v3.py "$CASE" --plan --expect 1252
  python3 reprocesar_por_bloques_v3.py "$CASE" --expect 1252 \\
      --block-size 50 --jobs 2 --prune-flow --min-free-gib 20

La limpieza elimina UNICAMENTE los dos arrays nuevos por frame, tras
validar TPS, puntos, metadatos y escribir un checkpoint atomico.
Los metadatos, huellas, puntos, TPS y estadisticas permanecen.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Import from the installed v3 runner, next to this script.
from reprocesar_refinamiento_v3 import (
    ALGORITHM, find_frames, required_inputs, source_digest,
    validate_corrected, validate_flow, validate_geotiff,
    estimate_bytes, atomic_json, sha256_file,
)

GB = 1024 ** 3
CHECKPOINT_VERSION = 1
DEFAULT_KEEP = "327470"


def parse_keep(raw: str):
    out = set()
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        if not value.isdigit():
            raise ValueError(f"ID de --keep-flow no valido: {value}")
        out.add(int(value))
    return out


def frame_number(frame):
    return int(frame.rsplit("-", 1)[-1])


def snapshot(path: Path, hash_contents=False):
    st = path.stat()
    result = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def source_fingerprint(case: Path, frame: str):
    """Detects upstream modifications before accepting a previous checkpoint."""
    from reprocesar_refinamiento_v3 import source_image
    pic = source_image(case / "pics", frame)
    if pic is None:
        raise FileNotFoundError(f"Falta imagen fuente {frame}")
    paths = {
        "filtered_points": case / "filtered_points" / f"{frame}.points",
        "first_tps": case / "geo" / f"{frame}_rect.tiff",
        "pixel_mapping": case / "geo" / f"{frame}_pixel_mapping.csv",
        "viirs": case / "viirs_cropped_aligned" / f"{frame}_viirs.tiff",
        "viirs_roi": case / "viirs_cropped_aligned" / f"{frame}_roi.json",
        "source_image": pic,
    }
    return {key: {"name": p.name, **snapshot(p, hash_contents=key in
            ("filtered_points", "pixel_mapping", "viirs_roi"))}
            for key, p in paths.items()}


def output_fingerprint(root: Path, frame: str):
    paths = {
        "corrected": root / "corrected_points" / f"{frame}_corrected.points",
        "stats": root / "corrected_points" / f"{frame}_correction_stats.json",
        "second_tps": root / "geo_corrected" / f"{frame}_rect.tiff",
        "second_mapping": root / "geo_corrected" / f"{frame}_pixel_mapping.csv",
        "flow_meta": root / "flow" / f"{frame}_flow_meta.json",
        "footprint": root / "footprints" / f"iss_geometric_footprint_{frame_number(frame)}.tif",
    }
    return {key: snapshot(p, hash_contents=key in ("corrected", "stats", "second_mapping", "flow_meta"))
            for key, p in paths.items()}


def checkpoint_path(root: Path, frame: str):
    return root / "batch_checkpoints" / f"{frame}.json"


def check_completed(case: Path, root: Path, frame: str, check_fingerprints=True):
    """Strict check: no completion without an atomic checkpoint and valid final outputs."""
    p = checkpoint_path(root, frame)
    if not p.is_file():
        return False
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
        if record.get("version") != CHECKPOINT_VERSION or record.get("frame") != frame:
            return False
        if not validate_corrected(root / "corrected_points", frame):
            return False
        if not validate_geotiff(root / "geo_corrected", frame):
            return False
        if check_fingerprints:
            if record["inputs"] != source_fingerprint(case, frame):
                raise RuntimeError(f"Los INPUTS de {frame} cambiaron desde su checkpoint")
            if record["outputs"] != output_fingerprint(root, frame):
                raise RuntimeError(f"Los OUTPUTS de {frame} cambiaron desde su checkpoint")
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Checkpoint dañado para {frame}: {exc}") from exc


def mark_completed(case, root, frame, margin, feather, histogram):
    flow = root / "flow"
    corr = root / "corrected_points"
    geo = root / "geo_corrected"
    if not validate_flow(flow, frame, margin, feather, histogram):
        raise RuntimeError(f"No se ha validado el flujo de {frame}; NO se limpia")
    if not validate_corrected(corr, frame):
        raise RuntimeError(f"No se han validado los GCPs de {frame}; NO se limpia")
    if not validate_geotiff(geo, frame):
        raise RuntimeError(f"No se ha validado la TPS de {frame}; NO se limpia")
    stats = json.loads((corr / f"{frame}_correction_stats.json").read_text(encoding="utf-8"))
    if stats.get("n_points", 0) < 6 or stats.get("applied_count", -1) < 0:
        raise RuntimeError(f"Estadisticas invalidas para {frame}; NO se limpia")
    record = {
        "version": CHECKPOINT_VERSION,
        "frame": frame,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "algorithm": ALGORITHM,
        "inputs": source_fingerprint(case, frame),
        "outputs": output_fingerprint(root, frame),
    }
    atomic_json(checkpoint_path(root, frame), record)
    if not check_completed(case, root, frame):
        raise RuntimeError(f"Checkpoint no valido para {frame}; NO se limpia")


def prune_intermediate(root: Path, frame: str, keep: set[int], dry_run=False):
    """Only TWO precisely named arrays under staged flow/, never original flow/."""
    if frame_number(frame) in keep:
        return 0
    flowdir = root / "flow"
    deleted = 0
    for suffix in ("_flow.npy", "_flow_valid.npy"):
        target = flowdir / (frame + suffix)
        if target.is_symlink():
            raise RuntimeError(f"Rechazo borrar un enlace simbolico: {target}")
        if target.is_file():
            if target.resolve().parent != flowdir.resolve():
                raise RuntimeError(f"Ruta inesperada: {target}")
            deleted += target.stat().st_size
            if not dry_run:
                target.unlink()
    return deleted


def batches_of_pending(frames, complete, batch_size):
    """Never include a completed/pruned frame inside a runner's numeric range."""
    result, current = [], []
    for frame in frames:
        if frame in complete:
            if current:
                result.append(current)
                current = []
            continue
        current.append(frame)
        if len(current) == batch_size:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def rebuild_summary(root: Path, frames: list[str], completed: set[str]):
    """The upstream runner overwrites its summary per block; rebuild globally."""
    rows = []
    for frame in frames:
        ready = frame in completed
        stats_path = root / "corrected_points" / f"{frame}_correction_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if ready else {}
        med = (stats.get("correction_mag_percentiles") or {}).get("p50")
        rows.append({
            "frame": frame,
            # Historical column name kept for compatibility; means successfully
            # computed and checkpointed, not that the dense array is retained.
            "flow_ready": int(ready),
            "flow_file_retained": int((root / "flow" / f"{frame}_flow.npy").is_file()),
            "corrected_ready": int(ready),
            "second_tps_ready": int(ready),
            "n_gcps": stats.get("n_points"),
            "n_flow_valid": stats.get("valid_flow_count"),
            "n_flow_applied": stats.get("applied_count"),
            "median_correction_all_gcps_px": med,
            "max_correction_px": (stats.get("correction_mag_percentiles") or {}).get("max"),
        })
    root.mkdir(parents=True, exist_ok=True)
    output = root / "resumen_por_frame.csv"
    temp = output.with_name(output.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, output)
    medians = np.array([r["median_correction_all_gcps_px"] for r in rows
                        if r["median_correction_all_gcps_px"] is not None], dtype=float)
    summary = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frame_candidates": len(frames),
        "frames_completed": len(completed),
        "flow_computed_and_checkpointed": len(completed),
        "flow_arrays_retained": sum(r["flow_file_retained"] for r in rows),
        "corrected_point_files": len(completed),
        "second_tps_files": len(completed),
        "median_of_per_frame_medians_px": float(np.median(medians)) if len(medians) else None,
        "iqr_of_per_frame_medians_px": [float(x) for x in np.percentile(medians, [25, 75])] if len(medians) else None,
        "note": "Flow arrays may have been pruned after verified TPS. Flow magnitude is not independent geolocation accuracy.",
    }
    atomic_json(root / "resumen_global.json", summary)
    return summary


def config_signature(args, case, root, frames):
    return {
        "version": CHECKPOINT_VERSION,
        "algorithm": ALGORITHM,
        "case": str(case),
        "output_root": str(root),
        "code_sha256": source_digest(),
        "margin": args.margin,
        "feather": args.feather,
        "histogram": args.histogram,
        "frames_sha256": hashlib.sha256("\n".join(frames).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--start", type=int, default=327041)
    parser.add_argument("--end", type=int, default=328344)
    parser.add_argument("--expect", type=int, default=1252)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--margin", type=int, default=96)
    parser.add_argument("--feather", type=int, default=32)
    parser.add_argument("--histogram", choices=("masked", "legacy"), default="masked")
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--keep-flow", default=DEFAULT_KEEP,
                        help="IDs separados por comas cuyos arrays se conservan (por defecto 327470)")
    parser.add_argument("--prune-flow", action="store_true",
                        help="Autoriza borrar SOLO los .npy NUEVOS tras checkpoint valido")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=0,
                        help="0=todos; 1=solo primer bloque pendiente para prueba")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if args.start > args.end or args.expect < 0 or args.block_size < 1 or args.jobs < 1 or args.min_free_gib < 5:
        parser.error("Rango/expect/bloque/jobs/min-free no validos")
    if not (1 <= args.feather <= args.margin):
        parser.error("Se requiere 1 <= feather <= margin")
    case = args.case.expanduser().resolve()
    root = (args.output_root or case / "paper" / "reprocessing_v3").expanduser().resolve()
    if not case.is_dir() or case not in root.parents or root == case:
        parser.error("CASE u output-root no validos")
    if case / "paper" not in root.parents:
        parser.error("--output-root debe estar dentro de CASE/paper, nunca en carpetas originales")
    if (root / "flow").is_symlink() or (root / "corrected_points").is_symlink():
        parser.error("No se permiten enlaces simbolicos en carpetas staged")
    keep = parse_keep(args.keep_flow)
    frames = find_frames(case, args.start, args.end)
    if args.expect and len(frames) != args.expect:
        raise SystemExit(f"ABORTADO: hay {len(frames)} frames, esperados {args.expect}")
    if not frames:
        raise SystemExit("No hay frames")
    invalid = {fr: required_inputs(case, fr) for fr in frames}
    invalid = {fr: missing for fr, missing in invalid.items() if missing}
    if invalid:
        for fr, missing in list(invalid.items())[:10]:
            print("FALTAN INPUTS:", fr, missing)
        raise SystemExit(f"ABORTADO: faltan inputs para {len(invalid)} frames")

    config = config_signature(args, case, root, frames)
    config_path = root / "_batch_processing_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise SystemExit("ABORTADO: version/inputs/rango/parametros cambiaron. Usa otro --output-root.")
    elif not args.plan:
        atomic_json(config_path, config)

    complete = set()
    for fr in frames:
        if check_completed(case, root, fr):
            complete.add(fr)
    batches = batches_of_pending(frames, complete, args.block_size)
    free = shutil.disk_usage(root if root.exists() else case).free
    print(f"Frames: {len(frames)} | completados con checkpoint: {len(complete)} | pendientes: {len(frames)-len(complete)}")
    print(f"Bloques pendientes: {len(batches)} | tamano maximo: {args.block_size} | libre: {free/GB:.1f} GiB")
    print(f"Carpeta staged: {root}")
    print(f"Flujo a conservar: {sorted(keep)} | limpieza de arrays nuevos: {'SI' if args.prune_flow else 'NO'}")
    if not args.prune_flow and batches:
        checks_all = {
            "flow": lambda f: validate_flow(root / "flow", f, args.margin, args.feather, args.histogram),
            "georef": lambda f: validate_geotiff(root / "geo_corrected", f),
        }
        total_est, _, _ = estimate_bytes(case, [fr for block in batches for fr in block], root, checks_all)
        print(f"Sin limpieza, almacenamiento pendiente estimado: {total_est/GB:.1f} GiB")
    else:
        total_est = 0
    if args.plan:
        for i, block in enumerate(batches[:5], 1):
            print(f"  bloque {i}: {block[0]} .. {block[-1]} ({len(block)} frames)")
        if len(batches) > 5:
            print(f"  ... y {len(batches)-5} bloques mas")
        print("PLAN: no se ha modificado ningun producto")
        return

    if not args.prune_flow and total_est + args.min_free_gib * GB > free:
        raise SystemExit("ABORTADO: el espacio estimado no alcanza para conservar todos los flujos. "
                         "Ejecuta con --prune-flow o usa otro disco.")

    # Recover from interruption after a successful checkpoint but before pruning.
    if args.prune_flow:
        recovered = sum(prune_intermediate(root, fr, keep) for fr in complete)
        if recovered:
            print(f"Limpieza recuperada de checkpoints anteriores: {recovered/GB:.2f} GiB")

    runner = Path(__file__).with_name("reprocesar_refinamiento_v3.py")
    if not runner.is_file():
        raise SystemExit(f"No se encuentra runner v3 junto al script: {runner}")
    processed = 0
    for idx, block in enumerate(batches, 1):
        if args.max_blocks and processed >= args.max_blocks:
            break
        lo, hi = frame_number(block[0]), frame_number(block[-1])
        checks = {
            "flow": lambda f: validate_flow(root / "flow", f, args.margin, args.feather, args.histogram),
            "georef": lambda f: validate_geotiff(root / "geo_corrected", f),
        }
        est, _, _ = estimate_bytes(case, block, root, checks)
        free = shutil.disk_usage(root).free
        if free < est + int(args.min_free_gib * GB):
            raise SystemExit(f"DISCO INSUFICIENTE antes del bloque {idx}: libre={free/GB:.1f} GiB, "
                             f"estimado={est/GB:.1f} GiB, reserva={args.min_free_gib:.1f} GiB. "
                             "No se ha borrado ningun archivo antiguo.")
        print(f"\n=== BLOQUE {idx}/{len(batches)}: {block[0]}..{block[-1]} "
              f"({len(block)} frames), libre={free/GB:.1f} GiB ===", flush=True)
        cmd = [sys.executable, str(runner), str(case), "--start", str(lo), "--end", str(hi),
               "--expect", str(len(block)), "--jobs", str(args.jobs),
               "--margin", str(args.margin), "--feather", str(args.feather),
               "--histogram", args.histogram, "--output-root", str(root)]
        subprocess.run(cmd, check=True)
        # Stage 1: verify ALL products, write ALL checkpoints. No deletion before this.
        for fr in block:
            mark_completed(case, root, fr, args.margin, args.feather, args.histogram)
        # Stage 2: optionally delete ONLY new flow arrays after verified checkpoints.
        if args.prune_flow:
            reclaimed = sum(prune_intermediate(root, fr, keep) for fr in block)
            print(f"Limpieza segura del bloque: {reclaimed/GB:.2f} GiB recuperados", flush=True)
        complete.update(block)
        summary = rebuild_summary(root, frames, complete)
        print(f"Progreso: {summary['frames_completed']}/{len(frames)} TPS finales", flush=True)
        processed += 1

    # If no new blocks, still restore global summary after a previous interrupted run.
    summary = rebuild_summary(root, frames, complete)
    print("\nRESUMEN:", json.dumps(summary, indent=2, ensure_ascii=False))
    if len(complete) == len(frames):
        print("COMPLETADO. Puedes generar la figura temporal con figura_estadisticas_flujo_v3.py")
    else:
        print("PARCIAL. Repite exactamente el mismo comando para continuar")
    print("Las carpetas originales flow/, corrected_points/ y geo_corrected/ no se han tocado")


if __name__ == "__main__":
    main()
