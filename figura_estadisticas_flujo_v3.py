#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura resumida para el timelapse: fraccion de GCPs corregidos y desplazamientos.

Calcula estadisticas DESCRIPTIVAS de flujo (NO accuracy).
Lee CASE/paper/reprocessing_v3/resumen_por_frame.csv; no recalcula ningun dato.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", type=Path)
    ap.add_argument("--reprocess-dir", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=250)
    args = ap.parse_args()
    case = args.case.expanduser().resolve()
    staged = args.reprocess_dir.expanduser().resolve() if args.reprocess_dir else case / "paper" / "reprocessing_v3"
    outdir = args.outdir.expanduser().resolve() if args.outdir else case / "paper" / "figures" / "reprocessing_v3"
    src = staged / "resumen_por_frame.csv"
    if not src.is_file():
        raise SystemExit(f"Falta {src}; ejecuta primero reprocesar_refinamiento_v3.py")
    df = pd.read_csv(src)
    need = ("frame", "n_gcps", "n_flow_applied", "median_correction_all_gcps_px", "second_tps_ready")
    if any(c not in df.columns for c in need):
        raise ValueError(f"Faltan columnas: {set(need) - set(df.columns)}")
    df["id"] = df["frame"].str.extract(r"E-(\d+)")[0].astype(int)
    df = df.sort_values("id")
    ready = df.loc[(df.second_tps_ready == 1) & (df.n_gcps > 0)].copy()
    if len(ready) < 3:
        raise ValueError("La figura temporal requiere al menos 3 frames con TPS final")
    xx = ready.id.to_numpy(int)
    fraction = 100 * ready.n_flow_applied.to_numpy(float) / ready.n_gcps.to_numpy(float)
    medians = ready.median_correction_all_gcps_px.to_numpy(float)
    mask = np.isfinite(fraction) & np.isfinite(medians)
    xx, fraction, medians = xx[mask], fraction[mask], medians[mask]
    if len(xx) < 3:
        raise ValueError("No hay suficientes valores finitos")
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True, constrained_layout=True)
    axs[0].scatter(xx, fraction, c="#2c77a6", s=6, alpha=0.75, rasterized=True)
    axs[0].axhline(float(np.median(fraction)), ls="--", lw=1, color="0.3")
    axs[0].set_ylabel("GCPs with applied flow (%)")
    axs[0].set_title("(a) Spatial-support filtering across the ISS timelapse")
    axs[1].scatter(xx, medians, c="#9a4485", s=6, alpha=0.75, rasterized=True)
    axs[1].axhline(float(np.median(medians)), ls="--", lw=1, color="0.3")
    axs[1].set_ylabel("Median displacement per frame (px)")
    axs[1].set_xlabel("ISS frame identifier")
    axs[1].set_title("(b) Median GCP correction magnitude (including uncorrected GCPs)")
    for ax in axs:
        ax.grid(alpha=0.15, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / "timelapse_refinement_statistics_v3"
    for ext in ("pdf", "png"):
        fig.savefig(base.with_suffix("." + ext), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    report = {
        "n_frames_with_final_tps": int(len(xx)),
        "fraction_gcps_applied_median_pct": float(np.median(fraction)),
        "median_per_frame_correction_px": float(np.median(medians)),
        "iqr_per_frame_correction_px": [float(v) for v in np.percentile(medians, [25, 75])],
        "source": str(src),
        "note": "These are optical-flow correction magnitudes, not independent georeferencing accuracy.",
    }
    base.with_name(base.name + "_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Figuras:", base.with_suffix(".png"), "y", base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
