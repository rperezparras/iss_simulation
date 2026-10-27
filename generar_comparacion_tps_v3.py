#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera comparacion VIIRS + TPS antigua/nueva reprocessing_v3 para ISS067-E-327470.

No modifica ningun producto del pipeline. Misma extension, rejilla, contraste
para la TPS antigua y nueva en cada figura. La escala radiometrica se fija
usando la vista de Oporto y se reutiliza en todas las figuras.

Ejemplo:
    python3 generar_comparacion_tps_327470.py \
        ~/iss_simulation/paper_caso_estudio/ISS067-E-327041-328344

Subir a ChatGPT el ZIP creado en paper/validation/comparacion_tps_327470/.
Dependencias: numpy, pandas, rasterio, matplotlib (ya usadas en el proyecto).
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

FRAME = 'ISS067-E-327470'
PORTO = (-8.95, -8.25, 40.85, 41.55)
PORTO_CENTRO = (-8.78, -8.47, 41.04, 41.28)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'No existe: {path}')
    return path


def read_on_grid(path: Path, bbox, width: int, rgb: bool = False):
    """Lee/reproyecta un TIFF a rejilla geografica EPSG:4326 sin cargarlo entero."""
    lon0, lon1, lat0, lat1 = map(float, bbox)
    if not (lon0 < lon1 and lat0 < lat1):
        raise ValueError(f'Extension incorrecta: {bbox}')
    # Matriz de alto proporcional, acotada para limitar memoria.
    height = max(550, min(1100, round(width * (lat1 - lat0) / (lon1 - lon0))))
    transform = from_bounds(lon0, lat0, lon1, lat1, width, height)
    with rasterio.open(require_file(path)) as src:
        if src.crs is None:
            raise ValueError(f'{path}: TIFF sin sistema de coordenadas')
        indexes = [1, 2, 3] if rgb and src.count >= 3 else [1]
        out = np.zeros((len(indexes), height, width), dtype=np.float32)
        for j, idx in enumerate(indexes):
            reproject(
                source=rasterio.band(src, idx),
                destination=out[j],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs='EPSG:4326',
                src_nodata=src.nodata,
                dst_nodata=0,
                resampling=Resampling.bilinear,
                num_threads=2,
            )
    if len(indexes) == 3:
        gray = 0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2]
    else:
        gray = out[0]
    gray[~np.isfinite(gray)] = 0.0
    return gray


def limits_from_images(arrays, p_low=2., p_high=99.5):
    positive = [a[np.isfinite(a) & (a > 0)] for a in arrays]
    positive = [a[::max(1, len(a)//150000)] for a in positive if len(a)]
    if not positive:
        raise ValueError('Todas las capas estan vacias en Oporto: revisa las rutas y la extension.')
    values = np.concatenate(positive)
    lo, hi = np.percentile(values, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return (float(lo), float(hi))


def normalize(a, bounds, gamma=0.75):
    lo, hi = bounds
    return np.clip((a - lo) / (hi - lo), 0., 1.) ** gamma


def overlay_viirs_iss(viirs, iss, alpha=0.75):
    """VIIRS gris y fuentes ISS magenta. Oscuridad no tapa VIIRS."""
    bg = np.repeat((viirs * 0.88)[..., None], 3, axis=-1)
    strength = np.clip((iss - 0.065) / 0.66, 0, 1) ** 0.8
    aa = np.minimum(1., alpha * strength)
    tint = iss[..., None] * np.array((1.0, 0.10, 0.72), dtype=np.float32)
    return np.clip(bg * (1. - aa[..., None]) + tint * aa[..., None], 0, 1)


def overlay_old_new(old, new):
    """Solo ISS: antigua cian, nueva magenta, coincidentes aproximadamente claras."""
    cyan = np.array((0.08, 0.90, 0.90), dtype=np.float32)
    magenta = np.array((0.92, 0.14, 0.75), dtype=np.float32)
    return np.clip(old[..., None] * cyan + new[..., None] * magenta, 0, 1)


def load_gcp_data(case):
    old_path = require_file(case/'corrected_points'/f'{FRAME}_corrected.points')
    new_path = require_file(case/'paper'/'reprocessing_v3'/'corrected_points'/f'{FRAME}_corrected.points')
    filtered_path = require_file(case/'filtered_points'/f'{FRAME}.points')
    old = pd.read_csv(old_path, comment='#')
    new = pd.read_csv(new_path, comment='#')
    original = pd.read_csv(filtered_path, comment='#')
    required = ['mapX','mapY','sourceX','sourceY']
    for df, label in [(old,'old'), (new,'new'), (original,'filtered')]:
        absent = [c for c in required if c not in df]
        if absent:
            raise ValueError(f'Columnas faltantes en {label}: {absent}')
        df['key'] = df['sourceX'].round(3).astype(str) + '/' + df['sourceY'].round(3).astype(str)
        if df['key'].duplicated().any():
            raise ValueError(f'Hay GCPs con sourceX/sourceY duplicados en {label}.')
    comp = old.merge(new, on='key', suffixes=('_old','_new'), validate='one_to_one')
    comp = comp.merge(original[['key','mapX','mapY']], on='key',validate='one_to_one')
    if len(comp) != len(old) or len(comp) != len(new) or len(comp) != len(original):
        raise ValueError('Los tres ficheros de puntos no tienen los mismos GCPs.')
    for suffix in ('old','new'):
        for c in ('dX','dY','correction_mag','flow_applied'):
            if f'{c}_{suffix}' not in comp:
                raise ValueError(f'Falta columna {c}_{suffix}')
    comp['change_in_flow_vector_px'] = np.hypot(
        comp['dX_new']-comp['dX_old'],comp['dY_new']-comp['dY_old'])
    comp['flow_removed'] = (comp['flow_applied_old'] == 1) & (comp['flow_applied_new'] == 0)
    comp['geo_shift_old_m_approx'] = np.hypot(
        (comp['mapX_old'] - comp['mapX'])*np.cos(np.radians(comp['mapY'])),
        comp['mapY_old'] - comp['mapY'])*111320.
    comp['geo_shift_new_m_approx'] = np.hypot(
        (comp['mapX_new'] - comp['mapX'])*np.cos(np.radians(comp['mapY'])),
        comp['mapY_new'] - comp['mapY'])*111320.
    return comp


def render_figure(paths, extent, fig_path, title, size, limits, marker=None):
    """Seis paneles con MISMA rejilla, extension, contraste y tamano."""
    raw = {k: read_on_grid(path, extent, size, rgb=(k != 'viirs'))
           for k, path in paths.items()}
    viirs = normalize(raw['viirs'], limits['viirs'])
    old = normalize(raw['old'], limits['iss'])
    new = normalize(raw['new'], limits['iss'])
    if float(np.max(viirs)) < 0.02:
        print(f'AVISO {title}: VIIRS casi vacio en esta zona.')
    if float(np.max(old)) < 0.02 or float(np.max(new)) < 0.02:
        print(f'AVISO {title}: una TPS presenta pocos pixeles iluminados.')
    fig, axs = plt.subplots(2, 3, figsize=(17.8, 10.6), constrained_layout=True)
    fig.suptitle(f'{FRAME} | {title}', fontsize=16, fontweight='bold')
    panels = [
        ('(a) VIIRS | referencia', viirs, 'gray'),
        ('(b) ISS | TPS anterior', old, 'gray'),
        ('(c) ISS | TPS corregida v2', new, 'gray'),
        ('(d) VIIRS gris + TPS anterior magenta', overlay_viirs_iss(viirs, old), None),
        ('(e) VIIRS gris + TPS corregida magenta', overlay_viirs_iss(viirs, new), None),
        ('(f) TPS anterior cian + TPS nueva magenta', overlay_old_new(old, new), None),
    ]
    for ax, (label, img, cmap) in zip(axs.flat, panels):
        ax.imshow(img, extent=extent, origin='upper', cmap=cmap, vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
        ax.set_title(label,fontsize=11)
        ax.set_xlabel('Longitud (°)',fontsize=9)
        ax.set_ylabel('Latitud (°)',fontsize=9)
        ax.tick_params(labelsize=8)
        if marker is not None and extent[0] <= marker[0] <= extent[1] and extent[2] <= marker[1] <= extent[3]:
            ax.scatter([marker[0]], [marker[1]], marker='+', c='yellow',
                       s=125, linewidths=1.6, zorder=10)
            ax.add_patch(plt.Circle(marker, radius=(extent[1]-extent[0])*0.025,
                                    fill=False, ec='yellow', lw=1.3))
    fig.savefig(fig_path, dpi=145, facecolor='white')
    plt.close(fig)
    print('FIGURA:', fig_path)
    return {
        'extension_lon_min_lon_max_lat_min_lat_max': [float(x) for x in extent],
        'viirs_max_display':float(viirs.max()),
        'old_max_display':float(old.max()),
        'new_max_display':float(new.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_dir',type=Path,help='Carpeta ISS067-E-327041-328344')
    parser.add_argument('--size',type=int,default=880,help='Resolucion horizontal de cada panel (default 880)')
    parser.add_argument('--porto',nargs=4,type=float,default=PORTO,
                        metavar=('LON_MIN','LON_MAX','LAT_MIN','LAT_MAX'))
    parser.add_argument('--centro-porto',nargs=4,type=float,default=PORTO_CENTRO,
                        metavar=('LON_MIN','LON_MAX','LAT_MIN','LAT_MAX'))
    args=parser.parse_args()
    if args.size < 350 or args.size > 1800:
        parser.error('--size debe estar entre 350 y 1800')
    case=args.case_dir.expanduser().resolve()
    paths={
        'viirs': require_file(case/'viirs_cropped_aligned'/f'{FRAME}_viirs.tiff'),
        'old': require_file(case/'geo_corrected'/f'{FRAME}_rect.tiff'),
        'new': require_file(case/'paper'/'reprocessing_v3'/'geo_corrected'/f'{FRAME}_rect.tiff'),
    }
    print('FUENTES:')
    for kind,path in paths.items():
        with rasterio.open(path) as src:
            print(f'  {kind:5s}: {path} | {src.width}x{src.height} | {src.crs}')

    outdir=case/'paper'/'figures'/'reprocessing_v3'/f'comparacion_tps_{FRAME.split("-")[-1]}'
    outdir.mkdir(parents=True,exist_ok=True)

    # Calcular los limites radiometricos UNA SOLA VEZ sobre el zoom grande de Oporto.
    porto=tuple(args.porto)
    centro=tuple(args.centro_porto)
    porto_images={k:read_on_grid(p,porto,max(450,args.size//2),rgb=(k!='viirs'))
                  for k,p in paths.items()}
    limits={
        'viirs': limits_from_images([porto_images['viirs']],2,99.5),
        'iss': limits_from_images([porto_images['old'],porto_images['new']],2,99.6),
    }
    print('ESCALAS FIJAS:', limits)
    comp=load_gcp_data(case)
    maxrow=comp.sort_values('correction_mag_new',ascending=False).iloc[0]
    location=(float(maxrow['mapX']),float(maxrow['mapY']))
    correction=float(maxrow['correction_mag_new'])
    print(f'MAYOR CORRECCION v2: {correction:.3f} px | '
          f'lon={location[0]:.6f} lat={location[1]:.6f} | '
          f'sourceX={maxrow["sourceX_new"]}, sourceY={maxrow["sourceY_new"]}')
    # Vista centrada en el GCP de mayor correccion.
    roi_outlier=(location[0]-.12,location[0]+.12,location[1]-.10,location[1]+.10)
    all_views=[('01_oporto_general',porto,'Oporto | contexto regional',None),
               ('02_oporto_centro',centro,'Oporto | acercamiento',None),
               ('03_gcp_maxima_correccion',roi_outlier,
                f'GCP de maxima correccion ({correction:.2f} px) | cruz amarilla',location)]
    figures=[]
    audits={}
    for filename,extent,title,marker in all_views:
        output=outdir/f'{filename}.png'
        audits[filename]=render_figure(paths,extent,output,title,args.size,limits,marker)
        figures.append(output)

    csvpath=outdir/'comparacion_gcps.csv'
    selected=['sourceX_new','sourceY_new','mapX','mapY',
              'dX_old','dY_old','dX_new','dY_new',
              'correction_mag_old','correction_mag_new','flow_applied_old',
              'flow_applied_new','correction_reason_new','flow_removed',
              'change_in_flow_vector_px','geo_shift_old_m_approx',
              'geo_shift_new_m_approx']
    if 'flow_valid_new' in comp:
        selected.append('flow_valid_new')
    if 'correction_reason_new' not in comp:
        comp['correction_reason_new'] = 'not_available'
    comp[selected].sort_values('change_in_flow_vector_px',ascending=False).to_csv(csvpath,index=False)
    print('CSV:',csvpath)
    report={
        'frame':FRAME,
        'sources':{k:str(v) for k,v in paths.items()},
        'contrast_common_to_all_images':limits,
        'figures':audits,
        'n_gcps':int(len(comp)),
        'old_flow_applied':int((comp['flow_applied_old']==1).sum()),
        'new_flow_applied':int((comp['flow_applied_new']==1).sum()),
        'rejected_old_corrections':int(comp['flow_removed'].sum()),
        'old_median_correction_all_px':float(comp['correction_mag_old'].median()),
        'new_median_correction_all_px':float(comp['correction_mag_new'].median()),
        'outlier_new':{
            'sourceX':float(maxrow['sourceX_new']),
            'sourceY':float(maxrow['sourceY_new']),
            'original_lon':location[0], 'original_lat':location[1],
            'correction_new_px':correction,
            'correction_old_px':float(maxrow['correction_mag_old']),
            'correction_reason_new':str(maxrow.get('correction_reason_new', 'not_available')),
        },
        'important_note':(
          'Visual comparison only; cannot infer independent geolocation accuracy '
          'from photometric similarity or optical flow magnitude. Images without '
          'alpha can include black regions with unknown coverage.'
        ),
    }
    report_path=outdir/'resumen_comparacion.json'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    readme_path=outdir/'LEEME.txt'
    readme_path.write_text(
        'Comparacion ISS067-E-327470\n'
        '01_oporto_general.png: mismas coordenadas, VIIRS + TPS original y nueva\n'
        '02_oporto_centro.png: acercamiento al centro de Oporto\n'
        '03_gcp_maxima_correccion.png: entorno del GCP con mayor flujo aplicado (cruz)\n'
        'comparacion_gcps.csv: coordenadas y vectores anteriores/nuevos para cada GCP\n'
        'resumen_comparacion.json: rutas, contadores, extents y escalas radiometricas\n'
        'No se cambian los datos del pipeline. La comparacion es visual.\n',
        encoding='utf-8',
    )
    zip_path=outdir/f'comparacion_tps_{FRAME.split("-")[-1]}.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in [*figures,csvpath,report_path,readme_path]:
            archive.write(p,arcname=p.name)
    print('\n' + '='*70)
    print('COMPARACION COMPLETADA')
    print('CARPETA:',outdir)
    print('ZIP PARA SUBIR A CHATGPT:',zip_path)
    print('No se ha modificado ningun archivo de entrada.')
    print('='*70)


if __name__=='__main__':
    main()
