#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura 2x2 para el paper: color Middlebury (flow_vis) y superposicion ISS/VIIRS.

Lee SOLO productos ya calculados del reprocesado v3; no recalcula flujo ni TPS.
Las regiones sin soporte segun *_flow_valid.npy se representan en gris claro.

Uso:
  python3 figura_viirs_flowvis_v3.py CASE_DIR --frame ISS067-E-327470
  python3 figura_viirs_flowvis_v3.py CASE_DIR --porto -8.95 -8.25 40.85 41.55

Salidas: CASE_DIR/paper/figures/reprocessing_v3/viirs_flow_porto_four_panel_flowvis_327470.{pdf,png}
         CASE_DIR/paper/figures/reprocessing_v3/viirs_flow_porto_four_panel_flowvis_327470_audit.json

Requisitos: numpy, pandas, matplotlib, rasterio, flow_vis (ya usado por optical_flow.py).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import flow_vis
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


COLORS = {
    'magenta': np.array([1.0, 0.12, 0.68], dtype=np.float32),
    'cyan': np.array([0.05, 0.88, 0.92], dtype=np.float32),
    'amber': np.array([1.0, 0.55, 0.08], dtype=np.float32),
}


def required(p):
    if not p.is_file():
        raise FileNotFoundError(f'Falta archivo del pipeline: {p}')
    return p


def check_extent(ext, name):
    l, r, b, t = ext
    if not (l < r and b < t):
        raise ValueError(f'{name}: usa lon_min lon_max lat_min lat_max')
    return tuple(map(float, ext))


def read_geo(path, extent, shape, bands=(1,)):
    """Remuestreo geoespacial a la MISMA rejilla EPSG:4326 para todas las capas."""
    l, r, b, t = extent
    h, w = shape
    dst_t = from_bounds(l, b, r, t, w, h)
    with rasterio.open(required(path)) as ds:
        if ds.crs is None:
            raise ValueError(f'{path.name}: GeoTIFF sin CRS')
        if max(bands) > ds.count:
            raise ValueError(f'{path.name}: se solicitan bandas {bands}, tiene {ds.count}')
        out = np.zeros((len(bands), h, w), dtype=np.float32)
        for i, band in enumerate(bands):
            reproject(
                source=rasterio.band(ds, band), destination=out[i],
                src_transform=ds.transform, src_crs=ds.crs,
                dst_transform=dst_t, dst_crs='EPSG:4326',
                src_nodata=ds.nodata, dst_nodata=0,
                resampling=Resampling.bilinear, num_threads=2,
            )
        mask = np.zeros((h, w), dtype=np.uint8)
        reproject(
            source=ds.read_masks(1), destination=mask,
            src_transform=ds.transform, src_crs=ds.crs,
            dst_transform=dst_t, dst_crs='EPSG:4326',
            src_nodata=0, dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return out, (mask > 0)


def to_gray(a):
    if a.shape[0] >= 3:
        return (0.299*a[0] + 0.587*a[1] + 0.114*a[2]).astype(np.float32)
    return a[0]


def stretch(a, valid, p1=2, p2=99.6):
    good = np.isfinite(a) & valid & (a > 0)
    if good.sum() < 25:
        raise ValueError('No hay suficientes pixeles validos para ajustar contraste')
    lo, hi = np.percentile(a[good], (p1, p2))
    return float(lo), float(max(hi, lo+1e-6))


def normalize(a, limits, gamma=0.72):
    lo, hi = limits
    return np.clip((a-lo)/(hi-lo), 0, 1)**gamma


def make_overlay(viirs_gray, viirs_valid, iss_gray, iss_valid,
                 viirs_limits, iss_limits, color, alpha):
    """VIIRS en gris + ISS en color semitransparente; fondo negro no tapa VIIRS."""
    v = normalize(viirs_gray, viirs_limits, gamma=0.76)
    i = normalize(iss_gray, iss_limits, gamma=0.76)
    v = np.where(viirs_valid, v, 0)
    base = np.repeat(v[..., None], 3, axis=2) * 0.94
    # El alpha es nulo en los pixeles oscuros: no tapamos el VIIRS con cielo negro.
    strength = np.clip((i - 0.075) / 0.52, 0, 1)**0.8
    aa = np.where(iss_valid, alpha*strength, 0)
    foreground = i[..., None] * color[None, None, :]
    return np.clip(base*(1-aa[..., None]) + foreground*aa[..., None], 0, 1)


def flow_geo_window(flow, ds, meta, extent, stride):
    """Muestrea flow[y-y0,x-x0] y devuelve vectores y extent geografico REAL."""
    x0, x1 = int(meta['x0']), int(meta['x1'])
    y0, y1 = int(meta['y0']), int(meta['y1'])
    if abs(ds.transform.b) > 1e-12 or abs(ds.transform.d) > 1e-12:
        raise ValueError('Se requiere GeoTIFF inicial north-up para mapear la ROI del flujo')
    inv = ~ds.transform
    l, r, b, t = extent
    pix = [inv*(x, y) for x, y in [(l,b),(l,t),(r,b),(r,t)]]
    xa = max(x0, int(np.floor(min(p[0] for p in pix))))
    xb = min(x1, int(np.ceil(max(p[0] for p in pix))))
    ya = max(y0, int(np.floor(min(p[1] for p in pix))))
    yb = min(y1, int(np.ceil(max(p[1] for p in pix))))
    if xb <= xa or yb <= ya:
        raise ValueError(f'No hay flujo optico en la extension {extent}')
    sub = np.asarray(flow[ya-y0:yb-y0:stride, xa-x0:xb-x0:stride, :],
                     dtype=np.float32)
    # Limites geograficos de la matriz submuestreada, respetando el stride.
    nx = sub.shape[1]; ny = sub.shape[0]
    lon0, lat0 = ds.transform*(xa, ya)
    lon1, lat1 = ds.transform*(min(xb, xa + nx*stride), min(yb, ya + ny*stride))
    geo_ext = (min(lon0, lon1), max(lon0, lon1),
               min(lat0, lat1), max(lat0, lat1))
    return sub, geo_ext


def vector_to_rgb(vec, vmax):
    """Color Middlebury con escala comun, compatible con flow_vis.flow_to_color.

    flow_to_color normaliza internamente por el maximo de cada matriz. Para
    mantener la misma escala en todos los paneles, se normalizan y saturan
    los vectores respecto de vmax y se agrega una fila de calibracion con
    un vector de modulo exactamente 1. Esa fila se elimina tras colorear.
    """
    if vmax <= 0:
        raise ValueError('vmax debe ser positivo')
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim != 3 or vec.shape[-1] != 2:
        raise ValueError('El flujo debe tener forma (alto, ancho, 2)')
    u = np.nan_to_num(vec[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(vec[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.hypot(u, v)
    scale = np.maximum(1.0, mag / vmax)
    normalized = np.stack((u / (vmax * scale), v / (vmax * scale)), axis=-1)
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)

    # flow_vis.flow_to_color existe en la version usada por optical_flow.py.
    # La fila de calibracion fuerza max_rad=1 para todos los recortes.
    calibration = np.zeros((1, normalized.shape[1], 2), dtype=np.float32)
    calibration[0, 0, 0] = 1.0
    padded = np.concatenate((normalized, calibration), axis=0)
    rgb = flow_vis.flow_to_color(padded, convert_to_bgr=False)[:-1]
    return np.asarray(rgb, dtype=np.float32) / 255.0, mag


def add_flow_legend(ax, vmax):
    """Leyenda generada por la MISMA funcion flow_vis que colorea los mapas.

    Eje horizontal: u = dx en pixeles (izquierda < 0, derecha > 0).
    Eje vertical: v = dy en pixeles (arriba < 0, abajo > 0).
    El eje vertical crece hacia abajo, igual que los pixeles de OpenCV.
    """
    inset = ax.inset_axes([0.075, 0.63, 0.27, 0.28])
    n = 151
    u = np.linspace(-vmax, vmax, n, dtype=np.float32)
    v = np.linspace(-vmax, vmax, n, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    rgb, _ = vector_to_rgb(np.stack((U, V), axis=-1), vmax)
    inset.imshow(rgb, origin='upper', extent=(-1, 1, 1, -1), interpolation='nearest')
    inset.set_xlim(-1, 1)
    inset.set_ylim(1, -1)
    inset.set_xticks([-1, 0, 1], labels=['left', '0', 'right'])
    inset.set_yticks([-1, 0, 1], labels=['up', '0', 'down'])
    inset.tick_params(labelsize=6, pad=1, colors='black', length=0)
    inset.set_xlabel('u = dx', fontsize=7, labelpad=1)
    inset.set_ylabel('v = dy', fontsize=7, labelpad=1)
    inset.set_facecolor('white')
    for spine in inset.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(0.6)
    return inset


def global_flow_max(flow, block_rows=256):
    """Maximo global del flow.npy, como la normalizacion de optical_flow.py.

    Lectura por bloques para evitar copiar el campo completo a memoria.
    """
    m = 0.0
    for start in range(0, flow.shape[0], block_rows):
        chunk = np.asarray(flow[start:start + block_rows], dtype=np.float32)
        finite = np.isfinite(chunk).all(axis=-1)
        if np.any(finite):
            a = chunk[..., 0][finite]
            b = chunk[..., 1][finite]
            m = max(m, float(np.max(np.hypot(a, b))))
    if not np.isfinite(m) or m <= 0:
        raise ValueError('El campo de flujo no contiene vectores finitos no nulos')
    return m


def describe_direction(u, v):
    horiz = 'left' if u < -0.05 else ('right' if u > 0.05 else 'approximately zero')
    vert = 'up' if v < -0.05 else ('down' if v > 0.05 else 'approximately zero')
    return horiz + ' / ' + vert


def flow_roi_extent(ds, meta):
    x0, x1 = int(meta['x0']), int(meta['x1'])
    y0, y1 = int(meta['y0']), int(meta['y1'])
    if not (0 <= x0 < x1 <= ds.width and 0 <= y0 < y1 <= ds.height):
        raise ValueError('Metadatos del flujo: ROI fuera de GeoTIFF inicial')
    corners = [ds.transform*(x,y) for x,y in
               [(x0,y0),(x1,y0),(x0,y1),(x1,y1)]]
    return (min(p[0] for p in corners), max(p[0] for p in corners),
            min(p[1] for p in corners), max(p[1] for p in corners))


def intersection(ext1, ext2):
    return (max(ext1[0],ext2[0]), min(ext1[1],ext2[1]),
            max(ext1[2],ext2[2]), min(ext1[3],ext2[3]))


def format_geo(ax, ext, xlabel=True, ylabel=True):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    if xlabel:
        ax.set_xlabel('Longitude (degrees)')
    if ylabel:
        ax.set_ylabel('Latitude (degrees)')
    ax.tick_params(labelsize=8)
    ax.set_aspect('auto')


def save(fig, base, dpi):
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        out = base.with_suffix('.'+suffix)
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
        print('Guardado:', out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_dir', type=Path)
    p.add_argument('--reprocess-dir', type=Path, default=None,
                   help='Directorio con flow/, corrected_points/ y geo_corrected/; default CASE/paper/reprocessing_v3')
    p.add_argument('--outdir', type=Path, default=None,
                   help='Carpeta de figuras; default CASE/paper/figures/reprocessing_v3')
    p.add_argument('--frame', default='ISS067-E-327470')
    p.add_argument('--porto', nargs=4, type=float, default=(-8.95,-8.25,40.85,41.55),
                   metavar=('LON_MIN','LON_MAX','LAT_MIN','LAT_MAX'),
                   help='Zoom de Oporto para paneles (b,c,d)')
    p.add_argument('--overview', nargs=4, type=float, default=None,
                   metavar=('LON_MIN','LON_MAX','LAT_MIN','LAT_MAX'),
                   help='Extension de panel (a). Por defecto: ROI del flujo intersectada con area de GCPs')
    p.add_argument('--iss-color', choices=tuple(COLORS), default='magenta')
    p.add_argument('--iss-alpha', type=float, default=0.70,
                   help='Opacidad maxima de luces ISS sobre VIIRS (0 a 1)')
    p.add_argument('--flow-alpha', type=float, default=0.95)
    p.add_argument('--flow-min-mag', type=float, default=0.08,
                   help='Umbral para describir la direccion mediana del flujo en los mensajes')
    p.add_argument('--flow-vmax', type=float, default=None,
                   help='Escala cromatica comun (px); default percentil 95 del flujo valido (saturacion a partir de ese valor)')
    p.add_argument('--flow-stride', type=int, default=3,
                   help='Muestreo del campo denso para visualizacion')
    p.add_argument('--size', type=int, default=1000,
                   help='Ancho de rejilla geografica de cada panel (px)')
    p.add_argument('--dpi', type=int, default=250)
    args = p.parse_args()

    if not (0 < args.iss_alpha <= 1 and 0 < args.flow_alpha <= 1):
        p.error('--iss-alpha y --flow-alpha deben estar en (0,1]')
    if args.flow_min_mag < 0:
        p.error('--flow-min-mag no puede ser negativo')
    if args.flow_stride < 1 or args.size < 300:
        p.error('--flow-stride >=1 y --size >=300')
    case = args.case_dir.expanduser().resolve()
    frame = args.frame
    processed = (args.reprocess_dir.expanduser().resolve() if args.reprocess_dir
                 else case/'paper'/'reprocessing_v3')
    first = required(case/'geo'/f'{frame}_rect.tiff')
    final = required(processed/'geo_corrected'/f'{frame}_rect.tiff')
    viirs = required(case/'viirs_cropped_aligned'/f'{frame}_viirs.tiff')
    flow_path = required(processed/'flow'/f'{frame}_flow.npy')
    meta = json.loads(required(processed/'flow'/f'{frame}_flow_meta.json').read_text())
    valid_mask_path = required(processed/'flow'/meta['flow_valid_mask_file'])
    flow_valid = np.load(valid_mask_path, mmap_mode='r').astype(bool)
    definition = meta.get('flow_definition', '')
    correction = meta.get('correction_to_apply_in_correct_points', '')
    if ('reference_iss' not in definition or 'distorted_viirs' not in definition
            or (correction and 'old_pixel + flow' not in correction)):
        raise ValueError('Convencion de flujo desconocida: revisar flow_meta.json ' +
                         f'(flow_definition={definition!r}, correction={correction!r})')
    corrected_path = required(processed/'corrected_points'/f'{frame}_corrected.points')
    filtered_path = required(case/'filtered_points'/f'{frame}.points')
    corrected = pd.read_csv(corrected_path, comment='#')
    filtered = pd.read_csv(filtered_path, comment='#')
    if len(corrected) != len(filtered):
        raise ValueError('Distinto numero de GCPs antes y despues de flujo')
    for col in ('correction_mag','flow_applied'):
        if col not in corrected.columns:
            raise ValueError(f'Falta columna {col} en {corrected_path.name}')
    mags = pd.to_numeric(corrected['correction_mag'], errors='raise').to_numpy(float)
    applied = pd.to_numeric(corrected['flow_applied'], errors='raise').to_numpy(float) > 0
    n_applied = int(applied.sum())
    median = float(np.median(mags))

    flow = np.load(flow_path, mmap_mode='r')
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f'Flujo debe tener forma (H,W,2): {flow.shape}')
    if flow_valid.shape != flow.shape[:2]:
        raise ValueError('flow_valid no coincide con las dimensiones del flujo')
    if flow.shape[:2] != (int(meta['y1'])-int(meta['y0']),
                          int(meta['x1'])-int(meta['x0'])):
        raise ValueError('Shape de flujo no coincide con flow_meta.json')
    porto = check_extent(args.porto, '--porto')
    with rasterio.open(first) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError(f'GeoTIFF inicial no EPSG:4326: {ds.crs}')
        if (ds.width,ds.height) != (int(meta['full_width']),int(meta['full_height'])):
            raise ValueError('flow_meta.json no corresponde a dimensiones de GeoTIFF inicial')
        roi = flow_roi_extent(ds,meta)
        if args.overview is not None:
            overview = check_extent(args.overview, '--overview')
        else:
            # El panel (a) muestra la zona de GCPs, evitando la zona no terrestre.
            xx = pd.to_numeric(filtered['mapX'],errors='raise').to_numpy(float)
            yy = pd.to_numeric(filtered['mapY'],errors='raise').to_numpy(float)
            l,r = np.quantile(xx,[0.01,0.99]); b,t = np.quantile(yy,[0.01,0.99])
            dx=max(0.3,(r-l)*0.10); dy=max(0.3,(t-b)*0.10)
            overview = intersection((l-dx,r+dx,b-dy,t+dy),roi)
            # Asegurar que el recuadro de Oporto aparezca en el mapa general.
            overview = (min(overview[0],porto[0]),max(overview[1],porto[1]),
                        min(overview[2],porto[2]),max(overview[3],porto[3]))
        for name, ext in (('overview',overview),('porto',porto)):
            if not (intersection(ext,roi)[0] < intersection(ext,roi)[1] and
                    intersection(ext,roi)[2] < intersection(ext,roi)[3]):
                raise ValueError(f'{name}: no intersecta ROI de flujo: {roi}')
        vec_broad, broad_flow_ext = flow_geo_window(flow,ds,meta,overview,args.flow_stride)
        valid_broad, _ = flow_geo_window(flow_valid[..., None], ds, meta,overview,args.flow_stride)
        valid_broad = valid_broad[..., 0].astype(bool)
        vec_porto, porto_flow_ext = flow_geo_window(flow,ds,meta,porto,
                                                   max(1,args.flow_stride//2))
        valid_porto, _ = flow_geo_window(flow_valid[..., None],ds,meta,porto,
                                       max(1,args.flow_stride//2))
        valid_porto = valid_porto[..., 0].astype(bool)
    with rasterio.open(final) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError(f'GeoTIFF final no EPSG:4326: {ds.crs}')
    with rasterio.open(viirs) as ds:
        if ds.crs is None:
            raise ValueError('VIIRS sin CRS')

    # Escala cromatica de flujo UNICA en los dos paneles superiores.
    mag_broad = np.hypot(vec_broad[..., 0], vec_broad[..., 1])
    mag_porto = np.hypot(vec_porto[..., 0], vec_porto[..., 1])
    valid_mag = mag_broad[np.isfinite(mag_broad) & valid_broad]
    if valid_mag.size == 0:
        raise ValueError('Flujo sin valores finitos en mapa general')
    if args.flow_vmax is not None:
        vmax = float(args.flow_vmax)
        scale_mode = 'fixed_user'
    else:
        sample_flow = np.asarray(flow[::8, ::8], dtype=np.float32)
        sample_valid = np.asarray(flow_valid[::8, ::8], dtype=bool)
        sample_mags = np.hypot(sample_flow[..., 0], sample_flow[..., 1])
        sample = sample_mags[sample_valid & np.isfinite(sample_mags)]
        if sample.size < 16:
            raise ValueError('No hay suficientes vectores validos para definir escala del flujo')
        vmax = float(np.percentile(sample, 95))
        scale_mode = 'valid_flow_p95'
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError('--flow-vmax debe ser un numero positivo')
    width = args.size
    broad_shape = (max(500, int(width*(overview[3]-overview[2])/(overview[1]-overview[0]))), width)
    porto_shape = (max(500, int(width*(porto[3]-porto[2])/(porto[1]-porto[0]))), width)
    broad_shape = (min(broad_shape[0],1500),width)
    porto_shape = (min(porto_shape[0],1500),width)
    vb, mb = read_geo(viirs,overview,broad_shape,bands=(1,))
    vp, mp = read_geo(viirs,porto,porto_shape,bands=(1,))
    ib, imb = read_geo(first,porto,porto_shape,bands=(1,2,3))
    fb, fmb = read_geo(final,porto,porto_shape,bands=(1,2,3))
    viirs_broad = to_gray(vb)
    viirs_porto = to_gray(vp)
    iss_before = to_gray(ib)
    iss_after = to_gray(fb)
    if (mb.sum()<50 or mp.sum()<50 or imb.sum()<50 or fmb.sum()<50):
        raise ValueError('Cobertura insuficiente en Oporto: revisa --porto y los GeoTIFFs')
    viirs_limits = stretch(viirs_porto,mp)
    both_iss = np.concatenate((iss_before[::3,::3],iss_after[::3,::3]))
    both_mask = np.concatenate((imb[::3,::3],fmb[::3,::3]))
    iss_limits = stretch(both_iss,both_mask)
    broad_limits = stretch(viirs_broad,mb)
    col = COLORS[args.iss_color]
    overlay_before = make_overlay(viirs_porto,mp,iss_before,imb,
                                  viirs_limits,iss_limits,col,args.iss_alpha)
    overlay_after = make_overlay(viirs_porto,mp,iss_after,fmb,
                                 viirs_limits,iss_limits,col,args.iss_alpha)
    if np.mean(imb & mp) < 0.015 or np.mean(fmb & mp) < 0.015:
        print('AVISO: interseccion de pixeles validos ISS/VIIRS en Oporto muy pequena.')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.titlesize':11,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axs = plt.subplots(2,2,figsize=(14.0,10.0),constrained_layout=True)
    axa, axb, axc, axd = axs.flat
    # Middlebury (flow_vis): tono = direccion, saturacion = magnitud.
    # NO confundir este campo denso con correcciones efectivamente aplicadas a GCPs.
    rgb_broad, mag_broad = vector_to_rgb(vec_broad, vmax)
    rgb_porto, mag_porto = vector_to_rgb(vec_porto, vmax)
    # Nunca confundir 0 de flujo con una region sin cobertura.
    rgb_broad[~valid_broad] = 0.91
    rgb_porto[~valid_porto] = 0.91
    # Fondo blanco fuera de ROI; soporte invalido en gris claro.
    # flow_vis codifica flujo cero como casi blanco: NO confundir con gris invalido.
    axa.set_facecolor('white')
    axb.set_facecolor('white')
    axa.imshow(rgb_broad,origin='upper',extent=broad_flow_ext,
               interpolation='nearest')
    axa.add_patch(Rectangle((porto[0],porto[2]),porto[1]-porto[0],porto[3]-porto[2],
                            fill=False,edgecolor='#ff5e5e',linewidth=2.0))
    axa.text(porto[0],porto[3], ' Porto ',fontsize=9,color='white',
             va='bottom',bbox={'facecolor':'#a02525','alpha':0.85,'pad':2,'edgecolor':'none'})
    axa.set_title('(a) Dense optical flow: direction and magnitude (regional)')
    format_geo(axa,overview)
    add_flow_legend(axa, vmax)
    # (b) Campo vectorial ampliado, misma codificacion de magnitud.
    axb.imshow(rgb_porto,origin='upper',extent=porto_flow_ext,
               interpolation='nearest')
    axb.set_title('(b) Dense optical flow: direction and magnitude (Porto)')
    format_geo(axb,porto)
    # (c,d) VIIRS abajo, ISS semitransparente arriba, misma region y contraste.
    for ax, image, title in (
        (axc,overlay_before,'(c) Initial TPS georeferencing + VIIRS (Porto)'),
        (axd,overlay_after,'(d) Refined TPS georeferencing + VIIRS (Porto)')):
        ax.imshow(image,origin='upper',extent=porto,interpolation='nearest')
        ax.set_title(title)
        format_geo(ax,porto)
    color_hex = {'magenta':'#ff1fad','cyan':'#0ce0eb','amber':'#ff8c14'}[args.iss_color]
    handles = [Patch(facecolor='0.70',edgecolor='0.45',label='VIIRS (grayscale)'),
               Patch(facecolor=color_hex,alpha=args.iss_alpha,
                     label=f'ISS (semi-transparent {args.iss_color})'),
               Patch(facecolor='0.91', edgecolor='0.6', label='Flow masked / unavailable')]
    fig.legend(handles=handles,loc='lower center',ncol=2,
               bbox_to_anchor=(0.5,-0.018),frameon=False,fontsize=10)
    outdir = (args.outdir.expanduser().resolve() if args.outdir
              else case/'paper'/'figures'/'reprocessing_v3')
    sid=frame.split('-')[-1]
    base=outdir/f'viirs_flow_porto_four_panel_flowvis_{sid}'
    save(fig,base,args.dpi)

    report={
        'frame':frame,
        'overview_lonmin_lonmax_latmin_latmax':list(overview),
        'porto_lonmin_lonmax_latmin_latmax':list(porto),
        'flow_roi_extent':list(roi),
        'flow_shape':list(flow.shape),
        'flow_definition':meta.get('flow_definition'),
        'flow_magnitude_scale_vmax_px':vmax,
        'flow_scale_mode':scale_mode,
        'flow_valid_fraction_roi':float(flow_valid.mean()),
        'flow_valid_fraction_overview':float(valid_broad.mean()),
        'flow_valid_fraction_porto':float(valid_porto.mean()),
        'flow_valid_mask':str(valid_mask_path),
        'flow_figure_invalid_areas':'light gray / no vector; distinguish from near-white zero flow',
        'flow_visualization':f'Middlebury flow_vis, invalid pixels light gray, common vector color scale ({scale_mode}, values above vmax saturated); u positive right, v positive down',
        'flow_min_mag_px':args.flow_min_mag,
        'overview_flow_p50_px':float(np.median(valid_mag)),
        'porto_flow_p50_px':float(np.median(mag_porto[np.isfinite(mag_porto) & valid_porto])) if valid_porto.any() else None,
        'gcp_count':int(len(corrected)),
        'corrected_gcp_count':n_applied,
        'all_gcp_median_applied_correction_px':median,
        'viirs_stretch_porto':list(viirs_limits),
        'iss_stretch_shared_before_after':list(iss_limits),
        'iss_color':args.iss_color,'iss_alpha':args.iss_alpha,
        'porto_iss_viirs_valid_overlap_before':float(np.mean(imb & mp)),
        'porto_iss_viirs_valid_overlap_after':float(np.mean(fmb & mp)),
        'note':'Dense flow and applied GCP corrections are different quantities. Neither is independent absolute geolocation error.',
    }
    audit=base.with_name(base.name+'_audit.json')
    audit.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('Auditoria:',audit)
    print(f'GCPs: {len(corrected)}, corregidos: {n_applied}, mediana incluyendo ceros: {median:.3f} px')
    print(f'Flujo Middlebury flow_vis: direccion por color; magnitud 0-{vmax:.2f} px por saturacion, misma escala en (a) y (b)')
    print('Oporto:',porto)
    active = np.isfinite(vec_porto).all(axis=-1) & valid_porto & (mag_porto > args.flow_min_mag)
    if np.any(active):
        u_med = float(np.median(vec_porto[...,0][active]))
        v_med = float(np.median(vec_porto[...,1][active]))
        print(f'Oporto - flujo denso mediano: u={u_med:+.3f} px, v={v_med:+.3f} px ({describe_direction(u_med, v_med)})')
        report['porto_dense_median_u_px'] = u_med
        report['porto_dense_median_v_px'] = v_med
    if all(c in corrected.columns for c in ('dX','dY')):
        inside = ((pd.to_numeric(filtered['mapX'], errors='coerce') >= porto[0]) &
                  (pd.to_numeric(filtered['mapX'], errors='coerce') <= porto[1]) &
                  (pd.to_numeric(filtered['mapY'], errors='coerce') >= porto[2]) &
                  (pd.to_numeric(filtered['mapY'], errors='coerce') <= porto[3]) & applied)
        if inside.any():
            u_gcp = float(np.median(pd.to_numeric(corrected.loc[inside,'dX'])))
            v_gcp = float(np.median(pd.to_numeric(corrected.loc[inside,'dY'])))
            print(f'Oporto - GCPs corregidos n={inside.sum()}: dX={u_gcp:+.3f}, dY={v_gcp:+.3f} px ({describe_direction(u_gcp, v_gcp)})')
        else:
            print('Oporto - no hay GCPs corregidos dentro del recorte seleccionado')

    audit.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'GCPs refinados con soporte: {n_applied}/{len(corrected)}. '
          'La mediana incluye GCPs sin correccion y no es precision absoluta.')


if __name__=='__main__':
    main()
