#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision.transforms as tfm
from skimage.metrics import structural_similarity as ssim
from skimage.transform import PolynomialTransform, SimilarityTransform

# -------------------------------------------------------------------
# IMPORTS DEL PROYECTO
# -------------------------------------------------------------------
from .iss_simulation import (
    reset_scene,
    list_tle_files,
    read_tle_from_files,
    find_closest_tle,
    get_iss_position_and_velocity,
    creaimagen,
)


# -------------------------------------------------------------------
# HELPERS DE MATCHING (adaptados de match_timelapse.py)
# -------------------------------------------------------------------

def image_loader(path: str, resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if resize is not None:
        img = img.resize((int(resize[1]), int(resize[0])), resample=Image.BILINEAR)
    return tfm.ToTensor()(img)


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        return x.numpy()
    return np.asarray(x)


def squeeze_points(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def result_get(result: Any, key: str):
    """Acceso robusto a dicts u objetos con atributos."""
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get(key, None)
    if hasattr(result, key):
        return getattr(result, key)
    return None


def result_has(result: Any, key: str) -> bool:
    """Comprueba si una salida de matcher contiene una clave/atributo."""
    if result is None:
        return False
    if isinstance(result, dict):
        return key in result
    return hasattr(result, key)


def sanitize_matched_points(
    mkpts0: Optional[np.ndarray],
    mkpts1: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convierte a float32 y elimina pares no finitos."""
    mkpts0 = squeeze_points(mkpts0)
    mkpts1 = squeeze_points(mkpts1)

    if mkpts0 is None or mkpts1 is None:
        return None, None

    n = min(len(mkpts0), len(mkpts1))
    if n <= 0:
        return None, None

    mkpts0 = np.asarray(mkpts0[:n], dtype=np.float32)
    mkpts1 = np.asarray(mkpts1[:n], dtype=np.float32)

    valid = np.isfinite(mkpts0).all(axis=1) & np.isfinite(mkpts1).all(axis=1)
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]

    if len(mkpts0) == 0:
        return None, None

    return mkpts0, mkpts1


def extract_mkpts_from_result(result: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extrae correspondencias de forma compatible con vismatch actual y con
    image-matching-models antiguo.

    Convención:
      imagen 0 = simulada
      imagen 1 = real
    """
    if result is None:
        return None, None

    # Prioridad alta: inliers ya filtrados si el backend los proporciona.
    direct_pairs = [
        ("inlier_kpts0", "inlier_kpts1"),
        ("inlier_mkpts0", "inlier_mkpts1"),
        ("mkpts0", "mkpts1"),
        ("m_kpts0", "m_kpts1"),
        ("matched_kpts0", "matched_kpts1"),
        ("matched_keypoints0", "matched_keypoints1"),
    ]

    for k0, k1 in direct_pairs:
        if result_has(result, k0) and result_has(result, k1):
            a = squeeze_points(to_numpy(result_get(result, k0)))
            b = squeeze_points(to_numpy(result_get(result, k1)))
            a, b = sanitize_matched_points(a, b)
            if a is not None and b is not None:
                return a, b

    # Formatos con keypoints + índices de matches.
    keypoint_pairs = [
        ("kpts0", "kpts1"),
        ("keypoints0", "keypoints1"),
        ("all_kpts0", "all_kpts1"),
    ]

    for k0, k1 in keypoint_pairs:
        if not (result_has(result, k0) and result_has(result, k1)):
            continue

        pts0 = squeeze_points(to_numpy(result_get(result, k0)))
        pts1 = squeeze_points(to_numpy(result_get(result, k1)))
        if pts0 is None or pts1 is None:
            continue

        # matches Nx2: columna 0 indexa pts0, columna 1 indexa pts1.
        if result_has(result, "matches"):
            m = to_numpy(result_get(result, "matches"))
            if m is not None:
                m = np.asarray(m)
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m[0]
                m = np.squeeze(m).astype(np.int64)
                if m.ndim == 2 and m.shape[1] == 2:
                    valid = (
                        (m[:, 0] >= 0) & (m[:, 0] < len(pts0)) &
                        (m[:, 1] >= 0) & (m[:, 1] < len(pts1))
                    )
                    if np.any(valid):
                        a, b = sanitize_matched_points(pts0[m[valid, 0]], pts1[m[valid, 1]])
                        if a is not None and b is not None:
                            return a, b

        # matches0: array (N0,) con índice en pts1 o -1.
        if result_has(result, "matches0"):
            m0 = to_numpy(result_get(result, "matches0"))
            if m0 is not None:
                m0 = np.asarray(m0)
                if m0.ndim == 2 and m0.shape[0] == 1:
                    m0 = m0[0]
                m0 = np.squeeze(m0).astype(np.int64)
                valid = (m0 >= 0) & (m0 < len(pts1))
                if np.any(valid):
                    a, b = sanitize_matched_points(pts0[valid], pts1[m0[valid]])
                    if a is not None and b is not None:
                        return a, b

        # matches1: array (N1,) con índice en pts0 o -1.
        if result_has(result, "matches1"):
            m1 = to_numpy(result_get(result, "matches1"))
            if m1 is not None:
                m1 = np.asarray(m1)
                if m1.ndim == 2 and m1.shape[0] == 1:
                    m1 = m1[0]
                m1 = np.squeeze(m1).astype(np.int64)
                valid = (m1 >= 0) & (m1 < len(pts0))
                if np.any(valid):
                    a, b = sanitize_matched_points(pts0[m1[valid]], pts1[valid])
                    if a is not None and b is not None:
                        return a, b

    return None, None


def orb_fallback_matches(img0_bgr: np.ndarray, img1_bgr: np.ndarray, max_matches: int = 2000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    g0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=7)
    kp0, des0 = orb.detectAndCompute(g0, None)
    kp1, des1 = orb.detectAndCompute(g1, None)

    if des0 is None or des1 is None or len(kp0) < 10 or len(kp1) < 10:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des0, des1, k=2)

    good = []
    for m in knn:
        if len(m) != 2:
            continue
        a, b = m
        if a.distance < 0.75 * b.distance:
            good.append(a)

    if len(good) < 20:
        return None, None

    good = sorted(good, key=lambda m: m.distance)[:max_matches]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 3.0)
    if mask is None:
        return None, None

    mask = mask.ravel().astype(bool)
    pts0_in = pts0[mask]
    pts1_in = pts1[mask]

    if len(pts0_in) < 20:
        return None, None

    return pts0_in, pts1_in


# -------------------------------------------------------------------
# DATACLASSES
# -------------------------------------------------------------------

@dataclass
class MatcherConfig:
    # Backend público recomendado: vismatch. El backend antiguo matching queda
    # sólo como compatibilidad opcional para instalaciones legacy.
    matcher_backend: str = "auto"      # auto | vismatch | matching | none
    model_name: Optional[str] = None    # None -> default según backend
    device: Optional[str] = None
    matching_repo: Optional[str] = None # ruta opcional indicada por CLI/env
    min_matches: int = 20
    no_orb_fallback: bool = False
    ransac_thresh_px: float = 3.0
    work_max_side: int = 1600


@dataclass
class WarpScoreConfig:
    poly_order: int = 2
    grid_step_px: int = 180
    coverage_bins: int = 6
    cv_repeats: int = 4
    cv_holdout_ratio: float = 0.25
    rng_seed: int = 1234
    use_similarity_baseline: bool = True
    use_masked_ssim: bool = True

    w_coverage: float = 1.50
    w_inlier_ratio: float = 0.60
    w_match_count: float = 0.80
    w_cv_rmse: float = 4.00
    w_disp_rmse: float = 3.00
    w_log_area: float = 1.75
    w_aniso: float = 1.75
    w_bending: float = 1.00
    w_fold: float = 4.00
    w_ssim: float = 0.35


@dataclass
class SearchConfig:
    yaw_range: Tuple[float, float] = (-180.0, 180.0)
    pitch_range: Tuple[float, float] = (30.0, 90.0)
    roll_range: Tuple[float, float] = (-15.0, 15.0)

    # En modo simplex/Nelder-Mead, coarse_steps ya NO significa paso de grid,
    # sino tamaño inicial del simplex en yaw, pitch y roll.
    coarse_steps: Tuple[float, float, float] = (20.0, 10.0, 5.0)

    # Se conservan estos parámetros por compatibilidad con llamadas antiguas.
    # refine_steps se usa como tolerancia angular aproximada de convergencia.
    fine_steps: Tuple[float, float, float] = (5.0, 2.5, 1.0)
    fine_windows: Tuple[float, float, float] = (15.0, 7.5, 4.0)
    refine_steps: Tuple[float, float, float] = (2.0, 1.0, 0.5)
    refine_windows: Tuple[float, float, float] = (4.0, 2.0, 1.0)
    topk_coarse: int = 5
    topk_fine: int = 3

    # Parámetros específicos del método Nelder-Mead/simplex.
    simplex_max_iter: int = 45
    simplex_max_evals: int = 140
    simplex_score_tol: float = 1e-3


# -------------------------------------------------------------------
# MATCHER FACTORY
# -------------------------------------------------------------------

def resolve_matching_repo(user_repo: Optional[str] = None) -> Optional[str]:
    """
    Devuelve una ruta local al repo de matching si el usuario la ha indicado.

    Soporta:
      - vismatch actual: repo con paquete ./vismatch
      - image-matching-models antiguo: repo con paquete ./matching

    Si vismatch está instalado por pip, no hace falta devolver ninguna ruta.
    No usa rutas personales tipo /home/usuario/... para que el código sea portable.
    """
    candidates: List[str] = []

    if user_repo:
        candidates.append(user_repo)

    candidates.extend([
        os.environ.get("VISMATCH_REPO", ""),
        os.environ.get("VISMATCH_DIR", ""),
        os.environ.get("IMAGE_MATCHING_MODELS_DIR", ""),
    ])

    for c in candidates:
        if not c:
            continue

        path = Path(c).expanduser().resolve()
        if not path.exists():
            continue

        # Caso normal: ruta al repo que contiene el paquete.
        if (path / "vismatch").exists() or (path / "matching").exists():
            return str(path)

        # Caso menos habitual: ruta directa al paquete vismatch/ o matching/.
        if path.name in {"vismatch", "matching"} and (path / "__init__.py").exists():
            return str(path.parent)

        # Si el usuario lo ha pasado explícitamente, se acepta igualmente.
        if user_repo:
            return str(path)

    return None


def add_matching_repo_to_syspath(repo_path: Optional[str]) -> None:
    """Añade el repo local al sys.path si se ha indicado y existe."""
    if not repo_path:
        return

    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        return

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"Repo local de matching añadido a sys.path: {path_str}")


def create_matcher(cfg: MatcherConfig):
    """
    Carga un matcher y devuelve (matcher, device).

    Backends:
      - auto: prueba vismatch y después matching antiguo.
      - vismatch: fuerza vismatch.
      - matching: fuerza image-matching-models antiguo.
      - none: no carga matcher; se usará ORB fallback.
    """
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU.")
        device = "cpu"

    backend = cfg.matcher_backend
    if backend == "none":
        print("Matcher neural desactivado. Se usará ORB fallback.")
        return None, device

    repo_path = resolve_matching_repo(cfg.matching_repo)
    add_matching_repo_to_syspath(repo_path)

    backends = [backend]
    if backend == "auto":
        backends = ["vismatch", "matching"]

    errors: List[str] = []

    for candidate_backend in backends:
        if candidate_backend == "vismatch":
            model_name = cfg.model_name or "superpoint-lightglue"
            try:
                from vismatch import get_matcher  # type: ignore
                import vismatch as vismatch_mod  # type: ignore

                print(f"vismatch importado desde: {vismatch_mod.__file__}")
                matcher = get_matcher(model_name, device=device)
                print(f"Matcher cargado: backend=vismatch | model={model_name}")
                return matcher, device
            except Exception as e:
                errors.append(f"vismatch({model_name}): {repr(e)}")
                continue

        if candidate_backend == "matching":
            model_name = cfg.model_name or "superpoint-lg"
            try:
                from matching import get_matcher  # type: ignore
                import matching as matching_mod  # type: ignore

                print(f"matching importado desde: {matching_mod.__file__}")
                matcher = get_matcher(model_name, device=device)
                print(f"Matcher cargado: backend=matching | model={model_name}")
                return matcher, device
            except Exception as e:
                errors.append(f"matching({model_name}): {repr(e)}")
                continue

    print("No se pudo cargar ningún matcher neural.")
    for err in errors:
        print(f"   - {err}")
    print("Se usará ORB fallback en todos los candidatos.")
    return None, device


def load_image_for_matcher(
    matcher,
    path: str,
    device: str,
    resize_hw: Optional[Tuple[int, int]] = None,
    resize_long_side: Optional[int] = None,
) -> Any:
    """
    Carga imagen para el matcher.

    vismatch actual suele exponer matcher.load_image(path, resize=...). Si no
    existe, se usa el loader tensorial clásico compatible con image-matching-models.
    """
    if matcher is not None and hasattr(matcher, "load_image"):
        try:
            if resize_long_side is not None:
                img = matcher.load_image(str(path), resize=int(resize_long_side))
            else:
                img = matcher.load_image(str(path))
            if torch.is_tensor(img):
                img = img.to(device)
            return img
        except TypeError:
            # Algunos wrappers no aceptan resize como argumento.
            img = matcher.load_image(str(path))
            if torch.is_tensor(img):
                img = img.to(device)
            return img

    return image_loader(str(path), resize=resize_hw).to(device)


def run_matcher(matcher, img0, img1):
    """Ejecuta matcher soportando wrappers con o sin dimensión batch."""
    with torch.inference_mode():
        try:
            return matcher(img0, img1)
        except Exception as first_error:
            if torch.is_tensor(img0) and torch.is_tensor(img1) and img0.ndim == 3 and img1.ndim == 3:
                return matcher(img0.unsqueeze(0), img1.unsqueeze(0))
            raise first_error


# -------------------------------------------------------------------
# GEOMETRÍA Y NORMALIZACIÓN
# -------------------------------------------------------------------

def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = 2.0 * (points[:, 0] / max(width - 1, 1)) - 1.0
    out[:, 1] = 2.0 * (points[:, 1] / max(height - 1, 1)) - 1.0
    return out


def denormalize_points(points_n: np.ndarray, width: int, height: int) -> np.ndarray:
    points_n = np.asarray(points_n, dtype=np.float64)
    out = np.empty_like(points_n, dtype=np.float64)
    out[:, 0] = 0.5 * (points_n[:, 0] + 1.0) * max(width - 1, 1)
    out[:, 1] = 0.5 * (points_n[:, 1] + 1.0) * max(height - 1, 1)
    return out


def generate_grid_points(width: int, height: int, step_px: int) -> np.ndarray:
    xs = np.arange(0, width, step_px, dtype=np.float64)
    ys = np.arange(0, height, step_px, dtype=np.float64)
    if len(xs) == 0 or xs[-1] != width - 1:
        xs = np.append(xs, width - 1)
    if len(ys) == 0 or ys[-1] != height - 1:
        ys = np.append(ys, height - 1)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def convex_hull_masked_grid(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return grid
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    keep = []
    for p in grid:
        inside = cv2.pointPolygonTest(hull.astype(np.float32), (float(p[0]), float(p[1])), False)
        if inside >= 0:
            keep.append(p)
    if len(keep) < 12:
        return grid
    return np.asarray(keep, dtype=np.float64)


# -------------------------------------------------------------------
# MODOLOS DE WARP Y SCORE
# -------------------------------------------------------------------

def point_coverage(points: np.ndarray, width: int, height: int, bins: int = 6) -> float:
    if points is None or len(points) == 0:
        return 0.0
    x = np.clip((points[:, 0] / max(width, 1) * bins).astype(int), 0, bins - 1)
    y = np.clip((points[:, 1] / max(height, 1) * bins).astype(int), 0, bins - 1)
    occ = {(int(ix), int(iy)) for ix, iy in zip(x, y)}
    return float(len(occ)) / float(bins * bins)


def estimate_similarity_baseline(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
    sim = SimilarityTransform()
    ok = sim.estimate(src, dst)
    if not ok:
        sim = SimilarityTransform(scale=1.0, rotation=0.0, translation=(0.0, 0.0))
    return sim


def residual_after_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    sim = estimate_similarity_baseline(src, dst)
    return sim.inverse(dst)


def finite_difference_jacobians(grid_src: np.ndarray, grid_dst_residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(grid_src[:, 0])
    ys = np.unique(grid_src[:, 1])
    nx = len(xs)
    ny = len(ys)
    if nx < 2 or ny < 2 or nx * ny != len(grid_src):
        raise ValueError("La rejilla debe ser rectangular para aproximar jacobianos.")

    P = grid_src.reshape(ny, nx, 2)
    Q = grid_dst_residual.reshape(ny, nx, 2)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dx = 1.0 if abs(dx) < 1e-12 else dx
    dy = 1.0 if abs(dy) < 1e-12 else dy

    J = np.zeros((ny - 1, nx - 1, 2, 2), dtype=np.float64)
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            q00 = Q[iy, ix]
            q10 = Q[iy, ix + 1]
            q01 = Q[iy + 1, ix]
            dqx = (q10 - q00) / dx
            dqy = (q01 - q00) / dy
            J[iy, ix, :, 0] = dqx
            J[iy, ix, :, 1] = dqy

    R = Q - P
    return J, P, R


def warp_deformation_metrics(
    transform: PolynomialTransform,
    width: int,
    height: int,
    grid_step_px: int,
    support_points_real: Optional[np.ndarray] = None,
    similarity_baseline: bool = True,
) -> Dict[str, float]:
    grid = generate_grid_points(width, height, grid_step_px)
    if support_points_real is not None and len(support_points_real) >= 3:
        grid = convex_hull_masked_grid(grid, support_points_real)

    # Si la rejilla no es rectangular tras filtrar por hull, evaluamos desplazamiento/bending
    # sobre puntos, y jacobianos en la rejilla rectangular completa como respaldo.
    grid_eval = grid
    if len(np.unique(grid[:, 0])) * len(np.unique(grid[:, 1])) != len(grid):
        grid_rect = generate_grid_points(width, height, grid_step_px)
    else:
        grid_rect = grid

    grid_n = normalize_points(grid_eval, width, height)
    pred_n = transform(grid_n)

    if similarity_baseline:
        pred_res_n = residual_after_similarity(grid_n, pred_n)
    else:
        pred_res_n = pred_n

    disp = pred_res_n - grid_n
    disp_norm = np.linalg.norm(disp, axis=1)
    disp_rmse = float(np.sqrt(np.mean(disp_norm ** 2))) if len(disp_norm) else 1e6
    disp_p95 = float(np.percentile(disp_norm, 95)) if len(disp_norm) else 1e6

    # Bending energy por segundas diferencias en rejilla rectangular completa.
    grid_rect_n = normalize_points(grid_rect, width, height)
    pred_rect_n = transform(grid_rect_n)
    if similarity_baseline:
        pred_rect_res_n = residual_after_similarity(grid_rect_n, pred_rect_n)
    else:
        pred_rect_res_n = pred_rect_n

    xs = np.unique(grid_rect[:, 0])
    ys = np.unique(grid_rect[:, 1])
    nx = len(xs)
    ny = len(ys)
    Qr = pred_rect_res_n.reshape(ny, nx, 2)
    Pr = grid_rect_n.reshape(ny, nx, 2)
    residual_field = Qr - Pr

    bend_x = np.diff(residual_field, n=2, axis=1) if nx >= 3 else np.zeros((ny, 0, 2), dtype=np.float64)
    bend_y = np.diff(residual_field, n=2, axis=0) if ny >= 3 else np.zeros((0, nx, 2), dtype=np.float64)
    bending_energy = 0.0
    n_terms = 0
    if bend_x.size:
        bending_energy += float(np.mean(np.linalg.norm(bend_x, axis=2) ** 2))
        n_terms += 1
    if bend_y.size:
        bending_energy += float(np.mean(np.linalg.norm(bend_y, axis=2) ** 2))
        n_terms += 1
    if n_terms:
        bending_energy /= n_terms

    # Jacobianos solo si la rejilla es rectangular.
    try:
        J, _, _ = finite_difference_jacobians(grid_rect_n, pred_rect_res_n)
        Jf = J.reshape(-1, 2, 2)
        dets = np.linalg.det(Jf)
        fold_ratio = float(np.mean(dets <= 0.0))

        valid = dets > 1e-9
        if np.any(valid):
            dets_valid = dets[valid]
            log_area_rmse = float(np.sqrt(np.mean(np.log(dets_valid) ** 2)))
            svals = np.linalg.svd(Jf[valid], compute_uv=False)
            aniso = np.abs(np.log(np.maximum(svals[:, 0], 1e-9) / np.maximum(svals[:, 1], 1e-9)))
            aniso_rmse = float(np.sqrt(np.mean(aniso ** 2)))
        else:
            log_area_rmse = 10.0
            aniso_rmse = 10.0
    except Exception:
        fold_ratio = 1.0
        log_area_rmse = 10.0
        aniso_rmse = 10.0

    return {
        "disp_rmse": disp_rmse,
        "disp_p95": disp_p95,
        "log_area_rmse": log_area_rmse,
        "aniso_rmse": aniso_rmse,
        "bending_energy": float(bending_energy),
        "fold_ratio": fold_ratio,
    }


def masked_ssim_on_earth(img_sim_bgr: np.ndarray, img_real_bgr: np.ndarray) -> float:
    if img_sim_bgr.shape[:2] != img_real_bgr.shape[:2]:
        img_real_bgr = cv2.resize(img_real_bgr, (img_sim_bgr.shape[1], img_sim_bgr.shape[0]), interpolation=cv2.INTER_AREA)

    sim_gray = cv2.cvtColor(img_sim_bgr, cv2.COLOR_BGR2GRAY)
    real_gray = cv2.cvtColor(img_real_bgr, cv2.COLOR_BGR2GRAY)

    # Máscara heurística para separar Tierra/cielo usando la simulada.
    thr = max(3, int(np.percentile(sim_gray, 15)))
    mask = sim_gray > thr
    if mask.mean() < 0.05:
        return float(ssim(sim_gray, real_gray))

    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    sim_crop = sim_gray[y:y + h, x:x + w]
    real_crop = real_gray[y:y + h, x:x + w]
    if sim_crop.size == 0 or real_crop.size == 0:
        return float(ssim(sim_gray, real_gray))
    return float(ssim(sim_crop, real_crop))


def repeated_cv_polynomial_rmse(
    src_n: np.ndarray,
    dst_n: np.ndarray,
    order: int,
    repeats: int,
    holdout_ratio: float,
    seed: int,
) -> float:
    n = len(src_n)
    if n < max(12, order * 4):
        tf = PolynomialTransform()
        ok = tf.estimate(src_n, dst_n, order=order)
        if not ok:
            return 1e6
        pred = tf(src_n)
        return float(np.sqrt(np.mean(np.sum((pred - dst_n) ** 2, axis=1))))

    rng = np.random.default_rng(seed)
    rmses: List[float] = []
    holdout_n = max(6, int(round(n * holdout_ratio)))
    holdout_n = min(holdout_n, n - 6)

    for _ in range(repeats):
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:holdout_n]
        tr_idx = idx[holdout_n:]

        tf = PolynomialTransform()
        ok = tf.estimate(src_n[tr_idx], dst_n[tr_idx], order=order)
        if not ok:
            rmses.append(1e6)
            continue
        pred = tf(src_n[val_idx])
        rmse = float(np.sqrt(np.mean(np.sum((pred - dst_n[val_idx]) ** 2, axis=1))))
        rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else 1e6


def robust_match_and_fit(
    img_sim_bgr: np.ndarray,
    img_real_bgr: np.ndarray,
    img_sim_path: str,
    img_real_path: str,
    matcher,
    matcher_cfg: MatcherConfig,
    warp_cfg: WarpScoreConfig,
    device: str,
) -> Optional[Dict[str, object]]:
    # Igualar resolución y opcionalmente reducir tamaño de trabajo.
    hs, ws = img_sim_bgr.shape[:2]
    hr, wr = img_real_bgr.shape[:2]
    if (hs, ws) != (hr, wr):
        img_real_bgr = cv2.resize(img_real_bgr, (ws, hs), interpolation=cv2.INTER_AREA)
        hr, wr = hs, ws

    scale = 1.0
    max_side = max(hs, ws)
    if max_side > matcher_cfg.work_max_side:
        scale = float(matcher_cfg.work_max_side) / float(max_side)
        work_w = max(32, int(round(ws * scale)))
        work_h = max(32, int(round(hs * scale)))
        sim_work = cv2.resize(img_sim_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        real_work = cv2.resize(img_real_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_w, work_h = ws, hs
        sim_work = img_sim_bgr
        real_work = img_real_bgr

    mkpts0 = mkpts1 = None

    if matcher is not None:
        # Para vismatch usamos matcher.load_image si existe; para el backend
        # antiguo usamos el loader tensorial clásico.
        resize_long_side = matcher_cfg.work_max_side if max_side > matcher_cfg.work_max_side else None
        sim_t = load_image_for_matcher(
            matcher,
            img_sim_path,
            device=device,
            resize_hw=(work_h, work_w),
            resize_long_side=resize_long_side,
        )
        real_t = load_image_for_matcher(
            matcher,
            img_real_path,
            device=device,
            resize_hw=(work_h, work_w),
            resize_long_side=resize_long_side,
        )
        result = run_matcher(matcher, sim_t, real_t)
        mkpts0, mkpts1 = extract_mkpts_from_result(result)

    n_raw = 0 if mkpts0 is None else len(mkpts0)
    if (mkpts0 is None or mkpts1 is None or n_raw < matcher_cfg.min_matches) and not matcher_cfg.no_orb_fallback:
        mk0_fb, mk1_fb = orb_fallback_matches(sim_work, real_work)
        if mk0_fb is not None and len(mk0_fb) >= matcher_cfg.min_matches:
            mkpts0, mkpts1 = mk0_fb, mk1_fb
            n_raw = len(mkpts0)

    if mkpts0 is None or mkpts1 is None or len(mkpts0) < matcher_cfg.min_matches:
        return None

    mkpts0 = np.asarray(mkpts0, dtype=np.float64)
    mkpts1 = np.asarray(mkpts1, dtype=np.float64)

    # RANSAC homography para depurar outliers antes de ajustar la polinómica.
    H, mask = cv2.findHomography(
        mkpts1.astype(np.float32),
        mkpts0.astype(np.float32),
        cv2.RANSAC,
        matcher_cfg.ransac_thresh_px,
    )
    if mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    mkpts0_in = mkpts0[inliers]
    mkpts1_in = mkpts1[inliers]
    if len(mkpts0_in) < matcher_cfg.min_matches:
        return None

    # Volver a tamaño original si hubo downscale.
    if scale != 1.0:
        inv_scale = 1.0 / scale
        mkpts0_in *= inv_scale
        mkpts1_in *= inv_scale
        mkpts0 *= inv_scale
        mkpts1 *= inv_scale

    src_n = normalize_points(mkpts1_in, ws, hs)
    dst_n = normalize_points(mkpts0_in, ws, hs)

    poly = PolynomialTransform()
    ok = poly.estimate(src_n, dst_n, order=warp_cfg.poly_order)
    if not ok:
        return None

    cv_rmse = repeated_cv_polynomial_rmse(
        src_n,
        dst_n,
        order=warp_cfg.poly_order,
        repeats=warp_cfg.cv_repeats,
        holdout_ratio=warp_cfg.cv_holdout_ratio,
        seed=warp_cfg.rng_seed,
    )

    coverage_real = point_coverage(mkpts1_in, ws, hs, bins=warp_cfg.coverage_bins)
    coverage_sim = point_coverage(mkpts0_in, ws, hs, bins=warp_cfg.coverage_bins)
    coverage = min(coverage_real, coverage_sim)
    inlier_ratio = float(len(mkpts0_in)) / float(len(mkpts0)) if len(mkpts0) else 0.0
    match_count_norm = float(np.tanh(len(mkpts0_in) / 120.0))

    deform = warp_deformation_metrics(
        transform=poly,
        width=ws,
        height=hs,
        grid_step_px=warp_cfg.grid_step_px,
        support_points_real=mkpts1_in,
        similarity_baseline=warp_cfg.use_similarity_baseline,
    )

    masked_ssim = masked_ssim_on_earth(img_sim_bgr, img_real_bgr) if warp_cfg.use_masked_ssim else 0.0

    score = (
        warp_cfg.w_coverage * coverage
        + warp_cfg.w_inlier_ratio * inlier_ratio
        + warp_cfg.w_match_count * match_count_norm
        + warp_cfg.w_ssim * masked_ssim
        - warp_cfg.w_cv_rmse * cv_rmse
        - warp_cfg.w_disp_rmse * deform["disp_rmse"]
        - warp_cfg.w_log_area * deform["log_area_rmse"]
        - warp_cfg.w_aniso * deform["aniso_rmse"]
        - warp_cfg.w_bending * deform["bending_energy"]
        - warp_cfg.w_fold * deform["fold_ratio"]
    )

    out = {
        "n_matches_raw": int(len(mkpts0)),
        "n_inliers": int(len(mkpts0_in)),
        "inlier_ratio": float(inlier_ratio),
        "coverage": float(coverage),
        "coverage_real": float(coverage_real),
        "coverage_sim": float(coverage_sim),
        "match_count_norm": float(match_count_norm),
        "cv_rmse": float(cv_rmse),
        "masked_ssim": float(masked_ssim),
        "score": float(score),
        "poly_params": poly.params.tolist(),
    }
    out.update({k: float(v) for k, v in deform.items()})
    return out


# -------------------------------------------------------------------
# ISS + RENDER
# -------------------------------------------------------------------

def get_iss_state(tle_data, time, orientation_mode: str = "forward"):
    sat = find_closest_tle(tle_data, time)
    lat, lon, alt, v_icrf, v_itrs = get_iss_position_and_velocity(sat, time)
    vel = v_itrs if orientation_mode == "forward" else v_itrs
    return lat, lon, alt, vel


def render_simulated_image(
    lat: float,
    lon: float,
    alt: float,
    vel,
    yaw: float,
    pitch: float,
    roll: float,
    time_str: str,
    output_dir: Path,
    sphere,
    focal_length: float,
    sensor_width: float,
    sensor_height: float,
    pixel_width: int,
    pixel_height: int,
    earth_radius: float,
    orientation_mode: str = "north",
) -> str:
    _, file_path = creaimagen(
        latitude=lat,
        longitude=lon,
        altitude_real=alt,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        velocity_vector=vel,
        sphere=sphere,
        focal_length=focal_length,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        time=time_str,
        output_directory=str(output_dir),
        earth_radius=earth_radius,
        render_image=True,
        orientation_mode=orientation_mode,
    )
    return file_path


# -------------------------------------------------------------------
# EVALUACIÓN DE CANDIDATOS
# -------------------------------------------------------------------

def evaluate_candidate(
    yaw: float,
    pitch: float,
    roll: float,
    real_image_path: str,
    real_image_bgr: np.ndarray,
    render_ctx: Dict[str, object],
    matcher,
    matcher_cfg: MatcherConfig,
    warp_cfg: WarpScoreConfig,
    device: str,
    metrics_cache: Dict[Tuple[float, float, float], Dict[str, object]],
    metrics_jsonl: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    key = (round(float(yaw), 6), round(float(pitch), 6), round(float(roll), 6))
    if key in metrics_cache:
        return metrics_cache[key]

    sim_path = render_simulated_image(
        lat=render_ctx["lat"],
        lon=render_ctx["lon"],
        alt=render_ctx["alt"],
        vel=render_ctx["vel"],
        yaw=float(yaw),
        pitch=float(pitch),
        roll=float(roll),
        time_str=render_ctx["time_str"],
        output_dir=render_ctx["output_dir"],
        sphere=render_ctx["sphere"],
        focal_length=render_ctx["focal_length"],
        sensor_width=render_ctx["sensor_width"],
        sensor_height=render_ctx["sensor_height"],
        pixel_width=render_ctx["pixel_width"],
        pixel_height=render_ctx["pixel_height"],
        earth_radius=render_ctx["earth_radius"],
        orientation_mode=render_ctx["orientation_mode"],
    )

    img_sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
    if img_sim is None:
        return None

    metrics = robust_match_and_fit(
        img_sim_bgr=img_sim,
        img_real_bgr=real_image_bgr,
        img_sim_path=str(sim_path),
        img_real_path=str(real_image_path),
        matcher=matcher,
        matcher_cfg=matcher_cfg,
        warp_cfg=warp_cfg,
        device=device,
    )
    if metrics is None:
        return None

    info: Dict[str, object] = {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "sim_path": str(sim_path),
    }
    info.update(metrics)
    metrics_cache[key] = info

    if metrics_jsonl is not None:
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

    return info


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    if step <= 0:
        raise ValueError("step debe ser > 0")
    n = int(math.floor((stop - start) / step + 1e-9))
    for i in range(n + 1):
        yield float(start + i * step)


# -------------------------------------------------------------------
# BÚSQUEDA DE ORIENTACIÓN CON NELDER-MEAD / SIMPLEX
# -------------------------------------------------------------------


def wrap_yaw_180(yaw: float) -> float:
    """
    Normaliza yaw al intervalo [-180, 180).
    """
    return ((float(yaw) + 180.0) % 360.0) - 180.0


def yaw_diff_deg(a: float, b: float) -> float:
    """
    Diferencia angular a - b en grados, respetando la circularidad de yaw.
    """
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vector a - b para [yaw, pitch, roll].
    El yaw se calcula como diferencia angular circular.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return np.array(
        [
            yaw_diff_deg(a[0], b[0]),
            a[1] - b[1],
            a[2] - b[2],
        ],
        dtype=np.float64,
    )


def centroid_angles(points: np.ndarray) -> np.ndarray:
    """
    Centroide de puntos [yaw, pitch, roll].
    Para yaw usa media circular; pitch y roll usan media ordinaria.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points debe tener shape (N, 3)")

    yaw_rad = np.deg2rad(points[:, 0])
    yaw_mean = np.rad2deg(
        np.arctan2(np.mean(np.sin(yaw_rad)), np.mean(np.cos(yaw_rad)))
    )

    return np.array(
        [
            wrap_yaw_180(yaw_mean),
            float(np.mean(points[:, 1])),
            float(np.mean(points[:, 2])),
        ],
        dtype=np.float64,
    )


def project_angle_candidate(x: np.ndarray, search_cfg: SearchConfig) -> np.ndarray:
    """
    Proyecta un candidato [yaw, pitch, roll] a los rangos permitidos.
    """
    x = np.asarray(x, dtype=np.float64).copy()

    yaw_min, yaw_max = sorted(map(float, search_cfg.yaw_range))
    pitch_min, pitch_max = sorted(map(float, search_cfg.pitch_range))
    roll_min, roll_max = sorted(map(float, search_cfg.roll_range))

    x[0] = wrap_yaw_180(x[0])
    x[0] = float(np.clip(x[0], yaw_min, yaw_max))
    x[1] = float(np.clip(x[1], pitch_min, pitch_max))
    x[2] = float(np.clip(x[2], roll_min, roll_max))

    return x


def candidate_key(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Clave estable para cachear evaluaciones de yaw/pitch/roll.
    """
    x = np.asarray(x, dtype=np.float64)
    return (
        round(float(x[0]), 6),
        round(float(x[1]), 6),
        round(float(x[2]), 6),
    )


def simplex_diameter_deg(simplex: np.ndarray) -> float:
    """
    Tamaño máximo del simplex en grados, usando norma euclídea en yaw/pitch/roll.
    """
    simplex = np.asarray(simplex, dtype=np.float64)
    if len(simplex) <= 1:
        return 0.0

    max_dist = 0.0
    for i in range(len(simplex)):
        for j in range(i + 1, len(simplex)):
            d = angle_delta(simplex[i], simplex[j])
            dist = float(np.linalg.norm(d))
            max_dist = max(max_dist, dist)
    return max_dist


def active_angle_dimensions(search_cfg: SearchConfig, eps: float = 1e-9) -> List[int]:
    """
    Devuelve las dimensiones activas del problema:
      0 = yaw, 1 = pitch, 2 = roll.
    Si un rango tiene min == max, se considera fijo.
    """
    ranges = [search_cfg.yaw_range, search_cfg.pitch_range, search_cfg.roll_range]
    active: List[int] = []
    for i, r in enumerate(ranges):
        r0, r1 = sorted(map(float, r))
        if abs(r1 - r0) > eps:
            active.append(i)
    return active


def build_initial_simplex(
    center: np.ndarray,
    steps: Tuple[float, float, float],
    search_cfg: SearchConfig,
) -> np.ndarray:
    """
    Construye el simplex inicial.

    En 3D tiene 4 vértices: centro + desplazamiento en yaw, pitch y roll.
    Si alguna dimensión está fija, se omite y el simplex pasa a ser de menor dimensión.
    """
    center = project_angle_candidate(center, search_cfg)
    vertices = [center]

    active_dims = active_angle_dimensions(search_cfg)
    for dim in active_dims:
        step = abs(float(steps[dim]))
        if step <= 0:
            continue

        v = center.copy()
        v[dim] += step
        v = project_angle_candidate(v, search_cfg)

        # Si el desplazamiento queda anulado por estar pegado al límite,
        # intentamos desplazar en la dirección contraria.
        if np.linalg.norm(angle_delta(v, center)) < 1e-9:
            v = center.copy()
            v[dim] -= step
            v = project_angle_candidate(v, search_cfg)

        if np.linalg.norm(angle_delta(v, center)) >= 1e-9:
            vertices.append(v)

    return np.vstack(vertices)


def make_default_simplex_center(search_cfg: SearchConfig) -> np.ndarray:
    """
    Centro inicial por defecto: punto medio de los rangos.
    """
    yaw_min, yaw_max = sorted(map(float, search_cfg.yaw_range))
    pitch_min, pitch_max = sorted(map(float, search_cfg.pitch_range))
    roll_min, roll_max = sorted(map(float, search_cfg.roll_range))

    return np.array(
        [
            0.5 * (yaw_min + yaw_max),
            0.5 * (pitch_min + pitch_max),
            0.5 * (roll_min + roll_max),
        ],
        dtype=np.float64,
    )


def search_best_orientation(
    real_image_path: str,
    search_output_dir: str,
    tle_dir: str,
    texture_path: str,
    obs_time: datetime,
    focal_length: float,
    sensor_width: float,
    sensor_height: float,
    pixel_width: int,
    pixel_height: int,
    orientation_mode: str = "forward",
    earth_radius: float = 10.0,
    search_cfg: Optional[SearchConfig] = None,
    matcher_cfg: Optional[MatcherConfig] = None,
    warp_cfg: Optional[WarpScoreConfig] = None,
    metrics_jsonl: Optional[str] = None,
):
    """
    Busca yaw/pitch/roll maximizando el score de evaluate_candidate mediante
    Nelder-Mead/simplex.

    Importante:
    - La función objetivo sigue siendo exactamente la misma que antes:
      render + matching + ajuste polinomial + métricas de deformación.
    - Lo único que cambia es la estrategia de propuesta de candidatos.
    - En 3D el simplex es un tetraedro de 4 vértices.
    """
    search_cfg = search_cfg or SearchConfig()
    matcher_cfg = matcher_cfg or MatcherConfig()
    warp_cfg = warp_cfg or WarpScoreConfig()

    real_image_path = str(real_image_path)
    search_output_dir = Path(search_output_dir)
    tle_dir = Path(tle_dir)
    texture_path = Path(texture_path)
    search_output_dir.mkdir(parents=True, exist_ok=True)

    real_bgr = cv2.imread(real_image_path, cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen real: {real_image_path}")

    sphere = reset_scene(earth_radius, str(texture_path))
    file_paths = list_tle_files(str(tle_dir))
    tle_data = read_tle_from_files(file_paths)
    lat, lon, alt, vel = get_iss_state(tle_data, obs_time, orientation_mode=orientation_mode)
    time_str = obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    matcher, device = create_matcher(matcher_cfg)

    render_ctx = {
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "vel": vel,
        "time_str": time_str,
        "output_dir": search_output_dir,
        "sphere": sphere,
        "focal_length": focal_length,
        "sensor_width": sensor_width,
        "sensor_height": sensor_height,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "earth_radius": earth_radius,
        "orientation_mode": orientation_mode,
    }

    metrics_jsonl_path = Path(metrics_jsonl) if metrics_jsonl else None
    if metrics_jsonl_path is not None:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_jsonl_path.write_text("", encoding="utf-8")

    # Cache de métricas válidas usado por evaluate_candidate.
    metrics_cache: Dict[Tuple[float, float, float], Dict[str, object]] = {}

    # Cache adicional que también recuerda candidatos inválidos, para no renderizar
    # dos veces un punto que no produjo suficientes matches.
    eval_cache: Dict[Tuple[float, float, float], Optional[Dict[str, object]]] = {}
    simplex_results: List[Dict[str, object]] = []

    invalid_score = -1.0e12
    n_evals = 0

    # Coeficientes clásicos de Nelder-Mead.
    alpha = 1.0   # reflexión
    gamma = 2.0   # expansión
    rho = 0.5     # contracción
    sigma = 0.5   # encogimiento

    def score_or_invalid(info: Optional[Dict[str, object]]) -> float:
        if info is None:
            return invalid_score
        return float(info["score"])

    def eval_vertex(x: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, object]], float]:
        """
        Evalúa un candidato [yaw, pitch, roll] usando la función objetivo actual.
        """
        nonlocal n_evals

        x = project_angle_candidate(x, search_cfg)
        key = candidate_key(x)

        if key in eval_cache:
            info_cached = eval_cache[key]
            return x, info_cached, score_or_invalid(info_cached)

        info = evaluate_candidate(
            yaw=float(x[0]),
            pitch=float(x[1]),
            roll=float(x[2]),
            real_image_path=real_image_path,
            real_image_bgr=real_bgr,
            render_ctx=render_ctx,
            matcher=matcher,
            matcher_cfg=matcher_cfg,
            warp_cfg=warp_cfg,
            device=device,
            metrics_cache=metrics_cache,
            metrics_jsonl=metrics_jsonl_path,
        )

        n_evals += 1
        eval_cache[key] = info

        if info is not None:
            simplex_results.append(info)

        return x, info, score_or_invalid(info)

    # -------------------------------------------------------------
    # 1) Simplex inicial
    # -------------------------------------------------------------
    x0 = make_default_simplex_center(search_cfg)
    simplex = build_initial_simplex(
        center=x0,
        steps=search_cfg.coarse_steps,
        search_cfg=search_cfg,
    )

    print("Comenzando búsqueda Nelder-Mead/simplex de yaw/pitch/roll...")
    print(
        f"  Centro inicial: yaw={x0[0]:.3f}, pitch={x0[1]:.3f}, roll={x0[2]:.3f}"
    )
    print(
        f"  Vértices iniciales: {len(simplex)} | max_iter={search_cfg.simplex_max_iter} | "
        f"max_evals={search_cfg.simplex_max_evals}"
    )

    infos: List[Optional[Dict[str, object]]] = []
    scores: List[float] = []

    for i in range(len(simplex)):
        simplex[i], info_i, score_i = eval_vertex(simplex[i])
        infos.append(info_i)
        scores.append(score_i)

    scores_arr = np.asarray(scores, dtype=np.float64)

    # Caso degenerado: todos los ángulos están fijados.
    if len(simplex) == 1:
        all_valid = sorted(simplex_results, key=lambda d: float(d["score"]), reverse=True)
        best = all_valid[0] if all_valid else None
        return best, (all_valid, [], [])

    if np.max(scores_arr) <= invalid_score / 10.0:
        print("No se encontró ningún resultado válido en el simplex inicial.")
        print(
            "Prueba a ampliar los rangos o a mover el centro inicial cambiando "
            "yaw_range/pitch_range/roll_range."
        )
        return None, ([], [], [])

    # Tolerancia angular de parada. Usamos refine_steps como tolerancia aproximada.
    simplex_tol_deg = max(float(v) for v in search_cfg.refine_steps)
    score_tol = float(search_cfg.simplex_score_tol)

    # -------------------------------------------------------------
    # 2) Iteraciones Nelder-Mead
    # -------------------------------------------------------------
    for it in range(int(search_cfg.simplex_max_iter)):
        # Ordenar de mejor a peor porque MAXIMIZAMOS score.
        order = np.argsort(scores_arr)[::-1]
        simplex = simplex[order]
        scores_arr = scores_arr[order]
        infos = [infos[int(i)] for i in order]

        best_x = simplex[0]
        best_score = float(scores_arr[0])
        worst_x = simplex[-1]
        second_worst_score = float(scores_arr[-2])
        worst_score = float(scores_arr[-1])

        diameter = simplex_diameter_deg(simplex)
        finite_scores = scores_arr[scores_arr > invalid_score / 10.0]
        if len(finite_scores):
            score_span = float(np.max(finite_scores) - np.min(finite_scores))
        else:
            score_span = float("inf")

        print(
            f"[SIMPLEX] iter={it:02d}, evals={n_evals:03d}, "
            f"best yaw={best_x[0]:.3f}, pitch={best_x[1]:.3f}, roll={best_x[2]:.3f}, "
            f"score={best_score:.5f}, diameter={diameter:.3f}, score_span={score_span:.6f}"
        )

        if diameter <= simplex_tol_deg and score_span <= score_tol:
            print(
                f"Convergencia simplex: diameter={diameter:.4f}, "
                f"score_span={score_span:.6f}"
            )
            break

        if n_evals >= int(search_cfg.simplex_max_evals):
            print(f"Parada por simplex_max_evals={search_cfg.simplex_max_evals}.")
            break

        # Centroide de todos los puntos salvo el peor.
        centroid = centroid_angles(simplex[:-1])

        # ---------------------------------------------------------
        # Reflexión
        # ---------------------------------------------------------
        worst_delta = angle_delta(worst_x, centroid)
        reflected_x = project_angle_candidate(
            centroid - alpha * worst_delta,
            search_cfg,
        )
        reflected_x, reflected_info, reflected_score = eval_vertex(reflected_x)

        # ---------------------------------------------------------
        # Expansión: si la reflexión mejora al mejor actual.
        # ---------------------------------------------------------
        if reflected_score > best_score:
            reflected_delta = angle_delta(reflected_x, centroid)
            expanded_x = project_angle_candidate(
                centroid + gamma * reflected_delta,
                search_cfg,
            )
            expanded_x, expanded_info, expanded_score = eval_vertex(expanded_x)

            if expanded_score > reflected_score:
                simplex[-1] = expanded_x
                scores_arr[-1] = expanded_score
                infos[-1] = expanded_info
            else:
                simplex[-1] = reflected_x
                scores_arr[-1] = reflected_score
                infos[-1] = reflected_info

        # ---------------------------------------------------------
        # Aceptar reflexión si mejora al segundo peor.
        # ---------------------------------------------------------
        elif reflected_score > second_worst_score:
            simplex[-1] = reflected_x
            scores_arr[-1] = reflected_score
            infos[-1] = reflected_info

        # ---------------------------------------------------------
        # Contracción o shrink.
        # ---------------------------------------------------------
        else:
            if reflected_score > worst_score:
                # Contracción externa: entre centroide y reflejado.
                reflected_delta = angle_delta(reflected_x, centroid)
                contracted_x = project_angle_candidate(
                    centroid + rho * reflected_delta,
                    search_cfg,
                )
                threshold_score = reflected_score
            else:
                # Contracción interna: entre centroide y peor.
                contracted_x = project_angle_candidate(
                    centroid + rho * worst_delta,
                    search_cfg,
                )
                threshold_score = worst_score

            contracted_x, contracted_info, contracted_score = eval_vertex(contracted_x)

            if contracted_score > threshold_score:
                simplex[-1] = contracted_x
                scores_arr[-1] = contracted_score
                infos[-1] = contracted_info
            else:
                # Shrink: encoger todo hacia el mejor punto.
                best_x = simplex[0].copy()

                new_simplex = [best_x]
                new_scores = [scores_arr[0]]
                new_infos = [infos[0]]

                for j in range(1, len(simplex)):
                    shrink_delta = angle_delta(simplex[j], best_x)
                    shrunk_x = project_angle_candidate(
                        best_x + sigma * shrink_delta,
                        search_cfg,
                    )
                    shrunk_x, shrunk_info, shrunk_score = eval_vertex(shrunk_x)

                    new_simplex.append(shrunk_x)
                    new_scores.append(shrunk_score)
                    new_infos.append(shrunk_info)

                simplex = np.vstack(new_simplex)
                scores_arr = np.asarray(new_scores, dtype=np.float64)
                infos = new_infos

    all_valid = sorted(
        simplex_results,
        key=lambda d: float(d["score"]),
        reverse=True,
    )
    best = all_valid[0] if all_valid else None

    print("Top simplex:")
    for r in all_valid[:5]:
        print(
            f"  yaw={r['yaw']:.3f}, pitch={r['pitch']:.3f}, roll={r['roll']:.3f}, "
            f"score={r['score']:.5f}, cv={r.get('cv_rmse', float('nan')):.5f}, "
            f"disp={r.get('disp_rmse', float('nan')):.5f}, "
            f"cov={r.get('coverage', float('nan')):.3f}, "
            f"inliers={r.get('n_inliers', 0)}"
        )

    # Para mantener compatibilidad con el código anterior, devolvemos una tupla
    # de tres listas. La primera contiene todos los candidatos válidos del simplex.
    return best, (all_valid, [], [])


# -------------------------------------------------------------------
# API COMPATIBLE CON LA PIPELINE ACTUAL
# -------------------------------------------------------------------


def search_best_yaw_pitch_roll(
    real_image_path,
    search_output_dir,
    tle_dir,
    texture_path,
    obs_time,
    focal_length,
    sensor_width,
    sensor_height,
    pixel_width,
    pixel_height,
    yaw_range=(-180, 180),
    pitch_range=(30, 90),
    roll_range=(-15, 15),
    coarse_steps=(20, 10, 5),
    fine_steps=(5, 2.5, 1),
    fine_windows=(15, 7.5, 4),
    refine_steps=(2, 1, 0.5),
    refine_windows=(4, 2, 1),
    orientation_mode="forward",
    earth_radius=10,
    metrics_jsonl=None,
    matcher_backend="auto",
    matcher_model=None,
    matching_repo=None,
    device=None,
    no_orb_fallback=False,
    simplex_max_iter=45,
    simplex_max_evals=140,
    simplex_score_tol=1e-3,
):
    """
    API principal: búsqueda de yaw, pitch y roll mediante Nelder-Mead/simplex.

    Devuelve:
        best, (simplex_results, [], [])

    Se mantiene la forma de retorno de la versión coarse/fine/refine para no
    romper código que espere tres listas.
    """
    search_cfg = SearchConfig(
        yaw_range=tuple(map(float, yaw_range)),
        pitch_range=tuple(map(float, pitch_range)),
        roll_range=tuple(map(float, roll_range)),
        coarse_steps=tuple(map(float, coarse_steps)),
        fine_steps=tuple(map(float, fine_steps)),
        fine_windows=tuple(map(float, fine_windows)),
        refine_steps=tuple(map(float, refine_steps)),
        refine_windows=tuple(map(float, refine_windows)),
        simplex_max_iter=int(simplex_max_iter),
        simplex_max_evals=int(simplex_max_evals),
        simplex_score_tol=float(simplex_score_tol),
    )
    return search_best_orientation(
        real_image_path=real_image_path,
        search_output_dir=search_output_dir,
        tle_dir=tle_dir,
        texture_path=texture_path,
        obs_time=obs_time,
        focal_length=focal_length,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        orientation_mode=orientation_mode,
        earth_radius=earth_radius,
        search_cfg=search_cfg,
        matcher_cfg=MatcherConfig(
            matcher_backend=matcher_backend,
            model_name=matcher_model,
            matching_repo=matching_repo,
            device=device,
            no_orb_fallback=no_orb_fallback,
        ),
        metrics_jsonl=metrics_jsonl,
    )


def search_best_yaw_pitch(
    real_image_path,
    search_output_dir,
    tle_dir,
    texture_path,
    obs_time,
    focal_length,
    sensor_width,
    sensor_height,
    pixel_width,
    pixel_height,
    yaw_range=(-180, 180),
    pitch_range=(30, 90),
    coarse_step=20,
    fine_step=5,
    fine_window=15,
    roll=0,
    orientation_mode="forward",
    earth_radius=10,
    metrics_jsonl=None,
    matcher_backend="auto",
    matcher_model=None,
    matching_repo=None,
    device=None,
    no_orb_fallback=False,
    simplex_max_iter=45,
    simplex_max_evals=140,
    simplex_score_tol=1e-3,
):
    """
    Wrapper compatible con la pipeline antigua, que buscaba solo yaw/pitch
    manteniendo roll fijo.

    Internamente llama a search_best_yaw_pitch_roll con roll_range=(roll, roll).

    Devuelve:
        best, (simplex_results, [])

    Así sigue funcionando código del tipo:
        best_angles, (coarse_res, fine_res) = angle_search.search_best_yaw_pitch(...)
    """
    best, (simplex_results, _, _) = search_best_yaw_pitch_roll(
        real_image_path=real_image_path,
        search_output_dir=search_output_dir,
        tle_dir=tle_dir,
        texture_path=texture_path,
        obs_time=obs_time,
        focal_length=focal_length,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        yaw_range=yaw_range,
        pitch_range=pitch_range,
        roll_range=(float(roll), float(roll)),
        coarse_steps=(float(coarse_step), float(coarse_step), 1.0),
        fine_steps=(float(fine_step), float(fine_step), 0.5),
        fine_windows=(float(fine_window), float(fine_window), 0.0),
        refine_steps=(float(fine_step), float(fine_step), 0.5),
        refine_windows=(float(fine_window), float(fine_window), 0.0),
        orientation_mode=orientation_mode,
        earth_radius=earth_radius,
        metrics_jsonl=metrics_jsonl,
        matcher_backend=matcher_backend,
        matcher_model=matcher_model,
        matching_repo=matching_repo,
        device=device,
        no_orb_fallback=no_orb_fallback,
        simplex_max_iter=simplex_max_iter,
        simplex_max_evals=simplex_max_evals,
        simplex_score_tol=simplex_score_tol,
    )
    return best, (simplex_results, [])


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Búsqueda automática de yaw/pitch/roll para una imagen ISS.")
    parser.add_argument("--real_image", required=True)
    parser.add_argument("--search_output_dir", required=True)
    parser.add_argument("--tle_dir", required=True)
    parser.add_argument("--texture_path", required=True)
    parser.add_argument("--obs_time", required=True, help="ISO, se asume UTC si no trae tz")

    parser.add_argument("--earth_radius", type=float, default=10.0)
    parser.add_argument("--focal_length", type=float, required=True)
    parser.add_argument("--sensor_width", type=float, default=36.0)
    parser.add_argument("--sensor_height", type=float, default=28.0)
    parser.add_argument("--pixel_width", type=int, required=True)
    parser.add_argument("--pixel_height", type=int, required=True)

    parser.add_argument("--yaw_min", type=float, default=-180)
    parser.add_argument("--yaw_max", type=float, default=180)
    parser.add_argument("--pitch_min", type=float, default=30)
    parser.add_argument("--pitch_max", type=float, default=90)
    parser.add_argument("--roll_min", type=float, default=-15)
    parser.add_argument("--roll_max", type=float, default=15)

    parser.add_argument("--coarse_yaw_step", type=float, default=20)
    parser.add_argument("--coarse_pitch_step", type=float, default=10)
    parser.add_argument("--coarse_roll_step", type=float, default=5)
    parser.add_argument("--fine_yaw_step", type=float, default=5)
    parser.add_argument("--fine_pitch_step", type=float, default=2.5)
    parser.add_argument("--fine_roll_step", type=float, default=1)
    parser.add_argument("--orientation_mode", choices=["north", "forward"], default="forward")
    parser.add_argument("--metrics_jsonl", default=None)
    parser.add_argument(
        "--matcher_backend",
        choices=["auto", "vismatch", "matching", "none"],
        default="auto",
        help="Backend de matching: auto usa vismatch si está instalado y si no prueba matching legacy.",
    )
    parser.add_argument(
        "--matcher_model",
        default=None,
        help="Nombre del modelo. Por defecto: superpoint-lightglue en vismatch, superpoint-lg en matching legacy.",
    )
    parser.add_argument(
        "--matching_repo",
        default=None,
        help="Ruta opcional a un repo local de vismatch o image-matching-models. También se puede usar VISMATCH_REPO o IMAGE_MATCHING_MODELS_DIR.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--no_orb_fallback", action="store_true")
    parser.add_argument("--simplex_max_iter", type=int, default=45)
    parser.add_argument("--simplex_max_evals", type=int, default=140)
    parser.add_argument("--simplex_score_tol", type=float, default=1e-3)
    args = parser.parse_args()

    obs_time = datetime.fromisoformat(args.obs_time)
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=timezone.utc)

    best, (coarse, fine, refine) = search_best_yaw_pitch_roll(
        real_image_path=args.real_image,
        search_output_dir=args.search_output_dir,
        tle_dir=args.tle_dir,
        texture_path=args.texture_path,
        obs_time=obs_time,
        focal_length=args.focal_length,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        pixel_width=args.pixel_width,
        pixel_height=args.pixel_height,
        yaw_range=(args.yaw_min, args.yaw_max),
        pitch_range=(args.pitch_min, args.pitch_max),
        roll_range=(args.roll_min, args.roll_max),
        coarse_steps=(args.coarse_yaw_step, args.coarse_pitch_step, args.coarse_roll_step),
        fine_steps=(args.fine_yaw_step, args.fine_pitch_step, args.fine_roll_step),
        orientation_mode=args.orientation_mode,
        metrics_jsonl=args.metrics_jsonl,
        matcher_backend=args.matcher_backend,
        matcher_model=args.matcher_model,
        matching_repo=args.matching_repo,
        device=args.device,
        no_orb_fallback=args.no_orb_fallback,
        simplex_max_iter=args.simplex_max_iter,
        simplex_max_evals=args.simplex_max_evals,
        simplex_score_tol=args.simplex_score_tol,
    )

    print("\nMejor orientación encontrada:")
    print(best)
