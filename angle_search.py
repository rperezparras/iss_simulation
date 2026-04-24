import os
import sys
import math
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def extract_mkpts_from_result(result: Optional[dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if result is None:
        return None, None

    for k0, k1 in [
        ("mkpts0", "mkpts1"),
        ("m_kpts0", "m_kpts1"),
        ("matched_kpts0", "matched_kpts1"),
    ]:
        if k0 in result and k1 in result:
            a = squeeze_points(to_numpy(result[k0]))
            b = squeeze_points(to_numpy(result[k1]))
            return a, b

    for k0, k1 in [
        ("kpts0", "kpts1"),
        ("keypoints0", "keypoints1"),
    ]:
        if k0 in result and k1 in result:
            pts0 = squeeze_points(to_numpy(result[k0]))
            pts1 = squeeze_points(to_numpy(result[k1]))

            if "matches" in result:
                m = to_numpy(result["matches"])
                if m is not None:
                    m = np.asarray(m).astype(np.int64)
                    if m.ndim == 2 and m.shape[1] == 2:
                        return pts0[m[:, 0]], pts1[m[:, 1]]

            if "matches0" in result:
                m0 = to_numpy(result["matches0"])
                if m0 is not None:
                    m0 = np.asarray(m0).astype(np.int64)
                    valid = m0 >= 0
                    return pts0[valid], pts1[m0[valid]]

            if "matches1" in result:
                m1 = to_numpy(result["matches1"])
                if m1 is not None:
                    m1 = np.asarray(m1).astype(np.int64)
                    valid = m1 >= 0
                    return pts0[m1[valid]], pts1[valid]

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
    model_name: str = "superpoint-lg"
    device: Optional[str] = None
    matching_repo: Optional[str] = None
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
    coarse_steps: Tuple[float, float, float] = (20.0, 10.0, 5.0)
    fine_steps: Tuple[float, float, float] = (5.0, 2.5, 1.0)
    fine_windows: Tuple[float, float, float] = (15.0, 7.5, 4.0)
    refine_steps: Tuple[float, float, float] = (2.0, 1.0, 0.5)
    refine_windows: Tuple[float, float, float] = (4.0, 2.0, 1.0)
    topk_coarse: int = 5
    topk_fine: int = 3


# -------------------------------------------------------------------
# MATCHER FACTORY
# -------------------------------------------------------------------

def create_matcher(cfg: MatcherConfig):
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    matching_repo = cfg.matching_repo
    if matching_repo is None:
        candidates = [
            os.environ.get("IMAGE_MATCHING_MODELS_DIR", ""),
            "/home/rpz/image-matching-models",
            "/home/raul/image-matching-models",
        ]
        for c in candidates:
            if c and Path(c).exists():
                matching_repo = c
                break

    if matching_repo and Path(matching_repo).exists() and matching_repo not in sys.path:
        sys.path.insert(0, matching_repo)

    try:
        from matching import get_matcher  # type: ignore
        matcher = get_matcher(cfg.model_name, device=device)
        return matcher, device
    except Exception:
        return None, device


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
        sim_t = image_loader(img_sim_path, resize=(work_h, work_w)).to(device)
        real_t = image_loader(img_real_path, resize=(work_h, work_w)).to(device)
        with torch.inference_mode():
            try:
                result = matcher(sim_t, real_t)
            except Exception:
                result = matcher(sim_t.unsqueeze(0), real_t.unsqueeze(0))
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
# BÚSQUEDA DE ORIENTACIÓN
# -------------------------------------------------------------------

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

    metrics_cache: Dict[Tuple[float, float, float], Dict[str, object]] = {}
    coarse_results: List[Dict[str, object]] = []
    fine_results: List[Dict[str, object]] = []
    refine_results: List[Dict[str, object]] = []

    yaw_step_c, pitch_step_c, roll_step_c = search_cfg.coarse_steps
    print("Comenzando búsqueda gruesa de yaw/pitch/roll...")
    for yaw in frange(search_cfg.yaw_range[0], search_cfg.yaw_range[1], yaw_step_c):
        for pitch in frange(search_cfg.pitch_range[0], search_cfg.pitch_range[1], pitch_step_c):
            for roll in frange(search_cfg.roll_range[0], search_cfg.roll_range[1], roll_step_c):
                info = evaluate_candidate(
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
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
                if info is None:
                    continue
                coarse_results.append(info)

    if not coarse_results:
        return None, ([], [], [])

    coarse_results = sorted(coarse_results, key=lambda d: float(d["score"]), reverse=True)
    top_coarse = coarse_results[: search_cfg.topk_coarse]

    print("Top coarse:")
    for r in top_coarse[:5]:
        print(
            f"  yaw={r['yaw']:.2f}, pitch={r['pitch']:.2f}, roll={r['roll']:.2f}, "
            f"score={r['score']:.4f}, cv={r['cv_rmse']:.4f}, disp={r['disp_rmse']:.4f}, "
            f"area={r['log_area_rmse']:.4f}, aniso={r['aniso_rmse']:.4f}, cov={r['coverage']:.3f}"
        )

    yaw_win_f, pitch_win_f, roll_win_f = search_cfg.fine_windows
    yaw_step_f, pitch_step_f, roll_step_f = search_cfg.fine_steps
    print("Comenzando búsqueda fina alrededor de los mejores coarse...")
    visited_fine = set()
    for center in top_coarse:
        for yaw in frange(center["yaw"] - yaw_win_f, center["yaw"] + yaw_win_f, yaw_step_f):
            for pitch in frange(center["pitch"] - pitch_win_f, center["pitch"] + pitch_win_f, pitch_step_f):
                for roll in frange(center["roll"] - roll_win_f, center["roll"] + roll_win_f, roll_step_f):
                    key = (round(float(yaw), 6), round(float(pitch), 6), round(float(roll), 6))
                    if key in visited_fine:
                        continue
                    visited_fine.add(key)
                    info = evaluate_candidate(
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
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
                    if info is None:
                        continue
                    fine_results.append(info)

    fine_results = sorted(fine_results, key=lambda d: float(d["score"]), reverse=True)
    top_fine = fine_results[: search_cfg.topk_fine] if fine_results else top_coarse[: search_cfg.topk_fine]

    yaw_win_r, pitch_win_r, roll_win_r = search_cfg.refine_windows
    yaw_step_r, pitch_step_r, roll_step_r = search_cfg.refine_steps
    print("Comenzando refinamiento final...")
    visited_refine = set()
    for center in top_fine:
        for yaw in frange(center["yaw"] - yaw_win_r, center["yaw"] + yaw_win_r, yaw_step_r):
            for pitch in frange(center["pitch"] - pitch_win_r, center["pitch"] + pitch_win_r, pitch_step_r):
                for roll in frange(center["roll"] - roll_win_r, center["roll"] + roll_win_r, roll_step_r):
                    key = (round(float(yaw), 6), round(float(pitch), 6), round(float(roll), 6))
                    if key in visited_refine:
                        continue
                    visited_refine.add(key)
                    info = evaluate_candidate(
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
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
                    if info is None:
                        continue
                    refine_results.append(info)

    all_valid = coarse_results + fine_results + refine_results
    all_valid = sorted(all_valid, key=lambda d: float(d["score"]), reverse=True)
    best = all_valid[0] if all_valid else None
    return best, (coarse_results, fine_results, refine_results)


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
):
    search_cfg = SearchConfig(
        yaw_range=tuple(map(float, yaw_range)),
        pitch_range=tuple(map(float, pitch_range)),
        roll_range=tuple(map(float, roll_range)),
        coarse_steps=tuple(map(float, coarse_steps)),
        fine_steps=tuple(map(float, fine_steps)),
        fine_windows=tuple(map(float, fine_windows)),
        refine_steps=tuple(map(float, refine_steps)),
        refine_windows=tuple(map(float, refine_windows)),
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
        metrics_jsonl=metrics_jsonl,
    )


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
    )

    print("\nMejor orientación encontrada:")
    print(best)
