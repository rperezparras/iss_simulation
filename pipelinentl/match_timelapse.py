#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empareja y alinea imagenes simuladas (output) y reales (pics) de un timelapse ISS.

Backends soportados:
  1) vismatch actual, recomendado para instalaciones nuevas:
       pip install vismatch
       modelo por defecto: superpoint-lightglue

  2) image-matching-models antiguo, para instalaciones legacy:
       export IMAGE_MATCHING_MODELS_DIR=/ruta/a/image-matching-models
       modelo por defecto: superpoint-lg

El script:
- Intenta cargar el matcher seleccionado.
- Extrae matches aunque cambien las keys del diccionario devuelto.
- Usa ORB+RANSAC como fallback si el matcher falla o devuelve pocos matches.
- Ajusta una PolynomialTransform entre imagen real y simulada.
- Genera una rejilla en la imagen real y la proyecta a la simulada.
- Guarda CSV con pares (sim_x, sim_y, real_x, real_y).
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFile, ImageOps
import torchvision.transforms as tfm

from skimage.metrics import structural_similarity as ssim
from skimage.transform import PolynomialTransform

import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------
# Helpers generales
# ----------------------------

def invert_y_coordinate(y: float, height: int) -> float:
    """Convierte y (origen arriba) -> y' (origen abajo)."""
    return float(height) - float(y)


def to_numpy(x: Any):
    """Convierte torch/numpy/list a np.ndarray."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach()
        if x.is_cuda:
            x = x.cpu()
        return x.numpy()
    return np.array(x)


def squeeze_points(arr: np.ndarray | None) -> np.ndarray | None:
    """Asegura shape (N, 2) cuando sea posible."""
    if arr is None:
        return None

    arr = np.asarray(arr)

    # Ejemplos habituales:
    #   (1, N, 2) -> (N, 2)
    #   (N, 2)    -> (N, 2)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    arr = np.squeeze(arr)

    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    return arr


def sanitize_matched_points(
    mkpts0: np.ndarray | None,
    mkpts1: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def image_loader(path: str, resize=None) -> torch.Tensor:
    """
    Carga una imagen como tensor float32 en [0, 1], shape (3, H, W).
    Aplica EXIF transpose.
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        # Fallback con cv2 si PIL se queja
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

    if resize is not None:
        # resize = (H, W)
        img = img.resize((int(resize[1]), int(resize[0])), resample=Image.BILINEAR)

    return tfm.ToTensor()(img)  # (3, H, W), float32 [0, 1]


# ----------------------------
# Backend de matching
# ----------------------------

def resolve_matching_repo(user_repo: str | None = None) -> str | None:
    """
    Devuelve una ruta local al repo de matching si el usuario la ha indicado.

    Soporta:
      - vismatch actual: repo con paquete ./vismatch
      - image-matching-models antiguo: repo con paquete ./matching

    Si vismatch esta instalado por pip, no hace falta devolver ninguna ruta.
    No usa rutas personales tipo /home/usuario/... para que el codigo sea portable.
    """
    candidates: list[str] = []

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

        # Si el usuario lo ha pasado explicitamente por CLI, se acepta igualmente.
        if user_repo:
            return str(path)

    return None


def add_matching_repo_to_syspath(repo_path: str | None) -> None:
    """Anade el repo local al sys.path si se ha indicado y existe."""
    if not repo_path:
        return

    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        return

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"Repo local de matching anadido a sys.path: {path_str}")


def load_matcher_backend(
    device: str,
    matcher_backend: str = "auto",
    matcher_model: str | None = None,
    matching_repo: str | None = None,
):
    """
    Carga un matcher y devuelve (matcher, backend_name, model_name).

    matcher_backend:
      - auto: prueba vismatch y despues matching antiguo.
      - vismatch: fuerza vismatch.
      - matching: fuerza image-matching-models antiguo.
      - none: no carga matcher; se usara ORB fallback.
    """
    if matcher_backend == "none":
        print("Matcher neural desactivado por --matcher_backend none. Se usara ORB fallback.")
        return None, "none", None

    repo_path = resolve_matching_repo(matching_repo)
    add_matching_repo_to_syspath(repo_path)

    backends = [matcher_backend]
    if matcher_backend == "auto":
        backends = ["vismatch", "matching"]

    errors: list[str] = []

    for backend in backends:
        if backend == "vismatch":
            model_name = matcher_model or "superpoint-lightglue"
            try:
                from vismatch import get_matcher  # type: ignore
                import vismatch as vismatch_mod  # type: ignore

                print(f"vismatch importado desde: {vismatch_mod.__file__}")
                matcher = get_matcher(model_name, device=device)
                print(f"Matcher cargado: backend=vismatch | model={model_name}")
                return matcher, "vismatch", model_name
            except Exception as e:
                errors.append(f"vismatch({model_name}): {repr(e)}")
                continue

        if backend == "matching":
            model_name = matcher_model or "superpoint-lg"
            try:
                from matching import get_matcher  # type: ignore
                import matching as matching_mod  # type: ignore

                print(f"matching importado desde: {matching_mod.__file__}")
                matcher = get_matcher(model_name, device=device)
                print(f"Matcher cargado: backend=matching | model={model_name}")
                return matcher, "matching", model_name
            except Exception as e:
                errors.append(f"matching({model_name}): {repr(e)}")
                continue

    print("No se pudo cargar ningun matcher neural.")
    for err in errors:
        print(f"   - {err}")
    print("Se usara ORB fallback en todos los frames.")
    return None, "none", None


def load_image_for_matcher(matcher, path: Path, device: str) -> Any:
    """
    Carga imagen para el matcher.

    vismatch actual suele exponer matcher.load_image(path). Si no existe,
    usamos el loader tensorial clasico compatible con image-matching-models.
    """
    if matcher is not None and hasattr(matcher, "load_image"):
        img = matcher.load_image(str(path))
        if torch.is_tensor(img):
            img = img.to(device)
        return img

    return image_loader(str(path)).to(device)


def run_matcher(matcher, img0, img1):
    """
    Ejecuta matcher soportando wrappers que esperan (3,H,W) o (1,3,H,W).
    """
    with torch.inference_mode():
        try:
            return matcher(img0, img1)
        except Exception as first_error:
            # Fallback para wrappers que requieren batch dimension.
            if torch.is_tensor(img0) and torch.is_tensor(img1) and img0.ndim == 3 and img1.ndim == 3:
                return matcher(img0.unsqueeze(0), img1.unsqueeze(0))
            raise first_error


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
    if result is None:
        return False
    if isinstance(result, dict):
        return key in result
    return hasattr(result, key)


def extract_mkpts_from_result(result: Any):
    """
    Soporta multiples formatos de salida del matcher.

    Devuelve:
      mkpts0 (simulada) np.ndarray (N, 2)
      mkpts1 (real)     np.ndarray (N, 2)

    Convencion de llamada del script:
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

    # Formatos con keypoints + indices de matches.
    keypoint_pairs = [
        ("kpts0", "kpts1"),
        ("keypoints0", "keypoints1"),
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
                    m = m[valid]
                    if len(m) > 0:
                        mk0 = pts0[m[:, 0]]
                        mk1 = pts1[m[:, 1]]
                        mk0, mk1 = sanitize_matched_points(mk0, mk1)
                        if mk0 is not None and mk1 is not None:
                            return mk0, mk1

        # matches0: array (N0,) con indice en pts1 o -1.
        if result_has(result, "matches0"):
            m0 = to_numpy(result_get(result, "matches0"))
            if m0 is not None:
                m0 = np.asarray(m0)
                if m0.ndim == 2 and m0.shape[0] == 1:
                    m0 = m0[0]
                m0 = np.squeeze(m0).astype(np.int64)
                if m0.ndim == 1:
                    n0 = min(len(m0), len(pts0))
                    m0 = m0[:n0]
                    valid = (m0 >= 0) & (m0 < len(pts1))
                    if valid.any():
                        mk0 = pts0[:n0][valid]
                        mk1 = pts1[m0[valid]]
                        mk0, mk1 = sanitize_matched_points(mk0, mk1)
                        if mk0 is not None and mk1 is not None:
                            return mk0, mk1

        # matches1: array (N1,) con indice en pts0 o -1.
        if result_has(result, "matches1"):
            m1 = to_numpy(result_get(result, "matches1"))
            if m1 is not None:
                m1 = np.asarray(m1)
                if m1.ndim == 2 and m1.shape[0] == 1:
                    m1 = m1[0]
                m1 = np.squeeze(m1).astype(np.int64)
                if m1.ndim == 1:
                    n1 = min(len(m1), len(pts1))
                    m1 = m1[:n1]
                    valid = (m1 >= 0) & (m1 < len(pts0))
                    if valid.any():
                        mk1 = pts1[:n1][valid]
                        mk0 = pts0[m1[valid]]
                        mk0, mk1 = sanitize_matched_points(mk0, mk1)
                        if mk0 is not None and mk1 is not None:
                            return mk0, mk1

    return None, None


# ----------------------------
# Fallback ORB
# ----------------------------

def orb_fallback_matches(img0_bgr, img1_bgr, max_matches=2000):
    """
    Fallback CPU: ORB + ratio test + RANSAC para filtrar inliers.

    Devuelve:
      mkpts0 (simulada), mkpts1 (real) en pixeles.
    """
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
    for pair in knn:
        if len(pair) != 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)

    if len(good) < 20:
        return None, None

    good = sorted(good, key=lambda m: m.distance)[:max_matches]

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])  # sim
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])  # real

    # Filtrado RANSAC con homografia real -> sim.
    H, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, 3.0)
    if mask is None:
        return None, None

    mask = mask.ravel().astype(bool)
    pts0_in = pts0[mask]
    pts1_in = pts1[mask]

    if len(pts0_in) < 20:
        return None, None

    return pts0_in, pts1_in


# ----------------------------
# Transformacion y rejilla
# ----------------------------

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


def apply_polynomial_transform(
    matches_src,
    matches_dst,
    grid_points,
    src_width,
    src_height,
    dst_width,
    dst_height,
    order=2,
):
    """
    Ajusta transformacion polinomica en coordenadas normalizadas.

    matches_src: puntos en imagen real.
    matches_dst: puntos correspondientes en imagen simulada.
    grid_points: rejilla definida en imagen real.
    """
    src_n = normalize_points(matches_src, src_width, src_height)
    dst_n = normalize_points(matches_dst, dst_width, dst_height)
    grid_n = normalize_points(grid_points, src_width, src_height)

    poly_transform = PolynomialTransform()
    ok = poly_transform.estimate(src_n, dst_n, order=order)

    if not ok:
        return None, None

    transformed_n = poly_transform(grid_n)
    transformed_points = denormalize_points(transformed_n, dst_width, dst_height)

    return poly_transform, transformed_points


def generate_grid_mesh(img_width, img_height, step=100, min_points=30):
    """
    Genera una rejilla aproximadamente regular que cubre toda la imagen.

    Usa linspace para:
      - llegar exactamente a los bordes;
      - evitar una ultima fila/columna mucho mas pequena;
      - devolver nx, ny para poder dibujar la rejilla como malla.
    """
    img_width = int(img_width)
    img_height = int(img_height)
    step = max(1, int(step))

    nx = max(2, int(round((img_width - 1) / step)) + 1)
    ny = max(2, int(round((img_height - 1) / step)) + 1)

    while nx * ny < min_points:
        if img_width / nx >= img_height / ny:
            nx += 1
        else:
            ny += 1

    xs = np.linspace(0, img_width - 1, nx, dtype=np.float64)
    ys = np.linspace(0, img_height - 1, ny, dtype=np.float64)

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points, nx, ny


def generate_grid_points(img_width, img_height, step=100, min_points=30):
    """Compatibilidad con codigo anterior: devuelve solo los puntos."""
    points, _, _ = generate_grid_mesh(
        img_width,
        img_height,
        step=step,
        min_points=min_points,
    )
    return points


# ----------------------------
# Plots diagnosticos
# ----------------------------

def draw_grid(ax, grid_points, nx, ny, color="lime", lw=0.9, alpha=0.95):
    """Dibuja una rejilla a partir de puntos ordenados como meshgrid."""
    grid = np.asarray(grid_points).reshape(ny, nx, 2)

    for iy in range(ny):
        pts = grid[iy, :, :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha)

    for ix in range(nx):
        pts = grid[:, ix, :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha)


def save_grid_overlay(
    image_bgr,
    grid_points,
    nx,
    ny,
    title,
    save_path,
    color="lime",
):
    """Guarda una imagen con una rejilla encima."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb, origin="upper")
    draw_grid(ax, grid_points, nx, ny, color=color)

    ax.set_title(title)
    ax.set_xlim(0, image_bgr.shape[1])
    ax.set_ylim(image_bgr.shape[0], 0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def draw_matches_colorful(
    img_real_bgr,
    img_sim_bgr,
    mkpts_real,
    mkpts_sim,
    title,
    save_path,
    max_matches=150,
    seed=1234,
):
    """
    Guarda un plot lado a lado:
      izquierda = imagen real
      derecha   = imagen simulada

    Dibuja matches inliers real -> simulada con colores visuales.
    Los colores no codifican calidad; solo ayudan a distinguir lineas.
    """
    real_rgb = cv2.cvtColor(img_real_bgr, cv2.COLOR_BGR2RGB)
    sim_rgb = cv2.cvtColor(img_sim_bgr, cv2.COLOR_BGR2RGB)

    h_real, w_real = real_rgb.shape[:2]
    h_sim, w_sim = sim_rgb.shape[:2]

    mkpts_real_plot_all = np.asarray(mkpts_real, dtype=np.float64).copy()
    mkpts_sim_plot_all = np.asarray(mkpts_sim, dtype=np.float64).copy()

    # Para visualizar lado a lado, si las alturas difieren, redimensionamos solo el plot.
    if h_real != h_sim:
        scale = h_real / max(h_sim, 1)
        new_w = int(round(w_sim * scale))
        sim_rgb = cv2.resize(sim_rgb, (new_w, h_real), interpolation=cv2.INTER_AREA)
        mkpts_sim_plot_all[:, 0] *= scale
        mkpts_sim_plot_all[:, 1] *= scale
        h_sim, w_sim = sim_rgb.shape[:2]

    n = len(mkpts_real_plot_all)
    if n == 0:
        return

    if n > max_matches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_matches, replace=False)
        mkpts_real_plot = mkpts_real_plot_all[idx]
        mkpts_sim_plot = mkpts_sim_plot_all[idx]
    else:
        mkpts_real_plot = mkpts_real_plot_all
        mkpts_sim_plot = mkpts_sim_plot_all

    canvas = np.concatenate([real_rgb, sim_rgb], axis=1)
    offset_x = w_real

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)

    cmap = plt.get_cmap("turbo")
    colors = cmap(np.linspace(0, 1, len(mkpts_real_plot)))

    for p_real, p_sim, color in zip(mkpts_real_plot, mkpts_sim_plot, colors):
        x0, y0 = float(p_real[0]), float(p_real[1])
        x1, y1 = float(p_sim[0]) + offset_x, float(p_sim[1])

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.7, alpha=0.65)
        ax.scatter([x0, x1], [y0, y1], color=color, s=8, alpha=0.85)

    ax.set_title(f"{title} | dibujados {len(mkpts_real_plot)}/{n} inliers")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

def ensure_arg_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Compatibilidad si main(args) se llama desde una pipeline con Namespace antiguo."""
    defaults = {
        "min_grid_points": 30,
        "plot_max_matches": 150,
        "device": None,
        "matching_repo": None,
        "matcher_backend": "auto",
        "matcher_model": None,
        "min_matches": 20,
        "no_orb_fallback": False,
    }

    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)

    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matching ISS simulado-real")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pictures_dir", type=str, required=True)
    parser.add_argument("--matches_output_dir", type=str, required=True)
    parser.add_argument("--grid_step", type=int, default=265)
    parser.add_argument("--show_every", type=int, default=100)
    parser.add_argument(
        "--min_grid_points",
        type=int,
        default=30,
        help="Numero minimo de puntos de rejilla validos deseado.",
    )
    parser.add_argument(
        "--plot_max_matches",
        type=int,
        default=150,
        help="Numero maximo de matches a dibujar en los plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo para el matcher neural. Por defecto: cuda si esta disponible.",
    )
    parser.add_argument(
        "--matching_repo",
        type=str,
        default=None,
        help=(
            "Ruta opcional a un repo local de vismatch o image-matching-models. "
            "Tambien puede definirse con VISMATCH_REPO, VISMATCH_DIR o "
            "IMAGE_MATCHING_MODELS_DIR."
        ),
    )
    parser.add_argument(
        "--matcher_backend",
        type=str,
        default="auto",
        choices=["auto", "vismatch", "matching", "none"],
        help=(
            "Backend de matching. 'auto' prueba vismatch y despues matching antiguo. "
            "'none' desactiva el matcher neural y usa ORB fallback."
        ),
    )
    parser.add_argument(
        "--matcher_model",
        type=str,
        default=None,
        help=(
            "Nombre del modelo. Por defecto: superpoint-lightglue para vismatch, "
            "superpoint-lg para matching antiguo."
        ),
    )
    parser.add_argument("--min_matches", type=int, default=20)
    parser.add_argument("--no_orb_fallback", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None):
    if args is None:
        args = parse_args()

    args = ensure_arg_defaults(args)

    output_dir = Path(args.output_dir).expanduser()
    pictures_dir = Path(args.pictures_dir).expanduser()
    matches_output_dir = Path(args.matches_output_dir).expanduser()
    matches_output_dir.mkdir(parents=True, exist_ok=True)

    output_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("render_output_") and f.lower().endswith(".png")
    )
    pictures_files = sorted(
        f for f in os.listdir(pictures_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )

    if not output_files:
        print(f"No se encontraron simuladas en {output_dir}")
        return
    if not pictures_files:
        print(f"No se encontraron reales en {pictures_dir}")
        return

    print(f"N de simuladas: {len(output_files)}")
    print(f"N de reales:    {len(pictures_files)}")

    if len(output_files) != len(pictures_files):
        print(
            "Aviso: el numero de simuladas y reales no coincide. "
            "Se procesaran pares por orden hasta el minimo comun."
        )

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA no disponible, usando CPU.")
            device = "cpu"

    print(f"torch: {torch.__version__} | device: {device}")

    matcher, backend_name, model_name = load_matcher_backend(
        device=device,
        matcher_backend=args.matcher_backend,
        matcher_model=args.matcher_model,
        matching_repo=args.matching_repo,
    )

    # Loop pares por orden
    for idx, (img0_file, img1_file) in enumerate(zip(output_files, pictures_files)):
        img0_path = output_dir / img0_file    # simulada
        img1_path = pictures_dir / img1_file  # real

        img0_bgr = cv2.imread(str(img0_path), cv2.IMREAD_COLOR)
        img1_bgr = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        if img0_bgr is None or img1_bgr is None:
            print(f"Frame {idx}: no se pudo leer una de las imagenes. Se salta.")
            continue

        h0, w0 = img0_bgr.shape[:2]
        h1, w1 = img1_bgr.shape[:2]

        mkpts0 = mkpts1 = None

        # 1) Intento matcher neural
        if matcher is not None:
            try:
                img0_t = load_image_for_matcher(matcher, img0_path, device)
                img1_t = load_image_for_matcher(matcher, img1_path, device)

                result = run_matcher(matcher, img0_t, img1_t)
                mkpts0, mkpts1 = extract_mkpts_from_result(result)
            except Exception as e:
                print(f"Frame {idx}: fallo el matcher neural ({backend_name}/{model_name}): {repr(e)}")
                mkpts0, mkpts1 = None, None

        n_m = 0 if mkpts0 is None else len(mkpts0)

        # 2) Fallback ORB si no hay matches suficientes
        if (mkpts0 is None or mkpts1 is None or n_m < args.min_matches) and not args.no_orb_fallback:
            mkpts0_fb, mkpts1_fb = orb_fallback_matches(img0_bgr, img1_bgr)
            if mkpts0_fb is not None and len(mkpts0_fb) >= args.min_matches:
                mkpts0, mkpts1 = mkpts0_fb, mkpts1_fb
                n_m = len(mkpts0)
                print(f"Frame {idx}: usando ORB fallback ({n_m} inliers)")
            else:
                print(f"Frame {idx}: matches insuficientes ({n_m}). No se genera CSV.")
                continue

        if mkpts0 is None or mkpts1 is None or len(mkpts0) < args.min_matches:
            print(
                f"Frame {idx}: matches insuficientes "
                f"({0 if mkpts0 is None else len(mkpts0)}). No se genera CSV."
            )
            continue

        mkpts0, mkpts1 = sanitize_matched_points(mkpts0, mkpts1)
        if mkpts0 is None or mkpts1 is None or len(mkpts0) < args.min_matches:
            print(f"Frame {idx}: matches no validos tras limpieza. No se genera CSV.")
            continue

        # 3) Filtrado RANSAC tambien para matches del matcher principal
        if len(mkpts0) < 4:
            print(f"Frame {idx}: menos de 4 matches, no se puede estimar homografia.")
            continue

        H, mask = cv2.findHomography(
            mkpts1.astype(np.float32),  # real
            mkpts0.astype(np.float32),  # sim
            cv2.RANSAC,
            3.0,
        )

        if mask is None:
            print(f"Frame {idx}: RANSAC fallo. No se genera CSV.")
            continue

        n_before = len(mkpts0)

        inliers = mask.ravel().astype(bool)
        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]

        n_after = len(mkpts0)
        print(
            f"Frame {idx}: matches={n_before}, "
            f"inliers={n_after}, "
            f"ratio={n_after / max(n_before, 1):.2f}"
        )

        if len(mkpts0) < args.min_matches:
            print(f"Frame {idx}: pocos inliers tras RANSAC ({len(mkpts0)}). No se genera CSV.")
            continue

        # 4) Grid en real
        real_points, grid_nx, grid_ny = generate_grid_mesh(
            w1,
            h1,
            step=args.grid_step,
            min_points=args.min_grid_points,
        )

        # 5) Transform real -> sim (src=real, dst=sim)
        poly_transform, transformed_grid_points = apply_polynomial_transform(
            matches_src=mkpts1,
            matches_dst=mkpts0,
            grid_points=real_points,
            src_width=w1,
            src_height=h1,
            dst_width=w0,
            dst_height=h0,
            order=2,
        )

        if poly_transform is None or transformed_grid_points is None:
            print(f"Frame {idx}: fallo el ajuste polinomico. No se genera CSV.")
            continue

        inside = (
            (transformed_grid_points[:, 0] >= 0) &
            (transformed_grid_points[:, 0] < w0) &
            (transformed_grid_points[:, 1] >= 0) &
            (transformed_grid_points[:, 1] < h0)
        )

        real_points_valid = real_points[inside]
        transformed_grid_points_valid = transformed_grid_points[inside]

        if len(real_points_valid) < args.min_matches:
            print(
                f"Frame {idx}: pocos puntos validos tras transformar "
                f"({len(real_points_valid)}). No se genera CSV."
            )
            continue

        # 6) Guardar CSV
        csv_filename = f"transformed_coordinates_{os.path.splitext(img0_file)[0]}.csv"
        csv_path = matches_output_dir / csv_filename

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sim_x", "sim_y", "real_x", "real_y"])

            for pt0, pt1 in zip(transformed_grid_points_valid, real_points_valid):
                sim_x = float(pt0[0])
                sim_y = float(invert_y_coordinate(pt0[1], h0))
                real_x = float(pt1[0])
                real_y = float(invert_y_coordinate(pt1[1], h1))
                writer.writerow([sim_x, sim_y, real_x, real_y])

        # 7) Plots cada N
        if args.show_every > 0 and idx % args.show_every == 0:
            img0_gray = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
            img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
            try:
                ssim_value = ssim(img1_gray, img0_gray)
            except Exception:
                ssim_value = None

            print(
                f"Procesando par {idx}: {img0_file} vs {img1_file} | "
                f"inliers={len(mkpts0)} | "
                f"grid_valid={len(real_points_valid)}/{len(real_points)} | "
                f"SSIM={ssim_value}"
            )

            # Matches real-sim con colores.
            match_img_path = matches_output_dir / f"matches_color_{idx}.png"
            draw_matches_colorful(
                img_real_bgr=img1_bgr,
                img_sim_bgr=img0_bgr,
                mkpts_real=mkpts1,
                mkpts_sim=mkpts0,
                title=f"Matches real-sim inliers (idx={idx})",
                save_path=str(match_img_path),
                max_matches=args.plot_max_matches,
            )

            # Rejilla regular sobre imagen real.
            real_grid_path = matches_output_dir / f"grid_real_{idx}.png"
            save_grid_overlay(
                image_bgr=img1_bgr,
                grid_points=real_points,
                nx=grid_nx,
                ny=grid_ny,
                title=f"Imagen real + rejilla regular (idx={idx})",
                save_path=str(real_grid_path),
                color="lime",
            )

            # Rejilla deformada sobre imagen simulada.
            sim_grid_path = matches_output_dir / f"grid_sim_deformed_{idx}.png"
            save_grid_overlay(
                image_bgr=img0_bgr,
                grid_points=transformed_grid_points,
                nx=grid_nx,
                ny=grid_ny,
                title=f"Imagen simulada + rejilla real transformada (idx={idx})",
                save_path=str(sim_grid_path),
                color="cyan",
            )

    print("Proceso terminado para todos los pares.")


if __name__ == "__main__":
    main()
