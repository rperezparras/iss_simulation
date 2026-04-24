import os
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as tfm
from skimage.metrics import structural_similarity as ssim

# -------------------------------------------------------------------
# IMPORTS DE TUS LIBRERÍAS
# -------------------------------------------------------------------

# Modelos de matching (ruta de tus modelos)
import sys
sys.path.append("/home/raul/image-matching-models")
from matching import get_matcher  # type: ignore

# Tu librería de simulación (ya limpia) en scripts_v3
from .iss_simulation import (
    reset_scene,
    list_tle_files,
    read_tle_from_files,
    find_closest_tle,
    get_iss_position_and_velocity,
    creaimagen,
)

# -------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL DEL MATCHER
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
matcher = get_matcher("superpoint-lg", device=device)


# -------------------------------------------------------------------
# FUNCIONES AUXILIARES DE IMAGEN
# -------------------------------------------------------------------

def image_loader(path, resize=None):
    """Carga una imagen como tensor (C,H,W) en [0,1] para el matcher."""
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = tfm.Resize(resize, antialias=True)(tfm.ToTensor()(img))
    else:
        img = tfm.ToTensor()(img)
    return img


def calculate_ssim_gray(imageA, imageB):
    """
    Calcula SSIM entre dos imágenes BGR (OpenCV) en escala de grises.
    Asume que tienen el mismo tamaño.
    """
    if imageA.shape != imageB.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño para calcular SSIM.")
    imgA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imgB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    return ssim(imgA_gray, imgB_gray)


def evaluate_pair(sim_path, real_path):
    """
    Evalúa qué tan parecida es una imagen simulada a una real:

    - Calcula SSIM en gris.
    - Ejecuta el matcher para obtener nº de inliers aproximado.
    - Devuelve un score combinado = SSIM + α * n_inliers.
    """
    img_sim = cv2.imread(sim_path, cv2.IMREAD_COLOR)
    img_real = cv2.imread(real_path, cv2.IMREAD_COLOR)

    if img_sim is None or img_real is None:
        return None

    # Redimensionar real a la resolución de la simulada si hace falta
    h_s, w_s = img_sim.shape[:2]
    h_r, w_r = img_real.shape[:2]
    if (h_s, w_s) != (h_r, w_r):
        img_real = cv2.resize(img_real, (w_s, h_s), interpolation=cv2.INTER_AREA)

    ssim_value = calculate_ssim_gray(img_sim, img_real)

    # Matcher (trabaja con tensores)
    img_sim_tensor = image_loader(sim_path)
    img_real_tensor = image_loader(real_path)

    with torch.no_grad():
        result = matcher(img_sim_tensor, img_real_tensor)

    mkpts0 = result.get("mkpts0", None)
    mkpts1 = result.get("mkpts1", None)

    if mkpts0 is None or mkpts1 is None:
        n_inliers = 0
    else:
        n_inliers = mkpts0.shape[0]

    # Score combinado (ajústalo si quieres dar más peso a unos u otros)
    score = float(ssim_value) + 0.0005 * float(n_inliers)

    return {
        "ssim": float(ssim_value),
        "n_inliers": int(n_inliers),
        "score": score,
    }


# -------------------------------------------------------------------
# ESTADO DE LA ISS Y RENDER SIMULADO
# -------------------------------------------------------------------

def get_iss_state(tle_data, time, orientation_mode="forward"):
    """
    Lat, lon, alt (km) y vector velocidad (km/s) adecuado para la orientación.

    - forward: usa v_itrs (ECEF/ITRS) -> estable respecto a la Tierra
    - north  : no depende de la velocidad, pero devolvemos v_itrs por consistencia
    """
    sat = find_closest_tle(tle_data, time)

    lat, lon, alt, v_icrf, v_itrs = get_iss_position_and_velocity(sat, time)

    if orientation_mode == "forward":
        vel = v_itrs
    else:
        vel = v_itrs  # o v_icrf, da igual si no lo usas

    return lat, lon, alt, vel

def render_simulated_image(
    lat,
    lon,
    alt,
    vel,
    yaw,
    pitch,
    roll,
    time_str,
    output_dir,
    sphere,
    focal_length,
    sensor_width,
    sensor_height,
    pixel_width,
    pixel_height,
    earth_radius,
    orientation_mode="north",
):
    """
    Llama a creaimagen y devuelve el path de la imagen simulada.
    """
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
# BÚSQUEDA COARSE-TO-FINE DE YAW / PITCH
# -------------------------------------------------------------------

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
    orientation_mode="north",
    earth_radius=10,
):
    """
    Busca los ángulos yaw y pitch que mejor reproducen una imagen real
    mediante:

      1) Búsqueda gruesa en grid yaw/pitch.
      2) Búsqueda fina alrededor del mejor resultado grueso.

    Parámetros clave:
    -----------------
    real_image_path : str o Path
        Ruta de la imagen real de referencia.
    search_output_dir : str o Path
        Carpeta donde se guardarán las imágenes simuladas generadas durante la búsqueda.
    tle_dir : str o Path
        Carpeta que contiene los ficheros TLE.
    texture_path : str o Path
        Textura nocturna de la Tierra usada por iss_simulation.reset_scene().
    obs_time : datetime
        Momento de la imagen real (en UTC, idealmente).
    orientation_mode : 'north' o 'forward'
        Convención de orientación de la cámara.
    yaw_range, pitch_range : tuplas (min, max)
        Rango de búsqueda en yaw y pitch (en grados).
    coarse_step, fine_step, fine_window :
        Parámetros de resolución de la búsqueda gruesa y fina.

    Devuelve:
    ---------
    best_fine : dict
        {'yaw': ..., 'pitch': ..., 'score': ...}
    (results_coarse, results_fine) : (list, list)
        Listas de diccionarios con info de todas las combinaciones evaluadas.
    """
    real_image_path = Path(real_image_path)
    search_output_dir = Path(search_output_dir)
    tle_dir = Path(tle_dir)
    texture_path = Path(texture_path)

    search_output_dir.mkdir(parents=True, exist_ok=True)

    # 0) Preparar escena y TLEs
    sphere = reset_scene(earth_radius, str(texture_path))
    file_paths = list_tle_files(str(tle_dir))
    tle_data = read_tle_from_files(file_paths)

    # 1) Estado de la ISS (lat/lon/alt/vel) fijo para esta foto
    lat, lon, alt, vel = get_iss_state(tle_data, obs_time, orientation_mode=orientation_mode)
    time_str = obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    results_coarse = []
    best = {"score": -1e9, "yaw": None, "pitch": None}

    # ---------- BÚSQUEDA GRUESA ----------
    print("Comenzando búsqueda gruesa...")
    for yaw in np.arange(yaw_range[0], yaw_range[1] + 1e-6, coarse_step):
        for pitch in np.arange(pitch_range[0], pitch_range[1] + 1e-6, coarse_step):
            sim_path = render_simulated_image(
                lat,
                lon,
                alt,
                vel,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                time_str=time_str,
                output_dir=search_output_dir,
                sphere=sphere,
                focal_length=focal_length,
                sensor_width=sensor_width,
                sensor_height=sensor_height,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                earth_radius=earth_radius,
                orientation_mode=orientation_mode,
            )

            metrics = evaluate_pair(str(sim_path), str(real_image_path))
            if metrics is None:
                continue

            info = {"yaw": float(yaw), "pitch": float(pitch), **metrics}
            results_coarse.append(info)

            if metrics["score"] > best["score"]:
                best = {"score": metrics["score"], "yaw": float(yaw), "pitch": float(pitch)}
                print(
                    f"[GRUESO] Nuevo mejor: yaw={yaw}, pitch={pitch}, "
                    f"score={metrics['score']:.4f}, "
                    f"SSIM={metrics['ssim']:.4f}, inliers={metrics['n_inliers']}"
                )

    if best["yaw"] is None:
        print("No se encontró ningún resultado válido en la búsqueda gruesa.")
        return None, (results_coarse, [])

    # ---------- BÚSQUEDA FINA ----------
    print("Comenzando búsqueda fina alrededor del mejor resultado grueso...")

    yaw_center = best["yaw"]
    pitch_center = best["pitch"]

    yaw_min = yaw_center - fine_window
    yaw_max = yaw_center + fine_window
    pitch_min = pitch_center - fine_window
    pitch_max = pitch_center + fine_window

    results_fine = []
    best_fine = dict(best)  # copia

    for yaw in np.arange(yaw_min, yaw_max + 1e-6, fine_step):
        for pitch in np.arange(pitch_min, pitch_max + 1e-6, fine_step):
            sim_path = render_simulated_image(
                lat,
                lon,
                alt,
                vel,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                time_str=time_str,
                output_dir=search_output_dir,
                sphere=sphere,
                focal_length=focal_length,
                sensor_width=sensor_width,
                sensor_height=sensor_height,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
                earth_radius=earth_radius,
                orientation_mode=orientation_mode,
            )

            metrics = evaluate_pair(str(sim_path), str(real_image_path))
            if metrics is None:
                continue

            info = {"yaw": float(yaw), "pitch": float(pitch), **metrics}
            results_fine.append(info)

            if metrics["score"] > best_fine["score"]:
                best_fine = {"score": metrics["score"], "yaw": float(yaw), "pitch": float(pitch)}
                print(
                    f"[FINO] Nuevo mejor: yaw={yaw}, pitch={pitch}, "
                    f"score={metrics['score']:.4f}, "
                    f"SSIM={metrics['ssim']:.4f}, inliers={metrics['n_inliers']}"
                )

    return best_fine, (results_coarse, results_fine)


# -------------------------------------------------------------------
# MAIN PARA PRUEBAS (OPCIONAL)
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Búsqueda automática de yaw/pitch para una imagen ISS."
    )
    parser.add_argument("--real_image", required=True, help="Ruta a la imagen real (JPG).")
    parser.add_argument(
        "--search_output_dir",
        required=True,
        help="Carpeta donde guardar las simulaciones usadas en la búsqueda.",
    )
    parser.add_argument(
        "--tle_dir",
        default="/home/raul/planb/ISS_Simulation/ISS_tle",
        help="Carpeta con los TLE de la ISS.",
    )
    parser.add_argument(
        "--texture_path",
        default="/home/raul/planb/ISS_Simulation/VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.sqrt.full.40k_20k.png",
        help="Textura nocturna para la simulación.",
    )
    parser.add_argument(
        "--obs_time",
        required=True,
        help="Fecha/hora de la foto real en formato ISO (p.ej. 2022-07-25T21:28:15). Se asume UTC.",
    )
    parser.add_argument("--earth_radius", type=float, default=10.0)

    parser.add_argument("--focal_length", type=float, default=28.0)
    parser.add_argument("--sensor_width", type=float, default=36.0)
    parser.add_argument("--sensor_height", type=float, default=28.0)
    parser.add_argument("--pixel_width", type=int, default=5568)
    parser.add_argument("--pixel_height", type=int, default=3712)

    parser.add_argument("--yaw_min", type=float, default=-180)
    parser.add_argument("--yaw_max", type=float, default=180)
    parser.add_argument("--pitch_min", type=float, default=30)
    parser.add_argument("--pitch_max", type=float, default=90)
    parser.add_argument("--coarse_step", type=float, default=10)
    parser.add_argument("--fine_step", type=float, default=5)
    parser.add_argument("--fine_window", type=float, default=15)

    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument(
        "--orientation_mode",
        choices=["north", "forward"],
        default="north",
    )

    args = parser.parse_args()

    obs_time = datetime.fromisoformat(args.obs_time).replace(tzinfo=timezone.utc)

    best_angles, (coarse_res, fine_res) = search_best_yaw_pitch(
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
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        fine_window=args.fine_window,
        roll=args.roll,
        orientation_mode=args.orientation_mode,
        earth_radius=args.earth_radius,
    )

    print("\nMejores ángulos encontrados:")
    print(best_angles)
