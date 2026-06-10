#!/usr/bin/env python3
"""
Descarga imágenes de la ISS desde el servidor de NASA EOL, en un rango de IDs,
con reintentos automáticos y soporte tanto como script CLI como módulo importable.

Uso como script:
    python -m pipelinentl.get_pics --mission ISS053 --start 462550 --end 462560 --output /ruta/a/pics

Uso como módulo (por ejemplo desde la pipeline):
    from pipelinentl import get_pics
    get_pics.download_all_images("ISS053", 462550, 462560, Path("/ruta/pics"))
"""

import argparse
import time
from pathlib import Path
from typing import List

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Número máximo de descargas concurrentes por defecto
MAX_THREADS = 10


def parse_args() -> argparse.Namespace:
    """
    Parser de argumentos para uso en línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Descarga todas las imágenes de la ISS por rango y misión, reintentando las fallidas"
    )
    parser.add_argument("--mission", required=True, help="Nombre de la misión (ejemplo: ISS053)")
    parser.add_argument("--start", type=int, required=True, help="ID inicial de imagen (entero)")
    parser.add_argument("--end", type=int, required=True, help="ID final de imagen (entero, inclusivo)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directorio de salida (opcional). "
             "Por defecto: ./<mission>-E-<start>-<end>/pics",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=MAX_THREADS,
        help=f"Número máximo de descargas concurrentes (por defecto {MAX_THREADS})",
    )
    return parser.parse_args()


def download_image(
    mission: str,
    image_id: int,
    output_dir: Path,
    retries: int = 3,
    delay: float = 5.0,
    timeout: float = 15.0,
) -> bool:
    """
    Descarga una única imagen de la ISS.

    Parámetros
    ----------
    mission : str
        Nombre de la misión (por ejemplo 'ISS053').
    image_id : int
        ID de la imagen (por ejemplo 462550).
    output_dir : Path
        Directorio donde guardar la imagen.
    retries : int
        Número máximo de reintentos en caso de fallo.
    delay : float
        Tiempo de espera (segundos) entre reintentos.
    timeout : float
        Timeout (segundos) para la petición HTTP.

    Devuelve
    --------
    bool
        True si la descarga fue exitosa, False en caso contrario.
    """
    url = f"https://eol.jsc.nasa.gov/DatabaseImages/ESC/large/{mission}/{mission}-E-{image_id}.JPG"
    base_filename = f"{mission}-E-{image_id}.JPG"
    image_path = output_dir / base_filename

    # Si ya existe, no hace falta descargar de nuevo
    if image_path.exists():
        print(f"Image {image_id} already exists, skipping.")
        return True

    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded image {image_id}")
                return True
            else:
                print(f"Failed to download image {image_id}: HTTP {response.status_code}")
                # Si el código no es 200, no solemos insistir mucho, pero respetamos 'retries'
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"Error downloading image {image_id}: {e}")

        attempt += 1
        if attempt < retries:
            print(f"Retrying download for image {image_id} (attempt {attempt + 1}/{retries})...")
            time.sleep(delay)
        else:
            print(f"Failed to download image {image_id} after {retries} attempts.")

    return False


def download_all_images(
    mission: str,
    start_id: int,
    end_id: int,
    output_dir: Path,
    max_threads: int = MAX_THREADS,
    retries: int = 3,
    delay: float = 5.0,
    timeout: float = 15.0,
) -> List[int]:
    """
    Descarga todas las imágenes de la ISS en el rango [start_id, end_id] para una misión dada.

    Parámetros
    ----------
    mission : str
        Nombre de la misión (por ejemplo 'ISS053').
    start_id : int
        ID inicial (inclusive).
    end_id : int
        ID final (inclusive).
    output_dir : Path
        Directorio donde guardar las imágenes.
    max_threads : int
        Número máximo de descargas concurrentes.
    retries : int
        Número de reintentos por imagen.
    delay : float
        Tiempo de espera (s) entre reintentos para una misma imagen.
    timeout : float
        Timeout (s) de la petición HTTP.

    Devuelve
    --------
    List[int]
        Lista de IDs de imágenes que no se pudieron descargar tras todos los intentos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_to_download = []
    for image_id in range(start_id, end_id + 1):
        image_path = output_dir / f"{mission}-E-{image_id}.JPG"
        if image_path.exists():
            print(f"Image {image_id} already exists, skipping.")
        else:
            ids_to_download.append(image_id)

    if not ids_to_download:
        print("All images already exist in the output directory.")
        return []

    print(f"Starting download of {len(ids_to_download)} images "
          f"from {mission}-E-{start_id} to {mission}-E-{end_id}...")

    failed_ids: List[int] = []

    # Descarga concurrente con ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_id = {
            executor.submit(
                download_image,
                mission,
                image_id,
                output_dir,
                retries,
                delay,
                timeout,
            ): image_id
            for image_id in ids_to_download
        }

        for future in as_completed(future_to_id):
            image_id = future_to_id[future]
            try:
                success = future.result()
                if not success:
                    failed_ids.append(image_id)
            except Exception as e:
                print(f"Unexpected error downloading image {image_id}: {e}")
                failed_ids.append(image_id)

    if failed_ids:
        print(f"Could not download the following images after retries: {failed_ids}")
    else:
        print("All images downloaded successfully.")

    return failed_ids


def main():
    """
    Punto de entrada cuando se ejecuta como script:
        python -m Scripts_v3.get_all_pics --mission ISS053 --start 462550 --end 462560
    """
    args = parse_args()
    mission = args.mission
    start_id = args.start
    end_id = args.end

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"./{mission}-E-{start_id}-{end_id}/pics")

    failed = download_all_images(
        mission=mission,
        start_id=start_id,
        end_id=end_id,
        output_dir=output_dir,
        max_threads=args.max_threads,
    )

    if failed:
        # Código de salida no cero si hay fallos, por si algún día quieres usarlo en un pipeline bash
        print("Some images could not be downloaded.")
    else:
        print("Download finished without errors.")


if __name__ == "__main__":
    main()
