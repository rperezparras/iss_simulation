#!/usr/bin/env python3
import os
from glob import glob
import argparse

import re 

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def filter_and_rename_points(
    input_folder: str,
    output_folder: str,
    radius_km: float,
    start_id: int,
    end_id: int,
    mission: str,
):
    """
    Filtra archivos *.points usando un BallTree (para eliminar puntos muy aislados)
    y los renombra para que sus nombres sean coherentes con los IDs de imágenes:

        <mission>-E-<ID>.points

    Asume que en input_folder hay archivos terminados en:
        *_real.points

    Cada archivo se procesa, se filtran puntos cuyo vecindario (radio radius_km)
    tiene 1 solo punto (ellos mismos), y el resultado se guarda en output_folder.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Radio en radianes para la métrica haversine
    radius_rad = radius_km / 6371.0

    point_files = sorted(glob(os.path.join(input_folder, "*_real.points")))

    expected_count = end_id - start_id + 1
    if len(point_files) != expected_count:
        print(
            f"⚠️ Aviso: se esperaban {expected_count} archivos *_real.points, "
            f"pero se encontraron {len(point_files)} en '{input_folder}'."
        )
        print("   Se continuará igualmente, usando los archivos ordenados por nombre.\n")

    for idx, file_path in enumerate(point_files):
        base = os.path.basename(file_path)

        parsed_id = extract_id_from_point_filename(base)
        if parsed_id is not None:
            current_id = parsed_id
        else:
            current_id = start_id + idx

        if current_id < start_id or current_id > end_id:
            print(f"⚠️ {base}: ID {current_id} fuera de rango, se salta.")
            continue

        new_filename = f"{mission}-E-{current_id}.points"
        output_path = os.path.join(output_folder, new_filename)

        try:
            df = pd.read_csv(file_path)

            # Necesitamos columnas mapY (latitud), mapX (longitud)
            if not {"mapY", "mapX"}.issubset(df.columns):
                print(f"⚠️ {os.path.basename(file_path)} no tiene columnas 'mapY' y 'mapX', se salta.")
                continue

            # Coordenadas en radianes para BallTree(haversine)
            coords = np.radians(df[["mapY", "mapX"]].values)

            tree = BallTree(coords, metric="haversine")
            neighbors = tree.query_radius(coords, r=radius_rad)

            # Conserva puntos que tienen al menos otro vecino en el radio
            mask = [len(n) > 1 for n in neighbors]
            filtered_df = df[mask]

            if filtered_df.empty:
                print(
                    f"⚠️ Tras filtrar, no quedan puntos en {os.path.basename(file_path)} "
                    f"(ID {current_id}). Se guarda igualmente un archivo vacío."
                )

            filtered_df.to_csv(output_path, index=False)
            print(f"✅ Procesado y renombrado: {new_filename} (de {os.path.basename(file_path)})")

        except Exception as e:
            print(f"❌ Error procesando {file_path}: {e}")

    print("\n✔️ Filtrado y renombrado completado.")

def extract_id_from_point_filename(name: str) -> int | None:
    """
    Extrae el ID real desde nombres como:
      ISS067-E-327041_real.points
      ISS067-E-327041.points
      coordinates_...   -> None
    """
    m = re.search(r"ISS\d+-E-(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtrado de puntos geográficos con BallTree y renombrado a <mission>-E-<ID>.points"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Carpeta con archivos *_real.points originales",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Carpeta para guardar archivos filtrados y renombrados",
    )
    parser.add_argument(
        "--radius_km",
        type=float,
        default=130.0,
        help="Radio en km para búsqueda de vecinos (BallTree, métrica haversine)",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
        help="ID inicial esperado para los archivos (coincide con el ID de imagen)",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        required=True,
        help="ID final esperado para los archivos",
    )
    parser.add_argument(
        "--mission",
        type=str,
        required=True,
        help="Nombre de la misión, p.ej. 'ISS053', 'ISS067', etc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filter_and_rename_points(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        radius_km=args.radius_km,
        start_id=args.start_id,
        end_id=args.end_id,
        mission=args.mission,
    )


if __name__ == "__main__":
    main()