#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime, timezone, timedelta
from skyfield.api import load
from skyfield.framelib import itrs

from pipelinentl.iss_simulation import list_tle_files, read_tle_from_files, find_closest_tle

#ejecutar con: python -m pipelinentl.debug_forward_drift

def neu_basis(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    U = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    E = np.array([-np.sin(lon), np.cos(lon), 0.0])
    N = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])

    # normalizar
    U = U / np.linalg.norm(U)
    E = E / np.linalg.norm(E)
    N = N / np.linalg.norm(N)
    return N, E, U

def bearing_from_forward(F, N, E):
    # bearing 0=N, 90=E
    fn = float(F.dot(N))
    fe = float(F.dot(E))
    ang = np.degrees(np.arctan2(fe, fn))
    return (ang + 360.0) % 360.0

def main():
    tle_dir = "/home/rpz/iss_simulation/ISS_tle"   # ajusta si hace falta
    tle_files = list_tle_files(tle_dir)
    tle_data = read_tle_from_files(tle_files)

    start = datetime(2012, 3, 28, 1, 50, 0, tzinfo=timezone.utc)
    end   = datetime(2012, 3, 28, 2,  5, 0, tzinfo=timezone.utc)
    step  = 1.0
    dt_v  = 1.0  # dt para la velocidad ECEF

    ts = load.timescale()

    times = []
    b_bug = []
    b_fix = []

    t = start
    while t <= end:
        sat = find_closest_tle(tle_data, t)

        t0 = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6)
        t1_dt = t + timedelta(seconds=dt_v)
        t1 = ts.utc(t1_dt.year, t1_dt.month, t1_dt.day, t1_dt.hour, t1_dt.minute, t1_dt.second + t1_dt.microsecond*1e-6)

        geo0 = sat.at(t0)
        geo1 = sat.at(t1)

        sp = geo0.subpoint()
        lat = sp.latitude.degrees
        lon = sp.longitude.degrees

        N, E, U = neu_basis(lat, lon)

        # BUG: usar velocidad inercial como si fuera ECEF (lo que hace tu código actual)
        v_icrf = np.array(geo0.velocity.km_per_s)
        vtan_bug = v_icrf - U * (v_icrf.dot(U))
        if np.linalg.norm(vtan_bug) < 1e-9:
            # si se vuelve degenerado
            vtan_bug = N.copy()
        F_bug = vtan_bug / np.linalg.norm(vtan_bug)

        # FIX: velocidad ECEF por diferencia finita en ITRS
        p0 = np.array(geo0.frame_xyz(itrs).km)
        p1 = np.array(geo1.frame_xyz(itrs).km)
        v_itrs = (p1 - p0) / dt_v
        vtan_fix = v_itrs - U * (v_itrs.dot(U))
        if np.linalg.norm(vtan_fix) < 1e-9:
            vtan_fix = N.copy()
        F_fix = vtan_fix / np.linalg.norm(vtan_fix)

        times.append(t)
        b_bug.append(bearing_from_forward(F_bug, N, E))
        b_fix.append(bearing_from_forward(F_fix, N, E))

        t += timedelta(seconds=step)

    # Plot
    x = np.arange(len(times))
    plt.figure(figsize=(12, 5))
    plt.plot(x, b_bug, label="bearing BUG (v_icrf proyectada)")
    plt.plot(x, b_fix, label="bearing FIX (v_itrs por diff ITRS)")
    plt.ylim(0, 360)
    plt.legend()
    plt.title("Drift de 'forward' por mezcla de marcos (ICRF vs ITRS)")
    plt.xlabel("frame")
    plt.ylabel("bearing (deg)")
    plt.tight_layout()
    plt.savefig("forward_drift_debug.png", dpi=160)

if __name__ == "__main__":
    main()
