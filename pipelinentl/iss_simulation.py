import os
import sys
import bpy
from math import radians, degrees, sin, cos, asin, atan2, sqrt, atan
import numpy as np
from mathutils import Vector
import csv
from skyfield.api import load, EarthSatellite
from datetime import datetime, timezone, timedelta
import mathutils
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from contextlib import contextmanager

#new
from skyfield.framelib import itrs

def setup_cycles_optix_only():
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences.addons["cycles"].preferences

    # OPTIX para RTX
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()

    target = prefs.compute_device_type  # "OPTIX"

    for d in prefs.devices:
        d.use = (d.type == target)

    scene.cycles.device = "GPU"

    print("compute_device_type:", prefs.compute_device_type)
    print("devices:", [(d.name, d.type, d.use) for d in prefs.devices])
    print("scene.cycles.device:", scene.cycles.device)
    print("render.engine:", scene.render.engine)

# ============================================================================
# UTILIDAD PARA SILENCIAR LA SALIDA DE BLENDER DURANTE EL RENDER
# ============================================================================

@contextmanager
def suppress_blender_output():
    """
    Context manager para silenciar stdout y stderr mientras se ejecuta un bloque,
    por ejemplo, la llamada a bpy.ops.render.render, que genera mensajes del tipo:
    "Fra:1 Mem:... Syncing Sphere / PointLight / Camera / Rendering X/64..."
    Tus propios print() fuera de este bloque se seguirán viendo normalmente.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


# ============================================================================
# CONVERSIONES Y GEOMETRÍA BÁSICA
# ============================================================================

def altitude_to_blender_units(altitude_km, earth_radius, earth_radius_km=6371.0):
    """
    Convierte una altitud en km a unidades de Blender, suponiendo que el radio
    de la Tierra en Blender es earth_radius.
    """
    blender_units = altitude_km / earth_radius_km * earth_radius
    return blender_units


def lat_lon_to_cartesian(lat_deg, lon_deg, altitude_real, earth_radius):
    """
    Convierte latitud, longitud (en grados) y altitud real (km) a coordenadas
    cartesianas (x, y, z) en el sistema de Blender, donde la Tierra es una
    esfera de radio earth_radius.
    """
    altitude = altitude_to_blender_units(altitude_real, earth_radius)
    lat_rad = radians(lat_deg)
    lon_rad = radians(lon_deg)
    r = earth_radius + altitude
    x = r * cos(lat_rad) * cos(lon_rad)
    y = r * cos(lat_rad) * sin(lon_rad)
    z = r * sin(lat_rad)
    return np.array([x, y, z])


def cartesian_to_geographic(x, y, z, earth_radius=10):
    """
    Convierte coordenadas cartesianas (x, y, z) al sistema geográfico (lat, lon, alt),
    asumiendo una esfera de radio earth_radius.
    """
    r = sqrt(x**2 + y**2 + z**2)
    lat = degrees(asin(z / r))
    lon = degrees(atan2(y, x))
    altitude = (r - earth_radius) / earth_radius * 6371.0  # km
    return lat, lon, altitude


# ============================================================================
# ESCENA DE BLENDER: ILUMINACIÓN Y TEXTURA
# ============================================================================

def create_uniform_lights_around_sphere(collection, num_lights=5, radius=100, energy=0):
    """
    Crea una malla de luces puntuales alrededor de la esfera, en longitudes y
    latitudes uniformemente espaciadas.
    """
    for i in range(num_lights):
        longitude = radians(360 / num_lights * i)
        for j in range(num_lights):
            latitude = radians(180 / num_lights * j - 90)

            x = radius * cos(latitude) * cos(longitude)
            y = radius * cos(latitude) * sin(longitude)
            z = radius * sin(latitude)

            light_data = bpy.data.lights.new(name=f"PointLight_{i}_{j}", type='POINT')
            light_object = bpy.data.objects.new(name=f"PointLight_{i}_{j}", object_data=light_data)
            collection.objects.link(light_object)

            light_object.location = (x, y, z)
            light_data.energy = energy  # 0 → no aportan realmente luz, sólo placeholder


def reset_scene(earth_radius, texture_path):
    """
    Limpia la escena, crea una esfera UV con la textura nocturna y un entorno
    negro. Devuelve el objeto esfera.
    """
    # Borrar objetos existentes
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Borrar cámaras existentes (por si acaso)
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Crear esfera UV de alta resolución
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=400,
        ring_count=400,
        radius=earth_radius,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
    )
    sphere = bpy.context.active_object

    # Desplegado UV para la textura
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.sphere_project()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Mundo negro (para resaltar la textura de emisión)
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0, 0, 0, 1)
    bg.inputs[1].default_value = 0.0

    # Material con textura nocturna en emisión
    mat = bpy.data.materials.new(name="EarthTextureMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    texture_node = nodes.new('ShaderNodeTexImage')
    texture_node.image = bpy.data.images.load(texture_path)

    emission_node = nodes.new('ShaderNodeEmission')
    output_node = nodes.get('Material Output')

    links.new(texture_node.outputs['Color'], emission_node.inputs['Color'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    sphere.data.materials.append(mat)
    # Rotación para alinear la textura (ajustada a tu conveniencia)
    sphere.rotation_euler = (radians(90), 0, radians(90))

    # Colección de luces (aunque con la textura en emisión prácticamente no hacen falta)
    light_collection = bpy.data.collections.new(name="LightCollection")
    bpy.context.scene.collection.children.link(light_collection)
    create_uniform_lights_around_sphere(light_collection)

    setup_cycles_optix_only()

    return sphere


# ============================================================================
# RAYOS DE CÁMARA Y INTERSECCIÓN CON LA ESFERA
# ============================================================================

def calculate_ray_directions(camera, x, y, width, height):
    """
    Calcula la dirección del rayo que pasa por el píxel (x, y) en el sistema
    de coordenadas del mundo.
    """
    camera_matrix = camera.matrix_world

    # Coordenadas normalizadas (NDC)
    ndc_x = (x + 0.5) / width * 2 - 1
    ndc_y = (y + 0.5) / height * 2 - 1

    # FOV horizontal de la cámara de Blender
    fov = camera.data.angle_x
    aspect_ratio = width / height

    px = ndc_x * np.tan(fov / 2)
    py = ndc_y * np.tan(fov / 2) / aspect_ratio
    pz = -1  # la cámara mira hacia -Z

    direction_camera_space = Vector((px, py, pz))
    direction_camera_space.normalize()

    direction_world_space = camera_matrix.to_quaternion() @ direction_camera_space
    return direction_world_space, px, py


def calculate_intersection(camera, direction, sphere):
    """
    Calcula la intersección del rayo (origen=posición de la cámara, dirección=direction)
    con la esfera (Tierra). Devuelve el punto de intersección o None si no hay.
    """
    origin = camera.location
    center = sphere.location
    radius = sphere.dimensions.x / 2

    a = direction.dot(direction)
    oc = origin - center
    b = 2.0 * oc.dot(direction)
    c = oc.dot(oc) - radius**2
    discriminant = b**2 - 4 * a * c

    if discriminant >= 0:
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        intersection = origin + direction * t
        return intersection
    return None


# ============================================================================
# ORIENTACIÓN DE CÁMARA: 'NORTH' Y 'FORWARD'
# ============================================================================

def rotate_around_axis(v, axis, angle):
    """
    Rotación de un vector v alrededor de un eje 'axis' un ángulo 'angle' (rad).
    Implementa la fórmula de rotación de Rodrigues.
    """
    axis = axis.normalized()
    return (
        v * cos(angle)
        + axis.cross(v) * sin(angle)
        + axis * axis.dot(v) * (1 - cos(angle))
    )


def set_camera_orientation_north(camera, lat_deg, lon_deg,
                                 off_nadir_deg, azimuth_deg, roll_deg):
    """
    Convención 'north':

    - off_nadir_deg (pitch):
        0°  = mirar al nadir (justo hacia abajo).
        >0° = se levanta hacia el horizonte en dirección NORTE geográfico.

    - azimuth_deg (yaw):
        rotación alrededor de la vertical local (Up). Con pitch y roll fijos,
        cambiar yaw solo rota la zona de Tierra visible, sin cambiar el corte
        Tierra/cielo ni la inclinación del horizonte.

    - roll_deg (roll):
        giro alrededor de la línea de visión. Es el único que inclina el horizonte.
    """
    # Ejes locales N-E-U
    lat = radians(lat_deg)
    lon = radians(lon_deg)

    U = Vector((cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)))        # Up
    E = Vector((-sin(lon),         cos(lon),         0.0))              # East
    N = Vector((-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)))      # North

    U.normalize(); E.normalize(); N.normalize()

    # Dirección base (yaw=0) en el plano Nadir-Norte
    theta = radians(off_nadir_deg)
    D = -U    # nadir
    H0 = N    # horizontal base = norte

    d0 = cos(theta)*D + sin(theta)*H0
    d0.normalize()

    # Up base antes de yaw/roll: proyectar U sobre el plano ⟂ d0
    u0 = U - d0 * U.dot(d0)
    if u0.length < 1e-6:
        u0 = E - d0 * E.dot(d0)
    u0.normalize()

    r0 = d0.cross(u0)
    r0.normalize()

    # Yaw: rotación de todo el frame alrededor de U
    phi = radians(azimuth_deg)
    d1 = rotate_around_axis(d0, U, phi)
    u1 = rotate_around_axis(u0, U, phi)
    r1 = rotate_around_axis(r0, U, phi)

    d1.normalize(); u1.normalize(); r1.normalize()

    # Roll alrededor de la línea de visión d1
    rho = radians(roll_deg)
    u = cos(rho)*u1 + sin(rho)*r1
    r = -sin(rho)*u1 + cos(rho)*r1

    u.normalize(); r.normalize()

    z_cam = -d1  # Blender mira en -Z

    rot_mat = mathutils.Matrix((
        (r.x, u.x, z_cam.x),
        (r.y, u.y, z_cam.y),
        (r.z, u.z, z_cam.z),
    ))

    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rot_mat.to_quaternion()


def set_camera_orientation_forward(camera, lat_deg, lon_deg,
                                   off_nadir_deg, azimuth_deg, roll_deg,
                                   velocity_vector):
    """
    Convención 'forward':

    - off_nadir_deg (pitch):
        0°  = nadir (mirar hacia abajo).
        >0° = se levanta hacia el horizonte en dirección de MOVIMIENTO
              (proyección de la velocidad en el plano tangente).

    - azimuth_deg (yaw):
        rotación alrededor de la vertical local (Up).

    - roll_deg (roll):
        giro de la imagen (inclina el horizonte).
    """
    # Ejes locales N-E-U
    lat = radians(lat_deg)
    lon = radians(lon_deg)

    U = Vector((cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)))        # Up
    E = Vector((-sin(lon),         cos(lon),         0.0))              # East
    N = Vector((-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)))      # North

    U.normalize(); E.normalize(); N.normalize()

    # Dirección "adelante" a partir del vector velocidad
    V = Vector(velocity_vector)
    V_tan = V - U * V.dot(U)  # componente tangente a la superficie
    if V_tan.length < 1e-6:
        V_tan = N.copy()
    F = V_tan.normalized()

    # Dirección base (yaw=0) en el plano Nadir-Forward
    theta = radians(off_nadir_deg)
    D = -U     # nadir
    H0 = F     # dirección "adelante"

    d0 = cos(theta)*D + sin(theta)*H0
    d0.normalize()

    # Up base antes de yaw/roll
    u0 = U - d0 * U.dot(d0)
    if u0.length < 1e-6:
        u0 = E - d0 * E.dot(d0)
    u0.normalize()

    r0 = d0.cross(u0)
    r0.normalize()

    # Yaw alrededor de U
    phi = radians(azimuth_deg)
    d1 = rotate_around_axis(d0, U, phi)
    u1 = rotate_around_axis(u0, U, phi)
    r1 = rotate_around_axis(r0, U, phi)

    d1.normalize(); u1.normalize(); r1.normalize()

    # Roll alrededor de la línea de visión d1
    rho = radians(roll_deg)
    u = cos(rho)*u1 + sin(rho)*r1
    r = -sin(rho)*u1 + cos(rho)*r1

    u.normalize(); r.normalize()

    z_cam = -d1  # Blender mira en -Z

    rot_mat = mathutils.Matrix((
        (r.x, u.x, z_cam.x),
        (r.y, u.y, z_cam.y),
        (r.z, u.z, z_cam.z),
    ))

    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = rot_mat.to_quaternion()


# ============================================================================
# FOV Y UTILIDADES CSV/POINTS
# ============================================================================

def calculate_horizontal_vertical_fov(focal_length, sensor_width, sensor_height):
    """
    Calcula el FOV horizontal y vertical (en grados) a partir de la focal y
    las dimensiones del sensor (en mm).
    """
    fov_horizontal_rad = 2 * atan(sensor_width / (2 * focal_length))
    fov_vertical_rad = 2 * atan(sensor_height / (2 * focal_length))

    fov_horizontal_deg = degrees(fov_horizontal_rad)
    fov_vertical_deg = degrees(fov_vertical_rad)

    print(f"Horizontal FoV: {fov_horizontal_deg:.2f} degrees")
    print(f"Vertical FoV: {fov_vertical_deg:.2f} degrees")

    return fov_horizontal_deg, fov_vertical_deg


def read_pixel_coordinates_from_csv(file_path):
    """
    Lee un CSV con columnas: sim_x, sim_y, real_x, real_y.
    """
    pixel_coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # saltar cabecera, si existe
        for row in reader:
            sim_x, sim_y, real_x, real_y = map(float, row)
            pixel_coordinates.append((sim_x, sim_y, real_x, real_y))
    return pixel_coordinates


def create_points(data, filepath, photo_height, source='real'):
    """
    Crea un archivo .points (QGIS) a partir de una lista de coordenadas:

        [sim_x, sim_y, real_x, real_y, lat, lon]

    photo_height se usa para invertir la Y (sistema de imagen → sistema QGIS).
    """
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mapX', 'mapY', 'sourceX', 'sourceY', 'enable', 'dX', 'dY', 'residual'])
        for coord in data:
            if None in coord:
                continue
            if any(isinstance(val, float) and np.isnan(val) for val in coord):
                continue

            sim_x, sim_y, real_x, real_y, lat, lon = coord
            mapX, mapY = lon, lat  # QGIS: X=lon, Y=lat

            if source == 'simulated':
                sourceX, sourceY = sim_x, -(photo_height - sim_y)
            else:
                sourceX, sourceY = real_x, -(photo_height - real_y)

            enable, dX, dY, residual = 1, 0, 0, 0
            writer.writerow([mapX, mapY, sourceX, sourceY, enable, dX, dY, residual])


def create_csv(data, filepath):
    """
    Guarda la lista de coordenadas [SimX, SimY, RealX, RealY, Lat, Lon] en CSV.
    """
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sim Pixel X', 'Sim Pixel Y', 'Real Pixel X', 'Real Pixel Y', 'Latitude', 'Longitude'])
        for coord in data:
            writer.writerow(coord)


# ============================================================================
# RENDER DE UNA IMAGEN: creaimagen
# ============================================================================

def get_or_create_camera():
    cam = bpy.data.objects.get("SimCamera")
    if cam and cam.type == "CAMERA":
        return cam

    cam_data = bpy.data.cameras.new("SimCameraData")
    cam = bpy.data.objects.new("SimCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    return cam

def creaimagen(latitude, longitude, altitude_real,
               yaw, pitch, roll, velocity_vector, sphere,
               focal_length, sensor_width, sensor_height,
               pixel_width, pixel_height,
               time="2023-03-20T12:00:00.0",
               output_directory=os.getcwd(),
               earth_radius=10,
               render_image=True,
               orientation_mode='north'):
    """
    Renderiza una imagen simulada de la Tierra desde la ISS para una configuración
    concreta de orientación.

    orientation_mode:
      - 'north'   -> pitch se levanta hacia el norte geográfico.
      - 'forward' -> pitch se levanta hacia la dirección de movimiento (velocity_vector).

    Devuelve:
      - camera: objeto cámara de Blender.
      - file_path: ruta de la imagen renderizada.
    """
    # Crear cámara
    camera = get_or_create_camera()
    camera_data = camera.data

    # Tipo de cámara y FOV
    camera_data.type = 'PERSP'
    camera_data.sensor_fit = 'AUTO'
    camera_data.sensor_width = sensor_width
    camera_data.lens = focal_length
    calculate_horizontal_vertical_fov(focal_length, sensor_width, sensor_height)

    # Posición de la cámara (ISS)
    position = lat_lon_to_cartesian(latitude, longitude, altitude_real, earth_radius)
    camera.location = tuple(position)

    # Orientación según modo
    if orientation_mode == 'north':
        set_camera_orientation_north(
            camera,
            lat_deg=latitude,
            lon_deg=longitude,
            off_nadir_deg=pitch,
            azimuth_deg=yaw,
            roll_deg=roll
        )
    elif orientation_mode == 'forward':
        set_camera_orientation_forward(
            camera,
            lat_deg=latitude,
            lon_deg=longitude,
            off_nadir_deg=pitch,
            azimuth_deg=yaw,
            roll_deg=roll,
            velocity_vector=velocity_vector
        )
    else:
        raise ValueError(f"orientation_mode desconocido: {orientation_mode}")

    # Cámara activa
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    engines = scene.render.bl_rna.properties['engine'].enum_items.keys()

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    scene.cycles.samples = 16
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05

    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transparent_max_bounces = 0

    scene.cycles.use_denoising = False

    bpy.context.scene.view_settings.exposure = 0.5
    bpy.context.scene.render.resolution_x = pixel_width
    bpy.context.scene.render.resolution_y = pixel_height
    bpy.context.scene.render.film_transparent = False

    # Nombre de fichero
    latitude_rounded = 'L' + str(round(latitude, 2)).replace('.', '-')
    longitude_rounded = 'G' + str(round(longitude, 2)).replace('.', '-')
    altitude_real_rounded = 'H' + str(round(altitude_real, 2)).replace('.', '-')
    yaw_rounded = 'Y' + str(round(yaw, 2)).replace('.', '-')
    pitch_rounded = 'P' + str(round(pitch, 2)).replace('.', '-')
    roll_rounded = 'R' + str(round(roll, 2)).replace('.', '-')

    datetime_obj = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    safe_time = datetime_obj.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

    file_path = os.path.join(
        output_directory,
        f"render_output_{safe_time}_{latitude_rounded}_{longitude_rounded}_{altitude_real_rounded}_{yaw_rounded}_{pitch_rounded}_{roll_rounded}.png"
    )
    bpy.context.scene.render.filepath = file_path

    # Render silencioso (solo se silencian los mensajes de Blender)
    if render_image:
        with suppress_blender_output():
            bpy.ops.render.render(write_still=True)
    else:
        # NO render. Solo asegura que Blender actualiza matrices y depsgraph
        bpy.context.view_layer.update()

    return camera, file_path


# ============================================================================
# PROYECCIÓN DE PÍXELES SOBRE LA ESFERA
# ============================================================================

def project_pixels(matches_path, latitude, longitude, altitude_real,
                   yaw, pitch, roll, velocity_vector, sphere, camera,
                   real_photo_height,
                   time="2023-03-20T12:00:00.0",
                   output_directory=os.getcwd(),
                   earth_radius=10):
    """
    Proyecta los píxeles (sim_x, sim_y, real_x, real_y) de un CSV sobre la Tierra,
    calculando lat/lon de intersección de los rayos de la cámara con la esfera.
    Genera dos archivos .points (real y simulado) para QGIS.
    """
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    pixel_coordinates = read_pixel_coordinates_from_csv(matches_path)
    pixel_coords = []

    for sim_x, sim_y, real_x, real_y in pixel_coordinates:
        direction, px, py = calculate_ray_directions(camera, sim_x, sim_y, width, height)
        intersection = calculate_intersection(camera, direction, sphere)

        if intersection:
            lat_inter, lon_inter, alt_inter = cartesian_to_geographic(*intersection, earth_radius=earth_radius)
            pixel_coords.append([sim_x, sim_y, real_x, real_y, lat_inter, lon_inter])
        else:
            pixel_coords.append([sim_x, sim_y, real_x, real_y, np.nan, np.nan])

    if not pixel_coords:
        pixel_coords.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    latitude_rounded = 'L' + str(round(latitude, 2)).replace('.', '-')
    longitude_rounded = 'G' + str(round(longitude, 2)).replace('.', '-')
    altitude_real_rounded = 'H' + str(round(altitude_real, 2)).replace('.', '-')
    yaw_rounded = 'Y' + str(round(yaw, 2)).replace('.', '-')
    pitch_rounded = 'P' + str(round(pitch, 2)).replace('.', '-')
    roll_rounded = 'R' + str(round(roll, 2)).replace('.', '-')

    datetime_obj = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    safe_time = datetime_obj.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

    points_file_path_real = os.path.join(
        output_directory,
        f"coordinates_{safe_time}_{latitude_rounded}_{longitude_rounded}_{altitude_real_rounded}_{yaw_rounded}_{pitch_rounded}_{roll_rounded}_real.points"
    )
    create_points(pixel_coords, points_file_path_real, real_photo_height, source='real')

    points_file_path_simulated = os.path.join(
        output_directory,
        f"coordinates_{safe_time}_{latitude_rounded}_{longitude_rounded}_{altitude_real_rounded}_{yaw_rounded}_{pitch_rounded}_{roll_rounded}_simulated.points"
    )
    create_points(pixel_coords, points_file_path_simulated, height, source='simulated')


# ============================================================================
# GESTIÓN DE TLE Y ÓRBITA
# ============================================================================

def list_tle_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]


def read_tle_from_files(file_paths):
    """
    Lee múltiples archivos .txt con TLEs y devuelve una lista de (satélite, epoch).
    """
    ts = load.timescale()
    tle_data = []
    for path in file_paths:
        with open(path, 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines) - 1:
                try:
                    if lines[i].strip().startswith('1') and lines[i+1].strip().startswith('2'):
                        line1 = lines[i].strip()
                        line2 = lines[i+1].strip()
                        satellite_name = line1.split()[1]
                        satellite = EarthSatellite(line1, line2, satellite_name, ts)
                        tle_data.append((satellite, satellite.epoch.utc_datetime()))
                        i += 2
                    else:
                        i += 1
                except IndexError:
                    break
    return tle_data


def find_closest_tle(tle_data, target_date):
    """
    Encuentra el TLE cuyo epoch está más cerca de target_date (UTC).
    """
    closest_tle = None
    min_time_diff = float('inf')
    target_date = target_date.replace(tzinfo=timezone.utc)
    for satellite, epoch in tle_data:
        time_diff = abs((target_date - epoch).total_seconds())
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_tle = satellite
    if closest_tle:
        print(f"Closest TLE found for {target_date}: {closest_tle.name} with epoch {closest_tle.epoch.utc_jpl()}")
        print(f"{closest_tle}")
    else:
        print("No close TLE found.")
    return closest_tle


def check_tle_validity(satellite, observation_date):
    """
    Imprime la diferencia en días entre el epoch del TLE y observation_date.
    Útil para debug, puede comentarse en producción para reducir ruido.
    """
    ts = load.timescale()
    t = ts.utc(observation_date.year, observation_date.month, observation_date.day)

    days_difference = abs(satellite.epoch.tt - t.tt)
    print(f"Days difference between TLE epoch and observation date: {days_difference} days")

    geocentric = satellite.at(t)
    subpoint = geocentric.subpoint()
    if subpoint.latitude.degrees is None:
        print("Failed to calculate valid subpoint. TLE might be outdated.")
    else:
        print(f"Latitude ISS: {subpoint.latitude.degrees}, Longitude ISS: {subpoint.longitude.degrees}")


def get_iss_position_and_velocity_old(satellite, time):
    """
    Devuelve lat, lon (grados), altitud (km) y vector velocidad (km/s) de la ISS
    para el satélite (TLE) y el instante dado.
    """
    ts = load.timescale()
    t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6)
    geocentric = satellite.at(t)
    subpoint = geocentric.subpoint()

    # Corregido: solo es error si son None, no si son 0
    if subpoint.latitude.degrees is None or subpoint.longitude.degrees is None:
        print("Error en cálculo del subpunto:", subpoint)

    return (
        subpoint.latitude.degrees,
        subpoint.longitude.degrees,
        subpoint.elevation.km,
        geocentric.velocity.km_per_s
    )

#new
def get_iss_position_and_velocity(satellite, time, dt_seconds=1.0):
    """
    Devuelve lat, lon (deg), altitud (km) y dos vectores:
      - v_icrf  : velocidad inercial (ICRF) como daba Skyfield
      - v_itrs  : velocidad en marco Tierra (ITRS/ECEF) por diferencia finita
    """
    ts = load.timescale()
    t0 = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6)
    t1_dt = (time + timedelta(seconds=dt_seconds))
    t1 = ts.utc(t1_dt.year, t1_dt.month, t1_dt.day, t1_dt.hour, t1_dt.minute, t1_dt.second + t1_dt.microsecond * 1e-6)

    geo0 = satellite.at(t0)
    geo1 = satellite.at(t1)

    subpoint = geo0.subpoint()
    lat = subpoint.latitude.degrees
    lon = subpoint.longitude.degrees
    alt = subpoint.elevation.km

    # 1) Velocidad inercial (lo de siempre)
    v_icrf = geo0.velocity.km_per_s  # OJO: marco inercial

    # 2) “Velocidad” en marco Tierra (ITRS/ECEF) por diferencia de posiciones
    p0 = np.array(geo0.frame_xyz(itrs).km)  # (3,)
    p1 = np.array(geo1.frame_xyz(itrs).km)  # (3,)
    v_itrs = (p1 - p0) / float(dt_seconds)  # km/s aprox en ITRS/ECEF

    return lat, lon, alt, v_icrf, v_itrs


# ============================================================================
# GENERACIÓN DE SERIES DE IMÁGENES (TIMELAPSE)
# ============================================================================

def generate_image_series(start_date, end_date, delta, tle_data,
                          yaw, pitch, roll, sphere,
                          focal_length, sensor_width, sensor_height,
                          pixel_width, pixel_height,
                          output_directory, earth_radius,
                          render_image=True,
                          orientation_mode='north'):
    """
    Genera un timelapse de imágenes simuladas entre start_date y end_date, con
    paso temporal delta (segundos), usando el TLE más cercano para cada minuto y
    los ángulos yaw, pitch, roll fijos.

    Devuelve:
      - latitudes: lista de latitudes de la ISS a cada paso.
      - longitudes: lista de longitudes de la ISS a cada paso.
    """
    latitudes = []
    longitudes = []
    current_date = start_date

    while current_date <= end_date:
        closest_tle = find_closest_tle(tle_data, current_date)
        # Esto es útil para debug; puedes comentar la siguiente línea si hace mucho ruido:
        check_tle_validity(closest_tle, start_date)

        if closest_tle:
            for second in np.arange(0, 60, delta):
                target_date = current_date + timedelta(seconds=float(second))
                if target_date > end_date:
                    break

                #new    
                latitude, longitude, altitude, v_icrf, v_itrs = get_iss_position_and_velocity(closest_tle, target_date)
                velocity_for_orientation = v_itrs

                print(f"Position of ISS on {target_date}: "
                      f"Latitude: {latitude:.4f}°, Longitude: {longitude:.4f}°, Altitude: {altitude:.2f} km")
                print(f"Velocity of ISS on {target_date}: {velocity_for_orientation} km/s")

                latitudes.append(latitude)
                longitudes.append(longitude)

                formatted_date = target_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

                creaimagen(
                    latitude, longitude, altitude,
                    yaw, pitch, roll, velocity_for_orientation, sphere,
                    focal_length, sensor_width, sensor_height,
                    pixel_width, pixel_height,
                    time=str(formatted_date),
                    output_directory=str(output_directory),
                    earth_radius=earth_radius,
                    render_image=render_image,
                    orientation_mode=orientation_mode
                )

            print(f"Completed images for minute starting at {current_date.isoformat()}")
        else:
            print(f"No TLE found close to {current_date.isoformat()}")

        current_date += timedelta(minutes=1)
        if current_date > end_date:
            break

    return latitudes, longitudes


# ============================================================================
# TRAZADO DE LA TRAYECTORIA EN MAPA Y GUARDAR ESCENA
# ============================================================================

def plot_iss_trajectory(latitudes, longitudes, output_directory, show=False):
    """
    Dibuja la trayectoria de la ISS sobre un mapa (proyección Robinson) y
    guarda la figura en output_directory.

    Parámetros
    ----------
    latitudes : list or array
        Latitudes de la ISS.
    longitudes : list or array
        Longitudes de la ISS.
    output_directory : str
        Carpeta donde se guardará la figura PNG.
    show : bool, optional
        Si True, muestra la figura en pantalla (plt.show()).
        Si False (por defecto), solo guarda y cierra (modo pipeline).
    """
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

    ax.plot(longitudes, latitudes,
            transform=ccrs.Geodetic(),
            marker='o', markersize=3, linewidth=1, color='blue', label='ISS Path')

    plt.title("ISS Path Over Earth Map")
    plt.legend()

    out_path = os.path.join(output_directory, 'iss_path_map.png')
    plt.savefig(out_path, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def save_timelapse(output_directory):
    """
    Empaqueta la escena y guarda un archivo .blend con todo el timelapse.
    """
    bpy.ops.file.pack_all()
    file_path = os.path.join(output_directory, "timelapse.blend")
    bpy.ops.wm.save_as_mainfile(filepath=file_path)
    print("Scene saved to", file_path)