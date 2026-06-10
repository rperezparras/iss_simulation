# iss_simulation
# Pipeline de georreferenciación de imágenes nocturnas de la ISS

Pipeline automatizada para georreferenciar timelapses nocturnos tomados desde la Estación Espacial Internacional (ISS), combinando:

- descarga de imágenes reales ISS,
- simulación geométrica en Blender a partir de TLEs,
- matching entre imágenes simuladas y reales,
- proyección de píxeles sobre la Tierra,
- filtrado de puntos de control,
- georreferenciación con GDAL + Thin Plate Spline,
- refinamiento opcional mediante imágenes VIIRS y flujo óptico.

El objetivo es transformar secuencias de imágenes ISS sin georreferenciación precisa en productos espaciales utilizables científicamente.

---

## Estructura general

La pipeline completa está orquestada desde:

- `timelapse_pipeline.py`

Los módulos principales son:

- `get_pics.py`: descarga de imágenes ISS desde NASA EOL.
- `generate_timelapse.py`: genera un timelapse simulado en Blender a partir de EXIF + TLEs.
- `iss_simulation.py`: núcleo de simulación geométrica, cámara, orientación y proyección.
- `angle_search.py`: búsqueda automática de yaw/pitch comparando imagen real y simulada.
- `match_timelapse.py`: matching entre imágenes simuladas y reales.
- `project_timelapse.py`: proyección de píxeles simulados sobre la Tierra y generación de `.points`.
- `filter_points.py`: filtrado espacial y renombrado de puntos.
- `georef_timelapse.py`: georreferenciación con GDAL usando GCPs y TPS.
- `viirs_roi_crop.py`: recorte y alineado de mosaicos VIIRS a las imágenes georreferenciadas.
- `optical_flow.py`: cálculo de flujo óptico entre ISS georreferenciada y VIIRS.
- `correct_points.py`: corrección de GCPs mediante el flujo óptico.
- `debug_forward_drift.py`: script de diagnóstico para depurar la orientación `forward`.

---

## Flujo de trabajo

### Pipeline base

1. **Descarga de imágenes ISS**
2. **Lectura de EXIF** para obtener tiempos, focal y resolución
3. **Búsqueda opcional de ángulos** (`yaw`, `pitch`) con `angle_search.py`
4. **Generación de timelapse simulado** en Blender
5. **Matching** entre imágenes simuladas y reales
6. **Proyección de píxeles** a coordenadas geográficas y generación de `.points`
7. **Filtrado de puntos** aislados y renombrado consistente
8. **Primera georreferenciación** con GDAL

### Refinamiento

9. **Recorte y alineado VIIRS**
10. **Cálculo de flujo óptico** ISS–VIIRS
11. **Corrección de puntos**
12. **Segunda georreferenciación** completa o de muestra

---

## Estructura de carpetas esperada

Para una secuencia de ejemplo como `ISS030-E-281044-281946/`, la pipeline genera algo como:

```text
ISS030-E-281044-281946/
├── pics/                    # imágenes reales descargadas
├── output/                  # renders simulados + .points iniciales
│   └── matches/             # CSVs con correspondencias sim-real
├── search_angles/           # renders usados para búsqueda de yaw/pitch
├── filtered_points/         # GCPs filtrados y renombrados
├── geo/                     # 1ª georreferenciación
├── viirs_cropped_aligned/   # recortes VIIRS alineados
├── flow/                    # flujo óptico (*.npy)
├── corrected_points/        # puntos corregidos con flujo
└── geo_corrected/           # 2ª georreferenciación
