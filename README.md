# ISS timelapse georeferencing pipeline

Pipeline for georeferencing nighttime ISS image sequences using Blender simulations, image matching, thin plate spline georeferencing, VIIRS data and optional optical-flow correction.

The main entry point is:

```bash
python3 -m pipelinentl.timelapse_pipeline
```

Most parameters are configured from this script through command-line arguments. The other scripts are used internally as processing modules.

## Repository

```bash
git clone https://github.com/rperezparras/iss_simulation.git
cd iss_simulation
```

## Recommended environment

The recommended Python version is **Python 3.11**.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the base requirements:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install .
```

Install the Blender-related requirements:

```bash
python3 -m pip install .[blender]
```

After activating the virtual environment, `python` and `python3` should normally point to the same environment. In this README, `python3` is used explicitly for Linux compatibility.

## Important dependency notes

The pipeline uses Blender through Python (`bpy`). For this reason, the requirements pin some packages to avoid known compatibility problems:

```text
numpy<2
opencv-python-headless<4.12
```

This is important because recent NumPy/OpenCV versions can be incompatible with some `bpy` builds.

If you see an error similar to:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

run:

```bash
python3 -m pip install "numpy<2" "opencv-python-headless<4.12" --force-reinstall
```

Then check:

```bash
python3 - <<'PY'
import numpy
import cv2
import bpy
import vismatch

print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("bpy OK")
print("vismatch OK")
PY
```

## External data

Large data files are not included in this repository. The pipeline expects a data root folder containing:

- ISS TLE files.
- The nighttime Earth texture used by Blender.
- The VIIRS mosaic.
- One folder per ISS timelapse.

Set the data root with:

```bash
export ISS_SIMULATION_DATA_ROOT=/path/to/your/iss_data
```

Expected structure:

```text
/path/to/your/iss_data/
├── ISS_tle/
├── VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.sqrt.full.40k_20k.png
├── VNL_v2_npp_2021_global_vcmslcfg_c202203152300.median_masked.tif
└── ISS067-E-327041-328344/
    ├── pics/
    ├── output/
    ├── filtered_points/
    ├── geo/
    ├── viirs_cropped_aligned/
    ├── flow/
    ├── corrected_points/
    └── geo_corrected/
```

The timelapse folder name is built as:

```text
<MISSION>-E-<START_ID>-<END_ID>
```

For example:

```text
ISS067-E-327041-328344
```

## Quick start

Example using the timelapse `ISS067-E-327041-328344`:

```bash
export ISS_SIMULATION_DATA_ROOT=/path/to/your/iss_data

python3 -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 327041 \
  --end-id 328344
```

To see all available options:

```bash
python3 -m pipelinentl.timelapse_pipeline --help
```

## Running without optical flow

To run only the first part of the pipeline, without VIIRS alignment, optical flow or second georeferencing:

```bash
python3 -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 327041 \
  --end-id 328344 \
  --no-optical-flow
```

## Running without automatic angle search

If you already know the camera orientation, you can skip `angle_search` and provide yaw, pitch and roll manually:

```bash
python3 -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 327041 \
  --end-id 328344 \
  --no-angle-search \
  --yaw 12.5 \
  --pitch 63.5 \
  --roll -1.0
```

## Matching backend

The default matching backend is `vismatch`.

For a normal installation, no local matching repository path is needed:

```bash
python3 -m pip install vismatch
```

The pipeline also keeps optional compatibility with the old `image-matching-models` repository. This is only for legacy local setups. If needed, define:

```bash
export IMAGE_MATCHING_MODELS_DIR=/path/to/image-matching-models
```

New users should normally use `vismatch`.

You can force a backend with:

```bash
python3 -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 327041 \
  --end-id 328344 \
  --matcher-backend vismatch
```

Available values are:

```text
auto
vismatch
matching
none
```

## Main outputs

For the example timelapse:

```text
ISS067-E-327041-328344/
```

the main output folders are:

```text
ISS067-E-327041-328344/
├── pics/                     # Original ISS images
├── output/                   # Blender renders and projected control points
├── output/matches/           # Matching CSVs and diagnostic plots
├── filtered_points/          # Filtered QGIS .points files
├── geo/                      # First georeferenced ISS GeoTIFFs
├── viirs_cropped_aligned/    # VIIRS crops aligned to the ISS georeferenced frames
├── flow/                     # Optical-flow fields
├── corrected_points/         # Control points corrected using optical flow
└── geo_corrected/            # Second georeferencing outputs
```

## Useful rerun options

The pipeline detects existing outputs and reuses them when possible. To force specific steps to run again:

```bash
python3 -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 327041 \
  --end-id 328344 \
  --rerun-matching
```

Other available rerun flags include:

```text
--rerun-simulation
--rerun-matching
--rerun-projection
--rerun-filtering
--rerun-first-georef
--rerun-viirs
--rerun-optical-flow
--rerun-correct-points
--rerun-second-georef
```

## Second georeferencing mode

The second georeferencing can be controlled with:

```bash
--second-georef-mode none
--second-georef-mode sample
--second-georef-mode full
```

Example:

```bash
python3 -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 327041 \
  --end-id 328344 \
  --second-georef-mode sample
```

## Design principle

The intended workflow is:

1. Configure the experiment from `timelapse_pipeline.py` or command-line arguments.
2. Let the pipeline call the auxiliary modules internally.
3. Avoid editing auxiliary scripts such as `match_timelapse.py`, `angle_search.py`, `filter_points.py`, `project_timelapse.py`, etc., unless developing the pipeline itself.

This makes the repository easier to install, reproduce and use on different machines.

## System dependencies

Some tools may need to be available at system level, especially:

- Blender or a compatible `bpy` Python package.
- GDAL command-line tools such as `gdal_translate` and `gdalwarp`.
- CUDA-compatible PyTorch if GPU acceleration is required.

For GDAL, make sure the command-line tools are visible in the environment:

```bash
gdalinfo --version
```

## Troubleshooting

### `cv2` cannot be imported in VS Code

Make sure VS Code is using the same virtual environment:

```text
Ctrl+Shift+P -> Python: Select Interpreter -> .venv/bin/python
```

### Data root not found

Check that `ISS_SIMULATION_DATA_ROOT` points to the folder that contains `ISS_tle`, the VIIRS mosaic and the timelapse folders:

```bash
echo $ISS_SIMULATION_DATA_ROOT
ls $ISS_SIMULATION_DATA_ROOT
```

### Help command

The help command should work without running the full pipeline:

```bash
python3 -m pipelinentl.timelapse_pipeline --help
```
