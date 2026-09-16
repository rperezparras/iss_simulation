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

The base installation includes the recommended neural matcher, `vismatch`, using SuperPoint + LightGlue.

Install the Blender-related requirements:

```bash
python3 -m pip install ".[blender]"
```

After activating the virtual environment, `python` and `python3` should normally point to the same environment. In this README, `python3` is used explicitly for Linux compatibility.

If you use IPython, install and launch it from the virtual environment:

```bash
python3 -m pip install ipython
python3 -m IPython
```

This avoids accidentally launching a system `ipython3` tied to a different Python installation.

## Important dependency notes

The pipeline uses Blender through Python (`bpy`) and `vismatch` as its recommended neural image matcher.

The following versions are intentionally constrained because they form the currently tested compatible environment:

```text
Python ~= 3.11
numpy < 2
opencv-python >= 4.5.4, < 4.12
lightning == 2.3.3
vismatch == 1.3.1
rerun-sdk < 0.23
setuptools >= 61, < 81
```

These constraints are important for two reasons:

1. Recent NumPy/OpenCV combinations can be incompatible with some `bpy` builds.
2. `vismatch 1.3.2` requires newer Lightning and Setuptools versions than the environment currently used by this project and can also cause pip to upgrade NumPy/OpenCV when installed independently.

For this reason, do **not** install or upgrade `vismatch` independently with an unconstrained command such as:

```bash
python3 -m pip install vismatch
```

Instead, install the project dependencies together:

```bash
python3 -m pip install .
```

This lets pip resolve the tested dependency set consistently.

The project uses `opencv-python` rather than `opencv-python-headless` because `vismatch` directly depends on `opencv-python`. Installing both OpenCV distributions in the same environment should be avoided because both provide the `cv2` module.

If an existing environment was modified by an incompatible `vismatch` installation, restore the tested versions with:

```bash
python3 -m pip install \
  "numpy<2" \
  "opencv-python>=4.5.4,<4.12" \
  "lightning==2.3.3" \
  "vismatch==1.3.1" \
  "rerun-sdk<0.23" \
  "setuptools>=61,<81"
```

Then check the main packages:

```bash
python3 - <<'PY'
import sys
import numpy
import cv2
import torch
import lightning
import vismatch

print("Python:", sys.version.split()[0])
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("torch:", torch.__version__)
print("lightning:", lightning.__version__)
print("vismatch:", vismatch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

To verify the actual neural matcher, not only the import:

```bash
python3 - <<'PY'
import torch
from vismatch import get_matcher

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

matcher = get_matcher("superpoint-lightglue", device=device)
print("SuperPoint + LightGlue loaded successfully")
PY
```

The first run may download model weights.

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

The default matching backend is `vismatch`, using `superpoint-lightglue`.

`vismatch` is installed together with the normal project dependencies. A separate `pip install vismatch` is neither required nor recommended because it can bypass the version constraints required by the rest of the project.

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

### `vismatch` or `torch` cannot be imported from IPython

Check which interpreter IPython is using:

```python
import sys
print(sys.executable)
```

It should point to the project's `.venv`.

If it does not, install and launch IPython from the virtual environment:

```bash
python3 -m pip install ipython
python3 -m IPython
```

### `pip check` reports `bpy 4.4.0 is not supported on this platform`

On some Linux installations, `pip check` can report:

```text
bpy 4.4.0 is not supported on this platform
```

If `bpy` imports correctly, verify the actual runtime directly:

```bash
python3 - <<'PY'
import sys
import platform
import bpy

print("Python:", sys.version)
print("System:", platform.system(), platform.machine())
print("glibc:", platform.libc_ver())
print("Blender/bpy:", bpy.app.version_string)
print("bpy:", bpy.__file__)
PY
```

If this reports Blender 4.4.0 from the project's `.venv`, the runtime installation is usable even if `pip check` still emits that platform warning.

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
