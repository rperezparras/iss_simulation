# ISS nighttime image georeferencing pipeline

Pipeline for automatic georeferencing of nighttime images acquired from the International Space Station (ISS). The workflow combines Blender-based ISS view simulation, image matching, control-point projection, filtering, Thin Plate Spline georeferencing, VIIRS alignment and optional optical-flow correction.

The main user-facing entry point is:

```bash
python -m pipelinentl.timelapse_pipeline
```

The auxiliary scripts in `pipelinentl/` are called internally by the pipeline and normally do not need to be edited.

## Repository layout

```text
scripts_v3/
├── README.md
├── requirements.txt
├── requirements-blender.txt
├── .gitignore
└── pipelinentl/
    ├── timelapse_pipeline.py
    ├── match_timelapse.py
    ├── angle_search.py
    ├── generate_timelapse.py
    ├── project_timelapse.py
    ├── filter_points.py
    ├── georef_timelapse.py
    ├── viirs_roi_crop.py
    ├── optical_flow.py
    └── correct_points.py
```

## Installation

The recommended Python version is **Python 3.11**.

```bash
git clone https://github.com/YOUR_USER/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-blender.txt
```

If `bpy` cannot be installed through pip on your system, install Blender separately and run the pipeline from a Blender-compatible Python environment. The non-Blender dependencies are listed in `requirements.txt`.

If you need a CUDA-enabled PyTorch build, install `torch` and `torchvision` using the official PyTorch selector for your CUDA version before installing the rest of the requirements.

## External data

Large external data files are not included in this repository. The pipeline expects a data root folder containing TLE files, the night-Earth texture for Blender, the VIIRS mosaic and the timelapse folders.

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
└── ISS067-E-362508-363421/
    ├── pics/
    ├── output/
    ├── filtered_points/
    ├── geo/
    ├── viirs_cropped_aligned/
    ├── flow/
    ├── corrected_points/
    └── geo_corrected/
```

The experiment folder name is built as:

```text
<MISSION>-E-<START_ID>-<END_ID>
```

For example:

```text
ISS067-E-362508-363421
```

## Quick start

Show all available options:

```bash
python -m pipelinentl.timelapse_pipeline --help
```

Run the full pipeline:

```bash
python -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421
```

Pass the data root directly instead of using an environment variable:

```bash
python -m pipelinentl.timelapse_pipeline \
  --data-root /path/to/your/iss_data \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421
```

## Matching backend

The default matching backend is `vismatch`, using SuperPoint + LightGlue. For a normal installation, no local matcher repository path is needed.

```bash
python -m pip install vismatch
```

The old `image-matching-models` repository is only kept as an optional legacy backend. New users should normally use `vismatch`. If you explicitly need the legacy backend, define:

```bash
export IMAGE_MATCHING_MODELS_DIR=/path/to/image-matching-models
```

and run:

```bash
python -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421 \
  --matcher-backend matching
```

## Useful options

Disable automatic angle search and use fallback yaw, pitch and roll:

```bash
python -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421 \
  --no-angle-search \
  --yaw 12.5 \
  --pitch 63.5 \
  --roll -1.0
```

Run without optical-flow correction:

```bash
python -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421 \
  --no-optical-flow
```

Run optical-flow correction but skip the second georeferencing:

```bash
python -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421 \
  --second-georef-mode none
```

Possible values for `--second-georef-mode` are:

```text
none
sample
full
```

Force selected steps to rerun:

```bash
python -m pipelinentl.timelapse_pipeline \
  --mission ISS067 \
  --start-id 362508 \
  --end-id 363421 \
  --rerun-matching \
  --rerun-filtering
```

## Outputs

For an experiment such as:

```text
ISS067-E-362508-363421
```

main outputs are written to:

```text
ISS067-E-362508-363421/
├── pics/                     # Original ISS images
├── output/                   # Blender renders and projected control points
├── output/matches/           # Matching CSVs and diagnostic plots
├── filtered_points/          # Filtered QGIS .points files
├── geo/                      # First georeferenced ISS GeoTIFFs
├── viirs_cropped_aligned/    # VIIRS crops aligned to ISS georeferenced frames
├── flow/                     # Optical-flow fields
├── corrected_points/         # Control points corrected using optical flow
└── geo_corrected/            # Second georeferencing outputs
```

## Notes

The pipeline uses GDAL command-line tools such as `gdal_translate` and `gdalwarp` during georeferencing. Make sure GDAL is available in your system environment.

Large datasets, rendered images, GeoTIFFs, `.points`, `.npy` flow files and cache folders should not be committed to GitHub.
