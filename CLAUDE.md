# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KIM-QA-Analysis is a Python GUI application for analyzing KIM (Kilo-voltage Imaging for Motion management) QA data. It compares measured fiducial marker positions from KIM tracking software against expected positions from centroid files and robot ground truth data. The app performs static localization analysis and treatment interrupt analysis for radiation therapy QA.

## Commands

### Run the application
```bash
cd python_app
pip install -r requirements.txt
python kim_analysis_app.py
```

### Run tests
```bash
python test_logic.py
```
Tests require the `Lyrebird Data/` directory (git-ignored sample data).

### Lint
```bash
pylint $(find . -name "*.py" ! -path "./.venv/*")
```
Pylint config is in `.pylintrc`. CI runs pylint against Python 3.8, 3.9, 3.10.

### Build standalone executable
```bash
cd python_app
pip install pyinstaller
pyinstaller KIM-QA-Analysis.spec
```
Output: `python_app/dist/KIM-QA-Analysis.exe`

## Architecture

### Core Python app (`python_app/`)

Two files with clear separation:

- **`kim_analysis_app.py`** — CustomTkinter GUI with two tabs (Static Analysis, Interrupt Analysis). Each tab has a sidebar for inputs and a main area with matplotlib plots. Uses `matplotlib.widgets.SpanSelector` for interactive time range selection.

- **`kim_analysis_logic.py`** — Pure analysis functions with no GUI dependencies. All data parsing, coordinate transforms, time alignment, and metric calculations live here.

### Key analysis functions in `kim_analysis_logic.py`

- `parse_centroid_file()` — Extracts seed/isocenter coordinates from centroid files, dynamically detects 2–9 seeds via regex
- `parse_trajectory_file()` — Reads KIM trajectory CSV data, dynamically detects marker columns (Marker_0_AP/LR/SI, etc.)
- `calculate_metrics()` — Computes deviations and statistics (mean, std, 5th/95th percentiles) with optional time filtering
- `parse_kim_data()` — Batch processes `MarkerLocationsGA_CouchShift_*.txt` files from a folder
- `parse_couch_shifts()` — Parses `couchShifts.txt` for couch movement data
- `parse_robot_file()` — Reads 7-column robot trace files (ground truth)
- `process_interrupt_data()` — Aligns KIM data to robot data via RMSE minimization with couch shift correction and 0.350s latency offset

### Coordinate system

Centroid files use (X, Y, Z) in cm. Trajectory files use (LR, SI, AP). Internal representation is (x, y, z) in mm. The transform applied is: x=x_centroid, y=z_centroid, z=-y_centroid.

### Legacy MATLAB code

`Elekta/` and `Varian Truebeam/` contain the original MATLAB App Designer implementations (.mlapp + .m files). The Python app is a refactored replacement. These directories serve as reference.

### Analysis progress (`C:\Users\kanke\Repo\phantom-progress`)

Separate repository where we document analysis progress for phantom experiments and QA results. Added as a working directory alongside this project.

### GitHub Pages (`docs/`)

- `index.html` — PRIME trial landing page
- `kv-simulator.html` — Interactive kV imaging simulator (browser-based)

## CI/CD

- **build-release.yml** — On GitHub release: builds Windows exe via PyInstaller (Python 3.11), uploads as release asset
- **pylint.yml** — On every push: runs pylint across Python 3.8/3.9/3.10
