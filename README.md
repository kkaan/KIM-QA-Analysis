# KIM QA Analysis - Python Application

A comprehensive GUI application for analyzing KIM (Kilo-voltage Imaging for Motion management) QA data using 6 DoF robot measurements. This application provides an intuitive interface for performing static localization and treatment interrupt analyses.

## Overview

The `python_app` is a modern Python-based GUI application built with CustomTkinter that replaces and extends the original MATLAB workflows. It provides two main analysis modules:

1. **Static Analysis** - Validates KIM's tracking accuracy during static conditions
2. **Interrupt Analysis** - Evaluates KIM's performance during treatment interruptions

## Features

### Static Analysis Module
- Load and parse centroid files (seed and isocenter coordinates)
- Load and visualize KIM trajectory data
- Interactive time-series plotting with deviation calculations
- Dual-axis plots showing both time and gantry angle
- Interactive time range selection using span selector
- Calculate statistical metrics (mean, std, 5th/95th percentiles)
- Export results and plots

### Interrupt Analysis Module
- Parse multiple KIM trajectory files from a folder
- Load robot trace data for comparison
- Auto-detect couch shift files
- Align and compare KIM measurements with robot ground truth
- Calculate tracking errors and performance metrics
- Pass/fail criteria evaluation
- Export analysis results and comparison plots

## Installation

### Requirements
- Python 3.7 or higher
- Dependencies listed in `requirements.txt`

### Setup

1. Navigate to the `python_app` directory:
```bash
cd python_app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

The required packages are:
- `customtkinter` - Modern GUI framework
- `matplotlib` - Plotting and visualization
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations

## Usage

### Running the Application

Launch the GUI application:
```bash
python kim_analysis_app.py
```

### Static Analysis Workflow

1. **Load Centroid File**: Click "Load Centroid File" and select your centroid data file (e.g., `Centroid_*.txt`)
2. **Load Trajectory File**: Click "Load Trajectory File" and select the KIM trajectory data
3. **View Plot**: The trajectory data will be automatically plotted with deviation from expected centroid
4. **Select Time Range**: 
   - Use the interactive span selector on the plot, or
   - Manually enter start/end times in the sidebar
5. **Calculate Metrics**: Click "Calculate Metrics" to compute statistics for the selected time interval
6. **Save Results**: Export metrics and plots to a directory of your choice

### Interrupt Analysis Workflow

1. **Load Trajectory Folder**: Select the folder containing `MarkerLocationsGA_CouchShift_*.txt` files
2. **Load Robot File**: Select the robot/hexa trace file for ground truth comparison
3. **Verify Couch Shifts**: The app will auto-detect `couchShifts.txt` in the trajectory folder
4. **Analyze**: Click "Analyze Interrupt" to process the data
5. **Review Results**: 
   - View aligned KIM vs Robot measurements in the plot
   - Check metrics and pass/fail status
6. **Save Results**: Export analysis results and comparison plots

## Application Architecture

### Main Components

- **`kim_analysis_app.py`**: GUI application with CustomTkinter interface
  - `KimAnalysisApp` class: Main application window with tabbed interface
  - Static analysis tab with interactive plotting
  - Interrupt analysis tab with multi-file processing
  
- **`kim_analysis_logic.py`**: Core analysis functions
  - `parse_centroid_file()`: Extract seed and isocenter coordinates
  - `parse_trajectory_file()`: Process KIM trajectory data
  - `calculate_metrics()`: Compute statistical metrics
  - `parse_kim_data()`: Parse multiple KIM data files
  - `parse_couch_shifts()`: Extract couch shift values
  - `parse_robot_file()`: Load robot trace data
  - `process_interrupt_data()`: Align and compare KIM vs Robot data

### Data Flow

**Static Analysis:**
```
Centroid File → Expected Centroid Calculation
                        ↓
Trajectory File → Deviation Calculation → Metrics → Results
```

**Interrupt Analysis:**
```
Trajectory Folder → KIM Data Parsing
Couch Shifts File → Shift Extraction     → Data Alignment → Comparison → Metrics
Robot File → Robot Data Parsing
```

## File Formats

### Input Files

- **Centroid Files**: Text files containing seed and isocenter coordinates
- **Trajectory Files**: CSV/TXT files with time-series position data
- **KIM Data Files**: `MarkerLocationsGA_CouchShift_*.txt` format
- **Couch Shifts**: `couchShifts.txt` with VRT, LNG, LAT shifts
- **Robot Files**: 7-column format with timestamp and position data

### Output Files

- **Metrics.txt**: Statistical analysis results (mean, std, percentiles)
- **Trace_Plot.png**: Visualization of trajectory data
- **Interrupt_Metrics.txt**: Interrupt analysis results with pass/fail status
- **Interrupt_Plot.png**: KIM vs Robot comparison plot

## Technical Details

- **GUI Framework**: CustomTkinter (modern themed Tkinter)
- **Plotting**: Matplotlib with interactive widgets
- **Data Processing**: Pandas DataFrames for efficient manipulation
- **Coordinate System**: LR (Left-Right), SI (Superior-Inferior), AP (Anterior-Posterior)

## Building Executables

### Automated Builds (GitHub Releases)

When you create a new release on GitHub, the workflow automatically:
1. Builds a standalone Windows executable using PyInstaller
2. Packages it as a ZIP file
3. Uploads it as a release asset

To trigger an automated build:
```bash
git tag v1.0.0
git push origin v1.0.0
# Then create a release from the tag on GitHub
```

### Manual Build (Local)

To build the executable locally:

1. Install PyInstaller:
```bash
pip install pyinstaller
```

2. Build using the spec file:
```bash
cd python_app
pyinstaller KIM-QA-Analysis.spec
```

3. The executable will be in `python_app/dist/KIM-QA-Analysis.exe`

Alternatively, build without the spec file:
```bash
cd python_app
pyinstaller --name="KIM-QA-Analysis" --windowed --onefile kim_analysis_app.py
```

## Legacy MATLAB Code


The original MATLAB implementations are still available in the repository:
- `Elekta/App_Static_loc.mlapp` - Static localization MATLAB app
- `Elekta/Staticloc.m` - Static localization MATLAB script
- See respective folders for linac-specific implementations (Elekta, Varian Truebeam)

## References

For detailed information about the KIM QA analysis methodology, see:
- `KIM QA publication.pdf` - Analysis procedure documentation
- `KIM commissioning report for Nepean Cancer Center.pdf` - Commissioning details
