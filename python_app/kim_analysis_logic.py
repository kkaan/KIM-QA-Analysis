import re
import pandas as pd
import numpy as np

def parse_centroid_file(filepath):
    """
    Parses the centroid file to extract seed and isocenter coordinates.
    Returns a dictionary with extracted values and the calculated expected centroid.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract Seed coordinates
    seeds = []
    for i in range(1, 4):
        match = re.search(f"Seed {i}, X=\s*([-\d\.]+), Y=\s*([-\d\.]+), Z=\s*([-\d\.]+)", content)
        if match:
            seeds.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    
    if len(seeds) != 3:
        raise ValueError("Could not find all 3 seeds in the centroid file.")

    # Extract Isocenter coordinates
    iso_match = re.search(r"Isocenter \(cm\), X=\s*([-\d\.]+), Y=\s*([-\d\.]+), Z=\s*([-\d\.]+)", content)
    if not iso_match:
        raise ValueError("Could not find Isocenter coordinates in the centroid file.")
    
    isocenter = [float(iso_match.group(1)), float(iso_match.group(2)), float(iso_match.group(3))]

    # Calculate Average Marker Position (Centroid of seeds)
    avg_marker = np.mean(seeds, axis=0)

    # Calculate Expected Centroid relative to Isocenter (in mm, assuming file is in cm? MATLAB code multiplies by 10)
    # MATLAB: Avg_marker_x_iso = 10*(Avg_marker_x - coordData(10));
    # Note: The MATLAB code seems to swap axes LATER. Let's stick to the file's X, Y, Z first.
    # The file says "Isocenter (cm)", so we multiply by 10 to get mm.
    
    expected_centroid_iso = 10 * (avg_marker - isocenter)

    # Apply Coordinate Transformation from MATLAB Staticloc.m
    # Avg_marker_x = Avg_marker_x_iso;
    # Avg_marker_y = Avg_marker_z_iso;
    # Avg_marker_z = -Avg_marker_y_iso;
    
    final_expected_centroid = {
        'x': expected_centroid_iso[0],
        'y': expected_centroid_iso[2],
        'z': -expected_centroid_iso[1]
    }

    return {
        'seeds': seeds,
        'isocenter': isocenter,
        'expected_centroid': final_expected_centroid
    }

def parse_trajectory_file(filepath):
    """
    Parses the trajectory file (CSV-like).
    Returns a pandas DataFrame with processed coordinates.
    """
    # Read the file, skipping the first line if it's just a header count or similar (MATLAB skips 1 line)
    # Based on the file content view, the first line IS the header.
    # "Frame No, Time (sec), ..."
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        # Fallback for potential formatting issues or if header is on second line
        df = pd.read_csv(filepath, skiprows=1)

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Ensure we have the necessary columns
    required_cols = ['Time (sec)', 'Gantry',
                     'Marker_0_AP', 'Marker_0_LR', 'Marker_0_SI',
                     'Marker_1_AP', 'Marker_1_LR', 'Marker_1_SI',
                     'Marker_2_AP', 'Marker_2_LR', 'Marker_2_SI']
    
    # Check if columns exist (handling potential naming variations if needed)
    # The file view shows "Marker_0_SI " (with space). The strip() above handles this.
    
    # Sort markers by SI position (Z) for each frame to ensure consistent ordering
    # MATLAB: Index the markers by SI position where 1 is the most cranial and 3 the most caudal
    # "most cranial" usually means highest SI value? 
    # MATLAB: sortedArray = sort(array, 'descend'); -> Yes, descending SI.
    
    # We need to do this row by row.
    
    processed_data = []

    for index, row in df.iterrows():
        # Extract markers
        m0 = {'ap': row['Marker_0_AP'], 'lr': row['Marker_0_LR'], 'si': row['Marker_0_SI']}
        m1 = {'ap': row['Marker_1_AP'], 'lr': row['Marker_1_LR'], 'si': row['Marker_1_SI']}
        m2 = {'ap': row['Marker_2_AP'], 'lr': row['Marker_2_LR'], 'si': row['Marker_2_SI']}
        
        markers = [m0, m1, m2]
        
        # Sort by SI (descending)
        markers.sort(key=lambda x: x['si'], reverse=True)
        
        # Calculate Centroid of measured markers
        # MATLAB: KIMData.xCent = (KIMData.x1 + KIMData.x2 + KIMData.x3)/3 ...
        # MATLAB Mapping:
        # x = LR (Column 5, 8, 11)
        # y = SI (Column 6, 9, 12)
        # z = AP (Column 4, 7, 10)
        
        # Let's compute the average LR, SI, AP first
        avg_lr = (m0['lr'] + m1['lr'] + m2['lr']) / 3.0
        avg_si = (m0['si'] + m1['si'] + m2['si']) / 3.0
        avg_ap = (m0['ap'] + m1['ap'] + m2['ap']) / 3.0
        
        # Map to Analysis Coordinates (X, Y, Z)
        # Based on MATLAB:
        # plot(..., KIMData.xCent, ..., KIMData.yCent, ..., KIMData.zCent)
        # legend('LAT (KIM)', 'LONG (KIM)', 'VRT (KIM)');
        # Wait, usually:
        # LAT = Lateral = LR = X
        # LONG = Longitudinal = SI = Y
        # VRT = Vertical = AP = Z
        
        # Let's re-verify MATLAB mapping in readKIMData:
        # eval(['KIMData.x' num2str(n) '= rawKIMData{5};']); -> 5th col is LR. So X = LR.
        # eval(['KIMData.y' num2str(n) '= rawKIMData{6};']); -> 6th col is SI. So Y = SI.
        # eval(['KIMData.z' num2str(n) '= rawKIMData{4};']); -> 4th col is AP. So Z = AP.
        
        # BUT in Staticloc.m:
        # Avg_marker_x = Avg_marker_x_iso;
        # Avg_marker_y = Avg_marker_z_iso;  <-- Y is Z_iso?
        # Avg_marker_z = -Avg_marker_y_iso; <-- Z is -Y_iso?
        
        # And KIMData.xCent = ... - Avg_marker_x;
        
        # Let's stick to the Trajectory File mapping first:
        # meas_x = LR
        # meas_y = SI
        # meas_z = AP
        
        processed_data.append({
            'time': row['Time (sec)'],
            'gantry': row['Gantry'],
            'meas_x': avg_lr,
            'meas_y': avg_si,
            'meas_z': avg_ap
        })

    return pd.DataFrame(processed_data)

def calculate_metrics(df, expected_centroid, time_interval=None):
    """
    Calculates deviations and metrics.
    df: DataFrame from parse_trajectory_file
    expected_centroid: dict from parse_centroid_file
    time_interval: tuple (start_time, end_time) or None
    """
    
    # Filter by time
    if time_interval:
        df = df[(df['time'] >= time_interval[0]) & (df['time'] <= time_interval[1])].copy()
    else:
        df = df.copy()

    # Calculate Deviations
    # Note: The MATLAB code subtracts the expected centroid from the measured centroid.
    # AND it seems to imply a coordinate swap for the EXPECTED centroid, but what about the MEASURED?
    # In readKIMData (MATLAB):
    # KIMData.xCent = (x1+x2+x3)/3 - Avg_marker_x;
    # KIMData.yCent = (y1+y2+y3)/3 - Avg_marker_y;
    # KIMData.zCent = (z1+z2+z3)/3 - Avg_marker_z;
    
    # So we must ensure Avg_marker_x/y/z matches the coordinate system of the measured data (LR, SI, AP).
    
    # In MATLAB:
    # Avg_marker_x = Avg_marker_x_iso; (Which was 10 * (SeedX - IsoX))
    # Avg_marker_y = Avg_marker_z_iso; (Which was 10 * (SeedZ - IsoZ))
    # Avg_marker_z = -Avg_marker_y_iso; (Which was -10 * (SeedY - IsoY))
    
    # This implies the Centroid File has a different coordinate system than the Trajectory File.
    # Centroid File: X, Y, Z
    # Trajectory File: LR, SI, AP
    
    # It looks like:
    # Traj LR (X) corresponds to Centroid X
    # Traj SI (Y) corresponds to Centroid Z ??
    # Traj AP (Z) corresponds to Centroid -Y ??
    
    # Let's apply this subtraction.
    
    df['dev_x'] = df['meas_x'] - expected_centroid['x']
    df['dev_y'] = df['meas_y'] - expected_centroid['y']
    df['dev_z'] = df['meas_z'] - expected_centroid['z']
    
    # Calculate Metrics
    metrics = {}
    for axis in ['x', 'y', 'z']:
        col = f'dev_{axis}'
        metrics[f'mean_{axis}'] = df[col].mean()
        metrics[f'std_{axis}'] = df[col].std()
        metrics[f'p5_{axis}'] = np.percentile(df[col], 5)
        metrics[f'p95_{axis}'] = np.percentile(df[col], 95)
        
    return df, metrics

import glob
import os

def parse_kim_data(folder_path):
    """
    Parses all MarkerLocationsGA_CouchShift_*.txt files in the folder.
    Returns a combined DataFrame with timestamps, gantry, and calculated centroids.
    """
    # Find all matching files
    files = glob.glob(os.path.join(folder_path, "MarkerLocationsGA_CouchShift_*.txt"))
    
    # Sort files by index (MarkerLocationsGA_CouchShift_0.txt, _1.txt, etc.)
    # Extract number from filename to sort correctly
    def get_file_index(filepath):
        match = re.search(r"MarkerLocationsGA_CouchShift_(\d+)\.txt", filepath)
        return int(match.group(1)) if match else -1
    
    files.sort(key=get_file_index)
    
    all_data = []
    
    for filepath in files:
        try:
            # Read file
            # Format: Frame No, Time, Gantry, ...
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            # Process each row to calculate centroid
            # Similar logic to parse_trajectory_file but we need to be robust
            
            for index, row in df.iterrows():
                # Extract markers
                m0 = {'ap': row['Marker_0_AP'], 'lr': row['Marker_0_LR'], 'si': row['Marker_0_SI']}
                m1 = {'ap': row['Marker_1_AP'], 'lr': row['Marker_1_LR'], 'si': row['Marker_1_SI']}
                m2 = {'ap': row['Marker_2_AP'], 'lr': row['Marker_2_LR'], 'si': row['Marker_2_SI']}
                
                markers = [m0, m1, m2]
                # Sort by SI (descending)
                markers.sort(key=lambda x: x['si'], reverse=True)
                
                avg_lr = (m0['lr'] + m1['lr'] + m2['lr']) / 3.0
                avg_si = (m0['si'] + m1['si'] + m2['si']) / 3.0
                avg_ap = (m0['ap'] + m1['ap'] + m2['ap']) / 3.0
                
                all_data.append({
                    'time': row['Time (sec)'],
                    'gantry': row['Gantry'],
                    'meas_x': avg_lr, # LR
                    'meas_y': avg_si, # SI
                    'meas_z': avg_ap, # AP
                    'file_index': get_file_index(filepath)
                })
                
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.DataFrame(all_data)
    
    # Normalize timestamps (start at 0)
    if not combined_df.empty:
        combined_df['time'] = combined_df['time'] - combined_df['time'].iloc[0]
        
    return combined_df

def parse_couch_shifts(filepath):
    """
    Parses couchShifts.txt to extract VRT, LNG, LAT shifts.
    Returns a list of shifts in mm.
    """
    shifts = []
    try:
        with open(filepath, 'r') as f:
            # Skip header
            next(f)
            # Read lines
            # Format: VRT, LNG, LAT
            # Example: -15.80, 125.50, -0.30
            
            vrt_vals = []
            lng_vals = []
            lat_vals = []
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    vrt_vals.append(float(parts[0]))
                    lng_vals.append(float(parts[1]))
                    lat_vals.append(float(parts[2]))
            
            # Calculate diffs and convert to mm (x10)
            # MATLAB: shiftsAP = diff(vrt) * 10;
            
            for i in range(len(vrt_vals) - 1):
                shift = {
                    'ap': (vrt_vals[i+1] - vrt_vals[i]) * 10,
                    'si': (lng_vals[i+1] - lng_vals[i]) * 10,
                    'lr': (lat_vals[i+1] - lat_vals[i]) * 10
                }
                shifts.append(shift)
                
    except Exception as e:
        print(f"Error parsing couch shifts: {e}")
        
    return shifts

def parse_robot_file(filepath):
    """
    Parses the Robot/Hexa trace file.
    Expects 7 columns. Returns DataFrame with time, x, y, z.
    """
    try:
        # Try reading with space delimiter
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
        
        # If only 1 column, maybe it's comma separated?
        if df.shape[1] < 4:
             df = pd.read_csv(filepath, header=None)
             
        # Rename columns (assuming first 4 are Time, X, Y, Z)
        # MATLAB: dataHexa.x = (1).*rawDataHexa{2}; y=3, z=4
        # Col 0: Time? MATLAB says rawDataHexa{1} is timestamps.
        
        df = df.iloc[:, :4]
        df.columns = ['time', 'x', 'y', 'z']
        
        return df
        
    except Exception as e:
        print(f"Error parsing robot file: {e}")
        return pd.DataFrame()

def process_interrupt_data(kim_df, robot_df, shifts, params=None):
    """
    Processes the data for interrupt analysis.
    """
    if kim_df.empty or robot_df.empty:
        return None, None

    # --- Step 1: Undo Couch Shifts from KIM Data ---
    # We need to know WHEN the shifts happened in the KIM data.
    # MATLAB uses 'shiftIndex' based on file lengths.
    # We have 'file_index' in kim_df.
    
    kim_df_processed = kim_df.copy()
    
    # Iterate through shifts
    # Shift 0 corresponds to transition from File 0 to File 1?
    # MATLAB: shiftIndex(n) = length(rawDataKIM{n}) + ...
    # So Shift 1 applies to everything AFTER File 0?
    # MATLAB loop:
    # for n = 1:noOfShifts
    #   dataKIM.yCent(shiftIndex(n):end) = dataKIM.yCent(...) - shiftsSI(n);
    
    # It seems shifts accumulate?
    # Or is it one shift per file transition?
    # "MarkerLocationsGA_CouchShift_0.txt" -> Initial
    # "MarkerLocationsGA_CouchShift_1.txt" -> After 1st shift?
    
    current_shift_ap = 0
    current_shift_si = 0
    current_shift_lr = 0
    
    # We assume shifts[i] happens before file[i+1]?
    # Let's assume shifts list length matches (num_files - 1).
    
    # Create a cumulative shift array for each row
    # But wait, MATLAB subtracts shiftsSI(n) from shiftIndex(n) to END.
    # This implies cumulative subtraction of EACH shift.
    
    # Let's identify the start index of each file > 0
    file_indices = sorted(kim_df['file_index'].unique())
    
    # Undo shifts
    for i, shift in enumerate(shifts):
        if i + 1 < len(file_indices):
            # Find index where file (i+1) starts
            file_idx = file_indices[i+1]
            mask = kim_df_processed['file_index'] >= file_idx
            
            kim_df_processed.loc[mask, 'meas_y'] -= shift['si']
            kim_df_processed.loc[mask, 'meas_x'] -= shift['lr']
            kim_df_processed.loc[mask, 'meas_z'] -= shift['ap']

    # --- Step 2: Time Alignment ---
    # Align KIM SI (y) with Robot Y (y)
    # MATLAB: findClosestSI(dataHexa, dataKIM)
    
    # Interpolate Robot Y to KIM timestamps
    from scipy.interpolate import interp1d
    
    # Ensure robot timestamps are sorted
    robot_df = robot_df.sort_values('time')
    
    # Create interpolator
    f_robot_y = interp1d(robot_df['time'], robot_df['y'], kind='linear', fill_value="extrapolate")
    
    # Search for optimal shift
    # MATLAB range: paramData(1):paramData(2):paramData(3) -> -400:0.01:20 ??
    # Let's use a reasonable range: -10s to +10s? Or larger if needed.
    # MATLAB default seems to be large.
    
    best_rmse = float('inf')
    best_shift = 0
    
    # Coarse search then fine search?
    # Let's try -50 to 50 seconds in 0.1s steps
    search_range = np.arange(-50, 50, 0.1)
    
    kim_si = kim_df_processed['meas_y'].values
    kim_time = kim_df_processed['time'].values
    
    for shift_val in search_range:
        shifted_time = kim_time + shift_val
        # Only compare where times overlap
        # But simpler to just interpolate robot at shifted times
        
        # We need robot_y at (kim_time + shift)
        # Wait, MATLAB: interp1(dataHexa.timestamps, dataHexa.y, dataKIM.timestamps + shiftValues(n))
        # So we shift KIM time to match Robot time?
        # If KIM starts at 0, and Robot starts at 0, but there is latency/offset.
        
        interp_robot_y = f_robot_y(kim_time + shift_val)
        
        # RMSE
        rmse = np.sqrt(np.mean((kim_si - interp_robot_y)**2))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift_val
            
    # Apply best shift + latency (0.350 from MATLAB)
    latency = 0.350
    total_shift = best_shift + latency
    kim_df_processed['time'] = kim_df_processed['time'] + total_shift
    
    # --- Step 3: Reapply Shifts ---
    # Add shifts back to KIM
    for i, shift in enumerate(shifts):
        if i + 1 < len(file_indices):
            file_idx = file_indices[i+1]
            mask = kim_df_processed['file_index'] >= file_idx
            
            kim_df_processed.loc[mask, 'meas_y'] += shift['si']
            kim_df_processed.loc[mask, 'meas_x'] += shift['lr']
            kim_df_processed.loc[mask, 'meas_z'] += shift['ap']
            
            # Add shifts to Robot data at corresponding times
            # We need to find WHEN this shift happens in Robot time
            # MATLAB: hexaShiftIndex = find(abs((dataHexa.timestamps - dataKIM.timestamps(shiftIndex(n)))) < ...)
            
            # Find timestamp of the first point of the file in KIM (shifted time)
            shift_time = kim_df_processed.loc[kim_df_processed['file_index'] == file_idx, 'time'].iloc[0]
            
            robot_mask = robot_df['time'] >= shift_time
            robot_df.loc[robot_mask, 'y'] += shift['si']
            robot_df.loc[robot_mask, 'x'] += shift['lr']
            robot_df.loc[robot_mask, 'z'] += shift['ap']

    # --- Step 4: Calculate Metrics ---
    # Interpolate Robot to KIM time (final)
    f_robot_x = interp1d(robot_df['time'], robot_df['x'], kind='linear', fill_value="extrapolate")
    f_robot_y = interp1d(robot_df['time'], robot_df['y'], kind='linear', fill_value="extrapolate")
    f_robot_z = interp1d(robot_df['time'], robot_df['z'], kind='linear', fill_value="extrapolate")
    
    kim_time_final = kim_df_processed['time']
    
    robot_interp_x = f_robot_x(kim_time_final)
    robot_interp_y = f_robot_y(kim_time_final)
    robot_interp_z = f_robot_z(kim_time_final)
    
    # Differences (KIM - Robot)
    diff_x = kim_df_processed['meas_x'] - robot_interp_x
    diff_y = kim_df_processed['meas_y'] - robot_interp_y
    diff_z = kim_df_processed['meas_z'] - robot_interp_z
    
    metrics = {
        'mean_lr': np.mean(diff_x),
        'mean_si': np.mean(diff_y),
        'mean_ap': np.mean(diff_z),
        'std_lr': np.std(diff_x),
        'std_si': np.std(diff_y),
        'std_ap': np.std(diff_z),
        'p5_lr': np.percentile(diff_x, 5),
        'p95_lr': np.percentile(diff_x, 95),
        'p5_si': np.percentile(diff_y, 5),
        'p95_si': np.percentile(diff_y, 95),
        'p5_ap': np.percentile(diff_z, 5),
        'p95_ap': np.percentile(diff_z, 95),
    }
    
    # Add interpolated robot data to dataframe for plotting
    kim_df_processed['robot_x'] = robot_interp_x
    kim_df_processed['robot_y'] = robot_interp_y
    kim_df_processed['robot_z'] = robot_interp_z
    
    return kim_df_processed, metrics
