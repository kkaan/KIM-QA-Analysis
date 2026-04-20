"""Generate a step-jump trajectory figure for a PRIME fraction.

Reads the MarkerLocationsGA_CouchShift_*.txt files plus couchShifts.txt from a
PRIME trajectory log directory, plots the raw centroid trajectory in a real-time
zoom window with beam-off pauses compressed on the x-axis, and highlights a
specific intra-fraction couch correction. Correction #1 is the initial
localisation shift and is never counted as an intra-fraction event.

This module exposes a reusable `make_figure(config)` function so the same
figure recipe can be driven for any PRIME fraction / intervention. The
`main()` wrapper below reproduces the PAT01 FX04 abstract figure committed at
`docs/prime_pat01_fx04_stepjumps.png`.

Run from the repo root:
    python abstract-figures/make_prime_stepjump_figure.py
"""

import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script lives in abstract-figures/ but reuses parsers from the main GUI
# package in python_app/. Add the package directory to sys.path so the import
# below works regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python_app"))

from kim_analysis_logic import parse_centroid_file  # noqa: E402


# Strip thousands-separator commas inside the Time field once t > 1000 s
# (e.g. "1,000.305" -> "1000.305"). The Time column is the only field in this
# file format that ever crosses 1000 with a decimal, so a comma sitting between
# a digit and three more digits followed by a decimal point is unambiguous.
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\.)")


# Legacy MATLAB convention: blue=LR, green=SI, red=AP
AXIS_SPECS = [
    ("meas_x", "LR deviation (mm)", "#1f77b4"),
    ("meas_y", "SI deviation (mm)", "#2ca02c"),
    ("meas_z", "AP deviation (mm)", "#d62728"),
]

# Absolute limit: frames with any centroid axis exceeding this magnitude are
# pre-tracking-lock sentinel placeholders (KIM exports a fixed ~135 mm value
# before the first 3D position is computed). Real PRIME PAT01 marker positions
# sit within roughly +/-25 mm of the treatment isocentre.
SANE_LIMIT_MM = 50.0

# Per-axis catastrophic single-frame tracking-glitch filter: drop frames whose
# centroid deviates from the per-axis median by more than these thresholds.
# LR is tight because PRIME LR motion is typically sub-mm; SI/AP are wider to
# preserve real respiratory excursions of ~6-8 mm.
GLITCH_DEV_MM = {"meas_x": 5.0, "meas_y": 12.0, "meas_z": 12.0}

# Within the locked-gantry sub-arcs, KIM only acquires in short bursts with
# multi-second gaps in between. Lines are broken at gaps longer than this so
# the figure does not falsely suggest continuous tracking.
GAP_BREAK_S = 5.0

# Beam-off / acquisition pauses longer than GAP_BREAK_S are visually compressed
# on the x-axis to this many seconds, with a hatched band marking the
# discontinuity. Keeps the trace dense without lying about timing.
GAP_COMPRESS_S = 3.0

OVERLAY_LABEL = "no-correction counterfactual"


def _file_index(filepath: str) -> int:
    match = re.search(r"MarkerLocationsGA_CouchShift_(\d+)\.txt", filepath)
    return int(match.group(1)) if match else -1


def read_couch_rows(couch_file: str) -> list:
    """Read absolute couch positions (VRT, LNG, LAT in cm) from a PRIME
    couchShifts.txt as a list of (vrt, lng, lat) float tuples.

    Tolerates:
      - A standard header line ("VRT (cm), LNG (cm), LAT (cm)").
      - FX01's anomaly where two recording sessions were concatenated without
        a newline separator so a data row like "-8.2,10.2,0.2VRT (cm), ..."
        embeds a second header mid-line.
    """
    with open(couch_file, "r") as fh:
        text = fh.read()
    fragments = re.split(r"VRT[^\n]*", text)
    rows = []
    for frag in fragments:
        for line in frag.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                vrt = float(parts[0])
                lng = float(parts[1])
                lat = float(parts[2])
            except ValueError:
                continue
            rows.append((vrt, lng, lat))
    return rows


def couch_rows_to_shifts(rows: list) -> list:
    """Convert a list of (VRT, LNG, LAT) absolute couch positions in cm to a
    list of inter-row shift dicts with keys {lr, si, ap} in mm. Matches the
    sign convention of kim_analysis_logic.parse_couch_shifts (Elekta convention:
    no axis negation).
    """
    shifts = []
    for i in range(len(rows) - 1):
        v0, l0, a0 = rows[i]
        v1, l1, a1 = rows[i + 1]
        shifts.append({
            "ap": (v1 - v0) * 10.0,
            "si": (l1 - l0) * 10.0,
            "lr": (a1 - a0) * 10.0,
        })
    return shifts


def compress_time_gaps(real_times_s, gap_threshold_s, compressed_width_s):
    """Map real timestamps to display timestamps, collapsing each pause longer
    than gap_threshold_s to a fixed compressed_width_s.

    Returns (display_times_s, gap_real_intervals, gap_display_intervals) where
    each interval list contains (start, end) tuples in seconds marking the
    pauses in real and display coordinates respectively.
    """
    real = np.asarray(real_times_s, dtype=float)
    if len(real) == 0:
        return real, [], []
    display = np.empty_like(real)
    display[0] = real[0]
    gap_real = []
    gap_display = []
    for i in range(1, len(real)):
        dt = real[i] - real[i - 1]
        if dt > gap_threshold_s:
            display[i] = display[i - 1] + compressed_width_s
            gap_real.append((real[i - 1], real[i]))
            gap_display.append((display[i - 1], display[i]))
        else:
            display[i] = display[i - 1] + dt
    return display, gap_real, gap_display


def load_kim_centroid(folder: str, expected_centroid: dict) -> pd.DataFrame:
    """Load and concatenate the MarkerLocationsGA trajectory files for this
    fraction and return the per-frame centroid as a *deviation* from the
    planned isocentre, by subtracting the expected centroid (LR/SI/AP in mm)
    computed by parse_centroid_file from the patient's seed/iso file.

    Quirks of the PRIME export handled here:
    - Only CouchShift_0.txt carries a header; later files are headerless
      continuations of the same column layout.
    - Once the time exceeds 1000 s, the Time field is written with a
      thousands-separator comma ("1,000.305") which would otherwise be parsed
      as an extra column.
    The repo's parse_kim_data assumes every file has a header and fails on
    these files, so we read the header once from file 0 and reuse it for all
    subsequent files.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "MarkerLocationsGA_CouchShift_*.txt")),
        key=_file_index,
    )
    if not files:
        raise FileNotFoundError(f"No MarkerLocationsGA_CouchShift_*.txt in {folder}")

    # Positional column layout for the first 9 fields. This is the same in
    # every PRIME PAT01 file we've seen so far AND in FX01's file 1 rows that
    # use the 3-marker extended layout (which has extra Marker_2 columns
    # appended after position 8, not inserted before). We intentionally read
    # only the first 9 fields and drop the rest to avoid misalignment when a
    # file mixes 22-column and 29-column rows (as FX01's file 1 does).
    BASE_COLS = [
        "Frame No",
        "Time (sec)",
        "Gantry",
        "Marker_0_AP",
        "Marker_0_LR",
        "Marker_0_SI",
        "Marker_1_AP",
        "Marker_1_LR",
        "Marker_1_SI",
    ]

    segments = []
    for filepath in files:
        fi = _file_index(filepath)
        with open(filepath, "r") as fh:
            raw = fh.read()
        # Drop the header line on file 0 (and on any other file that happens
        # to carry one).
        lines = raw.splitlines()
        if lines and lines[0].lstrip().startswith("Frame"):
            lines = lines[1:]
        rows = []
        for line in lines:
            if not line.strip():
                continue
            # Strip thousands-separator commas inside Time field (e.g.
            # "1,000.305" -> "1000.305").
            line = _THOUSANDS_COMMA_RE.sub("", line)
            parts = line.split(",")
            if len(parts) < 9:
                continue
            rows.append(parts[:9])
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=BASE_COLS)
        for c in BASE_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["file_index"] = fi
        segments.append(df)

    if not segments:
        raise SystemExit(f"No data rows in any trajectory file under {folder}")

    combined = pd.concat(segments, ignore_index=True)

    out = pd.DataFrame({
        "time": combined["Time (sec)"].astype(float),
        "gantry": combined["Gantry"].astype(float),
        "meas_x": combined[["Marker_0_LR", "Marker_1_LR"]].astype(float).mean(axis=1),
        "meas_y": combined[["Marker_0_SI", "Marker_1_SI"]].astype(float).mean(axis=1),
        "meas_z": combined[["Marker_0_AP", "Marker_1_AP"]].astype(float).mean(axis=1),
        "file_index": combined["file_index"].astype(int),
    })
    out["time"] = out["time"] - out["time"].iloc[0]
    out = out.dropna(subset=["time", "meas_x", "meas_y", "meas_z"])

    # Convert raw KIM imaging-frame centroid to a deviation from the planned
    # isocentre by subtracting the patient's expected centroid. parse_centroid_file
    # already returns the expected values in the same (LR, SI, AP) axis convention
    # as meas_x/meas_y/meas_z - subtract directly without further axis remapping.
    out["meas_x"] = out["meas_x"] - expected_centroid["x"]
    out["meas_y"] = out["meas_y"] - expected_centroid["y"]
    out["meas_z"] = out["meas_z"] - expected_centroid["z"]

    # Drop sentinel/placeholder frames. Before tracking locks (first ~30 frames
    # of file 0), KIM exports a fixed placeholder around (135, -4.7, 135) far
    # outside any plausible patient-frame position. After subtracting the expected
    # centroid these still sit at ~(148, -7, 134) so the |.| < 50 mm gate still
    # catches them.
    sane = (out[["meas_x", "meas_y", "meas_z"]].abs() < SANE_LIMIT_MM).all(axis=1)
    n_sentinel = (~sane).sum()
    if n_sentinel:
        print(f"Dropped {n_sentinel} sentinel frames (>|{SANE_LIMIT_MM:.0f}| mm)")
    out = out[sane].reset_index(drop=True)

    # Drop catastrophic single-frame glitches (per-axis median deviation).
    keep = pd.Series(True, index=out.index)
    for col, limit in GLITCH_DEV_MM.items():
        med = out[col].median()
        keep &= (out[col] - med).abs() <= limit
    n_glitch = (~keep).sum()
    if n_glitch:
        print(f"Dropped {n_glitch} glitch frames (per-axis median-deviation filter)")
    out = out[keep].reset_index(drop=True)

    out["time"] = out["time"] - out["time"].iloc[0]
    return out


def make_figure(config: dict) -> None:
    """Generate one step-jump trajectory figure from a config dict.

    See module docstring for the config-key contract. Config keys:
        kim_folder, couch_file, centroid_file, output_png,
        couch_row_count (optional, default None = use whole file),
        localisation_shift_idx (default 0),
        headline_shift_idx (which shift to highlight in gold),
        window_min (tuple or None = auto-compute),
        auto_window_half_width_min (default 3.5),
        y_limits (dict),
        overlay_from_cm, overlay_to_cm (tuples of 3 floats: VRT, LNG, LAT in cm),
        suptitle, subtitle.
    """
    kim_folder = config["kim_folder"]
    couch_file = config["couch_file"]
    centroid_file = config["centroid_file"]
    output_png = Path(config["output_png"])
    couch_row_count = config.get("couch_row_count")
    localisation_shift_idx = config.get("localisation_shift_idx", 0)
    headline_shift_idx = config["headline_shift_idx"]
    window_min = config.get("window_min")
    auto_half = config.get("auto_window_half_width_min", 3.5)
    y_limits = config["y_limits"]
    overlay_from_cm = config["overlay_from_cm"]
    overlay_to_cm = config["overlay_to_cm"]
    suptitle = config["suptitle"]
    subtitle = config["subtitle"]
    headline_label = config.get("headline_label", "Headlined correction")

    # --- load -----------------------------------------------------------------
    try:
        centroid = parse_centroid_file(centroid_file)
        expected = centroid["expected_centroid"]
        kim_df = load_kim_centroid(kim_folder, expected)
        couch_rows = read_couch_rows(couch_file)
    except Exception as exc:
        raise SystemExit(f"Failed to read PRIME logs from {kim_folder}: {exc}")

    print(
        f"Expected centroid (mm): "
        f"LR={expected['x']:+.2f} SI={expected['y']:+.2f} AP={expected['z']:+.2f}"
    )

    if kim_df.empty:
        raise SystemExit("parse_kim_data returned an empty DataFrame.")

    n_files = kim_df["file_index"].nunique()

    # Slice couch rows to the LAST couch_row_count entries if requested. This
    # handles fractions whose couchShifts.txt contains stale leading rows from
    # a prior recording session (e.g. PAT01 FX01): only the last n_files rows
    # of couchShifts.txt are real, yielding n_files - 1 deltas.
    if couch_row_count is not None:
        if len(couch_rows) < couch_row_count:
            raise SystemExit(
                f"couchShifts.txt has {len(couch_rows)} rows, "
                f"but couch_row_count={couch_row_count} requires at least that "
                f"many."
            )
        couch_rows = couch_rows[-couch_row_count:]

    shifts = couch_rows_to_shifts(couch_rows)

    expected_deltas = n_files - 1
    if len(shifts) != expected_deltas:
        raise SystemExit(
            f"Expected {expected_deltas} couch shifts for {n_files} trajectory "
            f"files, found {len(shifts)} (from {len(couch_rows)} couch rows)."
        )

    # Tag each contiguous burst (within the same file_index, no gaps > threshold)
    # so the plot can break lines across acquisition pauses.
    dt = kim_df["time"].diff()
    new_burst = (kim_df["file_index"].diff().fillna(0) != 0) | (dt > GAP_BREAK_S)
    kim_df["burst_id"] = new_burst.cumsum()

    # --- console summary ------------------------------------------------------
    print(f"Loaded {len(kim_df)} frames across {n_files} segments")
    for fi, seg in kim_df.groupby("file_index"):
        print(
            f"  segment {fi}: {len(seg):4d} frames, "
            f"{seg['time'].min():7.1f}-{seg['time'].max():7.1f} s"
        )
    print(f"Acquisition bursts (gaps > {GAP_BREAK_S:.0f} s): {kim_df['burst_id'].nunique()}")

    magnitudes = [np.sqrt(s["lr"] ** 2 + s["si"] ** 2 + s["ap"] ** 2) for s in shifts]
    intrafx_indices = [k for k in range(len(shifts)) if k != localisation_shift_idx]
    largest_intrafx_idx = (
        max(intrafx_indices, key=lambda k: magnitudes[k]) if intrafx_indices else None
    )

    print("Couch corrections (mm):")
    for k, (s, mag) in enumerate(zip(shifts, magnitudes)):
        if k == localisation_shift_idx:
            marker = "  (localisation, excluded)"
        elif k == headline_shift_idx and k == largest_intrafx_idx:
            marker = "  *** LARGEST INTRA-FRACTION ***"
        elif k == headline_shift_idx:
            marker = "  *** HEADLINED ***"
        elif k in intrafx_indices:
            marker = ""
        else:
            marker = ""
        print(
            f"  shift {k+1} (file {k}->{k+1}): "
            f"LR={s['lr']:+6.2f} SI={s['si']:+6.2f} AP={s['ap']:+6.2f}  "
            f"|D|={mag:5.2f}{marker}"
        )

    # --- compute boundary real times (seconds) --------------------------------
    boundaries_real_s = []
    file_indices = sorted(kim_df["file_index"].unique())
    for prev_fi, next_fi in zip(file_indices[:-1], file_indices[1:]):
        t_prev_end = kim_df.loc[kim_df["file_index"] == prev_fi, "time"].max()
        t_next_start = kim_df.loc[kim_df["file_index"] == next_fi, "time"].min()
        boundaries_real_s.append((t_prev_end + t_next_start) / 2.0)

    # --- resolve zoom window --------------------------------------------------
    if window_min is None:
        headline_t_min = boundaries_real_s[headline_shift_idx] / 60.0
        fraction_end_min = kim_df["time"].max() / 60.0
        window_min = (
            max(0.0, headline_t_min - auto_half),
            min(fraction_end_min, headline_t_min + auto_half),
        )
        print(
            f"Auto window: headline shift at {headline_t_min:.2f} min -> "
            f"window {window_min[0]:.2f}-{window_min[1]:.2f} min "
            f"(fraction end {fraction_end_min:.2f} min)"
        )

    # --- restrict to the zoom window and compress beam-off pauses -------------
    window_s = (window_min[0] * 60.0, window_min[1] * 60.0)
    win_df = kim_df[
        (kim_df["time"] >= window_s[0]) & (kim_df["time"] <= window_s[1])
    ].sort_values("time").reset_index(drop=True)
    if win_df.empty:
        raise SystemExit(f"No frames in window {window_min[0]}-{window_min[1]} min.")

    display_s, gap_real_intervals, gap_display_intervals = compress_time_gaps(
        win_df["time"].values, GAP_BREAK_S, GAP_COMPRESS_S
    )
    win_df["display_s"] = display_s

    # Real-second -> display-second interpolator built from the windowed samples.
    # Outside the data range, np.interp clamps, which is fine for the boundary
    # of the zoom window itself.
    def real_to_display(t_real_s):
        return np.interp(t_real_s, win_df["time"].values, display_s)

    print(
        f"Zoom window {window_min[0]:.0f}-{window_min[1]:.0f} min: "
        f"{len(win_df)} frames, {len(gap_display_intervals)} compressed pauses"
    )

    # --- compute overlay shift (mm in KIM frame) ------------------------------
    # VRT -> AP (meas_z), LNG -> SI (meas_y), LAT -> LR (meas_x). Couch
    # positions in cm, convert delta to mm.
    overlay_dap = (overlay_to_cm[0] - overlay_from_cm[0]) * 10.0
    overlay_dsi = (overlay_to_cm[1] - overlay_from_cm[1]) * 10.0
    overlay_dlr = (overlay_to_cm[2] - overlay_from_cm[2]) * 10.0
    overlay_delta = {"meas_x": overlay_dlr, "meas_y": overlay_dsi, "meas_z": overlay_dap}
    print(
        f"No-correction overlay (subtracts applied shift {overlay_from_cm} -> "
        f"{overlay_to_cm} cm): LR={overlay_dlr:+.1f} SI={overlay_dsi:+.1f} "
        f"AP={overlay_dap:+.1f} mm"
    )

    # --- figure ---------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True)

    # Real-time location of the highlighted couch correction. Defaults to the
    # midpoint of the file boundary corresponding to `headline_shift_idx`, but
    # can be overridden via `headline_t_override_s` for fractions where the
    # actual correction event happened mid-file rather than at a file boundary
    # (e.g. PAT01 FX05). When overridden, the post-correction overlay filter
    # also uses this time so frames after the in-file step jump are correctly
    # identified as "post-correction".
    headline_t_override_s = config.get("headline_t_override_s")
    if headline_t_override_s is not None:
        headline_t_real_s = float(headline_t_override_s)
    else:
        headline_t_real_s = float(boundaries_real_s[headline_shift_idx])
    headline_display_s = float(real_to_display(headline_t_real_s))
    band_half_width_s = 9.0  # ~18 s wide in real seconds; shown on display axis

    # X-axis tick locations: nice 1-minute real-time marks within the window,
    # placed at their corresponding display position. Drop any tick whose real
    # time falls inside a compressed gap (no data exists at that real time, so
    # the label would be misleading). Labels show real time.
    candidate_real_min = np.arange(
        np.ceil(window_min[0]), np.floor(window_min[1]) + 0.5, 1.0
    )

    def _inside_gap(t_s):
        return any(start < t_s < end for start, end in gap_real_intervals)

    real_tick_min = np.array(
        [t for t in candidate_real_min if not _inside_gap(t * 60.0)]
    )
    display_tick_min = real_to_display(real_tick_min * 60.0) / 60.0

    for ax, (col, ylabel, color) in zip(axes, AXIS_SPECS):
        # Hatched bands marking each compressed beam-off pause
        for d_start, d_end in gap_display_intervals:
            ax.axvspan(
                d_start / 60.0,
                d_end / 60.0,
                facecolor="lightgrey",
                edgecolor="grey",
                hatch="///",
                alpha=0.55,
                lw=0.0,
                zorder=0,
            )
        # Highlight band for the headlined intra-fraction correction
        ax.axvspan(
            (headline_display_s - band_half_width_s) / 60.0,
            (headline_display_s + band_half_width_s) / 60.0,
            color="gold",
            alpha=0.30,
            zorder=0,
        )
        # Dotted vertical line at every couch shift that falls inside the window
        # (excluding the localisation shift)
        for k, t_real in enumerate(boundaries_real_s):
            if k == localisation_shift_idx:
                continue
            if window_s[0] <= t_real <= window_s[1]:
                ax.axvline(
                    real_to_display(t_real) / 60.0,
                    color="grey",
                    ls=":",
                    lw=0.8,
                    alpha=0.6,
                    zorder=1,
                )
        # Translucent "no-correction counterfactual" overlay: the segment AFTER
        # the headlined correction, with the applied couch shift mathematically
        # REMOVED (subtracted). A couch shift physically moves the markers in
        # the imaging frame, so recorded_post = recorded_pre + applied_shift.
        # Subtracting the applied_shift from the post-correction trace therefore
        # gives the pre-correction deviation level, i.e. where the tumour would
        # have continued sitting had the operator NOT intervened. Drawn behind
        # the main trace so the overlay is visually subordinate but still
        # readable. The post-correction filter is time-based (rather than
        # file-index-based) so it works correctly when the headlined event is
        # mid-file rather than at a file boundary (FX05).
        if overlay_delta[col] != 0:
            post_df = win_df[win_df["time"] > headline_t_real_s]
            for _, burst in post_df.groupby("burst_id"):
                ax.plot(
                    burst["display_s"] / 60.0,
                    burst[col] - overlay_delta[col],
                    color=color,
                    lw=1.0,
                    alpha=0.30,
                    zorder=1.5,
                )
        # Plot each acquisition burst as its own line (no connecting line across
        # the compressed gaps).
        for _, burst in win_df.groupby("burst_id"):
            ax.plot(
                burst["display_s"] / 60.0,
                burst[col],
                color=color,
                lw=1.0,
                marker="o",
                ms=2.2,
                mew=0,
                zorder=2,
            )
        # +/- 2 mm tolerance lines
        for y in (-2.0, 2.0):
            ax.axhline(y, color="#555555", ls=(0, (1, 1.5)), lw=1.1, alpha=0.95, zorder=1)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*y_limits[col])

    axes[-1].set_xlabel("Time from kV beam-on (min)")
    axes[-1].set_xticks(display_tick_min)
    axes[-1].set_xticklabels([f"{t:.0f}" for t in real_tick_min])
    axes[-1].set_xlim(display_s[0] / 60.0, display_s[-1] / 60.0)

    # Optional headline-correction textbox in the LR panel. Disabled when the
    # figure is meant to ship with a published caption that already carries the
    # correction magnitudes (e.g. the FX04 abstract figure).
    if config.get("show_correction_textbox", True):
        s = shifts[headline_shift_idx]
        text = (
            f"{headline_label} (#{headline_shift_idx + 1})\n"
            f"$\\Delta$LR = {s['lr']:+.1f} mm\n"
            f"$\\Delta$SI = {s['si']:+.1f} mm\n"
            f"$\\Delta$AP = {s['ap']:+.1f} mm\n"
            f"$|\\Delta|$ = {magnitudes[headline_shift_idx]:.1f} mm"
        )
        axes[0].text(
            0.015,
            0.97,
            text,
            transform=axes[0].transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="goldenrod", lw=1.2),
        )

    show_suptitle = config.get("show_suptitle", True)
    if show_suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)
        if subtitle:
            fig.text(
                0.5,
                0.955,
                subtitle,
                ha="center",
                fontsize=9,
                style="italic",
                color="dimgrey",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            # Suptitle only (no subtitle): tighter top margin so the panels
            # reclaim the space the italic subtitle would have taken.
            fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        # No suptitle / subtitle: panels reclaim the top margin so the data
        # plots in each composite quadrant are visually larger.
        fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=config.get("dpi", 600), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure -> {output_png}")


def _fx04_config() -> dict:
    """Build the FX04 config for the abstract figure at
    docs/prime_pat01_fx04_stepjumps.png.

    The headlined event is shift #3 (intra-fraction correction 2 of 2): a clean
    AP-only correction where the recorded post-correction trace responds as
    expected. The other FX04 intra-fraction event (shift #2) is the one whose
    AP response is partially counteracted by rapid bladder filling and is less
    suitable for the abstract figure's main illustration.
    """
    kim_folder = r"L:\LEARN\GenesisCare\PRIME\Trajectory Logs\PAT01\FX04\Trajectory Logs"
    return {
        "kim_folder": kim_folder,
        "couch_file": kim_folder + r"\couchShifts.txt",
        "centroid_file": r"L:\LEARN\GenesisCare\PRIME\Patient Files\PAT01\Centroid_PAT01_BeamID_1.1_1.2_1.3_1.4.txt",
        "output_png": Path(__file__).resolve().parent.parent / "docs" / "prime_pat01_fx04_stepjumps.png",
        "couch_row_count": None,  # FX04 has no stale rows
        "localisation_shift_idx": 0,
        # FX04 shift #3 (index 2): intervention 2 of 2, the clean AP-only
        # correction (|Delta| = 2.0 mm).
        "headline_shift_idx": 2,
        "window_min": (7.0, 14.0),
        "y_limits": {
            "meas_x": (-5.0, 5.0),
            "meas_y": (-5.0, 5.0),
            "meas_z": (-5.0, 5.0),
        },
        # FX04 shift #3 absolute couch positions (rows 4 and 5 of couchShifts.txt)
        "overlay_from_cm": (-17.8, 66.3, 1.0),
        "overlay_to_cm": (-17.6, 66.3, 1.0),
        "suptitle": "KIM-detected trajectory with intervention event highlighted",
        "subtitle": "",  # info captured in the published caption
        "headline_label": "Intra-fraction correction 2 of 2",
        # Textbox info is captured in the published caption.
        "show_correction_textbox": False,
    }


def main() -> None:
    make_figure(_fx04_config())


if __name__ == "__main__":
    main()
