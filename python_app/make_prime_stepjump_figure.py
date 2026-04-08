"""Generate a step-jump trajectory figure for PRIME PAT01 FX04.

Reads four MarkerLocationsGA_CouchShift_*.txt files plus couchShifts.txt from the
PRIME trajectory log directory, plots the raw centroid trajectory in the 4-11 min
treatment window with beam-off pauses compressed on the x-axis, and highlights the
largest intra-fraction couch correction. Correction #1 is the initial localisation
shift and is not counted as an intra-fraction event.

Run from the repo root:
    python python_app/make_prime_stepjump_figure.py
"""

import glob
import io
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kim_analysis_logic import parse_couch_shifts, parse_centroid_file


# Strip thousands-separator commas inside the Time field once t > 1000 s
# (e.g. "1,000.305" -> "1000.305"). The Time column is the only field in this
# file format that ever crosses 1000 with a decimal, so a comma sitting between
# a digit and three more digits followed by a decimal point is unambiguous.
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\.)")


KIM_FOLDER = r"L:\LEARN\GenesisCare\PRIME\Trajectory Logs\PAT01\FX04\Trajectory Logs"
COUCH_FILE = KIM_FOLDER + r"\couchShifts.txt"
# TODO: parametrise via CLI when re-using this script for other patients.
CENTROID_FILE = r"L:\LEARN\GenesisCare\PRIME\Patient Files\PAT01\Centroid_PAT01_BeamID_1.1_1.2_1.3_1.4.txt"
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "docs" / "prime_pat01_fx04_stepjumps.png"

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

# Zoom window in real-time minutes since first imaging. Restricts the figure to
# the treatment phase of interest and excludes the initial localisation shift.
WINDOW_MIN = (4.0, 11.0)

# Per-axis y-limits (mm). After subtracting the expected centroid the trace is
# a deviation from planned isocentre, so all three axes are centred on zero with
# +/-5 mm headroom.
Y_LIMITS = {
    "meas_x": (-5.0, 5.0),  # LR deviation
    "meas_y": (-5.0, 5.0),  # SI deviation
    "meas_z": (-5.0, 5.0),  # AP deviation
}

# Index of the localisation correction to exclude from the "largest intra-fraction"
# ranking. Shift #1 (the file 0 -> file 1 correction) is the initial localisation.
LOCALISATION_SHIFT_IDX = 0

# Translucent "no-correction counterfactual" overlay. The post-correction segment
# of the trace is replotted with this shift mathematically REMOVED (subtracted),
# showing where the deviation would have continued sitting if the operator had
# NOT intervened with the couch correction. Set the FROM/TO values to the same
# numbers as the actual intra-fraction couch correction (in IEC patient frame
# VRT/LNG/LAT, cm) so subtracting them undoes the recorded shift exactly.
# VRT->AP, LNG->SI, LAT->LR. TODO: parametrise via CLI.
OVERLAY_FROM_CM = (-18.0, 66.1, 1.0)
OVERLAY_TO_CM = (-17.8, 66.3, 1.0)
OVERLAY_LABEL = "no-correction counterfactual"


def _file_index(filepath: str) -> int:
    match = re.search(r"MarkerLocationsGA_CouchShift_(\d+)\.txt", filepath)
    return int(match.group(1)) if match else -1


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
    """Load and concatenate the four PRIME trajectory files for this fraction
    and return the per-frame centroid as a *deviation* from the planned
    isocentre, by subtracting the expected centroid (LR/SI/AP in mm) computed
    by parse_centroid_file from the patient's seed/iso file.

    Quirks of the PRIME export handled here:
    - Only CouchShift_0.txt carries a header; files 1-3 are headerless
      continuations of the same column layout.
    - Once the time exceeds 1000 s (mid-way through file 3), the Time field is
      written with a thousands-separator comma ("1,000.305") which would
      otherwise be parsed as an extra column.
    The repo's parse_kim_data assumes every file has a header and fails on these
    files, so we read the header once from file 0 and reuse it for all four.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "MarkerLocationsGA_CouchShift_*.txt")),
        key=_file_index,
    )
    if not files:
        raise FileNotFoundError(f"No MarkerLocationsGA_CouchShift_*.txt in {folder}")

    with open(files[0], "r") as fh:
        header_line = fh.readline()
    cols = [c.strip() for c in header_line.rstrip("\n").split(",")]

    segments = []
    for filepath in files:
        fi = _file_index(filepath)
        with open(filepath, "r") as fh:
            text = fh.read()
        if fi == 0:
            # Drop the header line; we already extracted column names.
            text = text.split("\n", 1)[1] if "\n" in text else ""
        text = _THOUSANDS_COMMA_RE.sub("", text)
        df = pd.read_csv(
            io.StringIO(text),
            header=None,
            names=cols,
            engine="python",
        )
        df.columns = df.columns.str.strip()
        df["file_index"] = fi
        segments.append(df)

    combined = pd.concat(segments, ignore_index=True)

    # Per-frame centroid across the two markers (mean of LR/SI/AP)
    lr_cols = [c for c in combined.columns if re.fullmatch(r"Marker_\d+_LR", c)]
    si_cols = [c for c in combined.columns if re.fullmatch(r"Marker_\d+_SI", c)]
    ap_cols = [c for c in combined.columns if re.fullmatch(r"Marker_\d+_AP", c)]

    out = pd.DataFrame({
        "time": combined["Time (sec)"].astype(float),
        "gantry": combined["Gantry"].astype(float),
        "meas_x": combined[lr_cols].astype(float).mean(axis=1),
        "meas_y": combined[si_cols].astype(float).mean(axis=1),
        "meas_z": combined[ap_cols].astype(float).mean(axis=1),
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


def main() -> None:
    # --- load -----------------------------------------------------------------
    try:
        centroid = parse_centroid_file(CENTROID_FILE)
        expected = centroid["expected_centroid"]
        kim_df = load_kim_centroid(KIM_FOLDER, expected)
        shifts = parse_couch_shifts(COUCH_FILE)
    except Exception as exc:
        raise SystemExit(f"Failed to read PRIME logs from {KIM_FOLDER}: {exc}")

    print(
        f"Expected centroid (mm): "
        f"LR={expected['x']:+.2f} SI={expected['y']:+.2f} AP={expected['z']:+.2f}"
    )

    if kim_df.empty:
        raise SystemExit("parse_kim_data returned an empty DataFrame.")
    if kim_df["file_index"].nunique() != 4:
        raise SystemExit(
            f"Expected 4 sub-arcs, found {kim_df['file_index'].nunique()}."
        )
    if len(shifts) != 3:
        raise SystemExit(f"Expected 3 couch shifts, found {len(shifts)}.")

    # Tag each contiguous burst (within the same file_index, no gaps > threshold)
    # so the plot can break lines across acquisition pauses.
    dt = kim_df["time"].diff()
    new_burst = (kim_df["file_index"].diff().fillna(0) != 0) | (dt > GAP_BREAK_S)
    kim_df["burst_id"] = new_burst.cumsum()

    # --- console summary ------------------------------------------------------
    print(f"Loaded {len(kim_df)} frames across {kim_df['file_index'].nunique()} segments")
    for fi, seg in kim_df.groupby("file_index"):
        print(
            f"  segment {fi}: {len(seg):4d} frames, "
            f"{seg['time'].min():7.1f}-{seg['time'].max():7.1f} s"
        )
    print(f"Acquisition bursts (gaps > {GAP_BREAK_S:.0f} s): {kim_df['burst_id'].nunique()}")

    magnitudes = [np.sqrt(s["lr"] ** 2 + s["si"] ** 2 + s["ap"] ** 2) for s in shifts]
    # Rank intra-fraction corrections (excluding the localisation shift) by 3D
    # magnitude. The headlined "largest" correction is the largest of these.
    intrafx_indices = [k for k in range(len(shifts)) if k != LOCALISATION_SHIFT_IDX]
    headline_idx = max(intrafx_indices, key=lambda k: magnitudes[k])

    print("Couch corrections (mm):")
    for k, (s, mag) in enumerate(zip(shifts, magnitudes)):
        if k == LOCALISATION_SHIFT_IDX:
            marker = "  (localisation, excluded)"
        elif k == headline_idx:
            marker = "  *** LARGEST INTRA-FRACTION ***"
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

    # --- restrict to the zoom window and compress beam-off pauses -------------
    window_s = (WINDOW_MIN[0] * 60.0, WINDOW_MIN[1] * 60.0)
    win_df = kim_df[
        (kim_df["time"] >= window_s[0]) & (kim_df["time"] <= window_s[1])
    ].sort_values("time").reset_index(drop=True)
    if win_df.empty:
        raise SystemExit(f"No frames in window {WINDOW_MIN[0]}-{WINDOW_MIN[1]} min.")

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
        f"Zoom window {WINDOW_MIN[0]:.0f}-{WINDOW_MIN[1]:.0f} min: "
        f"{len(win_df)} frames, {len(gap_display_intervals)} compressed pauses"
    )

    # --- compute overlay shift (mm in KIM frame) ------------------------------
    # VRT -> AP (meas_z), LNG -> SI (meas_y), LAT -> LR (meas_x). Couch
    # positions in cm, convert delta to mm.
    overlay_dap = (OVERLAY_TO_CM[0] - OVERLAY_FROM_CM[0]) * 10.0
    overlay_dsi = (OVERLAY_TO_CM[1] - OVERLAY_FROM_CM[1]) * 10.0
    overlay_dlr = (OVERLAY_TO_CM[2] - OVERLAY_FROM_CM[2]) * 10.0
    overlay_delta = {"meas_x": overlay_dlr, "meas_y": overlay_dsi, "meas_z": overlay_dap}
    print(
        f"No-correction overlay (subtracts applied shift {OVERLAY_FROM_CM} -> "
        f"{OVERLAY_TO_CM} cm): LR={overlay_dlr:+.1f} SI={overlay_dsi:+.1f} "
        f"AP={overlay_dap:+.1f} mm"
    )

    # --- figure ---------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True)

    headline_display_s = float(real_to_display(boundaries_real_s[headline_idx]))
    band_half_width_s = 9.0  # ~18 s wide in real seconds; shown on display axis

    # X-axis tick locations: nice 1-minute real-time marks within the window,
    # placed at their corresponding display position. Drop any tick whose real
    # time falls inside a compressed gap (no data exists at that real time, so
    # the label would be misleading). Labels show real time.
    candidate_real_min = np.arange(
        np.ceil(WINDOW_MIN[0]), np.floor(WINDOW_MIN[1]) + 0.5, 1.0
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
        # Highlight band for the headline intra-fraction correction
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
            if k == LOCALISATION_SHIFT_IDX:
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
        # readable.
        if overlay_delta[col] != 0:
            post_df = win_df[win_df["file_index"] > headline_idx]
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
        ax.set_ylim(*Y_LIMITS[col])

    axes[-1].set_xlabel("Time since first imaging (min)")
    axes[-1].set_xticks(display_tick_min)
    axes[-1].set_xticklabels([f"{t:.0f}" for t in real_tick_min])
    axes[-1].set_xlim(display_s[0] / 60.0, display_s[-1] / 60.0)

    # Headline-correction textbox in the LR panel
    s = shifts[headline_idx]
    text = (
        f"Largest intra-fraction correction (#{headline_idx + 1})\n"
        f"$\\Delta$LR = {s['lr']:+.1f} mm\n"
        f"$\\Delta$SI = {s['si']:+.1f} mm\n"
        f"$\\Delta$AP = {s['ap']:+.1f} mm\n"
        f"$|\\Delta|$ = {magnitudes[headline_idx]:.1f} mm"
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

    fig.suptitle(
        "PRIME trajectory log - patient PAT01, fraction FX04",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "Marker centroid deviation from planned isocentre; hatched bands mark beam-off pauses (compressed for clarity)",
        ha="center",
        fontsize=9,
        style="italic",
        color="dimgrey",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
