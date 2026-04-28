"""Dynamic-marker centroid loader.

Replaces the hardcoded ``Marker_0`` + ``Marker_1`` BASE_COLS in
``abstract-figures/make_prime_stepjump_figure.load_kim_centroid`` so the app
handles 1..N implanted markers and lets the user deselect a migrated seed at
runtime without crashing.

Quirks preserved verbatim from the original loader:
 - Headerless continuation files (only ``CouchShift_0.txt`` carries a header).
 - Thousands-separator comma in the Time field once t > 1000 s
   (e.g. ``"1,000.305"``) — stripped via ``_THOUSANDS_COMMA_RE``.
 - Sentinel pre-tracking-lock placeholder rows (~135 mm), filtered via
   ``SANE_LIMIT_MM``.
 - Per-axis median-deviation glitch filter (``GLITCH_DEV_MM``).

The centroid is computed as the mean of the *active* markers only. The
expected centroid (subtracted from the measurements to produce a deviation
plot) is recomputed from the same subset of seeds so both sides of the
subtraction stay consistent.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants lifted from abstract-figures/make_prime_stepjump_figure.py so the
# new app shares one source of truth for tracking-quirk filtering thresholds.
# ----------------------------------------------------------------------------

# Strip thousands-separator commas inside the Time field once t > 1000 s
# (e.g. "1,000.305" -> "1000.305").
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\.)")

# Frames with any centroid axis exceeding this magnitude (mm) are sentinel
# placeholders KIM exports before the first 3D position is locked.
SANE_LIMIT_MM = 50.0

# Per-axis catastrophic single-frame glitch filter: drop frames whose centroid
# deviates from the per-axis median by more than these thresholds (mm).
GLITCH_DEV_MM = {"meas_x": 5.0, "meas_y": 12.0, "meas_z": 12.0}

# Acquisition pauses longer than this (s) break trace lines into bursts.
GAP_BREAK_S = 5.0

# Compress beam-off pauses to this many display-seconds.
GAP_COMPRESS_S = 3.0


_MARKER_TRIPLET_RE = re.compile(r"^Marker_(\d+)_(AP|LR|SI)$", re.IGNORECASE)
_FILE_INDEX_RE = re.compile(r"MarkerLocationsGA_CouchShift_(\d+)\.txt", re.IGNORECASE)


@dataclass
class LoadResult:
    """Bundle of everything the API layer needs from one fraction load."""

    df: pd.DataFrame
    """One row per accepted frame with columns:
        time, gantry, meas_x, meas_y, meas_z, file_index, n_active_markers,
        burst_id.
    Time is normalised to start at 0.
    """

    detected_markers: list[int]
    """Marker indices found in the trajectory files (e.g. [0, 1] or [0, 1, 2])."""

    active_markers: list[int]
    """Marker indices actually used for the centroid computation."""

    expected_centroid: dict
    """{'x': LR_mm, 'y': SI_mm, 'z': AP_mm} computed from the active seed subset."""

    warnings: list[str]
    """Non-fatal messages surfaced to the UI (sentinel/glitch counts, NaN frames)."""


def file_index(filepath: str) -> int:
    """Extract the integer N from ``MarkerLocationsGA_CouchShift_N.txt``."""
    match = _FILE_INDEX_RE.search(filepath)
    return int(match.group(1)) if match else -1


def discover_markers(folder: str) -> list[int]:
    """Read the header of ``MarkerLocationsGA_CouchShift_0.txt`` and return the
    sorted list of marker indices for which all three of AP/LR/SI columns
    appear. PRIME files store ``Marker_i_AP``, ``Marker_i_LR``, ``Marker_i_SI``.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "MarkerLocationsGA_CouchShift_*.txt")),
        key=file_index,
    )
    if not files:
        raise FileNotFoundError(
            f"No MarkerLocationsGA_CouchShift_*.txt found in {folder}"
        )

    with open(files[0], "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    cols = [c.strip() for c in header.split(",")]
    by_index: dict[int, set[str]] = {}
    for c in cols:
        m = _MARKER_TRIPLET_RE.match(c)
        if m:
            by_index.setdefault(int(m.group(1)), set()).add(m.group(2).upper())
    return sorted(i for i, axes in by_index.items() if {"AP", "LR", "SI"} <= axes)


def expected_centroid_from_seeds(
    seeds: list[list[float]],
    isocenter: list[float],
    active_markers: Iterable[int] | None = None,
) -> dict:
    """Compute the expected centroid in (LR, SI, AP) mm from a *subset* of seeds.

    Mirrors the axis swap baked into ``parse_centroid_file``:
        x = iso_x  (LR)
        y = iso_z  (SI)
        z = -iso_y (AP)

    ``seeds`` is a list of [X, Y, Z] (cm). ``isocenter`` is [X, Y, Z] (cm).
    ``active_markers`` selects which seed indices participate; ``None`` uses all.
    """
    if not seeds:
        raise ValueError("No seeds provided")
    if active_markers is None:
        chosen = list(range(len(seeds)))
    else:
        chosen = [i for i in active_markers if 0 <= i < len(seeds)]
    if not chosen:
        raise ValueError(
            f"No active markers selected among {len(seeds)} available seeds"
        )

    sel = np.asarray([seeds[i] for i in chosen], dtype=float)
    avg = sel.mean(axis=0)
    iso_mm = 10.0 * (avg - np.asarray(isocenter, dtype=float))
    return {
        "x": float(iso_mm[0]),       # LR
        "y": float(iso_mm[2]),       # SI
        "z": float(-iso_mm[1]),      # AP
    }


def _read_trajectory_segments(folder: str) -> list[pd.DataFrame]:
    """Read every ``MarkerLocationsGA_CouchShift_*.txt`` in ``folder`` into a
    list of per-file DataFrames. Each DataFrame has the file's full set of
    columns (after stripping thousands-separator commas) plus a ``file_index``
    column. Headerless continuation files reuse the header from file 0.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "MarkerLocationsGA_CouchShift_*.txt")),
        key=file_index,
    )
    if not files:
        raise FileNotFoundError(f"No MarkerLocationsGA_CouchShift_*.txt in {folder}")

    # Read the header once from file 0.
    with open(files[0], "r", encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline().strip()
    header_cols = [c.strip() for c in header_line.split(",")]
    if not header_cols or "Time (sec)" not in header_cols:
        raise ValueError(
            f"Unexpected header in {files[0]}: {header_line!r}"
        )

    segments: list[pd.DataFrame] = []
    for fp in files:
        fi = file_index(fp)
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        lines = text.splitlines()
        # Drop the header line if this file happens to carry one.
        if lines and lines[0].lstrip().startswith("Frame"):
            lines = lines[1:]
        rows: list[list[str]] = []
        for line in lines:
            if not line.strip():
                continue
            cleaned = _THOUSANDS_COMMA_RE.sub("", line)
            parts = [p.strip() for p in cleaned.split(",")]
            # Pad/truncate to header_cols length so concat-friendly.
            if len(parts) < len(header_cols):
                continue   # malformed row; skip
            rows.append(parts[: len(header_cols)])
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=header_cols)
        # Coerce the columns we care about to numeric. Any column whose name
        # matches Marker_i_(AP|LR|SI) or is Time/Gantry/Frame No is converted.
        for c in df.columns:
            if (
                c in ("Frame No", "Time (sec)", "Gantry")
                or _MARKER_TRIPLET_RE.match(c)
            ):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["file_index"] = fi
        segments.append(df)

    if not segments:
        raise ValueError(f"No data rows in any trajectory file under {folder}")
    return segments


def load_centroid(
    folder: str,
    expected_centroid: dict,
    active_markers: Iterable[int] | None = None,
) -> LoadResult:
    """Load and concatenate trajectory files for one fraction; compute the
    centroid trajectory as a deviation from the planned isocentre.

    Parameters
    ----------
    folder
        Directory containing ``MarkerLocationsGA_CouchShift_*.txt``.
    expected_centroid
        Dict ``{'x': LR_mm, 'y': SI_mm, 'z': AP_mm}`` to subtract from the
        measured centroid. Should be computed from the same subset of seeds
        that ``active_markers`` selects (use :func:`expected_centroid_from_seeds`).
    active_markers
        Iterable of marker indices to include in the centroid mean. ``None``
        uses every marker the trajectory header advertises.

    Returns
    -------
    LoadResult
    """
    segments = _read_trajectory_segments(folder)
    detected = sorted(
        {
            int(_MARKER_TRIPLET_RE.match(c).group(1))
            for seg in segments
            for c in seg.columns
            if _MARKER_TRIPLET_RE.match(c)
        }
    )
    if not detected:
        raise ValueError(f"No Marker_i_(AP|LR|SI) columns found in {folder}")

    if active_markers is None:
        active = list(detected)
    else:
        active = [i for i in active_markers if i in detected]
    if not active:
        raise ValueError(
            f"No active markers selected (detected={detected}, requested={list(active_markers)})"
        )

    combined = pd.concat(segments, ignore_index=True, sort=False)

    # Build per-axis centroid as the mean of the active markers' columns.
    # NaN-aware mean: if a frame is missing a particular marker's value, the
    # remaining markers still contribute. If every active marker is NaN for an
    # axis, that frame's axis becomes NaN and the row is dropped below.
    lr_cols = [f"Marker_{i}_LR" for i in active if f"Marker_{i}_LR" in combined.columns]
    si_cols = [f"Marker_{i}_SI" for i in active if f"Marker_{i}_SI" in combined.columns]
    ap_cols = [f"Marker_{i}_AP" for i in active if f"Marker_{i}_AP" in combined.columns]
    if not (lr_cols and si_cols and ap_cols):
        raise ValueError(
            f"Active markers {active} missing one of LR/SI/AP columns in trajectory file"
        )

    out = pd.DataFrame({
        "time": combined["Time (sec)"].astype(float),
        "gantry": combined["Gantry"].astype(float),
        "meas_x": combined[lr_cols].astype(float).mean(axis=1, skipna=True),
        "meas_y": combined[si_cols].astype(float).mean(axis=1, skipna=True),
        "meas_z": combined[ap_cols].astype(float).mean(axis=1, skipna=True),
        "file_index": combined["file_index"].astype(int),
    })
    # Per-frame count of how many active markers actually contributed (i.e. were
    # non-NaN across all three axes).
    triplet_present = pd.DataFrame({
        i: combined[[f"Marker_{i}_LR", f"Marker_{i}_SI", f"Marker_{i}_AP"]]
        .notna()
        .all(axis=1)
        for i in active
        if all(f"Marker_{i}_{ax}" in combined.columns for ax in ("LR", "SI", "AP"))
    })
    out["n_active_markers"] = triplet_present.sum(axis=1).astype(int)

    # Normalise time so the first frame is t=0.
    if not out.empty:
        out["time"] = out["time"] - out["time"].iloc[0]

    warnings: list[str] = []

    n_total = len(out)
    out = out.dropna(subset=["time", "meas_x", "meas_y", "meas_z"])
    n_after_nan = len(out)
    if n_total - n_after_nan:
        warnings.append(
            f"Dropped {n_total - n_after_nan} frames where all active markers were NaN"
        )

    # Subtract the expected centroid to convert raw imaging-frame positions to
    # deviations from the planned isocentre.
    out["meas_x"] = out["meas_x"] - expected_centroid["x"]
    out["meas_y"] = out["meas_y"] - expected_centroid["y"]
    out["meas_z"] = out["meas_z"] - expected_centroid["z"]

    # Drop sentinel pre-tracking-lock placeholders.
    sane = (out[["meas_x", "meas_y", "meas_z"]].abs() < SANE_LIMIT_MM).all(axis=1)
    n_sentinel = int((~sane).sum())
    if n_sentinel:
        warnings.append(
            f"Dropped {n_sentinel} sentinel frames (>|{SANE_LIMIT_MM:.0f}| mm)"
        )
    out = out[sane].reset_index(drop=True)

    # Drop catastrophic single-frame glitches (per-axis median deviation).
    keep = pd.Series(True, index=out.index)
    for col, limit in GLITCH_DEV_MM.items():
        med = out[col].median()
        keep &= (out[col] - med).abs() <= limit
    n_glitch = int((~keep).sum())
    if n_glitch:
        warnings.append(
            f"Dropped {n_glitch} glitch frames (per-axis median-deviation filter)"
        )
    out = out[keep].reset_index(drop=True)

    if out.empty:
        raise ValueError(
            f"All frames filtered out for {folder} (active markers={active})"
        )

    out["time"] = out["time"] - out["time"].iloc[0]

    # Tag bursts: same file, no gap > GAP_BREAK_S.
    dt = out["time"].diff()
    new_burst = (out["file_index"].diff().fillna(0) != 0) | (dt > GAP_BREAK_S)
    out["burst_id"] = new_burst.cumsum().astype(int)

    return LoadResult(
        df=out,
        detected_markers=detected,
        active_markers=active,
        expected_centroid=dict(expected_centroid),
        warnings=warnings,
    )
