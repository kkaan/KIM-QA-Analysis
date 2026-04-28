"""couchShifts.txt parsing.

Wraps the absolute-row reader from
``abstract-figures/make_prime_stepjump_figure.py`` so the FX01 anomaly (two
recording sessions concatenated into one file) is handled out of the box. The
``couch_row_count`` knob in the API lets the UI trim leading stale rows
without having to touch the file on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CouchShift:
    """One inter-row delta in mm. Axis convention matches
    ``kim_analysis_logic.parse_couch_shifts``: VRT->AP, LNG->SI, LAT->LR.
    """
    lr: float
    si: float
    ap: float

    @property
    def magnitude(self) -> float:
        return float((self.lr ** 2 + self.si ** 2 + self.ap ** 2) ** 0.5)


def read_couch_rows(couch_file: str) -> list[tuple[float, float, float]]:
    """Return absolute (VRT, LNG, LAT) couch positions in cm.

    Tolerates the FX01 mid-line embedded-header anomaly by splitting on any
    occurrence of ``VRT...`` first, then scanning each fragment for valid
    comma-separated numeric rows.
    """
    with open(couch_file, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    fragments = re.split(r"VRT[^\n]*", text)
    rows: list[tuple[float, float, float]] = []
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


def rows_to_shifts(rows: list[tuple[float, float, float]]) -> list[CouchShift]:
    """Convert adjacent (VRT, LNG, LAT) rows in cm to inter-row shifts in mm."""
    shifts: list[CouchShift] = []
    for i in range(len(rows) - 1):
        v0, l0, a0 = rows[i]
        v1, l1, a1 = rows[i + 1]
        shifts.append(
            CouchShift(
                ap=(v1 - v0) * 10.0,
                si=(l1 - l0) * 10.0,
                lr=(a1 - a0) * 10.0,
            )
        )
    return shifts


def slice_to_last_n(
    rows: list[tuple[float, float, float]],
    couch_row_count: int | None,
) -> list[tuple[float, float, float]]:
    """Return the last ``couch_row_count`` rows. If ``None``, return as-is.

    Used to trim leading stale rows (e.g. PAT01 FX01's merged-session file).
    """
    if couch_row_count is None:
        return rows
    if couch_row_count <= 0:
        return []
    return rows[-couch_row_count:]
