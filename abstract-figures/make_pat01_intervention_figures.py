"""Generate one step-jump trajectory figure per intra-fraction couch
intervention across PRIME PAT01's fractions (FX01, FX04, FX05). FX02 and FX03
have no intra-fraction interventions and are skipped.

For each fraction this script:
  1. Counts the MarkerLocationsGA_CouchShift_*.txt files (= n_files).
  2. Reads couchShifts.txt raw (tolerating FX01's concatenated stale rows).
  3. Keeps only the LAST n_files absolute-couch-position rows, yielding the
     n_files - 1 real deltas (first delta = localisation, excluded from the
     intra-fraction list).
  4. Builds one config per remaining intra-fraction shift and calls
     `make_figure` to render that shift's step-jump figure with a ~7-minute
     auto window centred on the shift.

Run from the repo root:
    python abstract-figures/make_pat01_intervention_figures.py
"""

import glob
import os
import re
import sys
from pathlib import Path

# Allow this script to import its sibling abstract-figures module regardless of
# the cwd it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_prime_stepjump_figure import make_figure  # noqa: E402


CENTROID_FILE = r"L:\LEARN\GenesisCare\PRIME\Patient Files\PAT01\Centroid_PAT01_BeamID_1.1_1.2_1.3_1.4.txt"
TRAJ_ROOT = r"L:\LEARN\GenesisCare\PRIME\Trajectory Logs\PAT01"
OUTPUT_DIR = Path(
    r"C:\Users\kankean.kandasamy\OneDrive - GenesisCare\Documents\phd-prime\prime-epsm-abstract\pat01-traces-interventions"
)

Y_LIMITS = {
    "meas_x": (-5.0, 5.0),
    "meas_y": (-5.0, 5.0),
    "meas_z": (-5.0, 5.0),
}

FRACTIONS = ["FX01", "FX02", "FX03", "FX04", "FX05"]


def _count_marker_files(kim_folder: str) -> int:
    return len(
        glob.glob(os.path.join(kim_folder, "MarkerLocationsGA_CouchShift_*.txt"))
    )


def _read_couch_rows(couch_file: str) -> list[tuple[float, float, float]]:
    """Read absolute couch positions (VRT, LNG, LAT in cm) from a PRIME
    couchShifts.txt.

    Tolerates:
      - A standard header line ("VRT (cm), LNG (cm), LAT (cm)").
      - FX01's anomaly where two recording sessions were concatenated without
        a newline separator so a data row like "-8.2,10.2,0.2VRT (cm), ..."
        embeds a second header mid-line.

    Returns a list of (vrt, lng, lat) float tuples in original file order.
    """
    with open(couch_file, "r") as fh:
        text = fh.read()
    # Split on both newlines and any "VRT" header token so embedded headers
    # (FX01 anomaly) are separated from their preceding data row.
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


def _build_fraction_configs(fraction: str) -> list[dict]:
    """Build one `make_figure` config per intra-fraction intervention in
    `fraction`. Returns an empty list for fractions without any intra-fraction
    interventions (so the caller can skip cleanly).
    """
    kim_folder = os.path.join(TRAJ_ROOT, fraction, "Trajectory Logs")
    couch_file = os.path.join(kim_folder, "couchShifts.txt")

    n_files = _count_marker_files(kim_folder)
    if n_files < 2:
        print(f"{fraction}: only {n_files} trajectory file(s), nothing to plot.")
        return []

    all_rows = _read_couch_rows(couch_file)
    if len(all_rows) < n_files:
        raise SystemExit(
            f"{fraction}: couchShifts.txt has {len(all_rows)} rows but "
            f"{n_files} trajectory files require at least {n_files}."
        )
    rows = all_rows[-n_files:]

    # Intra-fraction interventions are every delta after the localisation (#1).
    # delta k (1-indexed) spans rows k -> k+1.
    n_intrafx = n_files - 2  # total deltas = n_files - 1; minus the localisation
    if n_intrafx <= 0:
        print(
            f"{fraction}: {n_files} trajectory files -> only a localisation "
            f"shift, no intra-fraction interventions. Skipping."
        )
        return []

    print(
        f"{fraction}: {n_files} trajectory files, {len(all_rows)} couchShifts "
        f"rows ({len(all_rows) - n_files} stale leading rows discarded), "
        f"{n_intrafx} intra-fraction intervention(s)."
    )

    fx_num = fraction.lstrip("FX").lstrip("0") or "0"
    suptitle = f"PRIME trajectory log - patient PAT01, fraction {fraction}"
    subtitle = "Marker centroid deviation from planned isocentre; hatched bands mark beam-off pauses (compressed for clarity)"

    configs: list[dict] = []
    for j in range(n_intrafx):
        # Intra-fraction shift #j (1-indexed) is delta index 1 + j in the
        # sliced shifts list (0 = localisation, 1.. = intra-fraction).
        shift_idx = 1 + j
        # overlay_from/to are rows bracketing this shift (shift_idx -> shift_idx+1)
        overlay_from_cm = rows[shift_idx]
        overlay_to_cm = rows[shift_idx + 1]
        intervention_label = f"intervention_{j + 1}"
        output_png = OUTPUT_DIR / f"pat01_{fraction.lower()}_{intervention_label}.png"
        configs.append({
            "kim_folder": kim_folder,
            "couch_file": couch_file,
            "centroid_file": CENTROID_FILE,
            "output_png": output_png,
            "couch_row_count": n_files,
            "localisation_shift_idx": 0,
            "headline_shift_idx": shift_idx,
            "window_min": None,
            "auto_window_half_width_min": 3.5,
            "y_limits": Y_LIMITS,
            "overlay_from_cm": overlay_from_cm,
            "overlay_to_cm": overlay_to_cm,
            "suptitle": suptitle,
            "subtitle": subtitle,
            "headline_label": f"Intra-fraction correction {j + 1} of {n_intrafx}",
        })
    return configs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for fraction in FRACTIONS:
        print()
        print(f"========== {fraction} ==========")
        try:
            configs = _build_fraction_configs(fraction)
        except SystemExit as exc:
            print(f"{fraction}: {exc}")
            continue
        if not configs:
            print(f"Skipping {fraction}")
            continue
        for cfg in configs:
            make_figure(cfg)
            generated.append(Path(cfg["output_png"]))
    print()
    print("========== summary ==========")
    print(f"Generated {len(generated)} figure(s):")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
