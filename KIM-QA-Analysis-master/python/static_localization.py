#!/usr/bin/env python3
"""Python refactor of the KIM static localisation analysis."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class CentroidData:
    """Holds centroid (seed) and isocentre coordinates in cm."""

    seeds_cm: Sequence[Tuple[float, float, float]]
    isocenter_cm: Tuple[float, float, float]

    @property
    def avg_marker_offsets_mm(self) -> Tuple[float, float, float]:
        """Returns MATLAB-equivalent average marker offsets, expressed in mm."""
        sx = sum(seed[0] for seed in self.seeds_cm) / 3.0
        sy = sum(seed[1] for seed in self.seeds_cm) / 3.0
        sz = sum(seed[2] for seed in self.seeds_cm) / 3.0

        iso_x, iso_y, iso_z = self.isocenter_cm

        avg_marker_x_iso = 10.0 * (sx - iso_x)
        avg_marker_y_iso = 10.0 * (sy - iso_y)
        avg_marker_z_iso = 10.0 * (sz - iso_z)

        # Coordinate transform copied from Staticloc.m
        avg_marker_x = avg_marker_x_iso
        avg_marker_y = avg_marker_z_iso
        avg_marker_z = -avg_marker_y_iso
        return (avg_marker_x, avg_marker_y, avg_marker_z)


@dataclass
class TrajectoryData:
    """Centroid trajectory expressed relative to average marker offsets."""

    timestamps: List[float]
    x_centroid: List[float]
    y_centroid: List[float]
    z_centroid: List[float]
    source_file: Path


@dataclass
class AxisStats:
    mean: float
    std: float
    p05: float
    p95: float


@dataclass
class Metrics:
    lr: AxisStats
    si: AxisStats
    ap: AxisStats

    def as_row(self) -> List[float]:
        return [
            self.lr.mean,
            self.si.mean,
            self.ap.mean,
            self.lr.std,
            self.si.std,
            self.ap.std,
            self.lr.p05,
            self.lr.p95,
            self.si.p05,
            self.si.p95,
            self.ap.p05,
            self.ap.p95,
        ]


SEED_PATTERN = re.compile(
    r"Seed\s+(\d+).*?X\s*=\s*([-+]?\d*\.?\d+)"
    r".*?Y\s*=\s*([-+]?\d*\.?\d+).*?Z\s*=\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)
ISOCENTER_PATTERN = re.compile(
    r"Isocenter.*?X\s*=\s*([-+]?\d*\.?\d+)"
    r".*?Y\s*=\s*([-+]?\d*\.?\d+).*?Z\s*=\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_centroid_file(path: Path) -> CentroidData:
    """Extracts seed and isocentre coordinates from patient centroid files."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    seeds: Dict[int, Tuple[float, float, float]] = {}
    iso: Optional[Tuple[float, float, float]] = None

    for line in text.splitlines():
        seed_match = SEED_PATTERN.search(line)
        if seed_match:
            idx = int(seed_match.group(1))
            seeds[idx] = (
                float(seed_match.group(2)),
                float(seed_match.group(3)),
                float(seed_match.group(4)),
            )
            continue
        iso_match = ISOCENTER_PATTERN.search(line)
        if iso_match:
            iso = (
                float(iso_match.group(1)),
                float(iso_match.group(2)),
                float(iso_match.group(3)),
            )

    if len(seeds) == 3 and iso:
        ordered = [seeds[i] for i in sorted(seeds)]
        return CentroidData(ordered, iso)

    numbers = [float(value) for value in FLOAT_PATTERN.findall(text)]
    if len(numbers) < 12:
        raise ValueError(
            f"Centroid file {path} does not contain enough numeric entries."
        )
    # Some files contain patient IDs before the coordinates; keep the last 12.
    numbers = numbers[-12:]
    seeds_cm = [
        (numbers[0], numbers[1], numbers[2]),
        (numbers[3], numbers[4], numbers[5]),
        (numbers[6], numbers[7], numbers[8]),
    ]
    iso_cm = (numbers[9], numbers[10], numbers[11])
    return CentroidData(seeds_cm, iso_cm)


def locate_kim_file(folder: Path, override: Optional[Path]) -> Path:
    """Resolves the trajectory log to process."""
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = folder / path
        if not path.is_file():
            raise FileNotFoundError(f"KIM log {path} does not exist.")
        return path

    preferred = [
        "MarkerLocationsGA_CouchShift_2.txt",
        "MarkerLocationsGA_CouchShift_1.txt",
        "MarkerLocationsGA_CouchShift_0.txt",
        "MarkerLocations_CouchShift_0.txt",
    ]
    for candidate in preferred:
        path = folder / candidate
        if path.is_file():
            return path

    matches = sorted(folder.glob("MarkerLocations*.txt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No marker location files found in {folder}. "
        "Expected a file such as MarkerLocationsGA_CouchShift_0.txt."
    )


def _clean_fieldnames(fieldnames: Iterable[str]) -> List[str]:
    return [name.strip() for name in fieldnames]


def _parse_float(value: Optional[str]) -> float:
    if value is None:
        raise ValueError("Encountered missing numeric value.")
    return float(value.strip())


def load_kim_trajectory(
    file_path: Path, avg_marker_mm: Tuple[float, float, float], frame_average: int = 1
) -> TrajectoryData:
    """Reads marker trajectories and returns centroid traces."""
    with file_path.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"File {file_path} has no header.")
        reader.fieldnames = _clean_fieldnames(reader.fieldnames)

        timestamps: List[float] = []
        frames: List[float] = []
        markers: Dict[int, Dict[str, List[float]]] = {
            idx: {"LR": [], "SI": [], "AP": []} for idx in range(3)
        }

        for row in reader:
            if not row:
                continue
            try:
                frame = _parse_float(row["Frame No"])
                time = _parse_float(row["Time (sec)"])
            except (KeyError, ValueError):
                continue
            frames.append(frame)
            timestamps.append(time)
            for idx in markers:
                try:
                    markers[idx]["LR"].append(_parse_float(row[f"Marker_{idx}_LR"]))
                    markers[idx]["SI"].append(_parse_float(row[f"Marker_{idx}_SI"]))
                    markers[idx]["AP"].append(_parse_float(row[f"Marker_{idx}_AP"]))
                except (KeyError, ValueError):
                    raise ValueError(
                        f"Missing marker columns for Marker_{idx} in {file_path}."
                    )

    if not timestamps:
        raise ValueError(f"No trajectory samples found in {file_path}.")

    if frame_average == 1:
        frames = [value + 1 for value in frames]
    else:
        frames = [value * frame_average for value in frames]
        frames[0] = 1.0

    t0 = timestamps[0]
    timestamps = [value - t0 for value in timestamps]

    avg_x, avg_y, avg_z = avg_marker_mm
    x_centroid: List[float] = []
    y_centroid: List[float] = []
    z_centroid: List[float] = []
    for idx in range(len(timestamps)):
        x_mean = sum(markers[m]["LR"][idx] for m in markers) / 3.0
        y_mean = sum(markers[m]["SI"][idx] for m in markers) / 3.0
        z_mean = sum(markers[m]["AP"][idx] for m in markers) / 3.0
        x_centroid.append(x_mean - avg_x)
        y_centroid.append(y_mean - avg_y)
        z_centroid.append(z_mean - avg_z)

    return TrajectoryData(
        timestamps=timestamps,
        x_centroid=x_centroid,
        y_centroid=y_centroid,
        z_centroid=z_centroid,
        source_file=file_path,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    data = sorted(values)
    rank = (len(data) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return data[int(rank)]
    fraction = rank - low
    return data[low] + (data[high] - data[low]) * fraction


def _axis_stats(values: Sequence[float]) -> AxisStats:
    if len(values) == 0:
        raise ValueError("Requested statistics for an empty axis.")
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return AxisStats(mean=mean, std=std, p05=_percentile(values, 5), p95=_percentile(values, 95))


def compute_metrics(
    trajectory: TrajectoryData,
    lr_shift: float,
    si_shift: float,
    ap_shift: float,
    initial_skip: int = 10,
) -> Metrics:
    """Computes mean, std, and percentiles after discarding warm-up frames."""
    if initial_skip >= len(trajectory.x_centroid):
        raise ValueError(
            f"Not enough samples ({len(trajectory.x_centroid)}) "
            f"after skipping the first {initial_skip} frames."
        )

    lr_values = [
        value - lr_shift for value in trajectory.x_centroid[initial_skip:]
    ]
    si_values = [
        value - si_shift for value in trajectory.y_centroid[initial_skip:]
    ]
    ap_values = [
        value - ap_shift for value in trajectory.z_centroid[initial_skip:]
    ]

    return Metrics(
        lr=_axis_stats(lr_values),
        si=_axis_stats(si_values),
        ap=_axis_stats(ap_values),
    )


def write_metrics_file(metrics: Metrics, output_dir: Path) -> Path:
    """Writes metrics in the MATLAB-friendly layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "Metrics.txt"
    row = metrics.as_row()
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write("\tMean (mm)\t\t\tStd (mm)\t\t\tPercentile (5, 95)\n")
        handle.write("\tLR\tSI\tAP\tLR\tSI\tAP\tLR\tSI\tAP\n")
        handle.write(
            "\t{0:1.2f}\t{1:1.2f}\t{2:1.2f}\t{3:1.2f}\t{4:1.2f}\t{5:1.2f}\t"
            "({6:1.2f}, {7:1.2f})\t({8:1.2f}, {9:1.2f})\t({10:1.2f}, {11:1.2f})\n".format(
                *row
            )
        )
    return metrics_path


def plot_traces(trajectory: TrajectoryData, output_path: Path) -> Optional[Path]:
    """Renders centroid traces; returns path when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(trajectory.timestamps, trajectory.x_centroid, label="LAT (KIM)")
    ax.plot(trajectory.timestamps, trajectory.y_centroid, label="LONG (KIM)")
    ax.plot(trajectory.timestamps, trajectory.z_centroid, label="VRT (KIM)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Offset (mm)")
    ax.set_title("KIM Static Localisation Trace")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute KIM static localisation metrics (Python refactor)."
    )
    parser.add_argument(
        "--kim-folder",
        type=Path,
        required=True,
        help="Folder containing MarkerLocationsGA_CouchShift_*.txt files.",
    )
    parser.add_argument(
        "--centroid",
        type=Path,
        required=True,
        help="Centroid/patient coordinate file (raw export or stripped).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0,
        help="Static couch shift in the LR direction (mm).",
    )
    parser.add_argument(
        "--si",
        type=float,
        default=0.0,
        help="Static couch shift in the SI direction (mm).",
    )
    parser.add_argument(
        "--ap",
        type=float,
        default=0.0,
        help="Static couch shift in the AP direction (mm).",
    )
    parser.add_argument(
        "--kim-file",
        type=Path,
        help="Explicit MarkerLocations*.txt file. Defaults to GA_CouchShift_2 when present.",
    )
    parser.add_argument(
        "--frame-average",
        type=int,
        default=1,
        help="Frame averaging factor (matches the MATLAB option).",
    )
    parser.add_argument(
        "--initial-skip",
        type=int,
        default=10,
        help="Number of initial samples to discard before computing metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for Metrics.txt and plots (defaults to the KIM folder).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Disable matplotlib plot generation.",
    )
    parser.add_argument(
        "--trace-name",
        type=str,
        default="kim_trace.png",
        help="Filename for the generated trace plot.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    kim_folder = args.kim_folder.expanduser().resolve()
    centroid_path = args.centroid.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else kim_folder
    )

    if not kim_folder.is_dir():
        parser.error(f"KIM folder {kim_folder} does not exist or is not a directory.")
    if not centroid_path.is_file():
        parser.error(f"Centroid file {centroid_path} does not exist.")
    if args.frame_average < 1:
        parser.error("Frame averaging factor must be >= 1.")
    if args.initial_skip < 0:
        parser.error("Initial skip must be zero or positive.")

    centroid = parse_centroid_file(centroid_path)
    kim_file = locate_kim_file(kim_folder, args.kim_file)
    trajectory = load_kim_trajectory(
        kim_file, centroid.avg_marker_offsets_mm, frame_average=args.frame_average
    )
    metrics = compute_metrics(
        trajectory,
        lr_shift=args.lr,
        si_shift=args.si,
        ap_shift=args.ap,
        initial_skip=args.initial_skip,
    )
    metrics_path = write_metrics_file(metrics, output_dir)

    plot_path: Optional[Path] = None
    if not args.skip_plot:
        plot_path = plot_traces(trajectory, output_dir / args.trace_name)

    print(f"Processed {kim_file.name} (samples: {len(trajectory.timestamps)}).")
    print(f"Metrics written to {metrics_path}")
    if plot_path:
        print(f"Trace plot saved to {plot_path}")
    elif args.skip_plot:
        print("Plotting skipped by request.")
    else:
        print("matplotlib unavailable; skipping plot generation.")


if __name__ == "__main__":
    main()
