"""No-correction counterfactual overlay.

For each headlined intra-fraction couch correction, the post-correction segment
of the trace can be re-plotted with the applied shift mathematically removed
to show where the tumour would have continued sitting had the operator NOT
intervened.

Sign convention (preserved from
``abstract-figures/make_prime_stepjump_figure.py``):

    recorded_post = recorded_pre + applied_shift

So ``recorded_post - applied_shift`` reproduces the pre-correction deviation
level — the no-correction counterfactual. A corrective shift opposes the
deviation, so the overlay always pulls the post-correction trace back to where
the pre-correction trace was sitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class OverlaySegment:
    """One contiguous post-correction overlay band per axis."""
    axis: str           # "meas_x" | "meas_y" | "meas_z"
    burst_id: int
    display_s: list[float]
    values_mm: list[float]


def counterfactual_segments(
    df: pd.DataFrame,
    headline_t_real_s: float,
    delta_mm: dict,
    axis_cols: tuple[str, str, str] = ("meas_x", "meas_y", "meas_z"),
) -> list[OverlaySegment]:
    """Compute overlay segments showing each axis's no-correction counterfactual.

    Parameters
    ----------
    df
        DataFrame with at minimum ``time, display_s, burst_id, meas_x/y/z``.
    headline_t_real_s
        Real-time (seconds) at which the highlighted correction occurred.
        Frames *after* this time are subjected to the inverse-shift overlay.
    delta_mm
        Dict ``{'meas_x': dlr, 'meas_y': dsi, 'meas_z': dap}`` in mm — the
        applied couch correction.
    axis_cols
        Which deviation columns to produce overlays for.

    Returns
    -------
    list of OverlaySegment, one per (axis, burst) pair where the delta is
    non-zero. Bursts entirely before ``headline_t_real_s`` are skipped.
    """
    if df.empty:
        return []
    post = df[df["time"] > headline_t_real_s]
    if post.empty:
        return []
    out: list[OverlaySegment] = []
    for axis in axis_cols:
        d = float(delta_mm.get(axis, 0.0))
        if d == 0.0:
            continue
        for bid, burst in post.groupby("burst_id"):
            out.append(
                OverlaySegment(
                    axis=axis,
                    burst_id=int(bid),
                    display_s=[float(v) for v in burst["display_s"].tolist()],
                    values_mm=[float(v - d) for v in burst[axis].tolist()],
                )
            )
    return out


def couch_delta_to_axis_mm(
    from_cm: tuple[float, float, float],
    to_cm: tuple[float, float, float],
) -> dict:
    """Convert a couch (VRT, LNG, LAT) cm pair into a (meas_x, meas_y, meas_z)
    mm delta. VRT->AP (meas_z), LNG->SI (meas_y), LAT->LR (meas_x).
    """
    return {
        "meas_x": (to_cm[2] - from_cm[2]) * 10.0,
        "meas_y": (to_cm[1] - from_cm[1]) * 10.0,
        "meas_z": (to_cm[0] - from_cm[0]) * 10.0,
    }
