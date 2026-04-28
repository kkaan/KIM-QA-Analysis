"""Time-axis gap compression and intervention windowing.

Direct port of ``compress_time_gaps`` from
``abstract-figures/make_prime_stepjump_figure.py``. Pure functions — no
matplotlib dependency — so they're equally usable in the FastAPI route layer
and in the PDF renderer.

The compression maps real-time seconds to display-time seconds, collapsing
beam-off pauses longer than ``gap_threshold_s`` to a fixed
``compressed_width_s`` while leaving short bursts untouched. The frontend
plots in display-space and labels ticks with the original real-time values.
"""

from __future__ import annotations

import numpy as np


def compress_time_gaps(
    real_times_s,
    gap_threshold_s: float = 5.0,
    compressed_width_s: float = 3.0,
):
    """Map real-time stamps to display timestamps with gaps collapsed.

    Returns
    -------
    display_times_s : np.ndarray
        Same shape as ``real_times_s``. Display-space x-axis values.
    gap_real_intervals : list[tuple[float, float]]
        ``(start, end)`` pairs in real seconds for each compressed pause.
    gap_display_intervals : list[tuple[float, float]]
        ``(start, end)`` pairs in display seconds for the same pauses (used to
        position hatched bands).
    """
    real = np.asarray(real_times_s, dtype=float)
    if len(real) == 0:
        return real, [], []
    display = np.empty_like(real)
    display[0] = real[0]
    gap_real: list[tuple[float, float]] = []
    gap_display: list[tuple[float, float]] = []
    for i in range(1, len(real)):
        dt = real[i] - real[i - 1]
        if dt > gap_threshold_s:
            display[i] = display[i - 1] + compressed_width_s
            gap_real.append((float(real[i - 1]), float(real[i])))
            gap_display.append((float(display[i - 1]), float(display[i])))
        else:
            display[i] = display[i - 1] + dt
    return display, gap_real, gap_display


def real_to_display(t_real_s: float, real_times_s, display_times_s) -> float:
    """Interpolate a real-second timestamp into display seconds. Uses
    ``np.interp`` so it clamps at the boundaries of the windowed sample."""
    return float(np.interp(t_real_s, real_times_s, display_times_s))


def display_to_real(t_display_s: float, real_times_s, display_times_s) -> float:
    """Inverse of :func:`real_to_display`. Used when the user clicks somewhere
    on the compressed display axis (e.g. to add a manual intervention) and we
    need to translate that pixel back into real-time seconds."""
    return float(np.interp(t_display_s, display_times_s, real_times_s))
