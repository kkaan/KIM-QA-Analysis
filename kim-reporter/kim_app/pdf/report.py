"""Clinical PDF report generator.

Renders the three-panel deviation plot with matplotlib (using the same
``AXIS_SPECS`` palette and gap-compression logic as the on-screen view), then
embeds it into an A4-portrait ReportLab document together with a header,
intervention summary table, free-text physicist notes, and a footer carrying
the app version + generation timestamp.

Why matplotlib for the embedded plot rather than Plotly's image export? Two
reasons: (a) keeps the bundled binary smaller (Plotly's static export pulls in
``kaleido`` which adds ~150 MB) and (b) lets the PDF reuse the exact same
figure recipe that ``abstract-figures/make_prime_stepjump_figure.py`` already
ships, so the printed plot looks visually identical to the published abstract
figures.
"""

from __future__ import annotations

import datetime as _dt
from io import BytesIO
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from kim_app import __version__
from kim_app.api.schema import FractionResponse, HighlightSpec


# Same palette as abstract-figures/make_prime_stepjump_figure.AXIS_SPECS.
AXIS_SPECS = [
    ("meas_x", "LR deviation (mm)", "#1f77b4"),
    ("meas_y", "SI deviation (mm)", "#2ca02c"),
    ("meas_z", "AP deviation (mm)", "#d62728"),
]


def _build_plot_png(
    payload: FractionResponse,
    highlights: list[HighlightSpec],
    show_overlay: bool,
) -> BytesIO:
    """Render the three-panel deviation plot to an in-memory PNG buffer."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)

    real = np.asarray(payload.time_real_s, dtype=float)
    disp = np.asarray(payload.time_display_s, dtype=float)
    burst_id = np.asarray(payload.burst_id, dtype=int)
    series = {
        "meas_x": np.asarray(payload.meas_x, dtype=float),
        "meas_y": np.asarray(payload.meas_y, dtype=float),
        "meas_z": np.asarray(payload.meas_z, dtype=float),
    }

    # Display-axis range (minutes).
    if len(disp):
        x_min = disp.min() / 60.0
        x_max = disp.max() / 60.0
    else:
        x_min, x_max = 0.0, 1.0

    # Per-axis y-limits from the payload, falling back to ±5 mm.
    y_limits = payload.y_limits or {
        "meas_x": [-5.0, 5.0],
        "meas_y": [-5.0, 5.0],
        "meas_z": [-5.0, 5.0],
    }

    def _real_to_display(t_real: float) -> float:
        if len(disp) == 0:
            return 0.0
        return float(np.interp(t_real, real, disp))

    for ax, (col, ylabel, color) in zip(axes, AXIS_SPECS):
        # Hatched bands for compressed beam-off pauses.
        for d_start, d_end in payload.gap_display_intervals:
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
        # Gold highlight bands.
        for hl in highlights:
            d0 = _real_to_display(hl.t_start_real_s)
            d1 = _real_to_display(hl.t_end_real_s)
            ax.axvspan(d0 / 60.0, d1 / 60.0, color="gold", alpha=0.30, zorder=0)

        # Vertical line at every non-localisation intervention.
        for iv in payload.interventions:
            if iv.is_localisation:
                continue
            ax.axvline(
                iv.t_display_s / 60.0,
                color="grey",
                ls=":",
                lw=0.8,
                alpha=0.6,
                zorder=1,
            )

        # No-correction counterfactual overlay: for each non-localisation
        # intervention, subtract the applied delta from the post-intervention
        # segment of the trace.
        if show_overlay:
            for iv in payload.interventions:
                if iv.is_localisation:
                    continue
                delta = {"meas_x": iv.dlr_mm, "meas_y": iv.dsi_mm, "meas_z": iv.dap_mm}[col]
                if delta == 0.0:
                    continue
                mask = real > iv.t_real_s
                if not mask.any():
                    continue
                # Only overlay until the next intervention so successive shifts
                # don't paint compounding overlays on top of each other.
                next_iv_real = None
                for nxt in payload.interventions:
                    if (
                        not nxt.is_localisation
                        and nxt.t_real_s > iv.t_real_s
                        and (next_iv_real is None or nxt.t_real_s < next_iv_real)
                    ):
                        next_iv_real = nxt.t_real_s
                if next_iv_real is not None:
                    mask &= real < next_iv_real
                if not mask.any():
                    continue
                burst_post = burst_id[mask]
                disp_post = disp[mask]
                y_post = series[col][mask] - delta
                for bid in np.unique(burst_post):
                    bm = burst_post == bid
                    ax.plot(
                        disp_post[bm] / 60.0,
                        y_post[bm],
                        color=color,
                        lw=1.0,
                        alpha=0.30,
                        zorder=1.5,
                    )

        # Main trace, broken into bursts.
        for bid in np.unique(burst_id):
            bm = burst_id == bid
            ax.plot(
                disp[bm] / 60.0,
                series[col][bm],
                color=color,
                lw=1.0,
                marker="o",
                ms=2.2,
                mew=0,
                zorder=2,
            )

        # ±2 mm tolerance lines.
        for y in (-2.0, 2.0):
            ax.axhline(
                y, color="#555555", ls=(0, (1, 1.5)), lw=1.1, alpha=0.95, zorder=1
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(*y_limits.get(col, (-5.0, 5.0)))

    axes[-1].set_xlabel("Time from kV beam-on (min)")
    axes[-1].set_xlim(x_min, x_max)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _summarise_axis(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _interventions_table(
    payload: FractionResponse,
) -> Table:
    """Build the per-intervention summary table."""
    real = np.asarray(payload.time_real_s, dtype=float)
    series = {
        "lr": np.asarray(payload.meas_x, dtype=float),
        "si": np.asarray(payload.meas_y, dtype=float),
        "ap": np.asarray(payload.meas_z, dtype=float),
    }
    rows: list[list[str]] = [[
        "#", "Source", "t (s)", "ΔLR", "ΔSI", "ΔAP", "|Δ|", "Pre 20 s mean", "Post 20 s mean",
    ]]
    for iv in payload.interventions:
        if iv.is_localisation:
            label = f"{iv.label} (excluded)"
        else:
            label = iv.label
        pre_mask = (real >= iv.t_real_s - 20.0) & (real < iv.t_real_s)
        post_mask = (real >= iv.t_real_s) & (real <= iv.t_real_s + 20.0)
        pre_text = "—"
        post_text = "—"
        if pre_mask.any():
            pre_text = (
                f"LR {np.nanmean(series['lr'][pre_mask]):+0.2f}, "
                f"SI {np.nanmean(series['si'][pre_mask]):+0.2f}, "
                f"AP {np.nanmean(series['ap'][pre_mask]):+0.2f}"
            )
        if post_mask.any():
            post_text = (
                f"LR {np.nanmean(series['lr'][post_mask]):+0.2f}, "
                f"SI {np.nanmean(series['si'][post_mask]):+0.2f}, "
                f"AP {np.nanmean(series['ap'][post_mask]):+0.2f}"
            )
        rows.append([
            str(iv.index + 1),
            label,
            f"{iv.t_real_s:.1f}",
            f"{iv.dlr_mm:+0.2f}",
            f"{iv.dsi_mm:+0.2f}",
            f"{iv.dap_mm:+0.2f}",
            f"{iv.magnitude_mm:0.2f}",
            pre_text,
            post_text,
        ])

    table = Table(rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
            ("ALIGN", (2, 1), (6, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return table


def render_report(
    *,
    payload: FractionResponse,
    highlights: list[HighlightSpec],
    notes: str,
    show_overlay: bool,
    output_path: Path,
) -> Path:
    """Build the PDF and write it to ``output_path``. Returns the same path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the plot first so we can fail fast (and not leave a half-written PDF).
    plot_png = _build_plot_png(payload, highlights, show_overlay)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=2 * mm,
        textColor=colors.HexColor("#1F2937"),
    )
    subhead_style = ParagraphStyle(
        "SubHead",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        textColor=colors.HexColor("#374151"),
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10, leading=14
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#6B7280"),
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"KIM Report — {payload.patient_id} {payload.fraction_id}",
        author=f"KIM-QA Reporter {__version__}",
    )

    story = []

    # ---- Header --------------------------------------------------------
    story.append(Paragraph("KIM-Guided Couch Correction Report", title_style))
    header_rows = [
        ["Patient", payload.patient_id, "Fraction", payload.fraction_id],
        ["Active markers", ", ".join(f"#{i}" for i in payload.active_markers),
         "Detected markers", ", ".join(f"#{i}" for i in payload.detected_markers)],
    ]
    header_table = Table(header_rows, hAlign="LEFT", colWidths=[28*mm, 55*mm, 28*mm, 55*mm])
    header_table.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F3F4F6")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9CA3AF")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 4 * mm))

    # ---- Summary box ---------------------------------------------------
    n_intra = sum(1 for iv in payload.interventions if not iv.is_localisation)
    lr_lo, lr_hi = _summarise_axis(payload.meas_x)
    si_lo, si_hi = _summarise_axis(payload.meas_y)
    ap_lo, ap_hi = _summarise_axis(payload.meas_z)
    summary_lines = [
        f"<b>Intra-fraction interventions:</b> {n_intra}",
        f"<b>Peak deviation (mm):</b> "
        f"LR [{lr_lo:+0.2f}, {lr_hi:+0.2f}], "
        f"SI [{si_lo:+0.2f}, {si_hi:+0.2f}], "
        f"AP [{ap_lo:+0.2f}, {ap_hi:+0.2f}]",
        f"<b>Frames analysed:</b> {len(payload.time_real_s)}",
    ]
    if payload.warnings:
        summary_lines.append(
            f"<b>Loader warnings:</b> {'; '.join(payload.warnings)}"
        )
    for line in summary_lines:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 3 * mm))

    # ---- Plot ----------------------------------------------------------
    story.append(Paragraph("Marker centroid deviation from planned isocentre", subhead_style))
    plot_image = Image(plot_png, width=170 * mm, height=150 * mm, kind="proportional")
    story.append(plot_image)
    story.append(Spacer(1, 2 * mm))

    # ---- Intervention table -------------------------------------------
    if payload.interventions:
        story.append(Paragraph("Couch corrections", subhead_style))
        story.append(_interventions_table(payload))
        story.append(Spacer(1, 4 * mm))

    # ---- Physicist notes ----------------------------------------------
    story.append(Paragraph("Physicist notes", subhead_style))
    notes_text = notes.strip() if notes else "—"
    # Preserve user line breaks.
    for paragraph in notes_text.split("\n\n"):
        story.append(Paragraph(paragraph.replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 1.5 * mm))

    # ---- Footer (drawn on every page via onPage) ----------------------
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_text = (
        f"Generated by KIM-QA Reporter v{__version__} on {timestamp} "
        f"&middot; Patient dir: {Path(payload.fraction_id).parent}"
    )

    def _draw_footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(18 * mm, 8 * mm, footer_text)
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output_path
