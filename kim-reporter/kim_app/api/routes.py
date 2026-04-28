"""FastAPI routes for the KIM-QA Reporter desktop app.

Three endpoints:
  - POST /api/scan           — index a patient directory
  - POST /api/fraction       — load one fraction's deviation timeseries
  - POST /api/render-pdf     — produce a clinical PDF report

Plus auxiliary GETs used by the frontend:
  - GET  /api/health         — readiness probe
  - POST /api/pick-folder    — system folder picker (delegates to pywebview)
"""

from __future__ import annotations

import datetime as _dt
import glob
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

# Make ``python_app`` importable so we can reuse parse_centroid_file. The
# kim-reporter package lives at <repo>/kim-reporter/; python_app at <repo>/python_app/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON_APP = _REPO_ROOT / "python_app"
if str(_PYTHON_APP) not in sys.path:
    sys.path.insert(0, str(_PYTHON_APP))

from kim_analysis_logic import parse_centroid_file  # noqa: E402

from kim_app.core.loader import (  # noqa: E402
    discover_markers,
    expected_centroid_from_seeds,
    file_index,
    load_centroid,
)
from kim_app.core.shifts import (  # noqa: E402
    CouchShift,
    read_couch_rows,
    rows_to_shifts,
    slice_to_last_n,
)
from kim_app.core.window import compress_time_gaps, real_to_display  # noqa: E402

from kim_app.api.schema import (  # noqa: E402
    FractionRequest,
    FractionResponse,
    FractionStub,
    Intervention,
    ManualIntervention,
    RenderPdfRequest,
    RenderPdfResponse,
    ScanRequest,
    ScanResponse,
)


router = APIRouter(prefix="/api")


# ----------------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------------


def _find_centroid_file(patient_dir: Path) -> Optional[Path]:
    """Locate ``Centroid_*.txt`` for a patient. Looks inside the patient dir
    first, then in its parent (PRIME's sample-data layout puts the centroid
    file one directory above the patient folder).
    """
    for search in (patient_dir, patient_dir.parent):
        for cand in search.glob("Centroid_*.txt"):
            return cand
    return None


def _list_fractions(patient_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted ``(fraction_id, trajectory_logs_path)`` for every fraction
    folder under ``patient_dir`` that contains a ``Trajectory Logs`` subdir.
    """
    out: list[tuple[str, Path]] = []
    for entry in sorted(patient_dir.iterdir()):
        if not entry.is_dir():
            continue
        traj = entry / "Trajectory Logs"
        if traj.is_dir():
            out.append((entry.name, traj))
    return out


# ----------------------------------------------------------------------------
# /api/health
# ----------------------------------------------------------------------------


@router.get("/health")
def health() -> dict:
    return {"ok": True}


# ----------------------------------------------------------------------------
# /api/scan
# ----------------------------------------------------------------------------


@router.post("/scan", response_model=ScanResponse)
def scan_patient(req: ScanRequest) -> ScanResponse:
    pdir = Path(req.patient_dir).expanduser().resolve()
    if not pdir.is_dir():
        raise HTTPException(404, f"Patient directory not found: {pdir}")

    centroid_path = _find_centroid_file(pdir)
    if centroid_path is None:
        raise HTTPException(
            404,
            f"No Centroid_*.txt found in {pdir} or its parent.",
        )

    centroid = parse_centroid_file(str(centroid_path))
    fractions = _list_fractions(pdir)
    if not fractions:
        raise HTTPException(404, f"No fraction folders in {pdir}.")

    # Pick the first fraction with trajectory files to discover marker indices.
    detected: list[int] = []
    fraction_stubs: list[FractionStub] = []
    for fx_id, traj in fractions:
        ga_files = sorted(
            glob.glob(str(traj / "MarkerLocationsGA_CouchShift_*.txt")),
            key=file_index,
        )
        couch_file = traj / "couchShifts.txt"
        n_couch_rows = 0
        if couch_file.exists():
            n_couch_rows = len(read_couch_rows(str(couch_file)))
        if not detected and ga_files:
            try:
                detected = discover_markers(str(traj))
            except Exception:
                detected = []
        fraction_stubs.append(
            FractionStub(
                id=fx_id,
                n_files=len(ga_files),
                n_couch_rows=n_couch_rows,
            )
        )

    return ScanResponse(
        patient_id=pdir.name,
        patient_dir=str(pdir),
        centroid_file=str(centroid_path),
        seeds=[list(s) for s in centroid["seeds"]],
        isocenter=list(centroid["isocenter"]),
        detected_markers=detected,
        fractions=fraction_stubs,
    )


# ----------------------------------------------------------------------------
# /api/fraction
# ----------------------------------------------------------------------------


_DEFAULT_HIGHLIGHT_HALF_S = 20.0
"""Default ±N seconds around each intervention to draw the gold highlight band."""


_DEFAULT_Y_LIMITS = {
    "meas_x": [-5.0, 5.0],
    "meas_y": [-5.0, 5.0],
    "meas_z": [-5.0, 5.0],
}


def _load_fraction_payload(req: FractionRequest) -> FractionResponse:
    """Shared core of /api/fraction. Returns the assembled FractionResponse so
    the PDF route can also use it."""
    pdir = Path(req.patient_dir).expanduser().resolve()
    traj = pdir / req.fraction_id / "Trajectory Logs"
    if not traj.is_dir():
        raise HTTPException(404, f"Trajectory Logs folder not found: {traj}")

    centroid_path = Path(req.centroid_file).expanduser().resolve()
    if not centroid_path.is_file():
        raise HTTPException(404, f"Centroid file not found: {centroid_path}")

    centroid = parse_centroid_file(str(centroid_path))
    seeds = [list(s) for s in centroid["seeds"]]
    isocenter = list(centroid["isocenter"])

    # Recompute expected centroid using only the active seed subset so both
    # sides of the deviation subtraction stay consistent.
    expected = expected_centroid_from_seeds(
        seeds=seeds,
        isocenter=isocenter,
        active_markers=req.active_markers,
    )

    result = load_centroid(
        folder=str(traj),
        expected_centroid=expected,
        active_markers=req.active_markers,
    )

    df = result.df
    if df.empty:
        raise HTTPException(422, "No frames survived sentinel/glitch filtering")

    # Compress beam-off pauses for the display axis.
    display_s, gap_real, gap_display = compress_time_gaps(df["time"].values)
    df = df.copy()
    df["display_s"] = display_s

    # Couch shifts ----------------------------------------------------------
    couch_file = traj / "couchShifts.txt"
    couch_rows: list[tuple[float, float, float]] = []
    if couch_file.exists():
        couch_rows = read_couch_rows(str(couch_file))
    couch_rows = slice_to_last_n(couch_rows, req.couch_row_count)
    log_shifts = rows_to_shifts(couch_rows)

    # Boundaries: midpoint between last frame of file N and first frame of N+1.
    file_indices = sorted(df["file_index"].unique().tolist())
    boundaries_real_s: list[float] = []
    for prev_fi, next_fi in zip(file_indices[:-1], file_indices[1:]):
        t_prev_end = float(df.loc[df["file_index"] == prev_fi, "time"].max())
        t_next_start = float(df.loc[df["file_index"] == next_fi, "time"].min())
        boundaries_real_s.append((t_prev_end + t_next_start) / 2.0)

    # Build interventions list. Each log-derived shift maps to one boundary;
    # if shift count and boundary count disagree we still surface what we have.
    interventions: list[Intervention] = []
    n_log_pairs = min(len(log_shifts), len(boundaries_real_s))
    for k in range(n_log_pairs):
        sh: CouchShift = log_shifts[k]
        t_real = boundaries_real_s[k]
        t_disp = real_to_display(t_real, df["time"].values, display_s)
        interventions.append(
            Intervention(
                id=f"log-{k}",
                index=k,
                t_real_s=t_real,
                t_display_s=t_disp,
                dlr_mm=sh.lr,
                dsi_mm=sh.si,
                dap_mm=sh.ap,
                magnitude_mm=sh.magnitude,
                from_cm=list(couch_rows[k]),
                to_cm=list(couch_rows[k + 1]),
                is_localisation=(k == 0),
                source="log",
                label=("Localisation" if k == 0 else f"Intra-fraction #{k}"),
                highlight_t_start_real_s=max(0.0, t_real - _DEFAULT_HIGHLIGHT_HALF_S),
                highlight_t_end_real_s=t_real + _DEFAULT_HIGHLIGHT_HALF_S,
            )
        )

    # Manual interventions get appended in the order the UI provided them.
    for j, m in enumerate(req.manual_interventions):
        t_disp = real_to_display(m.t_real_s, df["time"].values, display_s)
        magnitude = float((m.dlr_mm ** 2 + m.dsi_mm ** 2 + m.dap_mm ** 2) ** 0.5)
        interventions.append(
            Intervention(
                id=f"manual-{j}",
                index=len(log_shifts) + j,
                t_real_s=m.t_real_s,
                t_display_s=t_disp,
                dlr_mm=m.dlr_mm,
                dsi_mm=m.dsi_mm,
                dap_mm=m.dap_mm,
                magnitude_mm=magnitude,
                from_cm=None,
                to_cm=None,
                is_localisation=False,
                source="manual",
                label=m.label,
                highlight_t_start_real_s=max(0.0, m.t_real_s - _DEFAULT_HIGHLIGHT_HALF_S),
                highlight_t_end_real_s=m.t_real_s + _DEFAULT_HIGHLIGHT_HALF_S,
            )
        )

    return FractionResponse(
        patient_id=pdir.name,
        fraction_id=req.fraction_id,
        detected_markers=result.detected_markers,
        active_markers=result.active_markers,
        expected_centroid=expected,
        time_real_s=df["time"].astype(float).tolist(),
        time_display_s=df["display_s"].astype(float).tolist(),
        gap_real_intervals=[list(g) for g in gap_real],
        gap_display_intervals=[list(g) for g in gap_display],
        burst_id=df["burst_id"].astype(int).tolist(),
        file_index=df["file_index"].astype(int).tolist(),
        n_active_markers=df["n_active_markers"].astype(int).tolist(),
        meas_x=df["meas_x"].astype(float).tolist(),
        meas_y=df["meas_y"].astype(float).tolist(),
        meas_z=df["meas_z"].astype(float).tolist(),
        interventions=interventions,
        y_limits=_DEFAULT_Y_LIMITS,
        warnings=result.warnings,
    )


@router.post("/fraction", response_model=FractionResponse)
def get_fraction(req: FractionRequest) -> FractionResponse:
    return _load_fraction_payload(req)


# ----------------------------------------------------------------------------
# /api/render-pdf
# ----------------------------------------------------------------------------


@router.post("/render-pdf", response_model=RenderPdfResponse)
def render_pdf(req: RenderPdfRequest) -> RenderPdfResponse:
    from kim_app.pdf.report import render_report

    fraction_req = FractionRequest(
        patient_dir=req.patient_dir,
        fraction_id=req.fraction_id,
        centroid_file=req.centroid_file,
        active_markers=req.active_markers,
        couch_row_count=req.couch_row_count,
        manual_interventions=req.manual_interventions,
    )
    payload = _load_fraction_payload(fraction_req)

    if req.output_path:
        out_path = Path(req.output_path).expanduser().resolve()
    else:
        date_tag = _dt.datetime.now().strftime("%Y%m%d")
        out_path = (
            Path(req.patient_dir).expanduser().resolve()
            / f"KIM_Report_{payload.patient_id}_{payload.fraction_id}_{date_tag}.pdf"
        )

    try:
        render_report(
            payload=payload,
            highlights=req.highlights,
            notes=req.notes,
            show_overlay=req.show_overlay,
            output_path=out_path,
        )
    except Exception as exc:   # noqa: BLE001 — surface every failure to the UI
        return RenderPdfResponse(ok=False, error=str(exc))

    return RenderPdfResponse(ok=True, path=str(out_path))


# ----------------------------------------------------------------------------
# /api/pick-folder — delegates to pywebview's native dialog
# ----------------------------------------------------------------------------


@router.post("/pick-folder")
def pick_folder() -> dict:
    """Open a native folder-picker. Returns ``{path: str|None}``.

    Uses pywebview's ``create_file_dialog(FOLDER_DIALOG)`` if a window is
    available; otherwise falls back to returning ``{path: None}`` so the UI can
    fall back to a manual text input.
    """
    try:
        import webview  # type: ignore

        windows = webview.windows
        if not windows:
            return {"path": None, "error": "No active webview window"}
        result = windows[0].create_file_dialog(webview.FOLDER_DIALOG)  # type: ignore[attr-defined]
        if not result:
            return {"path": None}
        # pywebview returns a tuple-like; take the first entry.
        return {"path": str(result[0])}
    except Exception as exc:  # noqa: BLE001
        return {"path": None, "error": str(exc)}
