"""Pydantic request/response models for the FastAPI surface.

Every exchange between the React frontend and the Python backend is JSON, so
the schemas below define the wire contract. Times are always seconds (float),
distances always millimetres (float), couch positions always centimetres
(float, 3-tuple of VRT/LNG/LAT).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------
# /api/scan
# ----------------------------------------------------------------------------


class ScanRequest(BaseModel):
    patient_dir: str = Field(..., description="Absolute path to a patient directory")


class FractionStub(BaseModel):
    id: str
    n_files: int
    n_couch_rows: int
    has_warnings: bool = False


class ScanResponse(BaseModel):
    patient_id: str
    patient_dir: str
    centroid_file: str
    seeds: list[list[float]]
    isocenter: list[float]
    detected_markers: list[int]
    fractions: list[FractionStub]


# ----------------------------------------------------------------------------
# /api/fraction
# ----------------------------------------------------------------------------


class ManualIntervention(BaseModel):
    """User-entered intervention used when an operator-applied shift was not
    written to ``couchShifts.txt`` (e.g. PRIME PAT01 FX05).
    """
    t_real_s: float
    dlr_mm: float
    dsi_mm: float
    dap_mm: float
    label: str = "Manual intervention"


class FractionRequest(BaseModel):
    patient_dir: str
    fraction_id: str
    centroid_file: str
    active_markers: Optional[list[int]] = None
    couch_row_count: Optional[int] = None
    manual_interventions: list[ManualIntervention] = Field(default_factory=list)


class Intervention(BaseModel):
    id: str
    index: int
    t_real_s: float
    t_display_s: float
    dlr_mm: float
    dsi_mm: float
    dap_mm: float
    magnitude_mm: float
    from_cm: Optional[list[float]] = None     # [VRT, LNG, LAT]; None for manual
    to_cm: Optional[list[float]] = None
    is_localisation: bool
    source: Literal["log", "manual"]
    label: str
    # Default highlight: ±20 s around t_real_s.
    highlight_t_start_real_s: float
    highlight_t_end_real_s: float


class FractionResponse(BaseModel):
    patient_id: str
    fraction_id: str
    detected_markers: list[int]
    active_markers: list[int]
    expected_centroid: dict[str, float]      # {x, y, z} mm
    time_real_s: list[float]
    time_display_s: list[float]
    gap_real_intervals: list[list[float]]    # [(start, end), ...]
    gap_display_intervals: list[list[float]]
    burst_id: list[int]
    file_index: list[int]
    n_active_markers: list[int]
    meas_x: list[float]
    meas_y: list[float]
    meas_z: list[float]
    interventions: list[Intervention]
    y_limits: dict[str, list[float]]         # {meas_x: [lo, hi], ...}
    warnings: list[str]


# ----------------------------------------------------------------------------
# /api/render-pdf
# ----------------------------------------------------------------------------


class HighlightSpec(BaseModel):
    """One drag-edited gold band that will be drawn on the PDF plots."""
    intervention_id: str
    t_start_real_s: float
    t_end_real_s: float
    label: str


class RenderPdfRequest(BaseModel):
    patient_dir: str
    fraction_id: str
    centroid_file: str
    active_markers: Optional[list[int]] = None
    couch_row_count: Optional[int] = None
    manual_interventions: list[ManualIntervention] = Field(default_factory=list)
    highlights: list[HighlightSpec] = Field(default_factory=list)
    notes: str = ""
    show_overlay: bool = True
    output_path: Optional[str] = None


class RenderPdfResponse(BaseModel):
    ok: bool
    path: Optional[str] = None
    error: Optional[str] = None
