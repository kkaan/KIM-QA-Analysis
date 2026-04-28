// Thin fetch wrappers around the FastAPI backend. In dev, the URL of the
// backend is passed as `?api=...` so the Vite dev server can talk to the
// in-process Python uvicorn server. In production builds, both ride on the
// same origin (the FastAPI server serves the static assets).

export type FractionStub = {
  id: string;
  n_files: number;
  n_couch_rows: number;
};

export type Intervention = {
  id: string;
  index: number;
  t_real_s: number;
  t_display_s: number;
  dlr_mm: number;
  dsi_mm: number;
  dap_mm: number;
  magnitude_mm: number;
  from_cm: number[] | null;
  to_cm: number[] | null;
  is_localisation: boolean;
  source: "log" | "manual";
  label: string;
  highlight_t_start_real_s: number;
  highlight_t_end_real_s: number;
};

export type FractionPayload = {
  patient_id: string;
  fraction_id: string;
  detected_markers: number[];
  active_markers: number[];
  expected_centroid: { x: number; y: number; z: number };
  time_real_s: number[];
  time_display_s: number[];
  gap_real_intervals: number[][];
  gap_display_intervals: number[][];
  burst_id: number[];
  file_index: number[];
  n_active_markers: number[];
  meas_x: number[];
  meas_y: number[];
  meas_z: number[];
  interventions: Intervention[];
  y_limits: Record<string, [number, number]>;
  warnings: string[];
};

export type ScanPayload = {
  patient_id: string;
  patient_dir: string;
  centroid_file: string;
  seeds: number[][];
  isocenter: number[];
  detected_markers: number[];
  fractions: FractionStub[];
};

export type ManualIntervention = {
  t_real_s: number;
  dlr_mm: number;
  dsi_mm: number;
  dap_mm: number;
  label: string;
};

export type HighlightSpec = {
  intervention_id: string;
  t_start_real_s: number;
  t_end_real_s: number;
  label: string;
};

const params = new URLSearchParams(window.location.search);
const apiOverride = params.get("api");
export const API_BASE = (apiOverride ?? "").replace(/\/$/, "");

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText} — ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  scan: (patientDir: string) =>
    postJson<ScanPayload>("/api/scan", { patient_dir: patientDir }),
  fraction: (req: {
    patient_dir: string;
    fraction_id: string;
    centroid_file: string;
    active_markers?: number[] | null;
    couch_row_count?: number | null;
    manual_interventions?: ManualIntervention[];
  }) => postJson<FractionPayload>("/api/fraction", req),
  renderPdf: (req: {
    patient_dir: string;
    fraction_id: string;
    centroid_file: string;
    active_markers?: number[] | null;
    couch_row_count?: number | null;
    manual_interventions?: ManualIntervention[];
    highlights: HighlightSpec[];
    notes: string;
    show_overlay: boolean;
    output_path?: string | null;
  }) =>
    postJson<{ ok: boolean; path?: string; error?: string }>(
      "/api/render-pdf",
      req,
    ),
  pickFolder: () =>
    postJson<{ path: string | null; error?: string }>("/api/pick-folder", {}),
};
