// Zustand store: the single source of truth the React tree reads from.
// The state shape mirrors the API response with light editorial layers on
// top (highlights, manual interventions, notes, overlay toggle) so that
// re-rendering the chart from local state is instant.

import { create } from "zustand";
import {
  FractionPayload,
  HighlightSpec,
  ManualIntervention,
  ScanPayload,
  api,
} from "./api";

type Status = "idle" | "loading" | "ready" | "error";

interface State {
  scan: ScanPayload | null;
  fraction: FractionPayload | null;
  status: Status;
  errorMessage: string | null;

  patientDir: string;
  fractionId: string;
  activeMarkers: number[] | null;
  couchRowCount: number | null;
  manualInterventions: ManualIntervention[];
  highlights: Record<string, HighlightSpec>;
  notes: string;
  showOverlay: boolean;
  pdfStatus: "idle" | "rendering" | "done" | "error";
  pdfMessage: string | null;
  pdfPath: string | null;

  setPatientDir: (s: string) => void;
  pickPatientFolder: () => Promise<void>;
  scanPatient: (patientDir?: string) => Promise<void>;
  selectFraction: (fractionId: string) => Promise<void>;
  toggleMarker: (markerIndex: number) => Promise<void>;
  setCouchRowCount: (n: number | null) => Promise<void>;
  addManualIntervention: (m: ManualIntervention) => Promise<void>;
  removeManualIntervention: (idx: number) => Promise<void>;
  updateHighlight: (id: string, range: [number, number]) => void;
  setNotes: (text: string) => void;
  toggleOverlay: () => void;
  renderPdf: () => Promise<void>;
}

export const useStore = create<State>((set, get) => ({
  scan: null,
  fraction: null,
  status: "idle",
  errorMessage: null,
  patientDir: "",
  fractionId: "",
  activeMarkers: null,
  couchRowCount: null,
  manualInterventions: [],
  highlights: {},
  notes: "",
  showOverlay: true,
  pdfStatus: "idle",
  pdfMessage: null,
  pdfPath: null,

  setPatientDir: (s) => set({ patientDir: s }),

  pickPatientFolder: async () => {
    try {
      const res = await api.pickFolder();
      if (res.path) {
        set({ patientDir: res.path });
        await get().scanPatient(res.path);
      }
    } catch (e) {
      set({ errorMessage: (e as Error).message });
    }
  },

  scanPatient: async (patientDir) => {
    const dir = patientDir ?? get().patientDir;
    if (!dir) return;
    set({ status: "loading", errorMessage: null });
    try {
      const scan = await api.scan(dir);
      set({
        scan,
        patientDir: scan.patient_dir,
        activeMarkers: scan.detected_markers,
        couchRowCount: null,
        fraction: null,
        manualInterventions: [],
        highlights: {},
        status: "ready",
      });
      // Auto-select the first fraction with at least 2 trajectory files.
      const auto =
        scan.fractions.find((f) => f.n_files >= 2) ?? scan.fractions[0];
      if (auto) await get().selectFraction(auto.id);
    } catch (e) {
      set({ status: "error", errorMessage: (e as Error).message });
    }
  },

  selectFraction: async (fractionId) => {
    const { scan, activeMarkers, couchRowCount, manualInterventions } = get();
    if (!scan) return;
    set({ status: "loading", fractionId });
    try {
      const fraction = await api.fraction({
        patient_dir: scan.patient_dir,
        fraction_id: fractionId,
        centroid_file: scan.centroid_file,
        active_markers: activeMarkers,
        couch_row_count: couchRowCount,
        manual_interventions: manualInterventions,
      });
      // Seed default highlights from each non-localisation intervention.
      const highlights: Record<string, HighlightSpec> = {};
      for (const iv of fraction.interventions) {
        if (iv.is_localisation) continue;
        highlights[iv.id] = {
          intervention_id: iv.id,
          t_start_real_s: iv.highlight_t_start_real_s,
          t_end_real_s: iv.highlight_t_end_real_s,
          label: iv.label,
        };
      }
      set({
        fraction,
        highlights,
        status: "ready",
        errorMessage: null,
        pdfStatus: "idle",
        pdfMessage: null,
        pdfPath: null,
      });
    } catch (e) {
      set({ status: "error", errorMessage: (e as Error).message });
    }
  },

  toggleMarker: async (markerIndex) => {
    const { scan, activeMarkers } = get();
    if (!scan) return;
    const cur = activeMarkers ?? scan.detected_markers;
    let next: number[];
    if (cur.includes(markerIndex)) {
      next = cur.filter((m) => m !== markerIndex);
    } else {
      next = [...cur, markerIndex].sort((a, b) => a - b);
    }
    if (next.length === 0) {
      set({ errorMessage: "At least one marker must remain active" });
      return;
    }
    set({ activeMarkers: next });
    if (get().fractionId) await get().selectFraction(get().fractionId);
  },

  setCouchRowCount: async (n) => {
    set({ couchRowCount: n });
    if (get().fractionId) await get().selectFraction(get().fractionId);
  },

  addManualIntervention: async (m) => {
    set({ manualInterventions: [...get().manualInterventions, m] });
    if (get().fractionId) await get().selectFraction(get().fractionId);
  },

  removeManualIntervention: async (idx) => {
    set({
      manualInterventions: get().manualInterventions.filter((_, i) => i !== idx),
    });
    if (get().fractionId) await get().selectFraction(get().fractionId);
  },

  updateHighlight: (id, range) => {
    const cur = get().highlights[id];
    if (!cur) return;
    set({
      highlights: {
        ...get().highlights,
        [id]: { ...cur, t_start_real_s: range[0], t_end_real_s: range[1] },
      },
    });
  },

  setNotes: (text) => set({ notes: text }),
  toggleOverlay: () => set({ showOverlay: !get().showOverlay }),

  renderPdf: async () => {
    const s = get();
    if (!s.scan || !s.fraction) return;
    set({ pdfStatus: "rendering", pdfMessage: null, pdfPath: null });
    try {
      const res = await api.renderPdf({
        patient_dir: s.scan.patient_dir,
        fraction_id: s.fractionId,
        centroid_file: s.scan.centroid_file,
        active_markers: s.activeMarkers,
        couch_row_count: s.couchRowCount,
        manual_interventions: s.manualInterventions,
        highlights: Object.values(s.highlights),
        notes: s.notes,
        show_overlay: s.showOverlay,
      });
      if (res.ok) {
        set({
          pdfStatus: "done",
          pdfPath: res.path ?? null,
          pdfMessage: `Saved to ${res.path}`,
        });
      } else {
        set({ pdfStatus: "error", pdfMessage: res.error ?? "Unknown error" });
      }
    } catch (e) {
      set({ pdfStatus: "error", pdfMessage: (e as Error).message });
    }
  },
}));
