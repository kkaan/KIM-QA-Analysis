import { AlertTriangle, Sparkles } from "lucide-react";
import { DeviationPlot } from "./components/DeviationPlot";
import { FractionTabs } from "./components/FractionTabs";
import { InterventionTable } from "./components/InterventionTable";
import { PatientPicker } from "./components/PatientPicker";
import { PhysicistNotes } from "./components/PhysicistNotes";
import { ReportButton } from "./components/ReportButton";
import { SeedToggleBar } from "./components/SeedToggleBar";
import { useStore } from "./lib/store";

export default function App() {
  const {
    scan,
    fraction,
    errorMessage,
    status,
    highlights,
    showOverlay,
  } = useStore();

  return (
    <div className="flex h-full flex-col bg-kim-bg text-kim-ink">
      {/* ---- Top bar ------------------------------------------------ */}
      <header className="flex items-center justify-between border-b border-kim-edge bg-kim-panel px-5 py-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-kim-accent" />
          <h1 className="text-lg font-semibold tracking-tight">
            KIM-QA Reporter
          </h1>
          <span className="ml-2 rounded-full border border-kim-edge px-2 py-0.5 text-[10px] uppercase tracking-wider text-kim-muted">
            v0.1
          </span>
        </div>
        {scan && (
          <div className="text-xs text-kim-muted">
            <span className="text-kim-ink">{scan.patient_id}</span>
            {fraction && (
              <>
                {" / "}
                <span className="text-kim-ink">{fraction.fraction_id}</span>
              </>
            )}
            {fraction && (
              <span className="ml-3 text-kim-muted">
                {fraction.time_real_s.length} frames ·{" "}
                {fraction.gap_display_intervals.length} compressed pauses
              </span>
            )}
          </div>
        )}
      </header>

      {/* ---- Body --------------------------------------------------- */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-[340px] flex-shrink-0 space-y-4 overflow-y-auto border-r border-kim-edge bg-kim-panel/60 p-4">
          <PatientPicker />
          <FractionTabs />
          <SeedToggleBar />
          <PhysicistNotes />
          <ReportButton />
        </aside>

        {/* Main pane */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {errorMessage && (
            <div className="flex items-start gap-2 border-b border-rose-700/60 bg-rose-900/30 px-4 py-2 text-sm text-rose-100">
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <span className="break-all">{errorMessage}</span>
            </div>
          )}
          {fraction ? (
            <>
              <div className="flex-1 px-4 pt-3">
                <DeviationPlot
                  fraction={fraction}
                  highlights={highlights}
                  showOverlay={showOverlay}
                />
              </div>
              {fraction.warnings.length > 0 && (
                <div className="border-t border-kim-edge bg-kim-panel/40 px-4 py-2 text-[11px] text-kim-muted">
                  {fraction.warnings.map((w, i) => (
                    <div key={i}>· {w}</div>
                  ))}
                </div>
              )}
              <div className="border-t border-kim-edge bg-kim-panel/60 p-4">
                <InterventionTable />
              </div>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center p-8 text-center text-kim-muted">
              {status === "loading" ? (
                <span>Loading…</span>
              ) : (
                <div className="max-w-md space-y-2">
                  <h2 className="text-base font-semibold text-kim-ink">
                    Pick a patient directory to begin
                  </h2>
                  <p className="text-sm">
                    The folder should contain per-fraction subfolders
                    (<code>FX01</code>, <code>FX02</code>…) each with a
                    <code className="px-1">Trajectory Logs</code> subfolder
                    holding <code>MarkerLocationsGA_CouchShift_*.txt</code> and
                    <code className="px-1">couchShifts.txt</code>. The
                    <code className="px-1">Centroid_*.txt</code> seed/iso file
                    can sit alongside the patient folder.
                  </p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
