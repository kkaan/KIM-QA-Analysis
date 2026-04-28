import { Atom, ToggleLeft, ToggleRight } from "lucide-react";
import { useStore } from "../lib/store";

export function SeedToggleBar() {
  const { scan, fraction, activeMarkers, toggleMarker, couchRowCount, setCouchRowCount } =
    useStore();
  if (!scan) return null;
  const detected = fraction?.detected_markers ?? scan.detected_markers;
  const active = activeMarkers ?? detected;

  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs uppercase tracking-wider text-kim-muted">
          Active markers
        </label>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {detected.map((idx) => {
            const isActive = active.includes(idx);
            return (
              <button
                key={idx}
                type="button"
                onClick={() => void toggleMarker(idx)}
                className={
                  "flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-sm transition " +
                  (isActive
                    ? "border-emerald-400 bg-emerald-400/10 text-emerald-300"
                    : "border-kim-edge text-kim-muted hover:border-kim-accent")
                }
                title={
                  isActive
                    ? `Marker ${idx} contributing to centroid — click to exclude`
                    : `Marker ${idx} excluded — click to include`
                }
              >
                {isActive ? (
                  <ToggleRight className="h-4 w-4" />
                ) : (
                  <ToggleLeft className="h-4 w-4" />
                )}
                <Atom className="h-3.5 w-3.5" /> Marker {idx}
              </button>
            );
          })}
        </div>
        <p className="mt-2 text-[11px] leading-snug text-kim-muted">
          Toggle to remove a migrated seed from the centroid calculation. The
          expected centroid is recomputed from the same subset of seeds in the
          centroid file so both sides of the deviation stay consistent.
        </p>
      </div>

      <div>
        <label className="text-xs uppercase tracking-wider text-kim-muted">
          Couch rows considered
        </label>
        <div className="mt-2 flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            id="trim-rows"
            checked={couchRowCount !== null}
            onChange={(e) => {
              if (!e.target.checked) {
                void setCouchRowCount(null);
                return;
              }
              // Default to the number of trajectory files (= unique file_index
              // values). For PRIME this gives n_files couch rows -> n_files-1
              // shifts, which is the canonical layout. The user can fine-tune
              // with the number input below.
              const nFiles = fraction
                ? new Set(fraction.file_index).size
                : 3;
              void setCouchRowCount(Math.max(2, nFiles));
            }}
          />
          <label htmlFor="trim-rows" className="text-sm">
            Trim leading rows
          </label>
          {couchRowCount !== null && (
            <input
              type="number"
              min={2}
              value={couchRowCount}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (!Number.isFinite(n) || n < 2) return;
                void setCouchRowCount(n);
              }}
              className="w-20 rounded-md border border-kim-edge bg-kim-bg px-2 py-1 text-sm font-mono"
            />
          )}
        </div>
        <p className="mt-1 text-[11px] leading-snug text-kim-muted">
          Use this for fractions whose <code>couchShifts.txt</code> contains
          stale leading rows from a previous session (e.g. PRIME PAT01 FX01).
        </p>
      </div>
    </div>
  );
}
