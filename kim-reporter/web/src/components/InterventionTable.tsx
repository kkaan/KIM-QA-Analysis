import { Pin, PlusCircle, Trash2 } from "lucide-react";
import { useState } from "react";
import { useStore } from "../lib/store";

function fmt(n: number): string {
  return (n >= 0 ? "+" : "") + n.toFixed(2);
}

export function InterventionTable() {
  const {
    fraction,
    highlights,
    updateHighlight,
    addManualIntervention,
    removeManualIntervention,
  } = useStore();
  const [draftT, setDraftT] = useState("");
  const [draftLR, setDraftLR] = useState("0");
  const [draftSI, setDraftSI] = useState("0");
  const [draftAP, setDraftAP] = useState("0");
  const [draftLabel, setDraftLabel] = useState("Manual intervention");

  if (!fraction) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs uppercase tracking-wider text-kim-muted">
          Interventions
        </label>
        <span className="text-[11px] text-kim-muted">
          {fraction.interventions.filter((i) => !i.is_localisation).length}{" "}
          intra-fraction
        </span>
      </div>

      <div className="overflow-x-auto rounded-md border border-kim-edge">
        <table className="w-full text-left text-xs">
          <thead className="bg-kim-bg/60 text-kim-muted">
            <tr>
              <th className="px-2 py-1.5 font-medium">#</th>
              <th className="px-2 py-1.5 font-medium">Source</th>
              <th className="px-2 py-1.5 font-medium">t (s)</th>
              <th className="px-2 py-1.5 font-medium">ΔLR</th>
              <th className="px-2 py-1.5 font-medium">ΔSI</th>
              <th className="px-2 py-1.5 font-medium">ΔAP</th>
              <th className="px-2 py-1.5 font-medium">|Δ|</th>
              <th className="px-2 py-1.5 font-medium">Highlight (s)</th>
              <th className="px-2 py-1.5"></th>
            </tr>
          </thead>
          <tbody>
            {fraction.interventions.map((iv) => {
              const hl = highlights[iv.id];
              return (
                <tr
                  key={iv.id}
                  className={
                    "border-t border-kim-edge/60 " +
                    (iv.is_localisation
                      ? "text-kim-muted"
                      : "text-kim-ink hover:bg-kim-bg/50")
                  }
                >
                  <td className="px-2 py-1.5 font-mono">{iv.index + 1}</td>
                  <td className="px-2 py-1.5">
                    <span className="inline-flex items-center gap-1">
                      {iv.source === "manual" && (
                        <Pin className="h-3 w-3 text-kim-accent" />
                      )}
                      {iv.label}
                      {iv.is_localisation && " (excl.)"}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 font-mono">
                    {iv.t_real_s.toFixed(1)}
                  </td>
                  <td className="px-2 py-1.5 font-mono">{fmt(iv.dlr_mm)}</td>
                  <td className="px-2 py-1.5 font-mono">{fmt(iv.dsi_mm)}</td>
                  <td className="px-2 py-1.5 font-mono">{fmt(iv.dap_mm)}</td>
                  <td className="px-2 py-1.5 font-mono">
                    {iv.magnitude_mm.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5">
                    {hl ? (
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          step={1}
                          value={hl.t_start_real_s.toFixed(1)}
                          onChange={(e) =>
                            updateHighlight(iv.id, [
                              parseFloat(e.target.value),
                              hl.t_end_real_s,
                            ])
                          }
                          className="w-14 rounded border border-kim-edge bg-kim-bg px-1 font-mono text-[11px]"
                        />
                        <span className="text-kim-muted">…</span>
                        <input
                          type="number"
                          step={1}
                          value={hl.t_end_real_s.toFixed(1)}
                          onChange={(e) =>
                            updateHighlight(iv.id, [
                              hl.t_start_real_s,
                              parseFloat(e.target.value),
                            ])
                          }
                          className="w-14 rounded border border-kim-edge bg-kim-bg px-1 font-mono text-[11px]"
                        />
                      </div>
                    ) : (
                      <span className="text-kim-muted">—</span>
                    )}
                  </td>
                  <td className="px-2 py-1.5">
                    {iv.source === "manual" && (
                      <button
                        type="button"
                        onClick={() => {
                          // Manual entries are appended in order, with index =
                          // log_count + j. Compute j from the index.
                          const logCount = fraction.interventions.filter(
                            (i) => i.source === "log",
                          ).length;
                          const j = iv.index - logCount;
                          if (j >= 0) void removeManualIntervention(j);
                        }}
                        className="text-kim-muted hover:text-rose-400"
                        title="Remove manual intervention"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Manual-intervention form ----------------------------------- */}
      <details className="rounded-md border border-kim-edge bg-kim-bg/40 p-2">
        <summary className="cursor-pointer text-xs text-kim-muted hover:text-kim-ink">
          <PlusCircle className="mr-1 inline h-3.5 w-3.5" />
          Add manual intervention (e.g. mid-file shift not in
          <code className="px-1">couchShifts.txt</code>)
        </summary>
        <div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-5">
          <div>
            <label className="text-kim-muted">Time (s)</label>
            <input
              type="number"
              value={draftT}
              onChange={(e) => setDraftT(e.target.value)}
              className="w-full rounded border border-kim-edge bg-kim-bg px-1.5 py-1 font-mono"
            />
          </div>
          <div>
            <label className="text-kim-muted">ΔLR (mm)</label>
            <input
              type="number"
              step={0.1}
              value={draftLR}
              onChange={(e) => setDraftLR(e.target.value)}
              className="w-full rounded border border-kim-edge bg-kim-bg px-1.5 py-1 font-mono"
            />
          </div>
          <div>
            <label className="text-kim-muted">ΔSI (mm)</label>
            <input
              type="number"
              step={0.1}
              value={draftSI}
              onChange={(e) => setDraftSI(e.target.value)}
              className="w-full rounded border border-kim-edge bg-kim-bg px-1.5 py-1 font-mono"
            />
          </div>
          <div>
            <label className="text-kim-muted">ΔAP (mm)</label>
            <input
              type="number"
              step={0.1}
              value={draftAP}
              onChange={(e) => setDraftAP(e.target.value)}
              className="w-full rounded border border-kim-edge bg-kim-bg px-1.5 py-1 font-mono"
            />
          </div>
          <div>
            <label className="text-kim-muted">Label</label>
            <input
              value={draftLabel}
              onChange={(e) => setDraftLabel(e.target.value)}
              className="w-full rounded border border-kim-edge bg-kim-bg px-1.5 py-1"
            />
          </div>
        </div>
        <button
          type="button"
          className="mt-2 rounded-md bg-kim-accent px-3 py-1 text-xs font-medium text-kim-bg hover:bg-amber-400"
          onClick={() => {
            const t = parseFloat(draftT);
            if (!isFinite(t)) return;
            void addManualIntervention({
              t_real_s: t,
              dlr_mm: parseFloat(draftLR) || 0,
              dsi_mm: parseFloat(draftSI) || 0,
              dap_mm: parseFloat(draftAP) || 0,
              label: draftLabel,
            });
            setDraftT("");
          }}
        >
          Add
        </button>
      </details>
    </div>
  );
}
