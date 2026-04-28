import { CheckCircle2, FileDown, Loader2, XCircle } from "lucide-react";
import { useStore } from "../lib/store";

export function ReportButton() {
  const {
    fraction,
    pdfStatus,
    pdfMessage,
    pdfPath,
    renderPdf,
    showOverlay,
    toggleOverlay,
  } = useStore();

  const disabled = !fraction || pdfStatus === "rendering";

  return (
    <div className="space-y-2 rounded-md border border-kim-edge bg-kim-panel/60 p-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">Export PDF report</h3>
          <p className="text-[11px] text-kim-muted">
            A4 portrait · saved next to the patient folder
          </p>
        </div>
        <button
          type="button"
          onClick={() => void renderPdf()}
          disabled={disabled}
          className="flex items-center gap-1.5 rounded-md bg-kim-accent px-3 py-2 text-sm font-medium text-kim-bg hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {pdfStatus === "rendering" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <FileDown className="h-4 w-4" />
          )}
          Render
        </button>
      </div>
      <label className="flex items-center gap-2 text-xs text-kim-muted">
        <input
          type="checkbox"
          checked={showOverlay}
          onChange={() => toggleOverlay()}
        />
        Include no-correction counterfactual overlay
      </label>
      {pdfStatus === "done" && (
        <div className="flex items-start gap-2 rounded border border-emerald-700/60 bg-emerald-900/30 p-2 text-xs text-emerald-200">
          <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span className="break-all">
            Saved to <code className="font-mono">{pdfPath}</code>
          </span>
        </div>
      )}
      {pdfStatus === "error" && (
        <div className="flex items-start gap-2 rounded border border-rose-700/60 bg-rose-900/30 p-2 text-xs text-rose-200">
          <XCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span>{pdfMessage}</span>
        </div>
      )}
    </div>
  );
}
