import { FolderOpen, Loader2, Search } from "lucide-react";
import { useStore } from "../lib/store";

export function PatientPicker() {
  const {
    patientDir,
    setPatientDir,
    pickPatientFolder,
    scanPatient,
    status,
  } = useStore();
  return (
    <div className="space-y-2">
      <label className="text-xs uppercase tracking-wider text-kim-muted">
        Patient directory
      </label>
      <input
        value={patientDir}
        onChange={(e) => setPatientDir(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") void scanPatient();
        }}
        placeholder="C:\path\to\PatientFolder"
        title={patientDir}
        className="w-full min-w-0 truncate rounded-md border border-kim-edge bg-kim-bg px-3 py-2 text-sm font-mono text-kim-ink placeholder:text-kim-muted focus:border-kim-accent focus:outline-none"
      />
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => void pickPatientFolder()}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-kim-edge bg-kim-bg px-3 py-2 text-sm hover:border-kim-accent hover:text-kim-accent"
          title="Pick folder"
        >
          <FolderOpen className="h-4 w-4" />
          Browse
        </button>
        <button
          type="button"
          onClick={() => void scanPatient()}
          disabled={!patientDir || status === "loading"}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-kim-accent px-3 py-2 text-sm font-medium text-kim-bg hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {status === "loading" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Search className="h-4 w-4" />
          )}
          Scan
        </button>
      </div>
    </div>
  );
}
