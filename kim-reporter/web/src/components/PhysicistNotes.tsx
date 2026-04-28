import { useStore } from "../lib/store";

export function PhysicistNotes() {
  const { notes, setNotes } = useStore();
  return (
    <div>
      <label className="text-xs uppercase tracking-wider text-kim-muted">
        Physicist notes
      </label>
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Brief observations to embed in the PDF — patient response, image quality, deviations from protocol, follow-up actions…"
        className="mt-1 w-full rounded-md border border-kim-edge bg-kim-bg px-3 py-2 text-sm text-kim-ink placeholder:text-kim-muted focus:border-kim-accent focus:outline-none"
        rows={5}
      />
    </div>
  );
}
