import { useStore } from "../lib/store";

// Tiny clsx-style helper (avoids a runtime dependency).
function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function FractionTabs() {
  const { scan, fractionId, selectFraction } = useStore();
  if (!scan) return null;
  return (
    <div>
      <label className="text-xs uppercase tracking-wider text-kim-muted">
        Fraction
      </label>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {scan.fractions.map((fx) => {
          const active = fx.id === fractionId;
          const disabled = fx.n_files < 2;
          return (
            <button
              key={fx.id}
              type="button"
              onClick={() => !disabled && void selectFraction(fx.id)}
              disabled={disabled}
              className={cx(
                "rounded-md border px-3 py-1.5 text-sm transition",
                active
                  ? "border-kim-accent bg-kim-accent/10 text-kim-accent"
                  : "border-kim-edge bg-kim-bg text-kim-ink hover:border-kim-accent/60",
                disabled && "cursor-not-allowed opacity-30",
              )}
              title={
                disabled
                  ? `${fx.id}: only ${fx.n_files} trajectory file — nothing to plot`
                  : `${fx.id}: ${fx.n_files} traj files, ${fx.n_couch_rows} couch rows`
              }
            >
              {fx.id}
              <span className="ml-1.5 text-xs text-kim-muted">
                {fx.n_files}f / {fx.n_couch_rows}r
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

