# KIM-QA Reporter — session handoff

This is the picking-up-where-we-left-off note for the `kim-reporter` desktop
app. Read this first when resuming work.

## What this is

A desktop app that lets a medical physicist:

1. Point at a patient directory containing PRIME trajectory logs.
2. See an interactive 3-panel deviation plot (LR / SI / AP) with pan, zoom,
   compressed beam-off pauses, and a default ±20 s gold band around every
   intra-fraction couch correction.
3. Toggle individual markers off (handles the “seed migrated, exclude it”
   case that the existing `abstract-figures/` scripts crash on).
4. Add manual interventions (handles operator-applied shifts that never made
   it to `couchShifts.txt`, e.g. PAT01 FX05).
5. Trim leading rows of `couchShifts.txt` (handles merged-session anomalies
   like PAT01 FX01).
6. Type a free-text physicist note.
7. Click **Render** to drop a clean A4 PDF next to the patient directory.

Stack: **Python (FastAPI + uvicorn + pywebview) + React/Vite/Plotly.js**, all
in one process, packaged with PyInstaller into a single Windows `.exe`.

## Current status

| Area | State |
|---|---|
| Backend `kim_app/core/loader.py` | Done. Dynamic-marker centroid loader; reuses `parse_centroid_file` and the abstract-figures parser quirks (thousands-comma regex, sentinel filter, glitch filter) |
| Backend `kim_app/core/shifts.py` | Done. couchShifts.txt parser with `slice_to_last_n` for the merged-session UI knob |
| Backend `kim_app/core/window.py` | Done. `compress_time_gaps` lifted verbatim |
| Backend `kim_app/core/overlay.py` | Done. No-correction counterfactual logic |
| Backend `kim_app/api/routes.py` | Done. `/api/scan`, `/api/fraction`, `/api/render-pdf`, `/api/pick-folder`, `/api/health` |
| Backend `kim_app/pdf/report.py` | Done. ReportLab clinical A4 PDF |
| Backend `kim_app/__main__.py` + `server.py` | Done. uvicorn-in-thread + pywebview window |
| Frontend `web/src/` | Done. React/Plotly UI; verified rendering on FX04 |
| PyInstaller `KIM-QA-Reporter.spec` | Done. Not yet exercised — `python -m kim_app` works, the frozen build hasn't been smoke-tested on a clean machine |
| Smoke tests | Manual on FX01/FX04/FX05 sample data + 3-marker synthetic test |

### Bugs fixed in the last working session
1. Frontend bundle missing (build instructions added to README, npm rollup-native-dep workaround documented).
2. `Trim leading rows` checkbox sent a non-integer float, returning HTTP 422 (now sends `n_files`).
3. Scan button hidden by long Windows paths (PatientPicker restructured: input on its own row, buttons below).

### Known issues — NOT YET FIXED
- **FX01 chart artefacts** caused by a non-monotonic Time column inside its
  trajectory files (a *second* merged-session anomaly, distinct from the
  couchShifts.txt one). See [issue #8](https://github.com/kkaan/KIM-QA-Analysis/issues/8) created in this session for the
  diagnostic detail and proposed fix. Code from the in-progress fix has been
  reverted from this branch so we can land a clean handoff.

## Code reuse map (from the live repo)

The new app does not duplicate parsers. It imports them from existing modules:

| Used module | Where reused |
|---|---|
| `python_app/kim_analysis_logic.parse_centroid_file` | `kim_app/api/routes.py` (via `sys.path.insert`) |
| Parser quirks lifted into `kim_app/core/loader.py` | `_THOUSANDS_COMMA_RE`, `SANE_LIMIT_MM`, `GLITCH_DEV_MM`, `GAP_BREAK_S` |
| `compress_time_gaps` from `abstract-figures/make_prime_stepjump_figure.py` | re-implemented verbatim in `kim_app/core/window.py` (no behavioural change) |
| Counterfactual overlay maths | re-implemented in `kim_app/core/overlay.py` |

The `abstract-figures/` and `python_app/` directories are NOT modified — the
reporter is purely additive.

## How to run

### Dev mode (Vite HMR + uvicorn in-process)
```powershell
cd kim-reporter\web
npm install
npm run dev   # http://localhost:5173

# in another terminal
cd kim-reporter
pip install -e ".[dev]"
$env:KIM_REPORTER_DEV_FRONTEND="http://localhost:5173"
python -m kim_app
```

### Production-style local run
```powershell
cd kim-reporter\web
npm run build      # outputs to ../kim_app/web_dist
cd ..
python -m kim_app
```

### Frozen build (not yet validated)
```powershell
cd kim-reporter\web; npm run build
cd ..; pyinstaller KIM-QA-Reporter.spec
# Output: dist/KIM-QA-Reporter.exe
```

## Remaining TODOs

### High priority
1. **Land the FX01 fix.** See [issue #8](https://github.com/kkaan/KIM-QA-Analysis/issues/8). The approach we were on:
   - In `kim_app/core/loader.py`, scan the concatenated trajectory stream for
     backward time-jumps > 5 s and drop everything before the LAST jump.
   - Add a `merged_session_drop_count: int` field to `LoadResult`.
   - In `kim_app/api/routes.py`, when a merge is detected and the user has not
     explicitly set `couch_row_count`, auto-set it to
     `n_unique_file_indexes_post_trim + 1` (matches PAT01 FX01’s “last 3 rows
     of couchShifts.txt are real” rule).
   - Extend the intervention-mapping logic so when `len(deltas) > n_boundaries`,
     the **first** `n_extra` deltas are pre-trajectory shifts (assign them
     synthetic negative times so the localisation event still appears in the
     intervention list).
   - The earlier in-progress code is in [issue #8](https://github.com/kkaan/KIM-QA-Analysis/issues/8) body for reference.

2. **Validate the PyInstaller frozen build** end-to-end on a clean Windows
   profile (ideally a VM with no Python installed). The spec file currently
   excludes `tkinter` / Qt / kaleido and explicitly bundles `matplotlib`'s
   data dir — there may be additional hidden imports needed for `uvicorn` or
   `pywebview` that only surface in the frozen environment.

3. **Intervention labels**: the current code labels delta `k=0` as
   "Localisation" and the rest as `Intra-fraction #k`. Once the FX01 fix lands,
   this needs revisiting (see TODO #1).

### Medium priority
4. **Drag-edit highlight bands in the chart.** Right now the user types start
   and end seconds in the intervention table. Plotly `editable: shape` mode
   would let them drag the band edges directly on the plot.

5. **Right-click → "Add intervention here"** on the plot. Currently manual
   interventions are added via a `<details>` form in the intervention table.
   A click-on-plot UX would feel a lot more direct, especially for FX05-style
   mid-file shifts.

6. **PDF preview pane** in the UI before the user hits Render. The plan
   mentions this as `ReportPreview.tsx` but it was never built.

### Low priority / nice-to-have
7. **Per-marker contribution sparkline** (small inset under the main chart)
   showing each active seed's individual deviation. Useful for spotting one
   seed drifting away from the others — could replace the seed-toggle UX
   eventually.

8. **CLI parametrisation** of the abstract-figures scripts (so the
   reporter and the abstract-figures pipeline can share a config). The
   abstract-figures scripts have hardcoded paths with `# TODO: parametrise via
   CLI` comments.

9. **`Robot trace overlay`** (cross-validation against the ground truth used
   by `python_app/kim_analysis_logic.process_interrupt_data`). Out of scope for
   the first reporter release, but a natural extension.

10. **Cross-platform bundles (macOS, Linux).** Currently Windows-only because
    pywebview's Edge WebView2 backend is the path of least resistance. WebKit
    and Qt backends exist if there's demand.

## File map (reference)

```
kim-reporter/
├── pyproject.toml
├── README.md
├── SESSION_HANDOFF.md   (this file)
├── KIM-QA-Reporter.spec
├── kim_app/
│   ├── __init__.py
│   ├── __main__.py
│   ├── server.py
│   ├── core/
│   │   ├── loader.py    ← dynamic-marker centroid loader
│   │   ├── shifts.py
│   │   ├── window.py
│   │   └── overlay.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schema.py
│   ├── pdf/
│   │   └── report.py
│   └── web_dist/        ← Vite build output (gitignored)
└── web/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── tsconfig.json
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css
        ├── components/
        │   ├── PatientPicker.tsx
        │   ├── FractionTabs.tsx
        │   ├── SeedToggleBar.tsx
        │   ├── DeviationPlot.tsx
        │   ├── InterventionTable.tsx
        │   ├── PhysicistNotes.tsx
        │   └── ReportButton.tsx
        ├── lib/
        │   ├── api.ts
        │   ├── store.ts
        │   └── plotlyPresets.ts
        └── types/
            └── plotly-dist-min.d.ts
```
