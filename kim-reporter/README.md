# KIM-QA Reporter

Desktop application for medical physicists to inspect KIM-guided couch corrections
and export clinical PDF reports for patient documentation.

The app reads a patient directory containing the centroid (seed/iso) file plus
per-fraction trajectory logs (`MarkerLocationsGA_CouchShift_*.txt`) and
`couchShifts.txt`, renders an interactive deviation plot with pan/zoom, and
exports an A4 PDF with the trajectory plots, intervention summary table, and a
free-text physicist-notes section.

## Architecture

- **Backend** (`kim_app/`): Python + FastAPI. Reuses parsers from
  `python_app/kim_analysis_logic.py` and the abstract-figures helpers, with a
  new dynamic-marker loader (`kim_app.core.loader`) that handles 1–N markers and
  per-marker deselection.
- **Frontend** (`web/`): React 18 + TypeScript + Vite + Tailwind +
  react-plotly.js. Built once with Vite into `kim_app/web_dist/` and shipped as
  static assets.
- **Shell**: pywebview (Edge WebView2 on Windows) hosts the React app pointing
  at the in-process FastAPI server.
- **PDF**: ReportLab with matplotlib-rendered figures embedded as PNGs.
- **Packaging**: PyInstaller `--onefile` produces `KIM-QA-Reporter.exe`.

## Development

### One-time setup

```powershell
# From repo root
cd kim-reporter
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cd web
npm install
```

> **Windows note**: npm 10.x has a [known bug](https://github.com/npm/cli/issues/4828)
> where it skips Rollup's platform-specific native binary on Windows. If
> `npm run build` fails with `Cannot find module @rollup/rollup-win32-x64-msvc`,
> run:
>
> ```powershell
> npm install --no-save @rollup/rollup-win32-x64-msvc
> ```

### Run in development

In one terminal:
```powershell
cd kim-reporter\web
npm run dev   # Vite dev server with HMR on http://localhost:5173
```

In another terminal:
```powershell
cd kim-reporter
$env:KIM_REPORTER_DEV_FRONTEND = "http://localhost:5173"
python -m kim_app
```

This launches uvicorn + pywebview pointing at the Vite dev server so frontend
edits hot-reload.

### Production build

```powershell
cd kim-reporter\web
npm run build      # outputs to kim_app/web_dist/

cd ..
pyinstaller KIM-QA-Reporter.spec
# Output: dist/KIM-QA-Reporter.exe
```

## Sample data

The repo ships a sample patient at
`Sample Patient Trajectory Logs/PAT01/` (PAT01 is a 2-marker case where one
implanted seed was removed from tracking). The reporter auto-detects the
centroid file at the parent directory.

## Project layout

```
kim-reporter/
├── pyproject.toml
├── KIM-QA-Reporter.spec
├── kim_app/
│   ├── __main__.py             # entrypoint: uvicorn thread + pywebview window
│   ├── server.py               # FastAPI app factory
│   ├── core/
│   │   ├── loader.py           # dynamic-marker centroid loader
│   │   ├── shifts.py           # couchShifts.txt parsing
│   │   ├── window.py           # gap compression
│   │   └── overlay.py          # no-correction counterfactual
│   ├── api/
│   │   ├── schema.py           # pydantic models
│   │   └── routes.py           # /api/scan /api/fraction /api/render-pdf
│   ├── pdf/
│   │   └── report.py           # ReportLab clinical template
│   └── web_dist/               # populated by `npm run build` (git-ignored)
└── web/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    └── src/                    # React app
```

## See also

- `abstract-figures/` — batch publication-figure pipeline (untouched).
- `python_app/` — original CustomTkinter analysis GUI (untouched).
