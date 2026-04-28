"""FastAPI app factory.

Mounts the API routes plus the Vite-built static frontend (when available). In
development we proxy/redirect to the Vite dev server (``npm run dev``); in a
PyInstaller build we serve the bundled ``web_dist/`` directory directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from kim_app import __version__
from kim_app.api.routes import router as api_router


def _web_dist_dir() -> Path | None:
    """Locate the built React bundle. Order of attempts:

    1. ``kim_app/web_dist/`` next to this module (production-package layout).
    2. ``kim-reporter/kim_app/web_dist/`` relative to repo root (when running
       ``python -m kim_app`` from a checkout).
    3. ``sys._MEIPASS/kim_app/web_dist/`` for PyInstaller frozen apps.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "web_dist",
    ]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "kim_app" / "web_dist")  # type: ignore[attr-defined]
    for c in candidates:
        if c.is_dir() and (c / "index.html").is_file():
            return c
    return None


def create_app() -> FastAPI:
    app = FastAPI(
        title="KIM-QA Reporter",
        version=__version__,
        description="Desktop reporter for KIM-guided couch corrections",
    )

    # In development we hot-reload the React bundle from Vite's dev server,
    # which means the frontend origin (http://localhost:5173) is different from
    # the API origin (http://127.0.0.1:<port>). Allow that explicitly.
    dev_frontend = os.environ.get("KIM_REPORTER_DEV_FRONTEND")
    allowed_origins = ["*"] if dev_frontend else []
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router)

    # Mount static assets if available. In dev, the user points pywebview at
    # the Vite server directly, so this branch is skipped.
    web_dist = _web_dist_dir()
    if web_dist is not None:
        app.mount(
            "/assets",
            StaticFiles(directory=web_dist / "assets"),
            name="assets",
        )

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(web_dist / "index.html")

        # SPA fallback: any non-API path falls back to index.html so client-side
        # routing keeps working on hard reloads.
        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            target = web_dist / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(web_dist / "index.html")

    else:
        # No bundle present — redirect ``/`` to the dev frontend if configured,
        # otherwise emit a small explanatory page.
        @app.get("/")
        def root_redirect():
            if dev_frontend:
                return RedirectResponse(dev_frontend)
            return {
                "ok": False,
                "error": (
                    "Frontend bundle not built. Run `cd web && npm run build` "
                    "or set KIM_REPORTER_DEV_FRONTEND to a Vite dev URL."
                ),
            }

    return app
