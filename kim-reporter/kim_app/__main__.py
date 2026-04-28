"""Entrypoint: launch uvicorn in a daemon thread, then open a pywebview window.

Both run in the same Python process so there's no IPC, no sidecar lifecycle
to manage, and a clean shutdown when the user closes the window. In dev mode
(``KIM_REPORTER_DEV_FRONTEND`` set) pywebview points at the Vite dev server so
frontend edits hot-reload while the FastAPI backend stays in-process for API
calls.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from contextlib import closing

import uvicorn
import webview

from kim_app import __version__


def _free_port(default: int = 8765) -> int:
    """Try ``default`` first, then ask the OS for any free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            pass
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_ready(host: str, port: int, timeout_s: float = 8.0) -> bool:
    """Poll the server's TCP port until it accepts connections or we time out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.25)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def main() -> None:
    # Lazy import so PyInstaller can pick up the FastAPI app via the kim_app
    # package without crashing if uvicorn isn't installed during pure unit tests.
    from kim_app.server import create_app

    host = "127.0.0.1"
    port = _free_port()
    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("KIM_REPORTER_LOG_LEVEL", "warning"),
        access_log=False,
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    server_thread.start()

    if not _wait_for_ready(host, port):
        print(f"Backend failed to start on {host}:{port}", file=sys.stderr)
        sys.exit(1)

    dev_frontend = os.environ.get("KIM_REPORTER_DEV_FRONTEND")
    backend_url = f"http://{host}:{port}"
    if dev_frontend:
        # In dev, the React app makes API calls to the backend via this env var.
        url = f"{dev_frontend.rstrip('/')}/?api={backend_url}"
    else:
        url = backend_url

    window = webview.create_window(
        title=f"KIM-QA Reporter v{__version__}",
        url=url,
        width=1320,
        height=860,
        min_size=(1024, 720),
        confirm_close=False,
    )

    try:
        webview.start(debug=bool(os.environ.get("KIM_REPORTER_DEBUG")))
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
