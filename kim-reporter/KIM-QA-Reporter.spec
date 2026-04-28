# PyInstaller spec for KIM-QA Reporter.
#
# Run from the kim-reporter/ directory:
#     pyinstaller KIM-QA-Reporter.spec
#
# Prerequisites:
#   - The web bundle has been built (`cd web && npm run build`), so
#     kim_app/web_dist/index.html exists.
#   - Edge WebView2 runtime is installed on the target machine (preinstalled
#     on Windows 10 21H2+ / Windows 11).
#
# Output: dist/KIM-QA-Reporter.exe (single-file build).

# pylint: disable=missing-module-docstring
# noqa: E501,F821 — `Analysis`, `EXE` etc are PyInstaller globals.

from pathlib import Path

import matplotlib

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

ROOT = Path(SPECPATH).resolve()
WEB_DIST = ROOT / "kim_app" / "web_dist"
PYTHON_APP = ROOT.parent / "python_app"

# matplotlib needs its data dir bundled or font lookups fail at runtime.
mpl_data = matplotlib.get_data_path()

datas = [
    (str(WEB_DIST), "kim_app/web_dist"),
    (mpl_data, "matplotlib/mpl-data"),
    # Include kim_analysis_logic.py as a data file alongside the package so
    # routes.py's sys.path.insert finds it inside the frozen bundle.
    (str(PYTHON_APP / "kim_analysis_logic.py"), "python_app"),
]

# reportlab ships its own data dir (fonts) that PyInstaller usually finds via
# its hooks; collect_data_files keeps us safe if reportlab's hook misses any.
datas += collect_data_files("reportlab")

hidden_imports = (
    collect_submodules("uvicorn")
    + collect_submodules("pydantic")
    + collect_submodules("matplotlib.backends")
    + ["uvicorn.logging", "uvicorn.protocols.websockets.auto"]
)

a = Analysis(
    ["kim_app/__main__.py"],
    pathex=[str(ROOT), str(PYTHON_APP)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "kaleido"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="KIM-QA-Reporter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
