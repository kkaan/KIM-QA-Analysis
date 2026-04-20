"""Compose the 4 PAT01 per-intervention figures into a single 2x2 grid for the
EPSM abstract submission. Each quadrant shows one intra-fraction couch
correction across PAT01's treatment course (FX01, FX04 x2, FX05).

The composite re-renders each panel into a temp directory with the per-panel
suptitle and the correction-magnitude textbox suppressed (the same information
is captured in the composite's suptitle and the published caption). The
standalone per-intervention PNGs in `pat01-traces-interventions/` are NOT
modified by this script.

Run from the repo root:
    python abstract-figures/make_pat01_intervention_grid.py
"""

import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Allow this script to import its sibling abstract-figures modules regardless
# of the cwd it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_prime_stepjump_figure import make_figure  # noqa: E402
from make_pat01_intervention_figures import build_fraction_configs, FRACTIONS  # noqa: E402


# TODO: parametrise via CLI when re-using for other patients.
OUTPUT_PNG = Path(
    r"C:\Users\kankean.kandasamy\OneDrive - GenesisCare\Documents\phd-prime"
    r"\prime-epsm-abstract\pat01_all_interventions_2x2.png"
)

# Quadrant labels in chronological order: top-left -> top-right -> bottom-left
# -> bottom-right. Filenames must match what `build_fraction_configs` would
# emit so the iteration order verifies (a)/(b)/(c)/(d) -> the correct panels.
QUADRANT_LABELS = ["(a)", "(b)", "(c)", "(d)"]
EXPECTED_PANEL_BASENAMES = [
    "pat01_fx01_intervention_1.png",
    "pat01_fx04_intervention_1.png",
    "pat01_fx04_intervention_2.png",
    "pat01_fx05_intervention_1.png",
]


def _render_chrome_stripped_panels(tmp_dir: Path) -> list[Path]:
    """Render the four per-intervention figures into `tmp_dir` with the
    per-panel suptitle and correction textbox suppressed. Returns the list of
    PNG paths in the canonical (a)/(b)/(c)/(d) order.
    """
    panel_paths: list[Path] = []
    for fraction in FRACTIONS:
        for cfg in build_fraction_configs(fraction):
            cfg = dict(cfg)  # don't mutate the source
            basename = Path(cfg["output_png"]).name
            cfg["output_png"] = tmp_dir / basename
            cfg["show_suptitle"] = False
            # show_correction_textbox is already False in build_fraction_configs,
            # but set it here too as a defensive belt-and-braces.
            cfg["show_correction_textbox"] = False
            make_figure(cfg)
            panel_paths.append(cfg["output_png"])

    if len(panel_paths) != 4:
        raise SystemExit(
            f"Expected 4 panel PNGs but got {len(panel_paths)}: "
            f"{[p.name for p in panel_paths]}"
        )
    actual_basenames = [p.name for p in panel_paths]
    if actual_basenames != EXPECTED_PANEL_BASENAMES:
        raise SystemExit(
            "Panel iteration order does not match expected (a)-(d) layout.\n"
            f"  Expected: {EXPECTED_PANEL_BASENAMES}\n"
            f"  Got:      {actual_basenames}"
        )
    return panel_paths


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        panel_paths = _render_chrome_stripped_panels(tmp_dir)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for ax, label, path in zip(axes.flat, QUADRANT_LABELS, panel_paths):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_axis_off()
            # Quadrant label in upper-left of each panel
            ax.text(
                0.01, 0.99, label,
                transform=ax.transAxes,
                fontsize=22,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc="white",
                    ec="dimgrey",
                    lw=0.8,
                    alpha=0.92,
                ),
            )

        fig.suptitle(
            "All KIM-guided intra-fraction couch corrections - PRIME trial first patient",
            fontsize=18,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
