"""Compose the 4 PAT01 per-intervention figures into a single 2x2 grid for the
EPSM abstract submission. Each quadrant shows one intra-fraction couch
correction across PAT01's treatment course (FX01, FX04 x2, FX05).

Run from the repo root:
    python abstract-figures/make_pat01_intervention_grid.py
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# TODO: parametrise via CLI when re-using for other patients.
INTERVENTION_DIR = Path(
    r"C:\Users\kankean.kandasamy\OneDrive - GenesisCare\Documents\phd-prime"
    r"\prime-epsm-abstract\pat01-traces-interventions"
)
OUTPUT_PNG = Path(
    r"C:\Users\kankean.kandasamy\OneDrive - GenesisCare\Documents\phd-prime"
    r"\prime-epsm-abstract\pat01_all_interventions_2x2.png"
)

# (label, filename) in chronological order: top-left -> top-right -> bottom-left -> bottom-right.
PANELS = [
    ("(a)", "pat01_fx01_intervention_1.png"),
    ("(b)", "pat01_fx04_intervention_1.png"),
    ("(c)", "pat01_fx04_intervention_2.png"),
    ("(d)", "pat01_fx05_intervention_1.png"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, (label, filename) in zip(axes.flat, PANELS):
        path = INTERVENTION_DIR / filename
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
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
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="dimgrey", lw=0.8, alpha=0.92),
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
