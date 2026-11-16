"""Shared figure style for paper v1: seaborn theme with the validated palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

ORDER = ["af3", "af3_no_template", "boltz", "boltz1x", "boltz2", "chai", "protenix"]
LABELS = {"af3": "AF3", "af3_no_template": "AF3 (no templates)", "boltz": "Boltz-1", "boltz1x": "Boltz-1x",
          "boltz2": "Boltz-2", "chai": "Chai-1", "protenix": "Protenix"}
# Palette after the Nature (NPG) journal palette, with the navy deepened to pass the
# chroma floor; navy, teal and vermillion validate all-pairs (worst CVD dE 11.6).
# Navy: what the model returns or decides; teal: best available and improvements;
# vermillion: deteriorations and highlighted operating points; greys: baselines.
NAVY, TEAL, VERMILION, LAVENDER = "#34539f", "#00a087", "#e64b35", "#8491b4"
INK, INK2, GRID, MUTED, PANEL = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d", "#f6f6f4"


def apply() -> None:
    sns.set_theme(context="paper", style="ticks", rc={
        "font.family": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42,
        "legend.frameon": False,
    })


def panel_label(ax, letter: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
