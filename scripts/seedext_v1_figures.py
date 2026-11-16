#!/usr/bin/env python3
"""Figure for E24 (Chai-1 with 20 seeds per system), written before any E24 result.

Reads results/seedext_v1/curves.csv and kstar.csv (scripts/seedext_v1_eval.py) and
writes fig_seedext to results/paper_v1/figures: (a-d) mean LDDT-PLI of the returned and
the best available pose against the number of seeds, over the pool (tiers weighted
equally) and per design confidence tier, with 95% cluster-bootstrap intervals (kept in the
figures when the paper moved to standard errors in tables, deviation D5); (e-h) the
gain from each seed with the 0.005 threshold and the marginal-gain threshold budget.
Run with python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paper_v1_style as st

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 0.005
TIERS = [("pool", "All systems"), ("low", "Low confidence"), ("mid", "Medium confidence"), ("high", "High confidence")]
SERIES = [("ret_q", st.NAVY, "Returned (native top-1)"), ("orc_q", st.TEAL, "Best available (oracle)")]


def plain(x: float) -> str:
    return "0.000" if abs(x) < 5e-4 else f"{x:.3f}"


def signed(x: float) -> str:
    return "0.000" if abs(x) < 5e-4 else f"{x:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/seedext_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/figures")
    args = parser.parse_args()
    st.apply()
    curves = pd.read_csv(args.input_dir / "curves.csv")
    kstar = pd.read_csv(args.input_dir / "kstar.csv")
    kstar = kstar[(kstar["scheme"] == "design") & (kstar["threshold"] == PRIMARY)]
    summary = json.loads((args.input_dir / "summary.json").read_text())
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.6), sharex=True,
                             gridspec_kw={"wspace": 0.42, "hspace": 0.32, "height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.11)
    levels = curves[curves["kind"] == "level"]
    gains = curves[curves["kind"] == "gain"]
    means = levels[levels["outcome"].isin(["ret_q", "orc_q"])]
    span = max(g["mean"].max() - g["mean"].min() for _, g in means.groupby("tier")) + 0.12
    gmax = gains[gains["outcome"].isin(["ret_q", "orc_q"])]["mean"].max()
    gmin = min(gains[gains["outcome"].isin(["ret_q", "orc_q"])]["mean"].min(), -0.002)
    for c, (tier, title) in enumerate(TIERS):
        top, bottom = axes[0, c], axes[1, c]
        lv = levels[levels["tier"] == tier]
        for outcome, color, _ in SERIES:
            g = lv[lv["outcome"] == outcome].sort_values("k")
            top.plot(g["k"], g["mean"], color=color, linewidth=1.5)
            gg = gains[(gains["tier"] == tier) & (gains["outcome"] == outcome)].sort_values("k")
            bottom.fill_between(gg["k"], gg["lo"], gg["hi"], color=color, alpha=0.15, linewidth=0)
            bottom.plot(gg["k"], gg["mean"], color=color, linewidth=1.3, marker="o", markersize=2.2)
            ks = kstar[(kstar["tier"] == tier) & (kstar["outcome"] == outcome)]
            if len(ks) and bool(ks["reached"].iloc[0]):
                k = int(ks["k_star"].iloc[0])
                y = lv[(lv["outcome"] == outcome) & (lv["k"] == k)]["mean"].iloc[0]
                top.scatter([k], [y], s=46, facecolors="none", edgecolors=color, linewidths=1.1, zorder=4)
        lo = lv[lv["outcome"].isin(["ret_q", "orc_q"])]["mean"].min() - 0.02
        hi = min(lo + span, 1.0)
        top.set_ylim(hi - span, hi)
        box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4}
        for i, (outcome, color, name) in enumerate([("orc_q", st.TEAL, "best"), ("ret_q", st.NAVY, "returned")]):
            d = summary[f"delta_5_20_{outcome}"][tier][:3]
            text = f"{name} {signed(d[0])} [{plain(d[1])}, {plain(d[2])}]".replace("-", "\u2212")
            top.text(0.04, 0.87 - 0.1 * i, text, transform=top.transAxes, ha="left", va="top", fontsize=6.0,
                     color=color, bbox=box)
        top.text(0.04, 0.97, "$\\Delta_{5\\to20}$ [95% CI]", transform=top.transAxes, ha="left", va="top", fontsize=6.0,
                 color=st.INK2, bbox=box)
        top.axvline(5, color=st.MUTED, linestyle=":", linewidth=0.9)
        top.set_title(f"{title}\n$n$ = {int(lv['n'].iloc[0])}", fontsize=8.5, linespacing=1.3)
        bottom.axhline(PRIMARY, color=st.INK2, linestyle=":", linewidth=0.9)
        bottom.axhline(0, color=st.GRID, linewidth=0.8)
        bottom.set_ylim(gmin, gmax * 1.08)
        bottom.set_xticks([1, 5, 10, 15, 20])
        bottom.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 1000:g}"))
    axes[0, 0].set_ylabel("Mean LDDT-PLI")
    axes[1, 0].set_ylabel("Gain from seed $k$\n($10^{-3}$ LDDT-PLI)")
    fig.text(0.535, 0.02, "Number of seeds $k$", ha="center", va="bottom", fontsize=8)
    handles = [Line2D([], [], color=color, linewidth=1.5, label=label) for _, color, label in SERIES]
    handles += [Line2D([], [], color=st.INK2, marker="o", markersize=6.5, markerfacecolor="none", linestyle="none",
                       label="Marginal-gain threshold budget $k^*$"),
                Line2D([], [], color=st.MUTED, linestyle=":", linewidth=0.9, label="Five seeds (benchmark)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.995), handletextpad=0.5,
               columnspacing=1.2)
    for r in range(2):
        for c in range(4):
            fig.text(axes[r, c].get_position().x0 - 0.06, axes[r, c].get_position().y1 + 0.012, "abcdefgh"[4 * r + c],
                     fontsize=10, fontweight="bold")
    st.save(fig, args.output_dir, "fig_seedext")


if __name__ == "__main__":
    main()
