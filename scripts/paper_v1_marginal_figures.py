#!/usr/bin/env python3
"""Figures for the marginal value of each seed (paper v1 follow-up).

Reads results/paper_v1/marginal/ (scripts/paper_v1_marginal_eval.py) and writes
fig_marginal (first-seed confidence tiers), fig_marginal_novelty (RNP novelty
bins) and fig_cost (Chai-1 quality against estimated inference time) to results/paper_v1/figures.
Run with python.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import paper_v1_style as st

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 0.005
SEEDS = [2, 3, 4, 5]
CMAP = mcolors.LinearSegmentedColormap.from_list("gain", [st.VERMILION, "#fbfaf8", st.TEAL])
CONF_TIERS = [("all", "all", "All systems"), ("confidence", "low", "Low"), ("confidence", "mid", "Medium"),
              ("confidence", "high", "High")]
NOVELTY_TIERS = [("novelty", "[0,30)", "$<$30"), ("novelty", "[30,50)", "30–50"), ("novelty", "[50,70)", "50–70"),
                 ("novelty", "[70,100]", "$\\geq$70")]
ROWS = [("returned", "Returned pose"), ("oracle", "Best available pose")]


def heat(ax, gains: pd.DataFrame, kstar: dict, norm, show_y: bool) -> None:
    data = gains.reindex(index=st.ORDER, columns=SEEDS).to_numpy()
    ax.imshow(data, cmap=CMAP, norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(data):
        if np.isfinite(v):
            dark = abs(norm(v) - 0.5) > 0.36
            ax.text(c, r, f"{v * 1000:.1f}", ha="center", va="center", fontsize=6.2,
                    fontweight="bold" if v >= PRIMARY else "normal", color="white" if dark else st.INK)
    for r, method in enumerate(st.ORDER):
        k, reached = kstar.get(method, (np.nan, True))
        text = "–" if not np.isfinite(k) else (f"{int(k)}" if reached else "≥5")
        ax.text(3.95, r, text, ha="center", va="center", fontsize=6.6, color=st.INK, clip_on=False)
    ax.text(3.95, -0.85, "$k^*$", ha="center", va="center", fontsize=7, color=st.INK2, clip_on=False)
    ax.set_xticks(range(len(SEEDS)), [str(k) for k in SEEDS])
    ax.set_yticks(range(len(st.ORDER)), [st.LABELS[m] for m in st.ORDER] if show_y else [])
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def marginal_figure(marginal, recommended, tiers, header, name, out) -> None:
    inc = marginal[(marginal["kind"] == "increment")]
    rec = recommended[recommended["threshold"] == PRIMARY]
    vmax = np.ceil(inc[inc["outcome"].isin(["returned", "oracle"])]["mean"].abs().max() * 200) / 200
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(2, len(tiers), figsize=(7.0, 4.35), gridspec_kw={"wspace": 0.36, "hspace": 0.42})
    fig.subplots_adjust(left=0.17, right=0.86, top=0.86, bottom=0.1)
    for r, (outcome, row_label) in enumerate(ROWS):
        for c, (stratifier, tier, title) in enumerate(tiers):
            ax = axes[r, c]
            sel = inc[(inc["outcome"] == outcome) & (inc["stratifier"] == stratifier) & (inc["tier"] == tier)]
            gains = sel.pivot(index="method", columns="k", values="mean")
            kr = rec[(rec["outcome"] == outcome) & (rec["stratifier"] == stratifier) & (rec["tier"] == tier)]
            kstar = {m: (k, ok) for m, k, ok in zip(kr["method"], kr["k_star"], kr["reached"])}
            heat(ax, gains, kstar, norm, show_y=c == 0)
            if r == 0:
                ax.set_title(title, pad=12)
            if r == 1:
                ax.set_xlabel("Seed $k$")
        axes[r, 0].set_ylabel(row_label, labelpad=6)
        fig.text(0.01, axes[r, 0].get_position().y1 + 0.03, "ab"[r], fontsize=10, fontweight="bold")
    x0 = axes[0, 1 if tiers[0][0] == "all" else 0].get_position().x0
    x1 = axes[0, -1].get_position().x1
    fig.text((x0 + x1) / 2, 0.955, header, ha="center", va="center", fontsize=8.5)
    cax = fig.add_axes([0.905, 0.2, 0.014, 0.6])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    bar.formatter = FuncFormatter(lambda v, _: f"{v * 1000:g}")
    bar.update_ticks()
    bar.outline.set_visible(False)
    bar.set_label("Gain from seed $k$ ($10^{-3}$ LDDT-PLI)", fontsize=7.5)
    bar.ax.tick_params(length=2, labelsize=7)
    st.save(fig, out, name)


def cost_figure(cost, out) -> None:
    pts = cost[cost["point"].str.startswith("k=")]
    rule = cost[cost["point"] == "confidence rule"]
    span = max((g[["ret_q", "orc_q"]].max().max() - g[["ret_q", "orc_q"]].min().min())
               for _, g in pts.groupby("tier")) + 0.03
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.45), sharex=True, gridspec_kw={"wspace": 0.42})
    fig.subplots_adjust(left=0.08, right=0.985, top=0.74, bottom=0.2)
    for ax, (stratifier, tier, title) in zip(axes, CONF_TIERS):
        g = pts[pts["tier"] == tier].sort_values("seeds")
        for col, color in (("ret_q", st.NAVY), ("orc_q", st.TEAL)):
            ax.plot(g["minutes"], g[col], color=color, marker="o", markersize=3.6, zorder=3)
        rg = rule[rule["tier"] == tier]
        ax.scatter(rg["minutes"], rg["ret_q"], marker="D", s=26, color=st.VERMILION, zorder=5)
        low = g[["ret_q", "orc_q"]].min().min() - 0.012
        ax.set_ylim(low, low + span)
        ax.set_title(title if stratifier == "all" else f"{title} confidence", fontsize=8.5)
        n = int(g["n_systems"].iloc[0])
        ax.text(0.04, 0.97, f"$n$ = {n:,}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=st.INK2)
    axes[0].set_ylabel("Mean LDDT-PLI")
    fig.text(0.535, 0.03, "Estimated inference time per complex (min)", ha="center", va="bottom", fontsize=8)
    handles = [Line2D([], [], color=st.NAVY, marker="o", markersize=3.6, label="Returned (native top-1)"),
               Line2D([], [], color=st.TEAL, marker="o", markersize=3.6, label="Best available (oracle)"),
               Line2D([], [], color=st.VERMILION, marker="D", markersize=4.5, linestyle="none",
                      label="Confidence rule")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.995), handletextpad=0.5,
               columnspacing=1.4)
    for i, ax in enumerate(axes):
        fig.text(ax.get_position().x0 - 0.06, 0.8, "abcd"[i], fontsize=10, fontweight="bold")
    st.save(fig, out, "fig_cost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/paper_v1/marginal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/figures")
    args = parser.parse_args()
    st.apply()
    marginal = pd.read_csv(args.input_dir / "marginal_gains.csv")
    recommended = pd.read_csv(args.input_dir / "recommended_seeds.csv")
    cost = pd.read_csv(args.input_dir / "chai_cost.csv")
    marginal_figure(marginal, recommended, CONF_TIERS, "First-seed confidence", "fig_marginal", args.output_dir)
    marginal_figure(marginal, recommended, NOVELTY_TIERS, "Similarity to the training data (RNP novelty score)",
                    "fig_marginal_novelty", args.output_dir)
    cost_figure(cost, args.output_dir)


if __name__ == "__main__":
    main()
