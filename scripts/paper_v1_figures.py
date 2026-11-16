#!/usr/bin/env python3
"""Paper v1 result figures, drawn with seaborn from results/paper_v1/tables.

Run with the plotting environment in environments/paperfig.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_v1_style as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = [st.LABELS[m] for m in st.ORDER]
Y = np.arange(len(st.ORDER))[::-1]


def method_axis(ax, labels: bool = True) -> None:
    ax.set_yticks(Y)
    ax.set_yticklabels(NAMES if labels else [])
    ax.set_ylim(-0.6, len(st.ORDER) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def shift_right(ax, dx: float = 0.075) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0, pos.width - dx, pos.height])


def fig_budget(t: Path, out: Path) -> None:
    g = pd.read_csv(t / "budget_gains.csv")
    g = g[(g["scope"] == "all") & (g["from_k"] == 1)]
    gap = pd.read_csv(t / "budget_curves.csv")
    gap = gap[(gap["split"] == "all") & (gap["k"] == 5)].set_index("method").loc[st.ORDER]
    n = pd.read_csv(t / "coverage.csv").set_index("method")["complete_systems"]
    sel = g[g["quantity"].isin(["ret_q", "orc_q"])]
    lo, hi = min(0.0, sel["lo"].min()), sel["hi"].max()
    pad = 0.08 * (hi - lo)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), gridspec_kw=dict(hspace=0.62, wspace=0.28))
    for i, (ax, m) in enumerate(zip(axes.flat, st.ORDER)):
        for qty, color, marker, label in (("orc_q", st.TEAL, "s", "Best available (oracle)"),
                                          ("ret_q", st.NAVY, "o", "Returned (native top-1)")):
            d = g[(g["method"] == m) & (g["quantity"] == qty)].sort_values("to_k")
            k = np.r_[1, d["to_k"].to_numpy()]
            ax.fill_between(k, np.r_[0, d["lo"]], np.r_[0, d["hi"]], color=color, alpha=0.18, linewidth=0)
            ax.plot(k, np.r_[0, d["mean"]], color=color, marker=marker, markeredgecolor="white",
                    markeredgewidth=0.6, label=label)
        ax.axhline(0, color=st.INK2, linewidth=0.7, zorder=1)
        ax.set_title(f"{st.LABELS[m]}  ($n$ = {int(n[m]):,})", fontsize=8)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(lo - pad, hi + pad)
        if i % 4:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("$\\Delta$ mean LDDT-PLI")
        if i >= 3:
            ax.set_xlabel("Seeds generated")
    handles, labels = axes.flat[0].get_legend_handles_labels()  # returned first, as in Figures 4 and 8
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.2)

    ax = axes.flat[7]
    ax.barh(Y, gap["gap_q"], color=st.LAVENDER, height=0.62)
    for y, v in zip(Y, gap["gap_q"]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=7, color=st.INK2)
    method_axis(ax)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, gap["gap_q"].max() * 1.6)
    ax.set_xlabel("LDDT-PLI")
    ax.set_title("Gap at five seeds", fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.86, bottom=0.12)
    shift_right(axes.flat[7])
    st.save(fig, out, "fig_budget")


def fig_selection(t: Path, out: Path) -> None:
    h = pd.read_csv(t / "robustness_headroom.csv").set_index("method").loc[st.ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.12, width_ratios=[1.15, 1]))

    ax = axes[0]
    ax.hlines(Y, h["near_best_random"] * 100, h["near_best_native"] * 100, color=st.GRID, linewidth=2.4, zorder=1)
    ax.scatter(h["near_best_random"] * 100, Y, marker="D", s=26, color=st.MUTED, edgecolor="white", linewidth=0.6,
               zorder=3, label="Random candidate")
    ax.scatter(h["near_best_native"] * 100, Y, marker="o", s=30, color=st.NAVY, edgecolor="white", linewidth=0.6,
               zorder=3, label="Native top-1")
    method_axis(ax)
    ax.set_xlim(20, 45)
    ax.set_xlabel("Near-best picks (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0,
              borderaxespad=0.1)

    ax = axes[1]
    err = np.vstack([h["headroom_share"] - h["headroom_lo"], h["headroom_hi"] - h["headroom_share"]]) * 100
    ax.barh(Y, h["headroom_share"] * 100, xerr=err, color=st.NAVY, alpha=0.85, height=0.62,
            error_kw=dict(ecolor=st.INK2, lw=0.8, capsize=0))
    ax.axvline(0, color=st.INK2, linewidth=0.7)
    method_axis(ax, labels=False)
    ax.set_xlim(min(-5, (h["headroom_lo"].min() * 100) - 2), max(40, h["headroom_hi"].max() * 100 + 3))
    ax.set_xlabel("Recovered headroom (%)")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.86, bottom=0.22)
    for letter, ax_ in zip("ab", axes):
        x0 = 0.008 if letter == "a" else ax_.get_position().x0 - 0.012
        fig.text(x0, 0.95, letter, fontsize=10, fontweight="bold", va="bottom")
    st.save(fig, out, "fig_selection")


def fig_predict(t: Path, out: Path) -> None:
    """Predictability of the next seed's value: pooled AUC (formerly Figure 5c) and AUC within each stage."""
    p = pd.read_csv(t / "predictability.csv")
    prior = p[p["model"] == "stage_prior"].set_index("method").loc[st.ORDER]
    lr = p[p["model"] == "logistic_all_features"].set_index("method").loc[st.ORDER]
    r = pd.read_csv(t / "robustness_predict.csv")
    r = r[r["stage"] != "pooled"].astype({"stage": int})
    within = r.pivot(index="method", columns="stage", values="auc_logistic").loc[st.ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.12, width_ratios=[1.15, 1]))

    ax = axes[0]
    ax.hlines(Y, prior["test_auc"], lr["test_auc"], color=st.GRID, linewidth=2.4, zorder=1)
    ax.scatter(prior["test_auc"], Y, marker="D", s=26, color=st.MUTED, edgecolor="white", linewidth=0.6, zorder=3,
               label="Stage prior")
    ax.scatter(lr["test_auc"], Y, marker="o", s=30, color=st.NAVY, edgecolor="white", linewidth=0.6, zorder=3,
               label="Logistic model")
    ax.axvline(0.5, color=st.INK2, linestyle=":", linewidth=0.9)
    method_axis(ax)
    ax.set_xlim(0.45, 0.72)
    ax.set_xlabel("Test AUC")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0,
              borderaxespad=0.1)

    ax = axes[1]
    norm = mcolors.TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.65)
    cmap = mcolors.LinearSegmentedColormap.from_list("auc", [st.VERMILION, "#fbfaf8", st.TEAL])
    data = within.to_numpy()
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=st.INK)
    ax.set_xticks(range(data.shape[1]), [str(k) for k in within.columns])
    ax.set_yticks(range(len(st.ORDER)), [])
    ax.set_ylim(len(st.ORDER) - 0.4, -0.6)
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Seeds generated")
    ax.set_title("Logistic model within each stage", fontsize=8)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.86, bottom=0.22)
    for letter, ax_ in zip("ab", axes):
        x0 = 0.008 if letter == "a" else ax_.get_position().x0 - 0.012
        fig.text(x0, 0.95, letter, fontsize=10, fontweight="bold", va="bottom")
    st.save(fig, out, "fig_predict")


def fig_changes(t: Path, out: Path) -> None:
    c = pd.read_csv(t / "next_batch_changes.csv")
    c = c[c["quantity"] == "returned"]
    xmax = 5 * np.ceil(max(c["frac_improve"].max(), c["frac_worsen"].max()) * 100 / 5 + 0.6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.08))
    for i, (ax, tr) in enumerate(zip(axes, ("1->2", "4->5"))):
        d = c[c["transition"] == tr].set_index("method").loc[st.ORDER]
        up, down = d["frac_improve"] * 100, d["frac_worsen"] * 100
        ax.barh(Y, up, color=st.TEAL, height=0.62, label="Improves by$>$0.01")
        ax.barh(Y, -down, color=st.VERMILION, height=0.62, label="Worsens by$>$0.01")
        for y, u, w in zip(Y, up, down):
            ax.text(u + 0.4, y, f"{u:.1f}", va="center", ha="left", fontsize=6.8, color=st.INK2)
            ax.text(-w - 0.4, y, f"{w:.1f}", va="center", ha="right", fontsize=6.8, color=st.INK2)
        ax.axvline(0, color=st.INK2, linewidth=0.8)
        ax.set_xlim(-xmax, xmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
        method_axis(ax, labels=(i == 0))
        a, b = tr.split("->")
        ax.set_title(f"Seed {a}$\\to${b}", fontsize=8.5)
        ax.set_xlabel("System–ordering pairs (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.56, 1.0), handlelength=1.4)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.8, bottom=0.18)
    for letter, ax_ in zip("ab", axes):
        x0 = 0.008 if letter == "a" else ax_.get_position().x0 - 0.012
        fig.text(x0, 0.86, letter, fontsize=10, fontweight="bold", va="bottom")
    st.save(fig, out, "fig_changes")


def fig_stopping(t: Path, out: Path) -> None:
    fr = pd.read_csv(t / "stopping_frontier.csv")
    op = pd.read_csv(t / "stopping_operating_points.csv")
    conf_op = op[op["family"] == "conf"].set_index("method").loc[st.ORDER]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), gridspec_kw=dict(hspace=0.62, wspace=0.28))
    ymax = 0
    for i, (ax, m) in enumerate(zip(axes.flat, st.ORDER)):
        conf = fr[(fr["method"] == m) & (fr["family"] == "conf")].sort_values("test_cost")
        fixed = fr[(fr["method"] == m) & (fr["family"] == "fixed")].sort_values("test_cost")
        ax.plot(conf["test_cost"], conf["test_loss"], color=st.NAVY, label="Confidence rule (all $\\tau$)")
        ax.scatter(fixed["test_cost"], fixed["test_loss"], s=18, color=st.MUTED, edgecolor="white", linewidth=0.5,
                   zorder=3, label="Fixed number of seeds")
        for _, r in fixed.iterrows():
            ax.annotate(f"{int(r['param_k'])}", (r["test_cost"], r["test_loss"]), textcoords="offset points",
                        xytext=(3, 3), fontsize=6.5, color=st.INK2)
        r = conf_op.loc[m]
        if np.isfinite(r["test_cost"]):
            ax.plot(r["test_cost"], r["test_loss"], linestyle="none", marker="D", color=st.VERMILION, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4,
                    label="Test performance at development-selected threshold")
        ax.axhline(0.005, color=st.INK2, linestyle=":", linewidth=0.9)
        ax.set_title(st.LABELS[m], fontsize=8)
        ax.set_xticks(range(1, 6))
        ymax = max(ymax, fixed["test_loss"].max(), conf["test_loss"].max())
        if i % 4:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("MPQS (LDDT-PLI)")
        if i >= 3:
            ax.set_xlabel("Mean seeds per system")
    for ax in axes.flat[:7]:
        ax.set_ylim(-0.0008, ymax * 1.12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.0)

    ax = axes.flat[7]
    saving = conf_op["test_saving_vs_fixed5"] * 100
    ax.barh(Y, saving, color=st.NAVY, alpha=0.85, height=0.62)
    for y, s_, loss in zip(Y, saving, conf_op["test_loss"]):
        ax.text(s_ + 1, y, f"{s_:.0f}%", va="center", fontsize=7, color=st.INK2)
    method_axis(ax)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, max(50, saving.max() * 1.3))
    ax.set_xticks([0, 20, 40])
    ax.set_xlabel("Seeds saved (%)")
    ax.set_title("Confidence rule", fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.86, bottom=0.12)
    shift_right(axes.flat[7])
    st.save(fig, out, "fig_stopping")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=ROOT / "results/paper_v1/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_v1/figures")
    args = parser.parse_args()
    st.apply()
    fig_budget(args.tables, args.output_dir)
    fig_selection(args.tables, args.output_dir)
    fig_predict(args.tables, args.output_dir)
    fig_changes(args.tables, args.output_dir)
    fig_stopping(args.tables, args.output_dir)
    print("figures written to", args.output_dir)


if __name__ == "__main__":
    main()
