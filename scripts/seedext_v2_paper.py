#!/usr/bin/env python3
"""Tables, macros and a figure for the paper writer: 20 seeds per system for four models.

Reads the frozen analyses of E24 (results/seedext_v1/, Chai-1) and E25
(results/seedext_v2/<method>/, Protenix, Boltz-2 and Boltz-1) and writes to
results/seedext_v2/paper/:

  seedext_models.tex               LaTeX macros (\\Xt...) for the text
  table_seedext_models.tex         one row per model: levels at 5 and 20 seeds, 5-to-20 changes
                                   (mean +/- SE) and k*
  table_seedext_models_tiers.tex   the same changes and k* per design confidence tier
  table_seedext_models_timing.tex  time per seed by padded input size
  fig_seedext_models.{pdf,png}     (a-d) mean LDDT-PLI of the returned and the best available pose
                                   against the number of seeds; (e-h) gain from each seed

Formatting only: every value is read from the analysis outputs; nothing is computed beyond
rounding, differences of reported levels and the GPU-hour sum of the timing tables.
Run with python.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_v1_style as st  # noqa: E402
from paper_v1_tables import f, rng  # noqa: E402
from seedext_v1_tables import kstar_text, msd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 0.005
# key, display name, macro word (LaTeX command names cannot contain digits), result folder
MODELS = [("chai", "Chai-1", "Chai", ROOT / "results/seedext_v1"),
          ("protenix", "Protenix", "Protenix", ROOT / "results/seedext_v2/protenix"),
          ("boltz2", "Boltz-2", "BoltzTwo", ROOT / "results/seedext_v2/boltz2"),
          ("boltz", "Boltz-1", "BoltzOne", ROOT / "results/seedext_v2/boltz")]
TIERS = [("pool", "All"), ("low", "Low"), ("mid", "Medium"), ("high", "High")]
SERIES = [("ret_q", st.NAVY, "Returned (native top-1)"), ("orc_q", st.TEAL, "Best available (oracle)")]


def plain(x: float) -> str:
    return "0.000" if abs(x) < 5e-4 else f"{x:.3f}"


def signed(x: float) -> str:
    return "0.000" if abs(x) < 5e-4 else f"{x:+.3f}"


def load(folder: Path) -> dict:
    return {"summary": json.loads((folder / "summary.json").read_text()),
            "curves": pd.read_csv(folder / "curves.csv"),
            "kstar": pd.read_csv(folder / "kstar.csv"),
            "timing": pd.read_csv(folder / "timing.csv")}


def level(d: dict, outcome: str, tier: str, k: int) -> float:
    c = d["curves"]
    r = c[(c.outcome == outcome) & (c.tier == tier) & (c.kind == "level") & (c.k == k)]
    return float(r["mean"].iloc[0])


def kstar_row(d: dict, outcome: str, tier: str, t: float = PRIMARY):
    k = d["kstar"]
    return k[(k.scheme == "design") & (k.outcome == outcome) & (k.tier == tier) & (k.threshold == t)].iloc[0]


def tier_n(d: dict, tier: str) -> int:
    c = d["curves"]
    return int(c[(c.tier == tier) & (c.kind == "level")]["n"].iloc[0])


def gpu_hours(d: dict) -> float:
    return float(d["timing"].eval("startup_s + first_s + steady_s * (n_seeds - 1)").sum() / 3600)


def write_macros(data: dict, out: Path) -> dict:
    macros = {"XtNModels": str(len(MODELS)), "XtNNewModels": str(len(MODELS) - 1)}
    new = [key for key, *_ in MODELS if key != "chai"]
    macros["XtNewSeeds"] = f"{sum(int(data[k]['timing']['n_seeds'].sum()) for k in new):,}"
    macros["XtNewGPUHours"] = f"{sum(gpu_hours(data[k]) for k in new):.0f}"
    for key, _, word, _ in MODELS:
        d, s = data[key], data[key]["summary"]
        p = s["primary"]
        orc = s["delta_5_20_orc_q"]["pool"]
        macros.update({
            f"XtNComplete{word}": str(s["n_complete"]), f"XtNSystems{word}": str(s["n_systems"]),
            f"XtRetDelta{word}": f(p["delta_ret_5_20"]), f"XtRetLo{word}": f(p["lo"]), f"XtRetHi{word}": f(p["hi"]),
            f"XtReading{word}": p["reading"],
            f"XtOrcDelta{word}": f(orc[0]), f"XtOrcLo{word}": f(orc[1]), f"XtOrcHi{word}": f(orc[2]),
            f"XtGapFive{word}": f(level(d, "orc_q", "pool", 5) - level(d, "ret_q", "pool", 5)),
            f"XtGapTwenty{word}": f(level(d, "orc_q", "pool", 20) - level(d, "ret_q", "pool", 20)),
            f"XtKstarOrc{word}": kstar_text(kstar_row(d, "orc_q", "pool")),
            f"XtKstarOrcLow{word}": kstar_text(kstar_row(d, "orc_q", "low")),
            f"XtRetSuccDeltaPts{word}": f(100 * s["delta_5_20_ret_s"]["pool"][0], 1),
            f"XtOrcSuccDeltaPts{word}": f(100 * s["delta_5_20_orc_s"]["pool"][0], 1),
            f"XtParityShare{word}": f"{100 * s['parity']['share_within']:.1f}",
            f"XtSpearmanRange{word}": rng(pd.DataFrame(s["ranking"]).T["spearman"], 2),
            f"XtRuleTwentySeedsLow{word}": f"{s['confidence_rule']['cap_20']['low']['mean_seeds']:.1f}",
        })
    keys = [key for key, *_ in MODELS]
    summ = {k: data[k]["summary"] for k in keys}
    macros["XtRetDeltaRange"] = rng([summ[k]["primary"]["delta_ret_5_20"] for k in keys])
    macros["XtRetHiMax"] = f(max(summ[k]["primary"]["hi"] for k in keys))
    macros["XtOrcDeltaRange"] = rng([summ[k]["delta_5_20_orc_q"]["pool"][0] for k in keys])
    macros["XtGapFiveRange"] = rng([level(data[k], "orc_q", "pool", 5) - level(data[k], "ret_q", "pool", 5) for k in keys])
    macros["XtGapTwentyRange"] = rng([level(data[k], "orc_q", "pool", 20) - level(data[k], "ret_q", "pool", 20) for k in keys])
    macros["XtKstarOrcRange"] = rng([int(kstar_row(data[k], "orc_q", "pool")["k_star"]) for k in keys], 0)
    macros["XtRetSuccDeltaPtsRange"] = rng([100 * summ[k]["delta_5_20_ret_s"]["pool"][0] for k in keys], 1)
    macros["XtOrcSuccDeltaPtsRange"] = rng([100 * summ[k]["delta_5_20_orc_s"]["pool"][0] for k in keys], 1)
    macros["XtSpearmanAllRange"] = rng([v["spearman"] for k in keys for v in summ[k]["ranking"].values()], 2)
    macros["XtRuleTwentySeedsLowRange"] = rng([summ[k]["confidence_rule"]["cap_20"]["low"]["mean_seeds"] for k in keys], 1)
    (out / "seedext_models.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    return macros


def write_tables(data: dict, out: Path) -> None:
    rows = []
    for key, name, _, _ in MODELS:
        d, s = data[key], data[key]["summary"]
        se_r, se_o = s["se_5_20_ret_q"]["pool"], s["se_5_20_orc_q"]["pool"]
        rows.append(" & ".join([
            name, f"{s['n_complete']}/{s['n_systems']}",
            f(level(d, "ret_q", "pool", 5)), f(level(d, "ret_q", "pool", 20)), msd(s["primary"]["delta_ret_5_20"], se_r),
            f(level(d, "orc_q", "pool", 5)), f(level(d, "orc_q", "pool", 20)), msd(s["delta_5_20_orc_q"]["pool"][0], se_o),
            f"{f(level(d, 'orc_q', 'pool', 5) - level(d, 'ret_q', 'pool', 5))} $\\to$ "
            f"{f(level(d, 'orc_q', 'pool', 20) - level(d, 'ret_q', 'pool', 20))}",
            kstar_text(kstar_row(d, "ret_q", "pool")), kstar_text(kstar_row(d, "orc_q", "pool"))]) + r" \\")
    (out / "table_seedext_models.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrlrrlccc}", r"\toprule",
        r" & & \multicolumn{3}{c}{Returned (native top-1)} & \multicolumn{3}{c}{Best available (oracle)} & & \multicolumn{2}{c}{$k^*$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){10-11}",
        r"Model & Complete & $k{=}5$ & $k{=}20$ & $\Delta_{5\to20}$ (mean $\pm$ SE) & $k{=}5$ & $k{=}20$ & "
        r"$\Delta_{5\to20}$ (mean $\pm$ SE) & Gap $5\to20$ & Ret. & Best \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    rows = []
    for key, name, _, _ in MODELS:
        d, s = data[key], data[key]["summary"]
        for i, (tier, label) in enumerate(TIERS):
            if tier == "pool":
                dr = s["primary"]["delta_ret_5_20"]
            else:
                dr = s["delta_5_20_ret_q"][tier][0]
            do = s["delta_5_20_orc_q"][tier][0]
            rows.append(" & ".join([
                name if i == 0 else "", label, str(tier_n(d, tier)),
                msd(dr, s["se_5_20_ret_q"][tier]), msd(do, s["se_5_20_orc_q"][tier]),
                kstar_text(kstar_row(d, "ret_q", tier)), kstar_text(kstar_row(d, "orc_q", tier))]) + r" \\")
        rows.append(r"\addlinespace")
    (out / "table_seedext_models_tiers.tex").write_text("\n".join([
        r"\begin{tabular}{llrllcc}", r"\toprule",
        r" & & & \multicolumn{2}{c}{$\Delta_{5\to20}$ (mean $\pm$ SE)} & \multicolumn{2}{c}{$k^*$} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & Tier & $n$ & Returned & Best available & Ret. & Best \\",
        r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}", ""]))

    steady = {key: data[key]["timing"].groupby("padded_tokens")["steady_s"].agg(["mean", "size"]) for key, *_ in MODELS}
    pads = sorted({int(p) for t in steady.values() for p in t.index})
    rows = []
    for pad in pads:
        cells = [str(pad)]
        for key, *_ in MODELS:
            t = steady[key]
            hit = t[t.index.astype(int) == pad]
            cells.append(f"{hit['mean'].iloc[0]:.0f} ({int(hit['size'].iloc[0])})" if len(hit) else "--")
        rows.append(" & ".join(cells) + r" \\")
    (out / "table_seedext_models_timing.tex").write_text("\n".join([
        r"\begin{tabular}{r" + "r" * len(MODELS) + "}", r"\toprule",
        r" & \multicolumn{" + str(len(MODELS)) + r"}{c}{Seconds per seed on an RTX A6000 (systems)} \\",
        r"\cmidrule(lr){2-" + str(len(MODELS) + 1) + "}",
        "Padded tokens & " + " & ".join(name for _, name, _, _ in MODELS) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def draw_figure(data: dict, out: Path) -> None:
    st.apply()
    fig, axes = plt.subplots(2, len(MODELS), figsize=(7.0, 4.6), sharex=True,
                             gridspec_kw={"wspace": 0.42, "hspace": 0.32, "height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.11)
    pooled = {key: data[key]["curves"][data[key]["curves"]["tier"] == "pool"] for key, *_ in MODELS}
    span = max(g[(g.kind == "level") & g.outcome.isin(["ret_q", "orc_q"])]["mean"].agg(lambda x: x.max() - x.min())
               for g in pooled.values()) + 0.12
    gains = pd.concat([g[(g.kind == "gain") & g.outcome.isin(["ret_q", "orc_q"])] for g in pooled.values()])
    gmax, gmin = gains["hi"].max(), min(gains["lo"].min(), -0.002)
    for c, (key, name, _, _) in enumerate(MODELS):
        d, s, cv = data[key], data[key]["summary"], pooled[key]
        top, bottom = axes[0, c], axes[1, c]
        lv, gn = cv[cv.kind == "level"], cv[cv.kind == "gain"]
        for outcome, color, _ in SERIES:
            g = lv[lv.outcome == outcome].sort_values("k")
            top.plot(g["k"], g["mean"], color=color, linewidth=1.5)
            gg = gn[gn.outcome == outcome].sort_values("k")
            bottom.fill_between(gg["k"], gg["lo"], gg["hi"], color=color, alpha=0.15, linewidth=0)
            bottom.plot(gg["k"], gg["mean"], color=color, linewidth=1.3, marker="o", markersize=2.2)
            ks = kstar_row(d, outcome, "pool")
            if bool(ks["reached"]):
                k = int(ks["k_star"])
                y = lv[(lv.outcome == outcome) & (lv.k == k)]["mean"].iloc[0]
                top.scatter([k], [y], s=46, facecolors="none", edgecolors=color, linewidths=1.1, zorder=4)
        lo = lv[lv.outcome.isin(["ret_q", "orc_q"])]["mean"].min() - 0.02
        hi = min(lo + span, 1.0)
        top.set_ylim(hi - span, hi)
        box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4}
        for i, (outcome, color, label) in enumerate([("orc_q", st.TEAL, "Best available"), ("ret_q", st.NAVY, "Returned")]):
            v = s[f"delta_5_20_{outcome}"]["pool"][:3]
            text = f"{label} {signed(v[0])}".replace("-", "−")
            top.text(0.04, 0.87 - 0.1 * i, text, transform=top.transAxes, ha="left", va="top", fontsize=6.0,
                     color=color, bbox=box)
        top.text(0.04, 0.97, "5$\\to$20 seeds", transform=top.transAxes, ha="left", va="top",
                 fontsize=6.0, color=st.INK2, bbox=box)
        top.axvline(5, color=st.MUTED, linestyle=":", linewidth=0.9)
        top.set_title(f"{name}\n$n$ = {s['n_complete']}", fontsize=8.5, linespacing=1.3)
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
                Line2D([], [], color=st.MUTED, linestyle=":", linewidth=0.9, label="Five-seed budget")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.995), handletextpad=0.5,
               columnspacing=1.2)
    letters = "abcdefgh"
    for r in range(2):
        for c in range(len(MODELS)):
            pos = axes[r, c].get_position()
            fig.text(pos.x0 - 0.06, pos.y1 + 0.012, letters[len(MODELS) * r + c], fontsize=10, fontweight="bold")
    st.save(fig, out, "fig_seedext_models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/seedext_v2/paper")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {key: load(folder) for key, _, _, folder in MODELS}
    macros = write_macros(data, args.output_dir)
    write_tables(data, args.output_dir)
    draw_figure(data, args.output_dir)
    for k, v in macros.items():
        print(f"{k:30s} {v}")


if __name__ == "__main__":
    main()
