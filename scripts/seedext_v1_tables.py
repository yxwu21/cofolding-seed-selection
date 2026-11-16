#!/usr/bin/env python3
"""Manuscript macros and SI tables for E24 (Chai-1 with 20 seeds per system).

Reads results/seedext_v1/ (scripts/seedext_v1_eval.py) and the system config, and
writes paper/generated/seedext.tex (macros), table_seedext_tiers.tex and
table_seedext_timing.tex. Formatting only; nothing is computed beyond rounding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paper_v1_tables import f, rng

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 0.005
TIERS = [("pool", "All systems"), ("low", "Low"), ("mid", "Medium"), ("high", "High")]
CAP = {"low": "Low", "mid": "Mid", "high": "High", "pool": "Pool"}
WORD = {1: "One", 5: "Five", 20: "Twenty"}  # LaTeX command names cannot contain digits


def msd(m, sd, d=3):
    while sd and d < 6 and abs(sd) < 0.5 * 10 ** -d:  # one more decimal while the error rounds to zero
        d += 1
    return f"{f(m, d)} $\\pm$ {f(abs(sd), d)}"


def kstar_text(row) -> str:
    return f"{int(row['k_star'])}" if bool(row["reached"]) else r"$\geq$20"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/seedext_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/seedext_v1_systems.json")
    parser.add_argument("--e22-timing", type=Path, default=ROOT / "results/paper_v1/tables/timing_systems.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated")
    args = parser.parse_args()
    d = args.input_dir
    s = json.loads((d / "summary.json").read_text())
    cfg = json.loads(args.config.read_text())
    curves = pd.read_csv(d / "curves.csv")
    kstar = pd.read_csv(d / "kstar.csv")
    timing = pd.read_csv(d / "timing.csv")
    e22 = pd.read_csv(args.e22_timing)

    def level(outcome, tier, k):
        r = curves[(curves.outcome == outcome) & (curves.tier == tier) & (curves.kind == "level") & (curves.k == k)]
        return float(r["mean"].iloc[0])

    def gain(outcome, tier, k):
        r = curves[(curves.outcome == outcome) & (curves.tier == tier) & (curves.kind == "gain") & (curves.k == k)]
        return float(r["mean"].iloc[0])

    def ks(outcome, tier, scheme="design", t=PRIMARY):
        r = kstar[(kstar.scheme == scheme) & (kstar.outcome == outcome) & (kstar.tier == tier) & (kstar.threshold == t)]
        return r.iloc[0]

    systems = cfg["systems"]
    macros = {
        "ExtNSystems": str(len(systems)), "ExtNClusters": str(len({x["cluster"] for x in systems})),
        "ExtNComplete": str(s["n_complete"]), "ExtNCandidates": f"{len(systems) * 100:,}",
        "ExtNLow": str(sum(x["tier"] == "low" for x in systems)),
        "ExtNMid": str(sum(x["tier"] == "mid" for x in systems)),
        "ExtNHigh": str(sum(x["tier"] == "high" for x in systems)),
        "ExtPoolSize": str(cfg["pool_size"]),
        "ExtGPUHours": f"{timing.eval('startup_s + first_s + steady_s * (n_seeds - 1)').sum() / 3600:.0f}",
        "ExtRetDelta": f(s["primary"]["delta_ret_5_20"]),
        "ExtReading": s["primary"]["reading"],
        "ExtOrcDelta": f(s["delta_5_20_orc_q"]["pool"][0]),
        "ExtParityShare": f"{100 * s['parity']['share_within']:.1f}" if "parity" in s else "n/a",
        "ExtParityMeanDiff": f(s["parity"]["mean_diff"], 5) if "parity" in s else "n/a",
        "ExtReplScoreCorr": f(s["replication"]["score_corr"]) if "replication" in s else "n/a",
        "ExtReplQShare": f"{100 * s['replication']['q_share_within_0.01']:.0f}" if "replication" in s else "n/a",
    }
    for tier in ("pool", "low", "mid", "high"):
        c = CAP[tier]
        for k in (1, 5, 20):
            macros[f"ExtRet{c}{WORD[k]}"] = f(level("ret_q", tier, k))
            macros[f"ExtOrc{c}{WORD[k]}"] = f(level("orc_q", tier, k))
        macros[f"ExtGap{c}Twenty"] = f(level("orc_q", tier, 20) - level("ret_q", tier, 20))
        macros[f"ExtGap{c}Five"] = f(level("orc_q", tier, 5) - level("ret_q", tier, 5))
        macros[f"ExtOrcStepFive{c}"] = f(gain("orc_q", tier, 5))
        macros[f"ExtOrcStepTwenty{c}"] = f(gain("orc_q", tier, 20))
        macros[f"ExtKstarRet{c}"] = kstar_text(ks("ret_q", tier))
        macros[f"ExtKstarOrc{c}"] = kstar_text(ks("orc_q", tier))
        if tier != "pool":
            dr, do = s["delta_5_20_ret_q"][tier], s["delta_5_20_orc_q"][tier]
            macros[f"ExtRetDelta{c}"] = f(dr[0])
            macros[f"ExtOrcDelta{c}"] = f(do[0])
    for k in (1, 5, 20):
        macros[f"ExtRetSucc{WORD[k]}"] = f"{100 * level('ret_s', 'pool', k):.1f}"
        macros[f"ExtOrcSucc{WORD[k]}"] = f"{100 * level('orc_s', 'pool', k):.1f}"
    rk = pd.DataFrame(s["ranking"]).T
    macros["ExtSpearmanRange"] = rng(rk["spearman"], 2)
    rule = s["confidence_rule"]
    macros["ExtRuleTwentySeedsLow"] = f"{rule['cap_20']['low']['mean_seeds']:.1f}"
    macros["ExtRuleTwentySeedsHigh"] = f"{rule['cap_20']['high']['mean_seeds']:.1f}"
    macros["ExtRuleTwentySeedsMid"] = f"{rule['cap_20']['mid']['mean_seeds']:.1f}"
    macros["ExtTierCuts"] = " and ".join(f"{x:.3f}" for x in cfg["tier_cuts"])
    macros["ExtNExcluded"] = str(len(systems) - s["n_complete"])
    mid, high = macros["ExtKstarOrcMid"], macros["ExtKstarOrcHigh"]
    macros["ExtKstarOrcMidHigh"] = mid if mid == high else f"{mid} and {high}"
    for name, key in (("ExtRetSuccDeltaPts", "delta_5_20_ret_s"), ("ExtOrcSuccDeltaPts", "delta_5_20_orc_s")):
        macros[name] = f(100 * s[key]["pool"][0], 1)
    steady = timing.groupby("padded_tokens")["steady_s"].mean()
    macros["ExtSteadyRange"] = f"{steady.min():.0f}--{steady.max():.0f}"
    macros["ExtSteadySDMax"] = f"{timing['steady_sd_s'].max():.0f}"
    out = args.output_dir
    (out / "seedext.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    def se_delta(prefix, tier):
        # Bootstrap standard error of the 5-to-20 change (cluster bootstrap within tiers; deviation D5).
        return float(s[f"se_5_20_{prefix}"][tier])

    rows = []
    for tier, label in TIERS:
        n = int(curves[(curves.tier == tier) & (curves.kind == "level")]["n"].iloc[0])
        if tier == "pool":
            dr, do = s["primary"], s["delta_5_20_orc_q"]["pool"]
            dr_txt, do_txt = msd(dr["delta_ret_5_20"], se_delta("ret_q", "pool")), msd(do[0], se_delta("orc_q", "pool"))
        else:
            dr_txt, do_txt = msd(s["delta_5_20_ret_q"][tier][0], se_delta("ret_q", tier)), msd(s["delta_5_20_orc_q"][tier][0], se_delta("orc_q", tier))
        rows.append(" & ".join([
            label, str(n), f(level("ret_q", tier, 1)), f(level("ret_q", tier, 5)), f(level("ret_q", tier, 20)), dr_txt,
            f(level("orc_q", tier, 1)), f(level("orc_q", tier, 5)), f(level("orc_q", tier, 20)), do_txt,
            kstar_text(ks("ret_q", tier)), kstar_text(ks("orc_q", tier))]) + r" \\")
    (out / "table_seedext_tiers.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrlrrrlcc}", r"\toprule",
        r" & & \multicolumn{4}{c}{Returned (native top-1)} & \multicolumn{4}{c}{Best available (oracle)} & \multicolumn{2}{c}{$k^*$} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}",
        r"Tier & $n$ & $k{=}1$ & $k{=}5$ & $k{=}20$ & $\Delta_{5\to20}$ (mean $\pm$ SE) & $k{=}1$ & $k{=}5$ & $k{=}20$ & $\Delta_{5\to20}$ (mean $\pm$ SE) & Ret. & Best \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    e22_steady = e22.groupby("padded_tokens")["steady_batch_s"].mean()
    rows = []
    for pad, g in timing.groupby("padded_tokens"):
        rows.append(" & ".join([str(int(pad)), str(len(g)), f"{g['steady_s'].mean():.0f}", f"{g['steady_sd_s'].mean():.1f}",
                                f(g["first_s"].mean() - g["steady_s"].mean(), 0), f"{g['peak_reserved_gb'].max():.1f}",
                                f"{e22_steady.get(pad, float('nan')):.0f}"]) + r" \\")
    (out / "table_seedext_timing.tex").write_text("\n".join([
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"Padded tokens & Systems & Per seed (s) & Within-system SD (s) & First-seed excess (s) & Peak memory (GB) & E22 per seed (s) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))
    for k, v in macros.items():
        print(f"{k:26s} {v}")


if __name__ == "__main__":
    main()
