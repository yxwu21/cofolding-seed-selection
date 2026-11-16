#!/usr/bin/env python3
"""Manuscript macros and SI tables for the marginal value of each seed (paper v1).

Reads results/paper_v1/marginal/ (scripts/paper_v1_marginal_eval.py) and writes
paper/generated/marginal.tex (macros), table_kstar.tex and table_rule_tiers.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_v1_tables import LABELS, ORDER, f, rng

ROOT = Path(__file__).resolve().parents[1]
PRIMARY, STRICT, LOOSE = 0.005, 0.002, 0.01
PROSE = {"af3": "AF3", "af3_no_template": "AF3 without templates", "boltz": "Boltz-1", "boltz1x": "Boltz-1x",
         "boltz2": "Boltz-2", "chai": "Chai-1", "protenix": "Protenix"}


def prose_list(methods) -> str:
    names = [PROSE[m] for m in ORDER if m in set(methods)]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def kstar_range(rec: pd.DataFrame) -> str:
    """Range of k*; values not reached within five seeds are shown as 5+."""
    values = [(int(k), bool(ok)) for k, ok in zip(rec["k_star"], rec["reached"])]
    lo = min(k for k, _ in values)
    hi_val = max(k for k, _ in values)
    hi = f"{hi_val}" + ("+" if any(k == hi_val and not ok for k, ok in values) else "")
    return str(lo) if str(lo) == hi else f"{lo}--{hi}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/paper_v1/marginal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated")
    args = parser.parse_args()
    m = pd.read_csv(args.input_dir / "marginal_gains.csv")
    rec = pd.read_csv(args.input_dir / "recommended_seeds.csv")
    rules = pd.read_csv(args.input_dir / "rule_by_tier.csv")
    cost = pd.read_csv(args.input_dir / "chai_cost.csv")

    def gains(outcome, stratifier, tier, k, kind="increment"):
        s = m[(m.outcome == outcome) & (m.stratifier == stratifier) & (m.tier == tier) & (m.kind == kind)
              & (m.k == k)]
        return s.set_index("method")["mean"].loc[ORDER]

    def ks(outcome, stratifier, tier, t=PRIMARY):
        return rec[(rec.outcome == outcome) & (rec.stratifier == stratifier) & (rec.tier == tier)
                   & (rec.threshold == t)].set_index("method").loc[ORDER]

    later = pd.concat([gains("returned", "all", "all", k) for k in (3, 4, 5)])
    ret_low2 = gains("returned", "confidence", "low", 2)
    low_two = ks("returned", "confidence", "low")
    low_two = low_two[low_two["k_star"] >= 2].index
    orc_low = ks("oracle", "confidence", "low")
    ret_high_total = gains("returned", "confidence", "high", 5, "level") - gains("returned", "confidence", "high", 1, "level")
    loose = rec[(rec.outcome == "returned") & (rec.threshold == LOOSE) & rec.stratifier.isin(["all", "confidence"])]
    rt = rules[rules.stratifier == "confidence"].set_index(["tier", "method"])
    c = cost[cost.stratifier == "all"].set_index("point")
    ex = "af3"  # representative variant quoted in the text (Section 3.2)

    def ex_level(outcome, k, scale=1.0, d=3):
        return f(scale * gains(outcome, "all", "all", k, "level")[ex], d)

    def ex_kstar(stratifier, tier):
        r = ks("oracle", stratifier, tier).loc[ex]
        return str(int(r["k_star"])) if r["reached"] else r"$\geq$5"

    macros = {
        "RetStepTwoRange": rng(gains("returned", "all", "all", 2)),
        "RetStepLaterMax": f(later.max()),
        "RetStepMax": f(pd.concat([gains("returned", "all", "all", k) for k in (2, 3, 4, 5)]).max()),
        "OrcStepTwoRange": rng(gains("oracle", "all", "all", 2)),
        "OrcStepFiveRange": rng(gains("oracle", "all", "all", 5)),
        "RetKstarAllRange": kstar_range(ks("returned", "all", "all")),
        "OrcKstarAllRange": kstar_range(ks("oracle", "all", "all")),
        "OrcKstarHighRange": kstar_range(ks("oracle", "confidence", "high")),
        "OrcKstarMidRange": kstar_range(ks("oracle", "confidence", "mid")),
        "OrcKstarLowRange": kstar_range(orc_low),
        "NOrcLowNotReached": str(int((~orc_low["reached"]).sum())),
        "OrcStepFiveHighRange": rng(gains("oracle", "confidence", "high", 5)),
        "OrcStepFiveLowRange": rng(gains("oracle", "confidence", "low", 5)),
        "OrcStepFiveLowMax": f(gains("oracle", "confidence", "low", 5).max()),
        "RetGainHighRange": rng(ret_high_total),
        "RetLowTwoSeedVariants": prose_list(low_two),
        "RetStepTwoLowSelRange": rng(ret_low2.loc[low_two]),
        "RetStepTwoLowOtherRange": rng(ret_low2.drop(low_two)),
        "RetKstarLooseRange": kstar_range(loose),
        "OrcStepFiveNovelRange": rng(gains("oracle", "novelty", "[0,30)", 5)),
        "OrcStepFiveFamiliarRange": rng(gains("oracle", "novelty", "[70,100]", 5)),
        "RuleSeedsHighRange": rng(rt.loc["high"]["mean_seeds"], 1),
        "RuleSeedsLowRange": rng(rt.loc["low"]["mean_seeds"], 1),
        "RulePosLossHighRange": rng(rt.loc["high"]["positive_loss"]),
        "CostNSystems": f"{int(c.loc['k=1', 'n_systems']):,}",
        "CostNUntimed": f"{int(c.loc['k=1', 'n_untimed_systems']):,}",
        "CostOneMin": f(c.loc["k=1", "minutes"], 1), "CostFiveMin": f(c.loc["k=5", "minutes"], 1),
        "CostRuleMin": f(c.loc["confidence rule", "minutes"], 1),
        "CostRuleSeeds": f(c.loc["confidence rule", "seeds"], 2),
        "CostRetOne": f(c.loc["k=1", "ret_q"]), "CostRetFive": f(c.loc["k=5", "ret_q"]),
        "CostRetRule": f(c.loc["confidence rule", "ret_q"]),
        "CostOrcOne": f(c.loc["k=1", "orc_q"]), "CostOrcFive": f(c.loc["k=5", "orc_q"]),
        "ExVariant": PROSE[ex],
        "ExRetOne": ex_level("returned", 1), "ExRetFive": ex_level("returned", 5),
        "ExOrcOne": ex_level("oracle", 1), "ExOrcFive": ex_level("oracle", 5),
        "ExRetSuccOne": ex_level("returned_success", 1, 100, 1),
        "ExRetSuccFive": ex_level("returned_success", 5, 100, 1),
        "ExOrcSuccOne": ex_level("oracle_success", 1, 100, 1),
        "ExOrcSuccFive": ex_level("oracle_success", 5, 100, 1),
        "ExRetStepTwo": f(gains("returned", "all", "all", 2)[ex]),
        "ExOrcStepTwo": f(gains("oracle", "all", "all", 2)[ex]),
        "ExOrcStepFive": f(gains("oracle", "all", "all", 5)[ex]),
        "ExOrcKstar": ex_kstar("all", "all"),
        "ExOrcStepFiveHigh": f(gains("oracle", "confidence", "high", 5)[ex]),
        "ExOrcStepFiveLow": f(gains("oracle", "confidence", "low", 5)[ex]),
        "ExOrcKstarHigh": ex_kstar("confidence", "high"),
        "ExOrcKstarLow": ex_kstar("confidence", "low"),
    }
    out = args.output_dir
    (out / "marginal.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    def cell(method, outcome, stratifier, tier):
        parts = []
        for t in (STRICT, PRIMARY, LOOSE):
            r = rec[(rec.method == method) & (rec.outcome == outcome) & (rec.stratifier == stratifier)
                    & (rec.tier == tier) & (rec.threshold == t)].iloc[0]
            parts.append(f"{int(r.k_star)}" if r.reached else r"$\geq$5")
        return "/".join(parts)

    tiers = [("all", "all"), ("confidence", "low"), ("confidence", "mid"), ("confidence", "high")]
    rows = [" & ".join([LABELS[mm], *[cell(mm, "returned", s, t) for s, t in tiers],
                        *[cell(mm, "oracle", s, t) for s, t in tiers]]) + r" \\" for mm in ORDER]
    (out / "table_kstar.tex").write_text("\n".join([
        r"\begin{tabular}{lcccccccc}", r"\toprule",
        r" & \multicolumn{4}{c}{Returned pose} & \multicolumn{4}{c}{Best available pose} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Method & All & Low & Medium & High & All & Low & Medium & High \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))
    rows = []
    for mm in ORDER:
        vals = [f"{rt.loc[(t, mm), 'mean_seeds']:.2f}" for t in ("low", "mid", "high")]
        vals += [f(rt.loc[(t, mm), "positive_loss"]) for t in ("low", "mid", "high")]
        rows.append(" & ".join([LABELS[mm], *vals]) + r" \\")
    (out / "table_rule_tiers.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r" & \multicolumn{3}{c}{Mean seeds} & \multicolumn{3}{c}{Positive loss} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Low & Medium & High & Low & Medium & High \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))
    for k, v in macros.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
