#!/usr/bin/env python3
"""Generate LaTeX tables and number macros for the paper v1 manuscript.

Every number quoted in paper/manuscript.tex comes from results/paper_v1/tables
through this script, so rerunning the analysis updates the text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["af3", "af3_no_template", "boltz", "boltz1x", "boltz2", "chai", "protenix"]
SEL_EXAMPLE = "af3_no_template"  # a typical variant on the selection measures, quoted in Section 3.3
LABELS = {"af3": "AF3", "af3_no_template": "AF3 (no templates)", "boltz": "Boltz-1", "boltz1x": "Boltz-1x",
          "boltz2": "Boltz-2", "chai": "Chai-1", "protenix": "Protenix"}


def f(x, d=3):
    x = 0.0 if abs(x) < 0.5 * 10 ** -d else x  # no negative zero
    return f"{x:.{d}f}".replace("-", "$-$")


def ci(m, lo, hi, d=3):
    return f"{f(m, d)} [{f(lo, d)}, {f(hi, d)}]"


def msd(m, sd, d=3):
    """Mean and bootstrap standard error, for the tables (deviation D5)."""
    while sd and d < 6 and abs(sd) < 0.5 * 10 ** -d:  # one more decimal while the error rounds to zero
        d += 1
    return f"{f(m, d)} $\\pm$ {f(abs(sd), d)}"


def rng(series, d=3, scale=1.0):
    s = (pd.Series(series, dtype=float) * scale).round(d)
    s = s.where(s.abs() >= 0.5 * 10 ** -d, 0.0)
    lo, hi = (f"{v:.{d}f}" for v in (s.min(), s.max()))
    lo, hi = (x.replace("-", "$-$") for x in (lo, hi))
    if lo == hi:
        return lo
    return f"{lo} to {hi}" if s.min() < 0 else f"{lo}--{hi}"


def timing_outputs(t: Path, out: Path, ops: pd.DataFrame) -> None:
    """RQ4 macros and table, written only once timing results exist."""
    sys_csv, conv_csv = t / "timing_systems.csv", t / "timing_conversion.csv"
    if not sys_csv.exists() or not conv_csv.exists():
        return
    ts = pd.read_csv(sys_csv)
    ts = ts[np.isfinite(ts["steady_batch_s"])].sort_values("residues")
    conv = pd.read_csv(conv_csv)
    chai = conv[conv.method == "chai"]
    conf = chai[chai.family == "conf"].set_index("system_id")
    fix1 = chai[(chai.family == "fixed") & (chai.param_k == 1)].set_index("system_id")
    fix2 = chai[(chai.family == "fixed") & (chai.param_k == 2)].set_index("system_id")
    chai_conf = ops[(ops.method == "chai") & (ops.family == "conf")].iloc[0]
    cold = ts.iloc[0]  # first process of the allocation (runs ordered by length): cold file cache
    warm = ts.iloc[1:]
    wid = warm["system_id"]
    by_pad = [f"{rng(g['steady_batch_s'], 0)} s at {p}" for p, g in ts.groupby("padded_tokens")]
    conf_min = conf["time_saving"] * ts.set_index("system_id")["T5_s"] / 60
    macros = {
        "TimingGPU": ts["gpu"].iloc[0].replace("NVIDIA ", ""),
        "TimingNSystems": str(len(ts)),
        "TimingResRange": f"{int(ts['residues'].min())}--{int(ts['residues'].max())}",
        "TimingPadMax": str(int(ts["padded_tokens"].max())),
        "ColdResidues": str(int(cold["residues"])),
        "ColdStartup": f"{cold['startup_s']:.0f}",
        "ColdFirstExtra": f"{cold['first_call_s'] - cold['steady_batch_s']:.0f}",
        "ColdOverhead": f"{cold['overhead_in_batches']:.1f}",
        "WarmStartupRange": rng(warm["startup_s"], 0),
        "WarmFirstDiffMax": f"{(warm['first_call_s'] - warm['steady_batch_s']).abs().max():.0f}",
        "WarmOverheadMax": f"{warm['overhead_in_batches'].max():.2f}",
        "SteadyRange": rng(ts["steady_batch_s"], 0),
        "SteadyByPad": ", ".join(by_pad[:-1]) + " and " + by_pad[-1] + " tokens",
        "SteadySDMax": f"{ts['steady_batch_sd_s'].max():.0f}",
        "MemRange": rng(ts["reserved_gb_max"], 1),
        "ChaiConfBatches": f"{chai_conf['test_cost']:.2f}",
        "ChaiConfBatchSaving": f"{chai_conf['test_saving_vs_fixed5'] * 100:.0f}",
        "ChaiConfLoss": f(chai_conf["test_loss"]),
        "ConfTimeSavingRange": rng(conf["time_saving"] * 100, 0),
        "ConfTimeSavingWarmRange": rng(conf.loc[wid, "time_saving"] * 100, 0),
        "ConfTimeSavingCold": f"{conf.loc[cold['system_id'], 'time_saving'] * 100:.0f}",
        "ConfMinSavedRange": rng(conf_min, 0),
        "FixOneTimeSavingWarmRange": rng(fix1.loc[wid, "time_saving"] * 100, 0),
        "FixOneTimeSavingCold": f"{fix1.loc[cold['system_id'], 'time_saving'] * 100:.0f}",
        "FixOneTimeSavingRange": rng(fix1["time_saving"] * 100, 0),
        "FixTwoTimeSavingRange": rng(fix2["time_saving"] * 100, 0),
    }
    (out / "timing.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    rows = []
    for _, r in ts.iterrows():
        rows.append(" & ".join([
            r["system_id"].split("__")[0], f"{int(r['residues'])}", f"{int(r['tokens'])} ({int(r['padded_tokens'])})",
            f"{r['startup_s']:.0f}", f"{r['first_call_s']:.0f}",
            f"{r['steady_batch_s']:.0f}", f"{r['overhead_in_batches']:.1f}", f"{r['T5_s'] / 60:.1f}",
            f"{conf.loc[r['system_id'], 'time_saving'] * 100:.0f}", f"{fix1.loc[r['system_id'], 'time_saving'] * 100:.0f}",
        ]) + r" \\")
    (out / "table_timing.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r" & & & \multicolumn{3}{c}{Measured, s} & & & \multicolumn{2}{c}{Wall-time saving vs 5 seeds, \%} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){9-10}",
        r"System & Residues & Tokens (padded) & Start-up & First seed & Later seeds & Overhead (seeds) & $T(5)$, min & Confidence stop & 1 seed \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def robustness_outputs(t: Path, out: Path, macros: dict) -> None:
    """Post hoc robustness macros and Supplementary Tables S3-S7."""
    def r(name):
        f = t / f"robustness_{name}.csv"
        return pd.read_csv(f) if f.exists() else None
    rm, ex, av, sub = r("rmsd"), r("excluded"), r("available"), r("subsets")
    ls, sd, pr, mt = r("losssweep"), r("stopdist"), r("predict"), r("material")
    hr, ps, sm = r("headroom"), r("perstudy"), r("samples")
    if rm is None:
        return
    rall = rm[rm.scope == "all"].set_index(["method", "quantity"])
    macros["RetRmsdPtsRange"] = rng([rall.loc[(m, "ret_r"), "gain"] * 100 for m in ORDER], 1)
    macros["OrcRmsdPtsRange"] = rng([rall.loc[(m, "orc_r"), "gain"] * 100 for m in ORDER], 1)
    exc = ex.pivot_table(index="method", columns="group", values=["n_systems", "mean_labelled_q", "mean_novelty", "multi_ligand_share"]).loc[ORDER]
    share = exc["n_systems"]["excluded"] / (exc["n_systems"]["excluded"] + exc["n_systems"]["complete"]) * 100
    macros["ExclShareRange"] = rng(share, 0)
    macros["ExclQRange"] = rng(exc["mean_labelled_q"]["excluded"], 2)
    macros["ComplQRange"] = rng(exc["mean_labelled_q"]["complete"], 2)
    macros["ExclNovRange"] = rng(exc["mean_novelty"]["excluded"], 0)
    macros["ComplNovRange"] = rng(exc["mean_novelty"]["complete"], 0)
    a = av.pivot_table(index="method", columns="quantity", values=["gain", "n_systems"]).loc[ORDER]
    macros["AvailNRange"] = f"{int(a['n_systems']['ret_q'].min()):,}--{int(a['n_systems']['ret_q'].max()):,}"
    macros["AvailRetGainRange"] = rng(a["gain"]["ret_q"], 3)
    macros["AvailOrcGainRange"] = rng(a["gain"]["orc_q"], 3)
    for label, key in (("common_6_methods", "CommonSix"), ("common_7_methods", "CommonSeven"), ("single_proper_ligand", "Single")):
        x = sub[sub.subset == label].pivot_table(index="method", columns="quantity", values=["gain", "n"])
        macros[f"{key}RetRange"] = rng(x["gain"]["ret_q"], 3)
        macros[f"{key}OrcRange"] = rng(x["gain"]["orc_q"], 3)
        macros[f"{key}N"] = f"{int(x['n']['ret_q'].max()):,}" if key != "Single" else f"{int(x['n']['ret_q'].min()):,}--{int(x['n']['ret_q'].max()):,}"
    l = ls[ls.limit.notna()]
    for lim, key in ((0.0025, "Low"), (0.005, "Mid"), (0.01, "High")):
        macros[f"Saving{key}Range"] = rng(l[l.limit == lim]["test_saving"] * 100, 0)
    sdp = sd.pivot_table(index="method", columns="stop_after_seed", values="share_of_test_runs").loc[ORDER]
    macros["StopOneRange"] = rng(sdp[1] * 100, 0)
    macros["RunAllRange"] = rng(sdp[5] * 100, 0)
    pooled = pr[pr.stage == "pooled"].set_index("method").loc[ORDER]
    macros["AUCGainRange"] = rng(pooled["auc_gain"], 3)
    macros["NAUCGainSignif"] = str(int(((pooled["auc_gain_lo"] > 0) | (pooled["auc_gain_hi"] < 0)).sum()))
    st = pr[pr.stage != "pooled"]
    macros["WithinStageAUCRange"] = rng(st["auc_logistic"], 2)
    macros["BestScoreWithinStageRange"] = rng(st["auc_best_score"], 2)
    m0 = mt[mt.threshold.astype(str) == "0.0"].set_index("method").loc[ORDER]
    m1 = mt[mt.threshold.astype(str) == "0.1"].set_index("method").loc[ORDER]
    mf = mt[mt.threshold.astype(str) == "success_flip"].set_index("method").loc[ORDER]
    macros["ImproveZeroRange"] = rng(m0["improve"] * 100, 0)
    macros["WorsenZeroRange"] = rng(m0["worsen"] * 100, 0)
    macros["ImproveTenthRange"] = rng(m1["improve"] * 100, 1)
    macros["WorsenTenthRange"] = rng(m1["worsen"] * 100, 1)
    macros["FlipImproveRange"] = rng(mf["improve"] * 100, 1)
    macros["FlipWorsenRange"] = rng(mf["worsen"] * 100, 1)
    h = hr.set_index("method").loc[ORDER]
    macros["NearBestRandomRange"] = rng(h["near_best_random"] * 100, 0)
    macros["NearBestRandomSpreadRange"] = rng(h["near_best_random_spread"] * 100, 0)
    macros["NHeadroomCIZero"] = str(int((h["headroom_lo"] <= 0).sum()))
    macros["SelExNearBestRandom"] = f(h.loc[SEL_EXAMPLE, "near_best_random"] * 100, 0)
    p = ps.set_index("method").loc[ORDER]
    macros["PerSysDiffRange"] = rng(p["share_abs_gt_0.01"] * 100, 0)
    macros["OneBetterRange"] = rng(p["share_1seed_better_gt_0.01"] * 100, 0)
    macros["FiveBetterRange"] = rng(p["share_5seed_better_gt_0.01"] * 100, 0)
    s5 = sm[sm.m_samples == 5].set_index("method").loc[ORDER]
    macros["WithinSeedOrcRange"] = rng(s5["within_seed_oracle"] - sm[sm.m_samples == 1].set_index("method").loc[ORDER]["within_seed_oracle"], 3)
    macros["AcrossSeedOrcRange"] = rng(s5["across_seeds_oracle"] - sm[sm.m_samples == 1].set_index("method").loc[ORDER]["across_seeds_oracle"], 3)

    # Table S3: robustness gains
    rows = []
    for m in ORDER:
        cells = [LABELS[m]]
        for frame, sel in ((av, None), (sub, "common_6_methods"), (sub, "common_7_methods"), (sub, "single_proper_ligand")):
            d = frame if sel is None else frame[frame.subset == sel]
            d = d[d.method == m]
            if d.empty:
                cells += ["--", "--"]
                continue
            g = d.set_index("quantity")
            nkey = "n_systems" if sel is None else "n"
            cells += [f"{int(g.loc['ret_q', nkey]):,}", f"{f(g.loc['ret_q', 'gain'])} / {f(g.loc['orc_q', 'gain'])}"]
        rows.append(" & ".join(cells) + r" \\")
    (out / "table_robustness.tex").write_text("\n".join([
        r"\begin{tabular}{lrlrlrlrl}", r"\toprule",
        r" & \multicolumn{2}{c}{Available-case} & \multicolumn{2}{c}{Common to 6 methods} & \multicolumn{2}{c}{Common to 7 methods} & \multicolumn{2}{c}{Single proper ligand} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Method & $n$ & ret. / oracle $\Delta_{1\to5}$ & $n$ & ret. / oracle & $n$ & ret. / oracle & $n$ & ret. / oracle \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))
    # Table S4: loss sweep + stop distribution
    rows = []
    for m in ORDER:
        for lim in (0.0025, 0.005, 0.01):
            x = l[(l.method == m) & (l.limit == lim)]
            if x.empty:
                continue
            x = x.iloc[0]
            stop = (f"{sdp.loc[m, 1] * 100:.0f} / {sdp.loc[m, 5] * 100:.0f}" if lim == 0.005 else "")
            rows.append(" & ".join([LABELS[m] if lim == 0.0025 else "", f"{lim:g}", f(x["tau"]),
                                    f"{x['test_saving'] * 100:.0f}", f(x["test_loss"]),
                                    msd(x["test_signed"], x["test_signed_se"]),
                                    msd(x["test_succ_diff"] * 100, x["test_succ_diff_se"] * 100, 1),
                                    stop]) + r" \\")
        rows.append(r"\addlinespace")
    (out / "table_losssweep.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrlll}", r"\toprule",
        r"Method & MPQS limit & $\tau$ & Saving, \% & MPQS & Signed loss $\pm$ SE & Success loss, pts $\pm$ SE & Stop at 1 / run all 5, \% \\",
        r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}", ""]))
    # Table S5: RMSD endpoint
    rows = []
    for m in ORDER:
        rr, oo = rall.loc[(m, "ret_r")], rall.loc[(m, "orc_r")]
        rows.append(" & ".join([LABELS[m], f"{int(rr['n']):,}", f"{rr['k1'] * 100:.1f} $\\to$ {rr['k5'] * 100:.1f}",
                                msd(rr["gain"] * 100, rr["se"] * 100, 1),
                                f"{oo['k1'] * 100:.1f} $\\to$ {oo['k5'] * 100:.1f}",
                                msd(oo["gain"] * 100, oo["se"] * 100, 1)]) + r" \\")
    (out / "table_rmsd.tex").write_text("\n".join([
        r"\begin{tabular}{lrllll}", r"\toprule",
        r" & & \multicolumn{2}{c}{Returned} & \multicolumn{2}{c}{Oracle} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Method & Systems & Success, \% & $\Delta_{1\to5}$, pts $\pm$ SE & Success, \% & $\Delta_{1\to5}$, pts $\pm$ SE \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))
    # Table S6: predictability
    rows = []
    for m in ORDER:
        x = st[st.method == m].set_index("stage")
        po = pooled.loc[m]
        rows.append(" & ".join([LABELS[m]] + [f"{x.loc[str(k), 'auc_logistic']:.2f}" for k in range(1, 5)]
                               + [f"{x.loc[str(k), 'auc_best_score']:.2f}" for k in range(1, 5)]
                               + [msd(po["auc_gain"], po["auc_gain_se"])]) + r" \\")
    (out / "table_predict.tex").write_text("\n".join([
        r"\begin{tabular}{lccccccccc}", r"\toprule",
        r" & \multicolumn{4}{c}{Logistic model} & \multicolumn{4}{c}{Best score alone} & \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Method & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=1$ & $k=2$ & $k=3$ & $k=4$ & AUC gain \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))
    # Table S7: samples within one seed vs across seeds
    rows = []
    s1 = sm[sm.m_samples == 1].set_index("method").loc[ORDER]
    for m in ORDER:
        rows.append(" & ".join([LABELS[m], f(s1.loc[m, "within_seed_returned"]),
                                f(s5.loc[m, "within_seed_returned"]), f(s5.loc[m, "within_seed_oracle"]),
                                f(s5.loc[m, "across_seeds_returned"]), f(s5.loc[m, "across_seeds_oracle"])]) + r" \\")
    (out / "table_samples.tex").write_text("\n".join([
        r"\begin{tabular}{lccccc}", r"\toprule",
        r" & 1 sample & \multicolumn{2}{c}{5 samples, 1 seed} & \multicolumn{2}{c}{5 seeds, 1 sample each} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Method & Mean & Returned & Best available & Returned & Best available \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=ROOT / "results/paper_v1/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/generated")
    args = parser.parse_args()
    t, out = args.tables, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(t / "coverage.csv").set_index("method").loc[ORDER]
    bud = pd.read_csv(t / "budget_curves.csv")
    gains = pd.read_csv(t / "budget_gains.csv")
    ops = pd.read_csv(t / "stopping_operating_points.csv")
    chg = pd.read_csv(t / "next_batch_changes.csv")
    rank = pd.read_csv(t / "ranking_diagnostics.csv").set_index("method").loc[ORDER]
    pred = pd.read_csv(t / "predictability.csv")
    nov = pd.read_csv(t / "novelty_strata.csv")

    def gain(method, qty, scope="all", a=1, b=5):
        r = gains[(gains.method == method) & (gains.quantity == qty) & (gains.scope == scope)
                  & (gains.from_k == a) & (gains.to_k == b)].iloc[0]
        return r["mean"], r["lo"], r["hi"]

    def gain_se(method, qty, scope="all", a=1, b=5):
        r = gains[(gains.method == method) & (gains.quantity == qty) & (gains.scope == scope)
                  & (gains.from_k == a) & (gains.to_k == b)].iloc[0]
        return r["se"]

    allb = bud[bud.split == "all"].set_index(["method", "k"])

    # ---------------------------------------------------------------- Table 1
    rows = []
    for m in ORDER:
        r1, r5 = allb.loc[(m, 1)], allb.loc[(m, 5)]
        dr, do = gain(m, "ret_q"), gain(m, "orc_q")
        rows.append(" & ".join([
            LABELS[m], f"{int(cov.loc[m, 'complete_systems']):,}", f(r1["ret_q"]), f(r5["ret_q"]),
            msd(dr[0], gain_se(m, "ret_q")), f(r5["orc_q"]), msd(do[0], gain_se(m, "orc_q")),
            f"{r1['ret_s'] * 100:.1f} $\\to$ {r5['ret_s'] * 100:.1f}",
            f"{r1['orc_s'] * 100:.1f} $\\to$ {r5['orc_s'] * 100:.1f}",
        ]) + r" \\")
    (out / "table_budget.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrlrlll}", r"\toprule",
        r" & & \multicolumn{3}{c}{Returned (native top-1)} & \multicolumn{2}{c}{Oracle best-of-$k$} & \multicolumn{2}{c}{Success, \% (LDDT-PLI $\geq 0.8$)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Method & Systems & $k{=}1$ & $k{=}5$ & $\Delta_{1\to5}$ (mean $\pm$ SE) & $k{=}5$ & $\Delta_{1\to5}$ (mean $\pm$ SE) & Returned & Oracle \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    # ---------------------------------------------------------------- Table 2
    rows = []
    for m in ORDER:
        o = ops[ops.method == m]
        conf = o[o.family == "conf"].iloc[0]
        f1 = o[(o.family == "fixed") & (o.param_k == 1)].iloc[0]
        f2 = o[(o.family == "fixed") & (o.param_k == 2)].iloc[0]
        rows.append(" & ".join([
            LABELS[m], f"{int(conf['n_test']):,}", f(conf["param_tau"], 3),
            f"{conf['test_cost']:.2f} $\\pm$ {conf['test_cost_se']:.2f}",
            f"{conf['test_saving_vs_fixed5'] * 100:.0f}",
            msd(conf["test_loss"], conf["test_loss_se"]),
            msd(conf["test_signed"], conf["test_signed_se"]),
            msd(f1["test_signed"], f1["test_signed_se"]),
            msd(f2["test_signed"], f2["test_signed_se"]),
        ]) + r" \\")
    (out / "table_stopping.tex").write_text("\n".join([
        r"\begin{tabular}{lrrlrllll}", r"\toprule",
        r" & & \multicolumn{5}{c}{Confidence stop (development-selected $\tau$)} & \multicolumn{2}{c}{Signed loss, fixed budget} \\",
        r"\cmidrule(lr){3-7}\cmidrule(lr){8-9}",
        r"Method & Test & $\tau$ & Mean seeds $\pm$ SE & Saving, \% & MPQS $\pm$ SE & Signed loss $\pm$ SE & 1 seed $\pm$ SE & 2 seeds $\pm$ SE \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    # ---------------------------------------------------------------- Table 3
    p = pred.set_index(["method", "model"])
    c12 = chg[(chg.quantity == "returned") & (chg.transition == "1->2")].set_index("method")
    rows = []
    for m in ORDER:
        rows.append(" & ".join([
            LABELS[m],
            f"{rank.loc[m, 'mean_spearman_score_vs_lddt']:.2f}", f"{rank.loc[m, 'mean_spearman_if_spread_gt_0.1']:.2f}",
            f"{rank.loc[m, 'frac_top1_within_0.01_of_best25'] * 100:.0f}",
            f"{rank.loc[m, 'frac_top1_within_0.01_of_best25_if_spread'] * 100:.0f}",
            f"{c12.loc[m, 'frac_improve'] * 100:.1f}", f"{c12.loc[m, 'frac_worsen'] * 100:.1f}",
            f"{p.loc[(m, 'stage_prior'), 'test_auc']:.3f}", f"{p.loc[(m, 'logistic_all_features'), 'test_auc']:.3f}",
        ]) + r" \\")
    (out / "table_selection.tex").write_text("\n".join([
        r"\begin{tabular}{lcccccccc}", r"\toprule",
        r" & \multicolumn{2}{c}{Spearman $\rho$} & \multicolumn{2}{c}{Near-best top-1 (\%)} & \multicolumn{2}{c}{Seed $1\to2$ (\%)} & \multicolumn{2}{c}{Test AUC} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Method & All & Spread $>0.1$ & All & Spread $>0.1$ & Improve & Worsen & Stage prior & Logistic \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    # ---------------------------------------------------------------- Tables S1, S2
    rows = []
    for m in ORDER:
        for i, b in enumerate(["[0,30)", "[30,50)", "[50,70)", "[70,100]"]):
            d = nov[(nov.method == m) & (nov.novelty_bin == b)].set_index("k")
            if d.empty:
                continue
            rows.append(" & ".join([LABELS[m] if i == 0 else "", f"${b}$",
                                    f"{int(d.loc[1, 'n_systems']):,}", f(d.loc[1, "ret_q"]), f(d.loc[5, "ret_q"]),
                                    f(d.loc[1, "orc_q"]), f(d.loc[5, "orc_q"]), f(d.loc[5, "orc_q"] - d.loc[5, "ret_q"])]) + r" \\")
        rows.append(r"\addlinespace")
    (out / "table_novelty.tex").write_text("\n".join([
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r" & & & \multicolumn{2}{c}{Returned} & \multicolumn{2}{c}{Oracle} & Gap \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Method & Similarity & Systems & $k{=}1$ & $k{=}5$ & $k{=}1$ & $k{=}5$ & $k{=}5$ \\",
        r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}", ""]))
    rows = [" & ".join([LABELS[m], f"{int(cov.loc[m, 'systems_in_table']):,}", f"{int(cov.loc[m, 'systems_with_proper_labels']):,}",
                        f"{int(cov.loc[m, 'complete_systems']):,}", f"{int(cov.loc[m, 'rmsd_evaluable_systems']):,}",
                        f"{int(cov.loc[m, 'dev']):,}", f"{int(cov.loc[m, 'test']):,}", f"{int(cov.loc[m, 'clusters']):,}"]) + r" \\"
            for m in ORDER]
    (out / "table_coverage.tex").write_text("\n".join([
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Method & In table & Labeled & Complete & RMSD-evaluable & Dev & Test & Clusters \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    # ---------------------------------------------------------------- macros
    dr = {m: gain(m, "ret_q") for m in ORDER}
    do = {m: gain(m, "orc_q") for m in ORDER}
    dts = {m: gain(m, "ret_q", "test") for m in ORDER}
    ret_s_pts = [(allb.loc[(m, 5)]["ret_s"] - allb.loc[(m, 1)]["ret_s"]) * 100 for m in ORDER]
    orc_s_pts = [(allb.loc[(m, 5)]["orc_s"] - allb.loc[(m, 1)]["orc_s"]) * 100 for m in ORDER]
    conf = ops[ops.family == "conf"].set_index("method").loc[ORDER]
    f1 = ops[(ops.family == "fixed") & (ops.param_k == 1)].set_index("method").loc[ORDER]
    novp = nov.pivot_table(index=["method", "novelty_bin"], columns="k", values=["ret_q", "orc_q"])
    orc_gain_novel = [novp.loc[(m, "[0,30)")]["orc_q"][5] - novp.loc[(m, "[0,30)")]["orc_q"][1] for m in ORDER]
    orc_gain_familiar = [novp.loc[(m, "[70,100]")]["orc_q"][5] - novp.loc[(m, "[70,100]")]["orc_q"][1] for m in ORDER]
    gap_novel = [novp.loc[(m, "[0,30)")]["orc_q"][5] - novp.loc[(m, "[0,30)")]["ret_q"][5] for m in ORDER]
    gap_familiar = [novp.loc[(m, "[70,100]")]["orc_q"][5] - novp.loc[(m, "[70,100]")]["ret_q"][5] for m in ORDER]
    novs = nov[nov.k == 5].set_index(["method", "novelty_bin"])
    lr = pred[pred.model == "logistic_all_features"].set_index("method").loc[ORDER]
    pr = pred[pred.model == "stage_prior"].set_index("method").loc[ORDER]
    macros = {
        "NMethods": str(len(ORDER)),
        "NSystemsRange": f"{int(cov['complete_systems'].min()):,}--{int(cov['complete_systems'].max()):,}",
        "NSystemsTotal": f"{int(cov['complete_systems'].sum()):,}",
        "RetGainRange": rng([v[0] for v in dr.values()]),
        "OrcGainRange": rng([v[0] for v in do.values()]),
        "RetGainTestRange": rng([v[0] for v in dts.values()]),
        "NRetGainTestSignif": str(sum(1 for v in dts.values() if v[1] > 0 or v[2] < 0)),
        "RetSuccPtsRange": rng(ret_s_pts, 1),
        "OrcSuccPtsRange": rng(orc_s_pts, 1),
        "GapFiveRange": rng([allb.loc[(m, 5)]["gap_q"] for m in ORDER]),
        "SpearmanRange": rng(rank["mean_spearman_score_vs_lddt"], 2),
        "SpearmanSpreadRange": rng(rank["mean_spearman_if_spread_gt_0.1"], 2),
        "TopNearBestRange": rng(rank["frac_top1_within_0.01_of_best25"] * 100, 0),
        "TopNearBestSpreadRange": rng(rank["frac_top1_within_0.01_of_best25_if_spread"] * 100, 0),
        "ImproveOneTwoRange": rng(c12.loc[ORDER, "frac_improve"] * 100, 1),
        "WorsenOneTwoRange": rng(c12.loc[ORDER, "frac_worsen"] * 100, 1),
        "AUCLogisticRange": rng(lr["test_auc"], 2),
        "AUCPriorRange": rng(pr["test_auc"], 2),
        "SavingRange": rng(conf["test_saving_vs_fixed5"] * 100, 0),
        "ConfLossRange": rng(conf["test_loss"], 3),
        "ConfMeanBatchesRange": rng(conf["test_cost"], 1),
        "FixOneSignedRange": rng(f1["test_signed"], 3),
        "FixOneLossRange": rng(f1["test_loss"], 3),
        "OrcGainNovelRange": rng(orc_gain_novel, 3),
        "OrcGainFamiliarRange": rng(orc_gain_familiar, 3),
        "GapNovelRange": rng(gap_novel, 3),
        "GapFamiliarRange": rng(gap_familiar, 3),
        "NativeMinusRandomRange": rng(rank["native_k5_minus_random"], 3),
        "HeadroomCapturedRange": rng(rank["headroom_captured_k5"] * 100, 0),
        "NPlateauFeasible": str(int((ops[ops.family == "plateau"]["test_cost"].notna()).sum())),
        "NoNoveltyRange": rng([int(cov.loc[m, "complete_systems"]) - int(nov[(nov.method == m) & (nov.k == 1)]["n_systems"].sum()) for m in ORDER], 0),
        "DevSavingRange": rng((1 - conf["dev_cost"] / 5) * 100, 0),
        "NComboFeasible": str(int((ops[ops.family == "conf_or_plateau"]["test_cost"].notna()).sum())),
        "NRetGainAllSignif": str(sum(1 for v in dr.values() if v[1] > 0 or v[2] < 0)),
        "RetGainAllUpperMax": f"{max(v[2] for v in dr.values()):.3f}",
        "NRetSuccAllSignif": str(sum(1 for m in ORDER if (lambda g: g[1] > 0 or g[2] < 0)(gain(m, "ret_s")))),
        "OrderSensMax": f"{(lambda o: (o['ret_q'] - o['perm_ret_q']).abs().max())(pd.read_csv(t / 'order_sensitivity.csv')):.4f}",
        "PosRateTestRange": rng(pr["pos_rate_test"] * 100, 1),
        "BestScoreAUCRange": rng(pred[pred.model == "single:best_score"].set_index("method").loc[ORDER, "test_auc"], 2),
        "BrierDiffMax": f"{(lr['test_brier'] - pr['test_brier']).abs().max():.3f}",
        "SelExVariant": "AF3 without templates",
        "SelExSpearman": f(rank.loc[SEL_EXAMPLE, "mean_spearman_score_vs_lddt"], 2),
        "SelExTopNearBest": f(rank.loc[SEL_EXAMPLE, "frac_top1_within_0.01_of_best25"] * 100, 0),
        "SelExHeadroom": f(rank.loc[SEL_EXAMPLE, "headroom_captured_k5"] * 100, 0),
        "OrcFiveNovelRange": rng([novs.loc[(m, "[0,30)"), "orc_q"] for m in ORDER], 3),
        "OrcFiveFamiliarRange": rng([novs.loc[(m, "[70,100]"), "orc_q"] for m in ORDER], 3),
        "OrcSuccFiveNovelRange": rng([novs.loc[(m, "[0,30)"), "orc_s"] * 100 for m in ORDER], 0),
        "OrcSuccFiveFamiliarRange": rng([novs.loc[(m, "[70,100]"), "orc_s"] * 100 for m in ORDER], 0),
    }
    robustness_outputs(t, out, macros)
    timing_outputs(t, out, ops)
    (out / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    (out / "numbers.json").write_text(json.dumps(macros, indent=2) + "\n")
    print(json.dumps(macros, indent=2))


if __name__ == "__main__":
    main()
