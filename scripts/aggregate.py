"""Aggregate runs -> summary table, paired tests vs a baseline, and plots.

    python -m scripts.aggregate --runs runs/main --baseline baseline_strong
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.metrics import overall_hmean


def load_runs(root):
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "frac*_seed*", "results.json")):
        r = json.load(open(f))
        exp, machine = f.split(os.sep)[-4], f.split(os.sep)[-3]
        scorer = r["primary"]
        m = r["metrics"][scorer]
        last = r["svd"][-1] if r["svd"] else {}
        rows.append({"exp": exp, "machine": machine, "fraction": r["config"]["data"]["fraction"],
                     "seed": r["config"]["seed"], "scorer": scorer, **m,
                     "eff_rank_train": last.get("train", {}).get("eff_rank", np.nan),
                     "eff_rank_heldout": last.get("heldout", {}).get("eff_rank", np.nan),
                     "eff_rank_gap": last.get("eff_rank_gap", np.nan)})
    return pd.DataFrame(rows)


def per_seed_overall(df):
    """Official-style score per (exp, fraction, seed): harmonic mean over all machines/domains."""
    out = []
    for (exp, frac, seed), g in df.groupby(["exp", "fraction", "seed"]):
        out.append({"exp": exp, "fraction": frac, "seed": seed, "n_machines": len(g),
                    "overall_hmean": overall_hmean(g.to_dict("records")),
                    "eff_rank_gap": g["eff_rank_gap"].mean()})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--baseline", default="baseline_strong")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or args.runs
    df = load_runs(args.runs)
    if df.empty:
        raise SystemExit("no results found")
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    ps = per_seed_overall(df)
    summ = ps.groupby(["exp", "fraction"]).agg(mean=("overall_hmean", "mean"), std=("overall_hmean", "std"),
                                                n_seeds=("seed", "nunique"),
                                                eff_rank_gap=("eff_rank_gap", "mean")).reset_index()

    # paired comparison vs baseline (matched by seed)
    from scipy import stats
    tests = []
    for (exp, frac), g in ps.groupby(["exp", "fraction"]):
        base = ps[(ps.exp == args.baseline) & (ps.fraction == frac)].set_index("seed")["overall_hmean"]
        cur = g.set_index("seed")["overall_hmean"]
        common = base.index.intersection(cur.index)
        if exp == args.baseline or len(common) < 3:
            tests.append((exp, frac, np.nan, np.nan))
            continue
        d = cur[common] - base[common]
        tests.append((exp, frac, float(d.mean()), float(stats.ttest_rel(cur[common], base[common]).pvalue)))
    summ = summ.merge(pd.DataFrame(tests, columns=["exp", "fraction", "delta_vs_base", "p_paired"]),
                      on=["exp", "fraction"])
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(summ.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for exp, g in summ.groupby("exp"):
        g = g.sort_values("fraction")
        ax.errorbar(g["fraction"] * 100, g["mean"] * 100, yerr=g["std"].fillna(0) * 100, marker="o", capsize=3, label=exp)
    ax.set_xscale("log")
    ax.set_xlabel("% of normal training clips")
    ax.set_ylabel("official-style score (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "perf_vs_data.png"), dpi=150)


if __name__ == "__main__":
    main()
