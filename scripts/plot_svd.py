"""Plot effective-rank trajectories (train vs held-out) for runs, one line per experiment.

    python -m scripts.plot_svd --runs runs/main --machine fan --fraction 0.1 --out svd_fan_0.1.png
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/main")
    p.add_argument("--machine", required=True)
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--out", default="svd_plot.png")
    args = p.parse_args()

    curves = defaultdict(list)
    for f in glob.glob(os.path.join(args.runs, "*", args.machine, f"frac{args.fraction}_seed*", "svd.jsonl")):
        exp = f.split(os.sep)[-4]
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            curves[exp].append(([r["epoch"] for r in recs], [r["train"]["eff_rank"] for r in recs],
                                [r["heldout"]["eff_rank"] for r in recs]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for exp, runs in curves.items():
        n = min(len(r[0]) for r in runs)
        ep = runs[0][0][:n]
        for ax, k, title in ((axes[0], 1, "train embeddings"), (axes[1], 2, "held-out embeddings")):
            arr = np.array([r[k][:n] for r in runs])
            ax.plot(ep, arr.mean(0), label=exp)
            ax.fill_between(ep, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.2)
            ax.set_title(f"effective rank: {title}")
            ax.set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
