"""Run experiments over machines x fractions x seeds (x lambda sweep). Skips finished runs.

    python -m scripts.run_grid --exps baseline_strong nuclear_emb --machines fan gearbox \
        --fractions 0.05 0.1 0.25 1.0 --seeds 0 1 2 3 4 --out runs/main \
        --sweep reg.nuclear_emb=1e-4,1e-3,1e-2,1e-1 --sweep-exp nuclear_emb
"""
import argparse
import os
import traceback

from src.config import load_config, set_nested
from src.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exps", nargs="+", required=True, help="config names in configs/ (without .yaml)")
    p.add_argument("--machines", nargs="+", required=True)
    p.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.1, 0.25, 1.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="runs/main")
    p.add_argument("--set", nargs="*", default=[], help="overrides applied to every run")
    p.add_argument("--sweep", default=None, help="key=v1,v2,... applied to --sweep-exp only")
    p.add_argument("--sweep-exp", default=None)
    args = p.parse_args()

    jobs = []  # (name, config_path, extra_overrides)
    for e in args.exps:
        path = os.path.join("configs", f"{e}.yaml")
        if args.sweep and e == args.sweep_exp:
            key, vals = args.sweep.split("=", 1)
            for v in vals.split(","):
                jobs.append((f"{e}_{key.split('.')[-1]}{v}", path, [f"{key}={v}"]))
        else:
            jobs.append((e, path, []))

    for name, path, extra in jobs:
        for machine in args.machines:
            for frac in args.fractions:
                for seed in args.seeds:
                    out_dir = os.path.join(args.out, name, machine, f"frac{frac}_seed{seed}")
                    if os.path.exists(os.path.join(out_dir, "results.json")):
                        continue
                    cfg = load_config(path, args.set + extra)
                    set_nested(cfg, "data.machine", machine)
                    set_nested(cfg, "data.fraction", frac)
                    cfg["seed"], cfg["out_dir"] = seed, out_dir
                    print(f"=== {name} | {machine} | frac={frac} | seed={seed}", flush=True)
                    try:
                        run(cfg)
                    except Exception:
                        traceback.print_exc()


if __name__ == "__main__":
    main()
