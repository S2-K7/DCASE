"""Train one (machine, data-fraction, seed, config) run and evaluate it.

    python -m src.train --config configs/baseline_strong.yaml --set data.machine=fan data.fraction=0.1
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .augment import mixup, smooth_onehot, soft_cross_entropy, spec_augment
from .config import load_config
from .data import (AudioDataset, assign_labels, build_label_map, compute_stats, scan_machine,
                   split_holdout, subsample)
from .metrics import asd_metrics, knn_scores, mahalanobis_scores
from .models import build_model
from .regularizers import (SpectralNormPenalty, apply_spectral_norm, nuclear_norm_penalty,
                           nuclear_norm_weights)
from .spectrum import singular_values, summarize
from .svd_logger import SVDLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_data(cfg):
    d, a, t = cfg["data"], cfg["audio"], cfg["train"]
    scan = scan_machine(d["root"], d["machine"], d.get("test_labels_csv"))
    pool, holdout = split_holdout(scan["train"], d["holdout_frac"])  # fixed holdout
    sub = subsample(pool, d["fraction"], cfg["seed"], d["min_per_group"])
    label_map = build_label_map(sub, d["label_mode"])
    if len(label_map) < 2:
        raise ValueError(
            f"Only {len(label_map)} proxy class(es) found with label_mode='{d['label_mode']}'. The proxy "
            "classification task needs >= 2 classes. Try another data.label_mode or edit parse_filename().")
    for lst in (sub, holdout, scan["test"]):
        assign_labels(lst, label_map, d["label_mode"])

    stats = compute_stats(AudioDataset(sub, a), n=100)
    ds_train = AudioDataset(sub, a, stats, train=True)
    ds_ref = AudioDataset(sub, a, stats)
    ds_hold = AudioDataset(holdout, a, stats)
    ds_test = AudioDataset(scan["test"], a, stats)

    bs, nw = t["batch_size"], t["num_workers"]
    steps = math.ceil(len(ds_train) / bs)
    repeat = max(1, math.ceil(t["min_steps_per_epoch"] / steps))  # equal optimisation budget across fractions
    train_ds = ConcatDataset([ds_train] * repeat)
    mk = lambda ds, shuffle, drop=False: DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop,
                                                    num_workers=nw)
    loaders = {
        "train": mk(train_ds, True, drop=len(train_ds) >= bs),
        "ref": mk(ds_ref, False), "hold": mk(ds_hold, True), "test": mk(ds_test, False),
    }
    info = {"n_train": len(sub), "n_holdout": len(holdout), "n_test": len(scan["test"]),
            "num_classes": len(label_map), "repeat": repeat}
    return loaders, info


@torch.no_grad()
def embed_loader(model, loader, device):
    model.eval()
    embs, anomaly, domain = [], [], []
    for b in loader:
        embs.append(model.embed(b["x"].to(device)).float().cpu().numpy())
        anomaly.append(b["anomaly"].numpy())
        domain.append(b["domain"].numpy())
    return np.concatenate(embs), np.concatenate(anomaly), np.concatenate(domain)


def run(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t, reg, ev = cfg["train"], cfg["reg"], cfg["eval"]

    loaders, info = build_data(cfg)
    in_ch = 2 if cfg["audio"]["channel_mode"] == "stack" else 1
    model = build_model(cfg, in_ch, info["num_classes"])
    if reg["spectral_norm"]:
        apply_spectral_norm(model, reg["scope"])
    model.to(device)
    spec_pen = SpectralNormPenalty(model, reg["scope"]) if reg["spectral_penalty"] > 0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    svd = SVDLogger(loaders["ref"], loaders["hold"], os.path.join(out_dir, "svd.jsonl"),
                    every=cfg["svd"]["every"], max_samples=cfg["svd"]["max_samples"], device=device)

    history, t0 = [], time.time()
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        sums = {"ce": 0.0, "nuc_emb": 0.0, "nuc_w": 0.0, "spec": 0.0}
        n_batches = 0
        for b in loaders["train"]:
            x, y = b["x"].to(device), b["y"].to(device)
            if t["specaugment"]:
                x = spec_augment(x, t["freq_mask"], t["time_mask"], t["n_masks"])
            y_soft = smooth_onehot(y, info["num_classes"], t["label_smoothing"])
            if t["mixup_alpha"] > 0:
                x, y_soft = mixup(x, y_soft, t["mixup_alpha"])
            emb, logits = model(x)
            ce = soft_cross_entropy(logits, y_soft)
            loss = ce
            sums["ce"] += ce.item()
            if reg["nuclear_emb"] > 0:
                r = nuclear_norm_penalty(emb)
                loss = loss + reg["nuclear_emb"] * r
                sums["nuc_emb"] += r.item()
            if reg["nuclear_weights"] > 0:
                r = nuclear_norm_weights(model, reg["scope"])
                loss = loss + reg["nuclear_weights"] * r
                sums["nuc_w"] += r.item()
            if spec_pen is not None:
                r = spec_pen()
                loss = loss + reg["spectral_penalty"] * r
                sums["spec"] += r.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_batches += 1
        sched.step()
        rec = {"epoch": epoch, **{k: v / max(1, n_batches) for k, v in sums.items()}}
        history.append(rec)
        svd.maybe_log(model, epoch, force=(epoch == t["epochs"]))
        print(f"[{epoch:3d}/{t['epochs']}] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch")
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    # ---- evaluation: final-epoch model only (no test-based model selection)
    ref, _, _ = embed_loader(model, loaders["ref"], device)
    test_emb, anomaly, domain = embed_loader(model, loaders["test"], device)
    metrics = {
        "knn": asd_metrics(knn_scores(ref, test_emb, ev["knn_k"], ev["metric"]), anomaly, domain),
        "maha": asd_metrics(mahalanobis_scores(ref, test_emb), anomaly, domain),
    }
    s_ref = singular_values(ref)
    results = {"config": cfg, "data_info": info, "metrics": metrics, "primary": ev["primary"],
               "final_ref_spectrum": summarize(s_ref), "history": history,
               "svd": svd.records, "train_seconds": time.time() - t0}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("RESULT", json.dumps(metrics[ev["primary"]]))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=10")
    args = p.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
