"""Logs the singular-value spectrum of embeddings on train vs held-out normal clips during training."""
import json

import numpy as np
import torch

from .spectrum import singular_values, summarize, spectrum_distance


class SVDLogger:
    def __init__(self, train_loader, heldout_loader, out_path, every=5, max_samples=512, device="cpu"):
        self.train_loader, self.heldout_loader = train_loader, heldout_loader
        self.out_path, self.every, self.max_samples, self.device = out_path, every, max_samples, device
        self.records = []
        open(out_path, "w").close()

    @torch.no_grad()
    def _collect(self, model, loader):
        embs, n = [], 0
        for b in loader:
            embs.append(model.embed(b["x"].to(self.device)).float().cpu().numpy())
            n += len(embs[-1])
            if n >= self.max_samples:
                break
        return np.concatenate(embs)[: self.max_samples]

    def maybe_log(self, model, epoch, force=False):
        if not force and epoch % self.every != 0:
            return None
        was_training = model.training
        model.eval()
        tr, ho = self._collect(model, self.train_loader), self._collect(model, self.heldout_loader)
        n = min(len(tr), len(ho))  # equal sample counts for a fair comparison
        s_tr, s_ho = singular_values(tr[:n]), singular_values(ho[:n])
        a, b = summarize(s_tr), summarize(s_ho)
        rec = {"epoch": epoch, "n": n,
               "train": a, "heldout": b,
               "eff_rank_gap": a["eff_rank"] - b["eff_rank"],
               "spectrum_dist": spectrum_distance(s_tr, s_ho)}
        self.records.append(rec)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if was_training:
            model.train()
        return rec
