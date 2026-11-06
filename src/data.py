"""DCASE-style data loading.

ASSUMED LAYOUT (matches DCASE 2023-2025 dev sets; VERIFY against the 2026 release):

    <root>/<machine>/train/section_00_source_train_normal_0001_<attrs>.wav
    <root>/<machine>/test/section_00_source_test_normal_0001_<attrs>.wav
    <root>/<machine>/test/section_00_target_test_anomaly_0001_<attrs>.wav

If your 2026 files are named or organised differently (e.g. the near/far channels are
separate files), change `parse_filename` / `scan_machine` / `AudioDataset._load` only.
Test labels can alternatively be supplied as a CSV (columns: filename,label; 1 = anomaly).
"""
import csv
import glob
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DOMAIN_TO_ID = {"source": 0, "target": 1}


# ---------------------------------------------------------------- file parsing
def parse_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    toks = name.split("_")
    low = [t.lower() for t in toks]
    info = {"path": path, "section": None, "domain": None, "anomaly": None, "attr": ""}
    if "section" in low:
        i = low.index("section")
        if i + 1 < len(toks):
            info["section"] = toks[i + 1]
    for t in low:
        if t in DOMAIN_TO_ID:
            info["domain"] = t
    for j, t in enumerate(low):
        if t in ("normal", "anomaly"):
            info["anomaly"] = int(t == "anomaly")
            info["attr"] = "_".join(toks[j + 2:])  # skip the clip index token
            break
    return info


def _read_labels_csv(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            labels[os.path.basename(row["filename"])] = int(row["label"])
    return labels


def scan_machine(root, machine, labels_csv=None):
    base = os.path.join(root, machine)
    out = {}
    for split in ("train", "test"):
        files = sorted(glob.glob(os.path.join(base, split, "*.wav")))
        out[split] = [parse_filename(f) for f in files]
    if not out["train"]:
        raise FileNotFoundError(f"No wav files under {base}/train - check data.root / data.machine")
    if labels_csv:
        lab = _read_labels_csv(labels_csv)
        for info in out["test"]:
            info["anomaly"] = lab.get(os.path.basename(info["path"]), info["anomaly"])
    return out


# ---------------------------------------------------------------- splits / labels
def _group_key(info):
    return (info["section"], info["domain"])


def split_holdout(infos, frac, seed=0):
    """Fixed (seed-independent) stratified holdout of normal training clips."""
    groups = defaultdict(list)
    for info in infos:
        groups[_group_key(info)].append(info)
    rng = random.Random(seed)
    train, hold = [], []
    for key in sorted(groups, key=str):
        g = groups[key][:]
        rng.shuffle(g)
        n_hold = int(math.ceil(frac * len(g))) if len(g) >= 10 else 0
        hold += g[:n_hold]
        train += g[n_hold:]
    return train, hold


def subsample(infos, fraction, seed, min_per_group=2):
    """Stratified (section, domain) subsample of the normal training clips."""
    if fraction >= 1.0:
        return list(infos)
    groups = defaultdict(list)
    for info in infos:
        groups[_group_key(info)].append(info)
    rng = random.Random(seed)
    keep = []
    for key in sorted(groups, key=str):
        g = groups[key]
        n_keep = min(len(g), max(min_per_group, int(round(fraction * len(g)))))
        keep += rng.sample(g, n_keep)
    return keep


def _label_key(info, mode):
    if mode == "attr":
        return f"{info['section']}|{info['attr']}"
    if mode == "section":
        return str(info["section"])
    if mode == "section_domain":
        return f"{info['section']}|{info['domain']}"
    raise ValueError(f"unknown label_mode {mode}")


def build_label_map(infos, mode):
    keys = sorted({_label_key(i, mode) for i in infos})
    return {k: n for n, k in enumerate(keys)}


def assign_labels(infos, label_map, mode):
    for info in infos:
        info["y"] = label_map.get(_label_key(info, mode), -1)


# ---------------------------------------------------------------- dataset
class AudioDataset(Dataset):
    def __init__(self, infos, audio_cfg, stats=None, train=False):
        import torchaudio
        self.infos, self.cfg, self.stats, self.train = infos, audio_cfg, stats, train
        self.sr = audio_cfg["sample_rate"]
        self.n_samples = int(self.sr * audio_cfg["duration"])
        self.mode = audio_cfg["channel_mode"]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=audio_cfg["n_fft"], hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mels"], f_min=audio_cfg["f_min"], f_max=audio_cfg["f_max"])

    def __len__(self):
        return len(self.infos)

    def _load(self, path):
        import soundfile as sf
        import torchaudio
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if self.mode == "ch1":
            wav = wav[:1]
        elif self.mode == "mean":
            wav = wav.mean(0, keepdim=True)
        elif self.mode == "stack":
            wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
        else:
            raise ValueError(f"unknown channel_mode {self.mode}")
        return wav

    def _fix_length(self, wav):
        n = wav.shape[1]
        if n >= self.n_samples:
            start = random.randint(0, n - self.n_samples) if self.train else (n - self.n_samples) // 2
            return wav[:, start:start + self.n_samples]
        return F.pad(wav, (0, self.n_samples - n))

    def __getitem__(self, i):
        info = self.infos[i]
        wav = self._fix_length(self._load(info["path"]))
        x = torch.log(self.mel(wav) + 1e-6)  # (C, n_mels, T)
        if self.stats is not None:
            x = (x - self.stats["mean"]) / (self.stats["std"] + 1e-6)
        anomaly = -1 if info["anomaly"] is None else info["anomaly"]
        return {
            "x": x,
            "y": torch.tensor(info.get("y", -1), dtype=torch.long),
            "domain": torch.tensor(DOMAIN_TO_ID.get(info["domain"], -1), dtype=torch.long),
            "anomaly": torch.tensor(anomaly, dtype=torch.long),
        }


def compute_stats(dataset, n=100, seed=0):
    """Global scalar mean/std of log-mel over up to n clips (use training clips only)."""
    idx = list(range(len(dataset)))
    random.Random(seed).shuffle(idx)
    vals = torch.cat([dataset[i]["x"].flatten() for i in idx[:n]])
    return {"mean": float(vals.mean()), "std": float(vals.std())}
