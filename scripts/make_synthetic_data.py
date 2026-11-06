"""Generate a tiny synthetic DCASE-like dataset to smoke-test the whole pipeline (not for results)."""
import argparse
import os

import numpy as np
import soundfile as sf


def make_clip(rng, sr, dur, f0, domain, anomaly):
    t = np.arange(int(sr * dur)) / sr
    sig = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3))
    noise = 0.3 if domain == "source" else 0.8
    if domain == "target":
        f0 *= 1.05
    near = sig + noise * rng.standard_normal(len(t))
    far = 0.4 * sig + 1.5 * noise * rng.standard_normal(len(t))
    if anomaly:
        burst = (rng.random(len(t)) < 0.0005) * rng.standard_normal(len(t)) * 8
        near = near + np.convolve(burst, np.hanning(200), mode="same") + 0.5 * np.sin(2 * np.pi * 2.7 * f0 * t)
    x = np.stack([near, far], axis=1).astype(np.float32)
    return x / (np.abs(x).max() + 1e-6) * 0.9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--machine", default="synthmachine")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--dur", type=float, default=2.0)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    speeds = {"1": 100.0, "2": 130.0, "3": 170.0}
    for split in ("train", "test"):
        os.makedirs(os.path.join(args.out, args.machine, split), exist_ok=True)

    def write(split, domain, status, i, speed):
        name = f"section_00_{domain}_{split}_{status}_{i:04d}_speed-{speed}.wav"
        clip = make_clip(rng, args.sr, args.dur, speeds[speed], domain, status == "anomaly")
        sf.write(os.path.join(args.out, args.machine, split, name), clip, args.sr)

    k = 0
    for domain, n in (("source", 120), ("target", 15)):
        for i in range(n):
            write("train", domain, "normal", k, str(1 + i % 3)); k += 1
    for domain in ("source", "target"):
        for status in ("normal", "anomaly"):
            for i in range(20):
                write("test", domain, status, k, str(1 + i % 3)); k += 1
    print("wrote synthetic data to", os.path.join(args.out, args.machine))


if __name__ == "__main__":
    main()
