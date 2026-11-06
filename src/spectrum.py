"""Pure-numpy singular-value spectrum metrics for embedding matrices (samples x dim)."""
import numpy as np

EPS = 1e-12


def singular_values(emb, center=True):
    X = np.asarray(emb, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return np.linalg.svd(X, compute_uv=False)


def effective_rank(s):
    """exp(entropy of normalised singular values) (Roy & Vetterli, 2007)."""
    p = s / (s.sum() + EPS)
    p = p[p > EPS]
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(s):
    return float((s ** 2).sum() / (s[0] ** 2 + EPS))


def participation_ratio(s):
    s2 = s ** 2
    return float(s2.sum() ** 2 / ((s2 ** 2).sum() + EPS))


def spectrum_distance(s1, s2):
    """L1 distance between normalised spectra (zero-padded to equal length)."""
    n = max(len(s1), len(s2))
    a = np.zeros(n)
    b = np.zeros(n)
    a[: len(s1)] = s1 / (s1.sum() + EPS)
    b[: len(s2)] = s2 / (s2.sum() + EPS)
    return float(np.abs(a - b).sum())


def summarize(s, topk=16):
    return {
        "eff_rank": effective_rank(s),
        "stable_rank": stable_rank(s),
        "part_ratio": participation_ratio(s),
        "top_sv": [float(x) for x in s[:topk]],
    }
