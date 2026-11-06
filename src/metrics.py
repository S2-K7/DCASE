"""Anomaly scoring + DCASE-style metrics (numpy / scikit-learn only)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def _l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_scores(ref, test, k=1, metric="cosine"):
    """Mean distance to the k nearest normal reference embeddings (higher = more anomalous)."""
    if metric == "cosine":
        ref, test = _l2norm(ref), _l2norm(test)
    k = min(k, len(ref))
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    dist, _ = nn.kneighbors(test)
    return dist.mean(axis=1)


def mahalanobis_scores(ref, test, shrinkage=1e-2):
    """Squared Mahalanobis distance to the reference distribution, with covariance shrinkage."""
    mu = ref.mean(axis=0)
    cov = np.cov(ref.T) if len(ref) > 1 else np.eye(ref.shape[1])
    d = cov.shape[0]
    cov = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * np.eye(d)
    prec = np.linalg.pinv(cov)
    diff = test - mu
    return np.einsum("ij,jk,ik->i", diff, prec, diff)


def hmean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(len(values) / np.sum(1.0 / np.maximum(values, 1e-12)))


def asd_metrics(scores, anomaly, domain, max_fpr=0.1):
    """DCASE-style metrics for ONE machine type.

    anomaly: 1 = anomalous, 0 = normal. domain: 0 = source, 1 = target.
    Returns AUC (source), AUC (target), pAUC over all test clips (max_fpr=0.1,
    McClish-standardised as in sklearn), and their harmonic mean."""
    scores, anomaly, domain = map(np.asarray, (scores, anomaly, domain))
    if (anomaly < 0).any():
        raise ValueError("Test labels missing (anomaly=-1). Provide data.test_labels_csv or "
                         "use files whose names contain 'normal'/'anomaly'.")
    out = {}
    for name, d in (("auc_source", 0), ("auc_target", 1)):
        m = domain == d
        out[name] = float(roc_auc_score(anomaly[m], scores[m])) if len(np.unique(anomaly[m])) == 2 else float("nan")
    out["pauc"] = float(roc_auc_score(anomaly, scores, max_fpr=max_fpr))
    vals = [v for v in out.values() if not np.isnan(v)]
    out["hmean"] = hmean(vals)
    return out


def overall_hmean(per_machine):
    """Harmonic mean of all (AUC_source, AUC_target, pAUC) values across machine types."""
    vals = []
    for m in per_machine:
        vals += [m[k] for k in ("auc_source", "auc_target", "pauc") if not np.isnan(m[k])]
    return hmean(vals)
