import numpy as np

from src.metrics import asd_metrics, hmean, knn_scores, mahalanobis_scores, overall_hmean
from src.spectrum import effective_rank, singular_values, spectrum_distance, stable_rank


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    full = rng.standard_normal((200, 16))
    low = rng.standard_normal((200, 1)) @ rng.standard_normal((1, 16)) + 1e-6 * rng.standard_normal((200, 16))
    assert effective_rank(singular_values(full)) > 12
    assert effective_rank(singular_values(low)) < 1.5
    assert stable_rank(singular_values(low)) < 1.5


def test_spectrum_distance_zero_for_same():
    s = singular_values(np.random.default_rng(1).standard_normal((50, 8)))
    assert spectrum_distance(s, s) == 0.0


def test_hmean():
    assert abs(hmean([0.5, 0.5]) - 0.5) < 1e-9
    assert abs(hmean([1.0, 0.5]) - 2 / 3) < 1e-9


def test_scoring_and_metrics_separate_anomalies():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((100, 8)) + 5.0
    normal = rng.standard_normal((40, 8)) + 5.0
    anom = rng.standard_normal((40, 8)) - 5.0
    test = np.concatenate([normal, anom])
    y = np.array([0] * 40 + [1] * 40)
    dom = np.array(([0] * 20 + [1] * 20) * 2)
    for scores in (knn_scores(ref, test, k=2, metric="euclid"), mahalanobis_scores(ref, test)):
        m = asd_metrics(scores, y, dom)
        assert m["auc_source"] > 0.95 and m["auc_target"] > 0.95 and m["pauc"] > 0.9
    assert overall_hmean([m, m]) > 0.9


def test_missing_labels_raise():
    try:
        asd_metrics(np.zeros(4), np.array([-1, -1, -1, -1]), np.array([0, 0, 1, 1]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
