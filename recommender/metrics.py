import math
import numpy as np


def calculate_ranking_metrics(recommended, relevant, k=10):
    """Tek kullanıcı için Recall@K ve NDCG@K hesapla."""
    if not relevant:
        raise ValueError("En az bir ilgili film gerekli.")

    top_k = recommended[:k]

    hits = [
        1 if movie_id in relevant else 0
        for movie_id in top_k
    ]

    recall = sum(hits) / len(relevant)

    dcg = sum(
        hit / math.log2(position + 2)
        for position, hit in enumerate(hits)
    )

    # İdeal sıralamada ilgili filmler ilk sıralarda bulunur.
    ideal_dcg = sum(
        1 / math.log2(position + 2)
        for position in range(min(k, len(relevant)))
    )

    ndcg = dcg / ideal_dcg

    return recall, ndcg

def paired_bootstrap_ci(
    baseline_scores,
    candidate_scores,
    n_bootstrap=5000,
    seed=42,
):
    baseline = np.asarray(baseline_scores, dtype=float)
    candidate = np.asarray(candidate_scores, dtype=float)

    if (
        baseline.ndim != 1
        or baseline.shape != candidate.shape
        or len(baseline) < 2
    ):
        raise ValueError("Aynı kullanıcılara ait eşleşmiş skorlar gerekli.")

    if not (
        np.isfinite(baseline).all()
        and np.isfinite(candidate).all()
    ):
        raise ValueError("Skorlar sonlu sayılar olmalı.")

    if n_bootstrap < 1:
        raise ValueError("Bootstrap tekrar sayısı pozitif olmalı.")

    differences = candidate - baseline

    rng = np.random.default_rng(seed)

    sampled_indices = rng.integers(
        low=0,
        high=len(differences),
        size=(n_bootstrap, len(differences)),
    )

    sampled_mean_differences = differences[
        sampled_indices
    ].mean(axis=1)

    lower, upper = np.quantile(
        sampled_mean_differences,
        [0.025, 0.975],
    )

    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }