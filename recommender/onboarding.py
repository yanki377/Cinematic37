import numpy as np

from .hybrid import combine_rankings


def recommend_from_ratings(
    bundle,
    movie_ratings,
    n=10,
    regularization=0.1,
):
    """Yeni kullanıcının puanlarından kişisel öneriler üret."""
    if len(movie_ratings) < 3:
        raise ValueError("En az 3 film seçmelisin.")

    if n < 1 or regularization <= 0:
        raise ValueError("Öneri sayısı ve düzenleme katsayısı pozitif olmalı.")

    preferences = {
        int(movie_id): float(rating)
        for movie_id, rating in movie_ratings.items()
    }

    selected_ids = sorted(preferences)
    movie_to_idx = bundle["movie_to_idx"]
    model = bundle["model"]

    if any(movie_id not in movie_to_idx for movie_id in selected_ids):
        raise ValueError("Seçilen filmlerden biri modelin kataloğunda yok.")

    ratings = np.array([
        preferences[movie_id]
        for movie_id in selected_ids
    ])

    if not np.isfinite(ratings).all() or np.any(
        (ratings < 0.5) | (ratings > 5.0)
    ):
        raise ValueError("Puanlar 0.5–5 arasında olmalı.")

    selected_indices = np.array([
        movie_to_idx[movie_id]
        for movie_id in selected_ids
    ])

    # Film vektörleri sabit; yalnızca yeni kullanıcı vektörünü bul.
    movie_vectors = model["movie_factors"][selected_indices]

    targets = (
        ratings
        - model["global_mean"]
        - model["movie_bias"][selected_indices]
    )

    factor_count = movie_vectors.shape[1]

    user_vector = np.linalg.solve(
        movie_vectors.T @ movie_vectors
        + regularization * np.eye(factor_count),
        movie_vectors.T @ targets,
    )

    # Kullanıcının seçtiği filmleri yeniden önerme.
    popularity_ranked = bundle["ranking"][
        ~bundle["ranking"]["movieId"].isin(selected_ids)
    ].copy()

    candidate_indices = (
        popularity_ranked["movieId"]
        .map(movie_to_idx)
        .to_numpy(dtype=np.int64)
    )

    personal_ranked = popularity_ranked.copy()

    personal_ranked["profile_score"] = (
        model["global_mean"]
        + model["movie_bias"][candidate_indices]
        + model["movie_factors"][candidate_indices].dot(user_vector)
    )

    personal_ranked = personal_ranked.sort_values(
        ["profile_score", "rating_count", "mean_rating", "movieId"],
        ascending=[False, False, False, True],
    )

    return combine_rankings(
        popularity_ranked=popularity_ranked,
        personal_ranked=personal_ranked,
        n=n,
        **bundle["hybrid_settings"],
    )