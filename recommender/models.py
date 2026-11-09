import math

def recommend_popular(user_id, train_ratings, ranking, n=10):
    """Kullanıcının geçmişinde bulunmayan popüler filmleri öner."""
    if n < 1:
        raise ValueError("Öneri sayısı en az 1 olmalı.")

    seen_movies = set(
        train_ratings.loc[
            train_ratings["userId"] == user_id,
            "movieId",
        ]
    )

    candidates = ranking[
        ~ranking["movieId"].isin(seen_movies)
    ]

    return candidates.head(n).copy()



def recommend_content(user_id, train_ratings, features, ranking, n=10):
    """Tür profili ve cosine similarity ile film öner."""
    if n < 1:
        raise ValueError("Öneri sayısı en az 1 olmalı.")

    user_history = train_ratings[
        train_ratings["userId"] == user_id
    ]

    seen_movies = set(user_history["movieId"])

    liked_ratings = (
        user_history[user_history["rating"] >= 4.0]
        .set_index("movieId")["rating"]
    )

    # Beğenilmiş film yoksa kişisel tür profili oluşturamayız.
    if liked_ratings.empty:
        return recommend_popular(
            user_id, train_ratings, ranking, n
        ).assign(similarity=float("nan"))

    # Ağırlıklar: 4 puan -> 1, 4.5 -> 1.5, 5 -> 2.
    weights = liked_ratings - 3.0

    user_profile = (
        features.loc[liked_ratings.index]
        .mul(weights, axis=0)
        .sum(axis=0)
    )

    profile_length = math.sqrt(user_profile.pow(2).sum())

    # Beğenilen filmlerin tamamının tür bilgisi eksik olabilir.
    if profile_length == 0:
        return recommend_popular(
            user_id, train_ratings, ranking, n
        ).assign(similarity=float("nan"))

    user_profile = user_profile / profile_length

    # İki taraf da normalize olduğu için noktasal çarpım
    # cosine similarity değerini verir.
    similarities = features.dot(user_profile).rename("similarity")

    candidates = ranking.join(
        similarities,
        on="movieId",
        validate="one_to_one",
    )

    candidates = candidates[
        ~candidates["movieId"].isin(seen_movies)
    ]

    return (
        candidates.sort_values(
            ["similarity", "rating_count", "mean_rating", "movieId"],
            ascending=[False, False, False, True],
        )
        .head(n)
        .copy()
    )