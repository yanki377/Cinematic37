import numpy as np
from .models import recommend_popular

def initialize_mf(
    encoded_train,
    n_users,
    n_movies,
    n_factors=20,
    seed=42,
):
    rng = np.random.default_rng(seed)

    model = {
        "global_mean": float(encoded_train["rating"].mean()),

        "user_factors": rng.normal(
            loc=0.0,
            scale=0.1,
            size=(n_users, n_factors),
        ),

        "movie_factors": rng.normal(
            loc=0.0,
            scale=0.1,
            size=(n_movies, n_factors),
        ),

        "user_bias": np.zeros(n_users),
        "movie_bias": np.zeros(n_movies),
    }

    return model


def predict_mf(model, user_idx, movie_idx):
    interaction = np.dot(
        model["user_factors"][user_idx],
        model["movie_factors"][movie_idx],
    )

    prediction = (
        model["global_mean"]
        + model["user_bias"][user_idx]
        + model["movie_bias"][movie_idx]
        + interaction
    )

    return float(prediction)

def train_mf(
    model,
    encoded_train,
    epochs=10,
    learning_rate=0.01,
    regularization=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)

    user_indices = encoded_train["user_idx"].to_numpy(dtype=np.int64)
    movie_indices = encoded_train["movie_idx"].to_numpy(dtype=np.int64)
    actual_ratings = encoded_train["rating"].to_numpy(dtype=np.float64)

    history = []

    for epoch in range(epochs):
        # Her epoch'ta kayıtları farklı sırayla dolaş.
        shuffled_indices = rng.permutation(len(actual_ratings))

        for row_idx in shuffled_indices:
            user_idx = user_indices[row_idx]
            movie_idx = movie_indices[row_idx]
            actual_rating = actual_ratings[row_idx]

            # İki güncellemede de eski vektörleri kullanacağız.
            user_vector = model["user_factors"][user_idx].copy()
            movie_vector = model["movie_factors"][movie_idx].copy()

            prediction = (
                model["global_mean"]
                + model["user_bias"][user_idx]
                + model["movie_bias"][movie_idx]
                + np.dot(user_vector, movie_vector)
            )

            error = actual_rating - prediction

            model["user_bias"][user_idx] += learning_rate * (
                error
                - regularization * model["user_bias"][user_idx]
            )

            model["movie_bias"][movie_idx] += learning_rate * (
                error
                - regularization * model["movie_bias"][movie_idx]
            )

            model["user_factors"][user_idx] += learning_rate * (
                error * movie_vector
                - regularization * user_vector
            )

            model["movie_factors"][movie_idx] += learning_rate * (
                error * user_vector
                - regularization * movie_vector
            )

        # Epoch tamamlandıktan sonra güncel modelin eğitim hatasını ölç.
        user_vectors = model["user_factors"][user_indices]
        movie_vectors = model["movie_factors"][movie_indices]

        predictions = (
            model["global_mean"]
            + model["user_bias"][user_indices]
            + model["movie_bias"][movie_indices]
            + np.sum(user_vectors * movie_vectors, axis=1)
        )

        train_rmse = float(
            np.sqrt(np.mean((actual_ratings - predictions) ** 2))
        )

        if not np.isfinite(train_rmse):
            raise ValueError("Eğitim sırasında sayısal kararsızlık oluştu.")

        history.append({
            "epoch": epoch + 1,
            "train_rmse": train_rmse,
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Eğitim RMSE: {train_rmse:.4f}"
        )

    return history

def recommend_mf(
    user_id,
    model,
    train_ratings,
    user_to_idx,
    movie_to_idx,
    ranking,
    n=10,
):
    if n < 1:
        raise ValueError("Öneri sayısı en az 1 olmalı.")

    # Eğitimde görülmeyen kullanıcılar için yedek yöntem.
    if user_id not in user_to_idx:
        return recommend_popular(
            user_id, train_ratings, ranking, n
        ).assign(predicted_rating=np.nan)

    user_idx = user_to_idx[user_id]

    seen_movies = set(
        train_ratings.loc[
            train_ratings["userId"] == user_id,
            "movieId",
        ]
    )

    # Diğer modellerle aynı eğitim aday kümesini kullan.
    candidates = ranking[
        ~ranking["movieId"].isin(seen_movies)
    ].copy()

    movie_indices = (
        candidates["movieId"]
        .map(movie_to_idx)
        .to_numpy(dtype=np.int64)
    )

    candidates["predicted_rating"] = (
        model["global_mean"]
        + model["user_bias"][user_idx]
        + model["movie_bias"][movie_indices]
        + model["movie_factors"][movie_indices].dot(
            model["user_factors"][user_idx]
        )
    )

    return (
        candidates.sort_values(
            ["predicted_rating", "rating_count", "mean_rating", "movieId"],
            ascending=[False, False, False, True],
        )
        .head(n)
        .copy()
    )