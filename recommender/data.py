def encode_training_data(train_ratings):
    # Eğitimde bulunan kullanıcı ve filmleri sırala.
    user_ids = sorted(train_ratings["userId"].unique())
    movie_ids = sorted(train_ratings["movieId"].unique())

    # Orijinal ID -> model indeksi eşlemeleri.
    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    movie_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }

    encoded = train_ratings[
        ["userId", "movieId", "rating"]
    ].copy()

    encoded["user_idx"] = encoded["userId"].map(user_to_idx)
    encoded["movie_idx"] = encoded["movieId"].map(movie_to_idx)

    return encoded, user_to_idx, movie_to_idx