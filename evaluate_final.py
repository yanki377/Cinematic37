from pathlib import Path

import pandas as pd

from recommender.config import MF_INITIALIZATION, MF_TRAINING
from recommender.data import encode_training_data
from recommender.matrix_factorization import initialize_mf, train_mf

from recommender.config import HYBRID_SETTINGS
from recommender.hybrid import combine_rankings
from recommender.matrix_factorization import recommend_mf
from recommender.metrics import calculate_ranking_metrics
from recommender.models import recommend_popular
import json
from hashlib import sha256

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ml-latest-small"

movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")

sorted_ratings = ratings.sort_values(
    ["timestamp", "userId", "movieId"]
).reset_index(drop=True)

# Önceki doğrulama bölümünün bitiş sınırını yeniden oluştur.
cutoff_position = int(len(sorted_ratings) * 0.90) - 1

development_cutoff = int(
    sorted_ratings["timestamp"].iloc[cutoff_position]
)

# Önceki eğitim + doğrulama bölümleri.
development = sorted_ratings[
    sorted_ratings["timestamp"] <= development_cutoff
].copy()

# Önceden ayırdığımız test bölümü.
final_test = sorted_ratings[
    sorted_ratings["timestamp"] > development_cutoff
].copy()

assert len(development) + len(final_test) == len(ratings)
assert (
    development["timestamp"].max()
    < final_test["timestamp"].min()
)

print("--- SON EĞİTİM VERİSİ ---")
print("Eğitim + doğrulama kaydı:", len(development))
print("Ayrılmış test kaydı:", len(final_test))

# Eklenen kullanıcı ve filmler için eşlemeleri yeniden oluştur.
encoded_development, final_user_to_idx, final_movie_to_idx = (
    encode_training_data(development)
)

print("Kullanıcı sayısı:", len(final_user_to_idx))
print("Film sayısı:", len(final_movie_to_idx))

# Yeni, başlangıç durumunda bir model oluştur.
final_mf_model = initialize_mf(
    encoded_train=encoded_development,
    n_users=len(final_user_to_idx),
    n_movies=len(final_movie_to_idx),
    **MF_INITIALIZATION,
)

print("\n--- SON MODELİN EĞİTİMİ ---")

final_training_history = train_mf(
    model=final_mf_model,
    encoded_train=encoded_development,
    **MF_TRAINING,
)

final_reports_dir = PROJECT_DIR / "reports" / "final_test"
final_reports_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(final_training_history).to_csv(
    final_reports_dir / "training_history.csv",
    index=False,
)

print("\nSon model eğitildi. Henüz test skoru hesaplanmadı.")

# Popülerliği de yeni eğitim verisinden yeniden hesapla.
final_popularity_ranking = (
    development.groupby("movieId")
    .agg(
        rating_count=("rating", "size"),
        mean_rating=("rating", "mean"),
    )
    .reset_index()
    .merge(
        movies[["movieId", "title"]],
        on="movieId",
        validate="one_to_one",
    )
    .sort_values(
        ["rating_count", "mean_rating", "movieId"],
        ascending=[False, False, True],
    )
    .reset_index(drop=True)
)

final_history = (
    development.groupby("userId")["movieId"]
    .agg(set)
    .to_dict()
)

test_positives = final_test[final_test["rating"] >= 4.0]

eligible_users = sorted(
    set(test_positives["userId"]) & set(final_history)
)

if not eligible_users:
    raise ValueError("Test değerlendirmesine uygun kullanıcı bulunamadı.")

candidate_movie_ids = set(final_popularity_ranking["movieId"])
full_ranking_size = len(final_popularity_ranking)

test_rows = []
total_relevant_pairs = 0
cold_relevant_pairs = 0

for user_id in eligible_users:
    relevant_movies = (
        set(
            test_positives.loc[
                test_positives["userId"] == user_id,
                "movieId",
            ]
        )
        - final_history[user_id]
    )

    if not relevant_movies:
        continue

    # Eğitimde bulunmayan pozitif filmleri hedeflerden çıkarmıyoruz.
    cold_count = len(relevant_movies - candidate_movie_ids)
    total_relevant_pairs += len(relevant_movies)
    cold_relevant_pairs += cold_count

    popularity_full = recommend_popular(
        user_id=user_id,
        train_ratings=development,
        ranking=final_popularity_ranking,
        n=full_ranking_size,
    )

    mf_full = recommend_mf(
        user_id=user_id,
        model=final_mf_model,
        train_ratings=development,
        user_to_idx=final_user_to_idx,
        movie_to_idx=final_movie_to_idx,
        ranking=final_popularity_ranking,
        n=full_ranking_size,
    )

    ranked_models = {
        "popularity": popularity_full.head(10),

        "hybrid_pop_mf": combine_rankings(
            popularity_ranked=popularity_full,
            personal_ranked=mf_full,
            n=10,
            **HYBRID_SETTINGS,
        ),
    }

    for model_name, recommendations in ranked_models.items():
        recommended_ids = recommendations["movieId"].tolist()

        assert len(recommended_ids) == len(set(recommended_ids))
        assert not (
            set(recommended_ids) & final_history[user_id]
        )

        recall, ndcg = calculate_ranking_metrics(
            recommended=recommended_ids,
            relevant=relevant_movies,
            k=10,
        )

        test_rows.append({
            "model": model_name,
            "userId": user_id,
            "relevant_count": len(relevant_movies),
            "cold_relevant_count": cold_count,
            "recall_at_10": recall,
            "ndcg_at_10": ndcg,
        })

final_user_metrics = pd.DataFrame(test_rows)

if final_user_metrics.empty:
    raise ValueError("Test değerlendirmesinde sonuç üretilemedi.")

final_summary = (
    final_user_metrics
    .groupby("model", as_index=False, sort=False)
    .agg(
        recall_at_10=("recall_at_10", "mean"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        evaluated_users=("userId", "nunique"),
    )
)

evaluated_users = set(final_user_metrics["userId"])

for model_name, rows in final_user_metrics.groupby("model"):
    assert set(rows["userId"]) == evaluated_users

test_users = set(final_test["userId"])
development_users = set(final_history)

print("\n--- TEST DEĞERLENDİRME KAPSAMI ---")
print("Toplam test kullanıcısı:", len(test_users))
print("Değerlendirilen kullanıcı:", len(evaluated_users))
print(
    "Eğitim geçmişi olmayan kullanıcı:",
    len(test_users - development_users),
)
print(
    "Geçmişi olup uygun pozitif hedefi olmayan kullanıcı:",
    len((test_users & development_users) - evaluated_users),
)
print("Pozitif kullanıcı-film hedefi:", total_relevant_pairs)
print("Eğitimde bulunmayan pozitif hedef çifti:", cold_relevant_pairs)

print("\n--- SON TEST SONUÇLARI ---")
print(final_summary.round(4).to_string(index=False))

final_summary.to_csv(
    final_reports_dir / "model_comparison.csv",
    index=False,
)

final_user_metrics.to_csv(
    final_reports_dir / "user_metrics.csv",
    index=False,
)


artifact_dir = PROJECT_DIR / "artifacts" / "final_model"
artifact_dir.mkdir(parents=True, exist_ok=True)

# Öğrenilmiş sayısal parametreler.
np.savez_compressed(
    artifact_dir / "mf_model.npz",
    **final_mf_model,
)

# JSON anahtarları metin olmalı.
mappings = {
    "user_to_idx": {
        str(int(user_id)): int(idx)
        for user_id, idx in final_user_to_idx.items()
    },
    "movie_to_idx": {
        str(int(movie_id)): int(idx)
        for movie_id, idx in final_movie_to_idx.items()
    },
}

(artifact_dir / "mappings.json").write_text(
    json.dumps(mappings, indent=2),
    encoding="utf-8",
)

# Film bilgileri ve eğitimden hesaplanan popülerlik sıralaması.
final_popularity_ranking.to_csv(
    artifact_dir / "popularity_ranking.csv",
    index=False,
)

# Önerilerden daha önce puanlanmış filmleri çıkarmak için.
development[["userId", "movieId"]].to_csv(
    artifact_dir / "seen_interactions.csv",
    index=False,
)

metadata = {
    "format_version": 1,
    "mf_initialization": MF_INITIALIZATION,
    "mf_training": MF_TRAINING,
    "hybrid_settings": HYBRID_SETTINGS,
    "training_cutoff_unix": int(development_cutoff),
    "training_ratings": len(development),
    "training_users": len(final_user_to_idx),
    "training_movies": len(final_movie_to_idx),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "source_sha256": {
        filename: sha256(
            (DATA_DIR / filename).read_bytes()
        ).hexdigest()
        for filename in ["movies.csv", "ratings.csv"]
    },
}

(artifact_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nModel dosyaları kaydedildi:")
print(artifact_dir)