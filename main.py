from pathlib import Path
import numpy as np
from recommender.hybrid import combine_rankings
from recommender.metrics import paired_bootstrap_ci
from recommender.config import (
    MF_INITIALIZATION,
    MF_TRAINING,
    HYBRID_SETTINGS,
)

import pandas as pd
from recommender.matrix_factorization import (
    initialize_mf,
    predict_mf,
    train_mf,
    recommend_mf

)
import matplotlib.pyplot as plt
import math
from recommender.data import encode_training_data

from recommender.models import recommend_popular, recommend_content
from recommender.metrics import calculate_ranking_metrics

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ml-latest-small"

movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")

print("Film tablosunun boyutu:", movies.shape)
print("Puan tablosunun boyutu:", ratings.shape)

print("\nİlk 5 film:")
print(movies.head())

print("\nİlk 5 puan:")
print(ratings.head())

print("\nBenzersiz kullanıcı sayısı:", ratings["userId"].nunique())

print("\n--- EKSİK DEĞERLER ---")
print("\nFilmler:")
print(movies.isna().sum())

print("\nPuanlar:")
print(ratings.isna().sum())


print("\n--- TEKRAR EDEN KAYITLAR ---")
print("Tekrarlanan film ID sayısı:", movies["movieId"].duplicated().sum())

duplicate_ratings = ratings.duplicated(subset=["userId", "movieId"]).sum()
print("Tekrarlanan kullanıcı-film çifti sayısı:", duplicate_ratings)


print("\n--- TABLOLAR ARASI TUTARLILIK ---")
unknown_movies = ~ratings["movieId"].isin(movies["movieId"])
print("Film tablosunda bulunmayan filmlere verilen puanlar:", unknown_movies.sum())


print("\n--- PUAN DEĞERLERİ ---")
print(sorted(ratings["rating"].unique().tolist()))


print("\n--- TÜR BİLGİSİ ---")
missing_genres = movies["genres"].eq("(no genres listed)").sum()
print("Tür bilgisi belirtilmemiş film sayısı:", missing_genres)


print("\n--- KULLANICI BAŞINA PUAN SAYISI ---")
ratings_per_user = ratings.groupby("userId").size()
print(ratings_per_user.describe())


print("\n--- PUANLANAN FİLMLER ---")
print("En az bir puanı olan film sayısı:", ratings["movieId"].nunique())


# Her filmin puan sayısını ve ortalama puanını hesapla.
movie_stats = (
    ratings.groupby("movieId")
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
)

top_movies = (
    movie_stats
    .sort_values(
        ["rating_count", "movieId"],
        ascending=[False, True],
    )
    .head(10)
)

print("\n--- EN ÇOK PUANLANAN 10 FİLM ---")
print(
    top_movies[["title", "rating_count", "mean_rating"]]
    .round(2)
    .to_string(index=False)
)

# Puan değerlerini küçükten büyüğe sırala.
rating_counts = ratings["rating"].value_counts().sort_index()

fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 6),
    constrained_layout=True,
)

axes[0].bar(rating_counts.index, rating_counts.values, width=0.35)
axes[0].set_title("Puan dağılımı")
axes[0].set_xlabel("Puan")
axes[0].set_ylabel("Kayıt sayısı")
axes[0].set_xticks(rating_counts.index)

# En çok puanlanan film grafiğin üstünde görünsün.
plot_movies = top_movies.iloc[::-1]

axes[1].barh(plot_movies["title"], plot_movies["rating_count"])
axes[1].set_title("En çok puanlanan 10 film")
axes[1].set_xlabel("Puan sayısı")

figures_dir = PROJECT_DIR / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

figure_path = figures_dir / "eda_overview.png"
fig.savefig(figure_path, dpi=150, bbox_inches="tight")

print("\nGrafik kaydedildi:", figure_path)
plt.show()

# Timestamp, puanın verildiği zamanı Unix saniyesi olarak tutuyor.
ratings["rated_at"] = pd.to_datetime(
    ratings["timestamp"],
    unit="s",
    utc=True,
)

sorted_ratings = ratings.sort_values(
    ["timestamp", "userId", "movieId"]
).reset_index(drop=True)

# Bölme sınırlarını belirle.
train_position = int(len(sorted_ratings) * 0.80) - 1
validation_position = int(len(sorted_ratings) * 0.90) - 1

train_cutoff = sorted_ratings.iloc[train_position]["timestamp"]
validation_cutoff = sorted_ratings.iloc[validation_position]["timestamp"]

# Aynı saniyede verilen puanları aynı bölümde tut.
train = sorted_ratings[
    sorted_ratings["timestamp"] <= train_cutoff
].copy()

validation = sorted_ratings[
    (sorted_ratings["timestamp"] > train_cutoff)
    & (sorted_ratings["timestamp"] <= validation_cutoff)
].copy()

test = sorted_ratings[
    sorted_ratings["timestamp"] > validation_cutoff
].copy()

# Kayıt kaybetmediğimizi ve zaman sırasını koruduğumuzu doğrula.
assert len(train) + len(validation) + len(test) == len(ratings)
assert train["timestamp"].max() < validation["timestamp"].min()
assert validation["timestamp"].max() < test["timestamp"].min()

print("\n--- VERİ BÖLÜMLERİ ---")

for name, frame in [
    ("Eğitim", train),
    ("Doğrulama", validation),
    ("Test", test),
]:
    print(f"\n{name}:")
    print("Puan sayısı:", len(frame))
    print("Kullanıcı sayısı:", frame["userId"].nunique())
    print("Başlangıç:", frame["rated_at"].min())
    print("Bitiş:", frame["rated_at"].max())

# Doğrulamadaki kullanıcıların kaçının geçmişi var?
train_users = set(train["userId"])
validation_users = set(validation["userId"])

print("\n--- DOĞRULAMA KULLANICILARI ---")
print(
    "Eğitimde geçmişi olan:",
    len(validation_users & train_users),
)
print(
    "Eğitimde geçmişi olmayan:",
    len(validation_users - train_users),
)

# Popülerlik istatistiklerini yalnızca eğitim verisinden hesapla.
popularity_ranking = (
    train.groupby("movieId")
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




# Eğitimde bulunan bir kullanıcıyla deneyelim.
example_user = int(train["userId"].min())

recommendations = recommend_popular(
    user_id=example_user,
    train_ratings=train,
    ranking=popularity_ranking,
    n=10,
)

# Kullanıcının daha önce puanladığı bir filmi önermediğimizi kontrol et.
seen_movies = set(
    train.loc[train["userId"] == example_user, "movieId"]
)

assert not recommendations["movieId"].isin(seen_movies).any()

print("\n--- POPÜLERLİK TABANLI ÖNERİLER ---")
print("Kullanıcı ID:", example_user)
print("Eğitimde puanladığı film sayısı:", len(seen_movies))

print(
    recommendations[["title", "rating_count", "mean_rating"]]
    .round(2)
    .to_string(index=False)
)



# Her kullanıcının eğitim dönemindeki geçmişini hazırla.
train_history = (
    train.groupby("userId")["movieId"]
    .agg(set)
    .to_dict()
)

# Doğrulama döneminde beğenildiği bilinen filmler.
validation_positives = validation[
    validation["rating"] >= 4.0
]

user_results = []

for user_id, user_rows in validation_positives.groupby("userId"):
    # Bu deneyde yalnızca eğitim geçmişi olanları değerlendir.
    if user_id not in train_history:
        continue

    relevant_movies = (
        set(user_rows["movieId"]) - train_history[user_id]
    )

    if not relevant_movies:
        continue

    recommended_ids = recommend_popular(
        user_id=user_id,
        train_ratings=train,
        ranking=popularity_ranking,
        n=10,
    )["movieId"].tolist()

    recall, ndcg = calculate_ranking_metrics(
        recommended=recommended_ids,
        relevant=relevant_movies,
        k=10,
    )

    user_results.append({
        "userId": user_id,
        "relevant_count": len(relevant_movies),
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
    })

evaluation_results = pd.DataFrame(user_results)

if evaluation_results.empty:
    raise ValueError("Değerlendirmeye uygun kullanıcı bulunamadı.")

print("\n--- BASELINE DOĞRULAMA SONUÇLARI ---")
print("Değerlendirilen kullanıcı:", len(evaluation_results))
print(
    "Recall@10:",
    round(evaluation_results["recall_at_10"].mean(), 4),
)
print(
    "NDCG@10:",
    round(evaluation_results["ndcg_at_10"].mean(), 4),
)

print("\nKullanıcı bazında ilk 5 sonuç:")
print(evaluation_results.head().round(4).to_string(index=False))

# Sonraki modellerle karşılaştırmak için baseline skorlarını sakla.
baseline_metrics = {
    "model": "popularity",
    "recall_at_10": evaluation_results["recall_at_10"].mean(),
    "ndcg_at_10": evaluation_results["ndcg_at_10"].mean(),
}

# Baseline ile aynı aday film kümesini kullan.
training_movie_ids = train["movieId"].unique()

content_catalog = (
    movies.loc[
        movies["movieId"].isin(training_movie_ids),
        ["movieId", "title", "genres"],
    ]
    .sort_values("movieId")
    .copy()
)

# Satırlar film ID'leri, sütunlar türler olacak.
genre_text = (
    content_catalog
    .set_index("movieId")["genres"]
    .replace("(no genres listed)", "")
)

genre_features = (
    genre_text
    .str.get_dummies(sep="|")
    .astype(float)
)

print("\n--- TÜR ÖZELLİK MATRİSİ ---")
print("Boyut:", genre_features.shape)
print("Türler:", genre_features.columns.tolist())

# Okunabilir olması için ilk üç filmin tablosunu ters çevir.
print("\nİlk 3 filmin tür göstergeleri:")
print(genre_features.head(3).T)

no_genre_count = genre_features.sum(axis=1).eq(0).sum()
print("\nTür vektörü tamamen sıfır olan film sayısı:", no_genre_count)


# Film vektörlerini uzunluklarına bölerek normalize et.
# Sıfır vektörlerde sıfıra bölmeyi önle.
genre_lengths = genre_features.pow(2).sum(axis=1).pow(0.5)

normalized_genres = genre_features.div(
    genre_lengths.replace(0, 1),
    axis=0,
)





content_recommendations = recommend_content(
    user_id=example_user,
    train_ratings=train,
    features=normalized_genres,
    ranking=popularity_ranking,
    n=10,
)

# Eğitim geçmişindeki filmler önerilmemeli.
example_seen_movies = set(
    train.loc[train["userId"] == example_user, "movieId"]
)

assert not content_recommendations["movieId"].isin(
    example_seen_movies
).any()

print("\n--- İÇERİK TABANLI ÖNERİLER ---")
print("Kullanıcı ID:", example_user)

print(
    content_recommendations[
        ["title", "similarity", "rating_count"]
    ]
    .round(4)
    .to_string(index=False)
)
content_user_results = []

# Baseline'da değerlendirdiğimiz kullanıcıları aynen kullan.
for user_id in evaluation_results["userId"]:
    relevant_movies = (
        set(
            validation_positives.loc[
                validation_positives["userId"] == user_id,
                "movieId",
            ]
        )
        - train_history[user_id]
    )

    recommended_ids = recommend_content(
        user_id=user_id,
        train_ratings=train,
        features=normalized_genres,
        ranking=popularity_ranking,
        n=10,
    )["movieId"].tolist()

    recall, ndcg = calculate_ranking_metrics(
        recommended=recommended_ids,
        relevant=relevant_movies,
        k=10,
    )

    content_user_results.append({
        "userId": user_id,
        "relevant_count": len(relevant_movies),
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
    })

content_evaluation_results = pd.DataFrame(content_user_results)

# İki modelin aynı kullanıcı grubunda ölçüldüğünü doğrula.
assert set(content_evaluation_results["userId"]) == set(
    evaluation_results["userId"]
)

content_metrics = {
    "model": "genre_content",
    "recall_at_10": content_evaluation_results["recall_at_10"].mean(),
    "ndcg_at_10": content_evaluation_results["ndcg_at_10"].mean(),
}

comparison = pd.DataFrame([
    {
        **baseline_metrics,
        "evaluated_users": len(evaluation_results),
    },
    {
        **content_metrics,
        "evaluated_users": len(content_evaluation_results),
    },
])

print("\n--- MODEL KARŞILAŞTIRMASI ---")
print(comparison.round(4).to_string(index=False))

# Portfolyoda kullanacağımız deney sonuçlarını kaydet.
reports_dir = PROJECT_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    reports_dir / "model_comparison.csv",
    index=False,
)

evaluation_results.to_csv(
    reports_dir / "popularity_user_metrics.csv",
    index=False,
)

content_evaluation_results.to_csv(
    reports_dir / "content_user_metrics.csv",
    index=False,
)

print("\nKarşılaştırma ve kullanıcı bazlı sonuçlar kaydedildi.")


encoded_train, user_to_idx, movie_to_idx = encode_training_data(train)

# Dönüşüm sırasında kayıt kaybolmamalı veya eşleşme eksik kalmamalı.
assert len(encoded_train) == len(train)
assert not encoded_train[["user_idx", "movie_idx"]].isna().any().any()

print("\n--- MATRIX FACTORIZATION HAZIRLIĞI ---")
print("Eğitim kaydı sayısı:", len(encoded_train))
print("Kullanıcı sayısı:", len(user_to_idx))
print("Film sayısı:", len(movie_to_idx))

print(
    "Kullanıcı indeks aralığı:",
    encoded_train["user_idx"].min(),
    "-",
    encoded_train["user_idx"].max(),
)

print(
    "Film indeks aralığı:",
    encoded_train["movie_idx"].min(),
    "-",
    encoded_train["movie_idx"].max(),
)

print("\nİlk 5 eğitim kaydı:")
print(encoded_train.head().to_string(index=False))


mf_model = initialize_mf(
    encoded_train=encoded_train,
    n_users=len(user_to_idx),
    n_movies=len(movie_to_idx),
    **MF_INITIALIZATION,
)

first_record = encoded_train.iloc[0]

initial_prediction = predict_mf(
    model=mf_model,
    user_idx=int(first_record["user_idx"]),
    movie_idx=int(first_record["movie_idx"]),
)

print("\n--- MATRIX FACTORIZATION BAŞLANGICI ---")
print("Kullanıcı vektörleri:", mf_model["user_factors"].shape)
print("Film vektörleri:", mf_model["movie_factors"].shape)
print("Eğitim puan ortalaması:", round(mf_model["global_mean"], 4))

print("\nİlk kayıt için:")
print("Gerçek puan:", first_record["rating"])
print("Eğitim öncesi tahmin:", round(initial_prediction, 4))

print("\n--- MATRIX FACTORIZATION EĞİTİMİ ---")

training_history = train_mf(
    model=mf_model,
    encoded_train=encoded_train,
    **MF_TRAINING,
)
pd.DataFrame(training_history).to_csv(
    reports_dir / "mf_training_history.csv",
    index=False,
)

trained_prediction = predict_mf(
    model=mf_model,
    user_idx=int(first_record["user_idx"]),
    movie_idx=int(first_record["movie_idx"]),
)

print("\nİlk eğitim kaydı için:")
print("Gerçek puan:", first_record["rating"])
print("Eğitim öncesi tahmin:", round(initial_prediction, 4))
print("Eğitim sonrası tahmin:", round(trained_prediction, 4))


known_user_mask = validation["userId"].isin(user_to_idx)
known_movie_mask = validation["movieId"].isin(movie_to_idx)

known_validation = validation[
    known_user_mask & known_movie_mask
].copy()

if known_validation.empty:
    raise ValueError("Değerlendirilebilecek doğrulama kaydı bulunamadı.")

val_user_indices = (
    known_validation["userId"]
    .map(user_to_idx)
    .to_numpy(dtype=np.int64)
)

val_movie_indices = (
    known_validation["movieId"]
    .map(movie_to_idx)
    .to_numpy(dtype=np.int64)
)

actual_values = known_validation["rating"].to_numpy(dtype=np.float64)

# Eğitilmiş modelle tahmin yap; parametreleri değiştirme.
mf_predictions = (
    mf_model["global_mean"]
    + mf_model["user_bias"][val_user_indices]
    + mf_model["movie_bias"][val_movie_indices]
    + np.sum(
        mf_model["user_factors"][val_user_indices]
        * mf_model["movie_factors"][val_movie_indices],
        axis=1,
    )
)

mf_validation_rmse = float(
    np.sqrt(np.mean((actual_values - mf_predictions) ** 2))
)

# Basit karşılaştırma: herkese eğitim puan ortalamasını tahmin et.
mean_validation_rmse = float(
    np.sqrt(
        np.mean((actual_values - mf_model["global_mean"]) ** 2)
    )
)

rating_comparison = pd.DataFrame([
    {
        "model": "global_mean",
        "validation_rmse": mean_validation_rmse,
        "evaluated_ratings": len(known_validation),
    },
    {
        "model": "matrix_factorization",
        "validation_rmse": mf_validation_rmse,
        "evaluated_ratings": len(known_validation),
    },
])

print("\n--- PUAN TAHMİNİ DOĞRULAMASI ---")
print("Toplam doğrulama kaydı:", len(validation))
print("Değerlendirilen kayıt:", len(known_validation))
print(
    "Kapsama oranı:",
    f"{len(known_validation) / len(validation):.1%}",
)

print(rating_comparison.round(4).to_string(index=False))

rating_comparison.to_csv(
    reports_dir / "mf_validation_rmse.csv",
    index=False,
)


mf_user_results = []

# Önceki modellerde değerlendirdiğimiz aynı kullanıcılar.
for user_id in evaluation_results["userId"]:
    relevant_movies = (
        set(
            validation_positives.loc[
                validation_positives["userId"] == user_id,
                "movieId",
            ]
        )
        - train_history[user_id]
    )

    recommended_ids = recommend_mf(
        user_id=user_id,
        model=mf_model,
        train_ratings=train,
        user_to_idx=user_to_idx,
        movie_to_idx=movie_to_idx,
        ranking=popularity_ranking,
        n=10,
    )["movieId"].tolist()

    assert not (
        set(recommended_ids) & train_history[user_id]
    )

    recall, ndcg = calculate_ranking_metrics(
        recommended=recommended_ids,
        relevant=relevant_movies,
        k=10,
    )

    mf_user_results.append({
        "userId": user_id,
        "relevant_count": len(relevant_movies),
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
    })

mf_evaluation_results = pd.DataFrame(mf_user_results)

mf_metrics = {
    "model": "matrix_factorization",
    "recall_at_10": mf_evaluation_results["recall_at_10"].mean(),
    "ndcg_at_10": mf_evaluation_results["ndcg_at_10"].mean(),
}

comparison = pd.DataFrame([
    baseline_metrics,
    content_metrics,
    mf_metrics,
])

comparison["evaluated_users"] = len(evaluation_results)

print("\n--- ÜÇ MODELİN KARŞILAŞTIRMASI ---")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    reports_dir / "model_comparison.csv",
    index=False,
)

mf_evaluation_results.to_csv(
    reports_dir / "mf_user_metrics.csv",
    index=False,
)
# Kullanıcı bazında iki modelin sonuçlarını eşleştir.
paired_results = evaluation_results.merge(
    mf_evaluation_results,
    on=["userId", "relevant_count"],
    suffixes=("_popularity", "_mf"),
    validate="one_to_one",
)

paired_results["recall_difference"] = (
    paired_results["recall_at_10_mf"]
    - paired_results["recall_at_10_popularity"]
)

print("\n--- KULLANICI BAZINDA KARŞILAŞTIRMA ---")
print(
    paired_results[
        [
            "userId",
            "relevant_count",
            "recall_at_10_popularity",
            "recall_at_10_mf",
            "recall_difference",
        ]
    ]
    .sort_values("recall_difference")
    .round(4)
    .to_string(index=False)
)

# MF'nin popülerliğe göre en fazla Recall kaybettiği örneği incele.
diagnostic_user = int(
    paired_results
    .sort_values(["recall_difference", "userId"])
    .iloc[0]["userId"]
)

diagnostic_recommendations = recommend_mf(
    user_id=diagnostic_user,
    model=mf_model,
    train_ratings=train,
    user_to_idx=user_to_idx,
    movie_to_idx=movie_to_idx,
    ranking=popularity_ranking,
    n=10,
)

relevant_movies = (
    set(
        validation_positives.loc[
            validation_positives["userId"] == diagnostic_user,
            "movieId",
        ]
    )
    - train_history[diagnostic_user]
)

diagnostic_recommendations["validation_positive"] = (
    diagnostic_recommendations["movieId"].isin(relevant_movies)
)

print("\n--- MF ÖNERİLERİNİN İNCELENMESİ ---")
print("Kullanıcı:", diagnostic_user)

print(
    diagnostic_recommendations[
        [
            "title",
            "predicted_rating",
            "rating_count",
            "mean_rating",
            "validation_positive",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

# Birleştirme için yalnızca ilk 10'u değil,
# kullanıcının tüm aday filmlerinin sıralamalarını al.
full_ranking_size = len(popularity_ranking)

popularity_full = recommend_popular(
    user_id=example_user,
    train_ratings=train,
    ranking=popularity_ranking,
    n=full_ranking_size,
)

mf_full = recommend_mf(
    user_id=example_user,
    model=mf_model,
    train_ratings=train,
    user_to_idx=user_to_idx,
    movie_to_idx=movie_to_idx,
    ranking=popularity_ranking,
    n=full_ranking_size,
)

hybrid_recommendations = combine_rankings(
    popularity_ranked=popularity_full,
    personal_ranked=mf_full,
    n=10,
    **HYBRID_SETTINGS,
)
# Ağırlık tamamen popülerlikteyse aynı listeyi üretmeliyiz.
popularity_control = combine_rankings(
    popularity_ranked=popularity_full,
    personal_ranked=mf_full,
    popularity_weight=1.0,
    n=10,
)

assert popularity_control["movieId"].tolist() == (
    popularity_full.head(10)["movieId"].tolist()
)

print("\n--- HİBRİT ÖNERİLER ---")
print("Kullanıcı:", example_user)

print(
    hybrid_recommendations[
        ["title", "popularity_rank", "personal_rank", "hybrid_score"]
    ]
    .round(6)
    .to_string(index=False)
)

hybrid_user_rows = []
full_ranking_size = len(popularity_ranking)

for user_id in evaluation_results["userId"]:
    relevant_movies = (
        set(
            validation_positives.loc[
                validation_positives["userId"] == user_id,
                "movieId",
            ]
        )
        - train_history[user_id]
    )

    # Bu kullanıcının bütün adaylarını popülerliğe göre sırala.
    popularity_full = recommend_popular(
        user_id=user_id,
        train_ratings=train,
        ranking=popularity_ranking,
        n=full_ranking_size,
    )

    # İki kişisel modelin tam sıralamalarını hazırla.
    personal_rankings = {
        "hybrid_pop_mf": recommend_mf(
            user_id=user_id,
            model=mf_model,
            train_ratings=train,
            user_to_idx=user_to_idx,
            movie_to_idx=movie_to_idx,
            ranking=popularity_ranking,
            n=full_ranking_size,
        ),

        "hybrid_pop_content": recommend_content(
            user_id=user_id,
            train_ratings=train,
            features=normalized_genres,
            ranking=popularity_ranking,
            n=full_ranking_size,
        ),
    }

    for model_name, personal_full in personal_rankings.items():
        hybrid_recommendations = combine_rankings(
            popularity_ranked=popularity_full,
            personal_ranked=personal_full,
            n=10,
            **HYBRID_SETTINGS,
        )

        recommended_ids = hybrid_recommendations["movieId"].tolist()

        assert not (
            set(recommended_ids) & train_history[user_id]
        )

        recall, ndcg = calculate_ranking_metrics(
            recommended=recommended_ids,
            relevant=relevant_movies,
            k=10,
        )

        hybrid_user_rows.append({
            "model": model_name,
            "userId": user_id,
            "relevant_count": len(relevant_movies),
            "recall_at_10": recall,
            "ndcg_at_10": ndcg,
        })

hybrid_user_results = pd.DataFrame(hybrid_user_rows)

hybrid_summary = (
    hybrid_user_results
    .groupby("model", as_index=False, sort=False)
    .agg(
        recall_at_10=("recall_at_10", "mean"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        evaluated_users=("userId", "nunique"),
    )
)

# Her hibrit aynı kullanıcı grubunda değerlendirilmiş olmalı.
expected_users = set(evaluation_results["userId"])

for model_name, rows in hybrid_user_results.groupby("model"):
    assert set(rows["userId"]) == expected_users

individual_summary = pd.DataFrame([
    baseline_metrics,
    content_metrics,
    mf_metrics,
])

individual_summary["evaluated_users"] = len(expected_users)

comparison = pd.concat(
    [individual_summary, hybrid_summary],
    ignore_index=True,
)

print("\n--- TÜM MODELLERİN DOĞRULAMA SONUÇLARI ---")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    reports_dir / "model_comparison.csv",
    index=False,
)

hybrid_user_results.to_csv(
    reports_dir / "hybrid_user_metrics.csv",
    index=False,
)

best_hybrid_results = hybrid_user_results[
    hybrid_user_results["model"] == "hybrid_pop_mf"
].copy()

paired_scores = evaluation_results.merge(
    best_hybrid_results,
    on=["userId", "relevant_count"],
    suffixes=("_baseline", "_hybrid"),
    validate="one_to_one",
)

assert set(paired_scores["userId"]) == set(
    evaluation_results["userId"]
)

bootstrap_rows = []

for metric in ["recall_at_10", "ndcg_at_10"]:
    result = paired_bootstrap_ci(
        baseline_scores=paired_scores[f"{metric}_baseline"],
        candidate_scores=paired_scores[f"{metric}_hybrid"],
    )

    bootstrap_rows.append({
        "metric": metric,
        **result,
    })

bootstrap_report = pd.DataFrame(bootstrap_rows)

print("\n--- HİBRİT EKSİ POPÜLERLİK: BOOTSTRAP ---")
print(bootstrap_report.round(5).to_string(index=False))

bootstrap_report.to_csv(
    reports_dir / "hybrid_bootstrap_comparison.csv",
    index=False,
)
