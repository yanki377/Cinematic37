from pathlib import Path

from recommender.artifacts import load_artifacts
from recommender.hybrid import combine_rankings
from recommender.matrix_factorization import recommend_mf
from recommender.models import recommend_popular


PROJECT_DIR = Path(__file__).resolve().parent

bundle = load_artifacts(
    PROJECT_DIR / "artifacts" / "final_model"
)

USER_ID = 1
N_RECOMMENDATIONS = 10

full_ranking_size = len(bundle["ranking"])

popularity_full = recommend_popular(
    user_id=USER_ID,
    train_ratings=bundle["history"],
    ranking=bundle["ranking"],
    n=full_ranking_size,
)

mf_full = recommend_mf(
    user_id=USER_ID,
    model=bundle["model"],
    train_ratings=bundle["history"],
    user_to_idx=bundle["user_to_idx"],
    movie_to_idx=bundle["movie_to_idx"],
    ranking=bundle["ranking"],
    n=full_ranking_size,
)

recommendations = combine_rankings(
    popularity_ranked=popularity_full,
    personal_ranked=mf_full,
    n=N_RECOMMENDATIONS,
    **bundle["hybrid_settings"],
)

print("\n--- KAYDEDİLMİŞ MODELDEN ÖNERİLER ---")
print("Kullanıcı:", USER_ID)

if USER_ID not in bundle["user_to_idx"]:
    print("Kullanıcı geçmişi bulunamadı; popülerlik önerileri gösteriliyor.")

print(
    recommendations[
        ["title", "hybrid_score", "popularity_rank", "personal_rank"]
    ]
    .round(6)
    .to_string(index=False)
)