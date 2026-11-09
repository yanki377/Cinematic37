import json
import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

token = os.getenv("TMDB_READ_TOKEN", "").strip()

if not token:
    raise RuntimeError("TMDB_READ_TOKEN bulunamadı.")

# Modelimizin desteklediği filmler.
ranking = pd.read_csv(
    PROJECT_DIR / "artifacts" / "final_model" / "popularity_ranking.csv"
)

# MovieLens film kimliklerini TMDB kimlikleriyle eşleştir.
links = pd.read_csv(
    PROJECT_DIR / "data" / "raw" / "ml-latest-small" / "links.csv",
    dtype={"tmdbId": "Int64"},
)

selected_movies = (
    ranking.merge(
        links[["movieId", "tmdbId"]],
        on="movieId",
        how="left",
        validate="one_to_one",
    )
    .dropna(subset=["tmdbId"])
    .head(30)
)

catalog = []

with httpx.Client(
    base_url="https://api.themoviedb.org/3/",
    headers={
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    },
    timeout=20,
) as client:
    # Afiş adreslerini oluşturmak için görsel ayarlarını al.
    configuration_response = client.get("configuration")
    configuration_response.raise_for_status()

    images = configuration_response.json()["images"]
    image_base = images["secure_base_url"]

    poster_size = (
        "w500"
        if "w500" in images["poster_sizes"]
        else "original"
    )

    for position, row in enumerate(
        selected_movies.itertuples(index=False),
        start=1,
    ):
        tmdb_id = int(row.tmdbId)

        response = client.get(
            f"movie/{tmdb_id}",
            params={"language": "tr-TR"},
        )

        if response.status_code == 404:
            print(f"Film bulunamadı, atlandı: {row.title}")
            continue

        response.raise_for_status()
        details = response.json()

        poster_path = details.get("poster_path")

        poster_url = (
            f"{image_base}{poster_size}{poster_path}"
            if poster_path
            else None
        )

        catalog.append({
            "movie_id": int(row.movieId),
            "tmdb_id": tmdb_id,
            "title": details.get("title") or row.title,
            "original_title": details.get("original_title") or row.title,
            "movielens_title": row.title,
            "release_date": details.get("release_date", ""),
            "overview": details.get("overview", ""),
            "genres": [
                genre["name"]
                for genre in details.get("genres", [])
            ],
            "runtime": details.get("runtime"),
            "poster_url": poster_url,
            "tmdb_rating": details.get("vote_average"),
            "tmdb_vote_count": details.get("vote_count"),
        })

        print(
            f"{position}/{len(selected_movies)} hazır: "
            f"{catalog[-1]['title']}"
        )

if not catalog:
    raise RuntimeError("Film kataloğu oluşturulamadı.")

output_dir = PROJECT_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "movie_catalog.json"

output_file.write_text(
    json.dumps(catalog, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

poster_count = sum(
    movie["poster_url"] is not None
    for movie in catalog
)

print("\n--- KATALOG HAZIR ---")
print("Hazırlanan film:", len(catalog))
print("Afişi bulunan film:", poster_count)
print("Dosya:", output_file)
print("İlk afiş adresi:", catalog[0]["poster_url"])
