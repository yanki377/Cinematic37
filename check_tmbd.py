import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

token = os.getenv("TMDB_READ_TOKEN", "").strip()

if not token:
    raise RuntimeError(".env dosyasında TMDB_READ_TOKEN bulunamadı.")

response = httpx.get(
    "https://api.themoviedb.org/3/movie/11",
    headers={
        "Authorization": f"Bearer {token}",
    },
    params={
        "language": "tr-TR",
    },
    timeout=20,
)

response.raise_for_status()
movie = response.json()

print("Film:", movie["title"])
print("Afiş yolu:", movie.get("poster_path"))
print("Açıklama:", movie.get("overview", "")[:200])