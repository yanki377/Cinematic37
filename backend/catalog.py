"""Yerel katalog ve TMDB'den tamamlanan film bilgileri."""
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd


logger = logging.getLogger(__name__)


def normalize(text):
    text = unicodedata.normalize("NFD", str(text).casefold().replace("ı", "i"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


GENRE_ALIASES = {
    "Action": "Aksiyon", "Adventure": "Macera", "Animation": "Animasyon",
    "Children": "Aile Çocuk", "Comedy": "Komedi", "Crime": "Suç",
    "Documentary": "Belgesel", "Drama": "Dram", "Fantasy": "Fantastik",
    "Film-Noir": "Kara Film", "Horror": "Korku", "IMAX": "IMAX",
    "Musical": "Müzikal", "Mystery": "Gizem", "Romance": "Romantik",
    "Sci-Fi": "Bilim Kurgu", "Thriller": "Gerilim", "War": "Savaş",
    "Western": "Western Vahşi Batı",
}


def title_keys(title):
    raw = str(title or "")
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    ordered = re.sub(r"^(.+), (The|A|An)$", r"\2 \1", clean)
    return tuple({normalize(raw), normalize(clean), normalize(ordered)} - {""})


def match_score(needle, keys):
    if any(needle == key for key in keys):
        return 100
    if any(key.startswith(needle) for key in keys):
        return 85
    if any(needle in key for key in keys):
        return 70
    return 0


class MovieCatalog:
    def __init__(self, project_dir: Path, bundle, token=""):
        self._lock = threading.Lock()
        self._token = token
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._cache_path = project_dir / "data/processed/tmdb_cache.json"
        self._base = bundle["ranking"].set_index("movieId").to_dict("index")
        self._rank = {
            int(movie_id): rank
            for rank, movie_id in enumerate(bundle["ranking"]["movieId"])
        }
        links = pd.read_csv(
            project_dir / "data/raw/ml-latest-small/links.csv",
            dtype={"tmdbId": "Int64"},
        ).dropna(subset=["tmdbId"])
        self._tmdb_ids = {
            int(row.movieId): int(row.tmdbId)
            for row in links.itertuples(index=False)
        }
        source_movies = pd.read_csv(project_dir / "data/raw/ml-latest-small/movies.csv")
        self._genres = source_movies.set_index("movieId")["genres"].to_dict()
        self._search_titles = {
            int(movie_id): title_keys(row["title"])
            for movie_id, row in self._base.items()
        }
        self._search_genres = {}
        for movie_id in self._base:
            genres = [
                value for value in str(self._genres.get(movie_id, "")).split("|")
                if value and value != "(no genres listed)"
            ]
            self._search_genres[int(movie_id)] = normalize(
                " ".join(genres + [GENRE_ALIASES.get(value, value) for value in genres])
            )
        # Aynı TMDB filmi birden fazla ID ile eşleşirse en popüler kaydı kullan.
        self._model_id_by_tmdb = {}
        for movie_id in self._base:
            if movie_id in self._tmdb_ids:
                self._model_id_by_tmdb.setdefault(self._tmdb_ids[movie_id], int(movie_id))
        self._query_cache = {}
        self._search_backoff_until = 0.0
        initial = json.loads(
            (project_dir / "data/processed/movie_catalog.json").read_text(encoding="utf-8")
        )
        self._cache = {
            int(row["movie_id"]): row for row in initial
            if int(row["movie_id"]) in self._base
        }
        self._extra = {}
        if self._cache_path.exists():
            extra = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._extra = {
                int(row["movie_id"]): row for row in extra
                if int(row["movie_id"]) in self._base
            }
            self._cache.update(self._extra)
        self._retry_at = {}
        self._client = httpx.Client(
            base_url="https://api.themoviedb.org/3/",
            headers={"Authorization": f"Bearer {token}", "accept": "application/json"},
            timeout=httpx.Timeout(8.0, connect=4.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )

    def listing(self, query="", limit=30):
        limit = min(30, max(1, int(limit)))
        needle = normalize(query)
        if not needle:
            return self.enrich(list(self._base)[:limit])
        if len(needle) < 2:
            return []

        with self._lock:
            cached = dict(self._cache)
        scores = {}
        # Arama bütün model kataloğunu kapsar; önbellekte olmak zorunlu değil.
        for movie_id, keys in self._search_titles.items():
            score = match_score(needle, keys)
            row = cached.get(movie_id)
            if row:
                localized = [
                    *title_keys(row.get("title", "")),
                    *title_keys(row.get("original_title", "")),
                ]
                score = max(score, match_score(needle, localized))
            genre_text = self._search_genres[movie_id]
            if row:
                genre_text += " " + normalize(" ".join(row.get("genres", [])))
            if needle in genre_text:
                score = max(score, 25)
            if score:
                scores[movie_id] = score

        # Henüz çevrilmemiş Türkçe/alternatif adları TMDB aramasıyla çöz.
        # Modelin desteklemediği TMDB filmlerini sonuçlara ekleme.
        if len(scores) < limit:
            for position, item in enumerate(self._search_tmdb(query)):
                movie_id = self._model_id_by_tmdb.get(item.get("id"))
                if movie_id is None:
                    continue
                keys = [
                    *title_keys(item.get("title", "")),
                    *title_keys(item.get("original_title", "")),
                ]
                score = max(match_score(needle, keys), 50 - position * 0.1)
                scores[movie_id] = max(scores.get(movie_id, 0), score)

        ordered = sorted(scores, key=lambda movie_id: (-scores[movie_id], self._rank[movie_id]))
        chosen, seen = [], set()
        for movie_id in ordered:
            key = self._tmdb_ids.get(movie_id, ("movielens", movie_id))
            if key not in seen:
                seen.add(key)
                chosen.append(movie_id)
                if len(chosen) == limit:
                    break
        return self.enrich(chosen)

    def _search_tmdb(self, query):
        key = normalize(query)
        now = time.monotonic()
        with self._lock:
            cached = self._query_cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
            if now < self._search_backoff_until:
                return []
        if not self._token:
            return []
        try:
            response = self._client.get(
                "search/movie",
                params={"query": query.strip(), "language": "tr-TR", "include_adult": "false"},
            )
            if response.status_code == 429:
                try:
                    delay = float(response.headers.get("Retry-After", "60"))
                except ValueError:
                    delay = 60.0
                if not math.isfinite(delay):
                    delay = 60.0
                with self._lock:
                    self._search_backoff_until = time.monotonic() + max(60, delay)
            response.raise_for_status()
            items = response.json().get("results", [])
            if not isinstance(items, list):
                return []
        except (httpx.HTTPError, ValueError):
            return []
        with self._lock:
            if len(self._query_cache) >= 256:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[key] = (time.monotonic() + 900, items)
        return items

    def _fallback(self, movie_id):
        title = str(self._base[movie_id]["title"])
        match = re.search(r"\((\d{4})\)\s*$", title)
        return {
            "movie_id": movie_id,
            "tmdb_id": self._tmdb_ids.get(movie_id),
            "title": title,
            "original_title": title,
            "movielens_title": title,
            "release_date": "",
            "year": match.group(1) if match else None,
            "overview": "",
            "genres": [
                value for value in str(self._genres.get(movie_id, "")).split("|")
                if value and value != "(no genres listed)"
            ],
            "runtime": None,
            "poster_url": None,
            "tmdb_rating": None,
            "tmdb_vote_count": None,
        }

    def get(self, movie_id):
        movie_id = int(movie_id)
        with self._lock:
            if movie_id in self._cache:
                return self._cache[movie_id]
            delayed = time.monotonic() < self._retry_at.get(movie_id, 0)

        fallback = self._fallback(movie_id)
        tmdb_id = self._tmdb_ids.get(movie_id)
        if delayed or not self._token or tmdb_id is None:
            return fallback

        retry_delay = 60.0
        try:
            for attempt in range(2):
                response = self._client.get(
                    f"movie/{tmdb_id}",
                    params={"language": "tr-TR"},
                )
                if response.status_code == 429 and attempt == 0:
                    try:
                        delay = float(response.headers.get("Retry-After", "1"))
                    except ValueError:
                        delay = 1.0
                    # Uzun bekleme isteniyorsa isteği şimdi tekrarlama.
                    if delay > 2:
                        retry_delay = max(60.0, delay)
                        break
                    time.sleep(max(0, delay))
                    continue
                break
            response.raise_for_status()
            details = response.json()
        except (httpx.HTTPError, ValueError):
            with self._lock:
                self._retry_at[movie_id] = time.monotonic() + retry_delay
            logger.warning("Film bilgisi tamamlanamadi: movie_id=%s", movie_id)
            return fallback

        path = details.get("poster_path")
        row = {
            **fallback,
            "title": details.get("title") or fallback["title"],
            "original_title": details.get("original_title") or fallback["original_title"],
            "release_date": details.get("release_date") or "",
            "overview": details.get("overview") or "",
            "genres": [genre["name"] for genre in details.get("genres", [])],
            "runtime": details.get("runtime"),
            "poster_url": f"https://image.tmdb.org/t/p/w500{path}" if path else None,
            "tmdb_rating": details.get("vote_average"),
            "tmdb_vote_count": details.get("vote_count"),
        }
        with self._lock:
            self._cache[movie_id] = row
            self._extra[movie_id] = row
            temporary = self._cache_path.with_suffix(".tmp")
            try:
                temporary.write_text(
                    json.dumps(list(self._extra.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary, self._cache_path)
            except OSError:
                # Kayıt izni yoksa da bellekteki katalogla önerileri sun.
                logger.warning("TMDB onbellegi diske kaydedilemedi.")
        return row

    def enrich(self, movie_ids):
        # map, modelin öneri sırasını korur.
        return list(self._pool.map(self.get, movie_ids))

    def close(self):
        self._pool.shutdown(wait=True)
        self._client.close()
