import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommender.artifacts import load_artifacts
from recommender.onboarding import recommend_from_ratings

from .catalog import MovieCatalog


PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(PROJECT_DIR / ".env")
    bundle = load_artifacts(PROJECT_DIR / "artifacts" / "final_model")
    # Paylaşılan film/model parametreleri her istek boyunca sabit kalır.
    for name in ("user_factors", "movie_factors", "user_bias", "movie_bias"):
        bundle["model"][name].setflags(write=False)

    catalog = MovieCatalog(
        PROJECT_DIR,
        bundle,
        token=os.getenv("TMDB_READ_TOKEN", "").strip(),
    )
    app.state.bundle = bundle
    app.state.catalog = catalog
    try:
        yield
    finally:
        catalog.close()


app = FastAPI(title="Film Atlası", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class MovieRating(BaseModel):
    movie_id: int = Field(gt=0, strict=True)
    rating: float = Field(ge=0.5, le=5.0, multiple_of=0.5)


class RecommendationRequest(BaseModel):
    ratings: list[MovieRating] = Field(min_length=3, max_length=10)
    n: int = Field(default=10, ge=1, le=20, strict=True)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/movies")
def movie_catalog(
    request: Request,
    response: Response,
    query: str = Query(default="", max_length=100),
    limit: int = Query(default=30, ge=1, le=30),
):
    response.headers["X-Catalog-Size"] = str(len(request.app.state.bundle["movie_to_idx"]))
    return request.app.state.catalog.listing(query, limit)


@app.get("/api/movies/{movie_id}")
def movie_details(movie_id: int, request: Request):
    if movie_id not in request.app.state.bundle["movie_to_idx"]:
        raise HTTPException(status_code=404, detail="Film modelin kataloğunda bulunmuyor.")
    return request.app.state.catalog.get(movie_id)


@app.post("/api/recommendations")
def recommendations(payload: RecommendationRequest, request: Request):
    movie_ids = [entry.movie_id for entry in payload.ratings]
    if len(set(movie_ids)) != len(movie_ids):
        raise HTTPException(status_code=422, detail="Aynı filmi bir kez seçmelisin.")

    bundle = request.app.state.bundle
    if any(movie_id not in bundle["movie_to_idx"] for movie_id in movie_ids):
        raise HTTPException(status_code=422, detail="Seçilen filmlerden biri modelin kataloğunda bulunmuyor.")

    preferences = {entry.movie_id: entry.rating for entry in payload.ratings}
    ranked = recommend_from_ratings(bundle, preferences, n=payload.n)
    result_ids = [int(value) for value in ranked["movieId"]]
    movies = request.app.state.catalog.enrich(result_ids)
    return {
        "movies": movies,
        "input_count": len(preferences),
        "metadata_missing": sum(not movie.get("poster_url") for movie in movies),
    }


@app.get("/", response_class=FileResponse)
def homepage():
    return FileResponse(WEB_DIR / "index.html")
