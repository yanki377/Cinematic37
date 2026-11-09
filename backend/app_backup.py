import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_DIR / "web"
CATALOG_PATH = PROJECT_DIR / "data" / "processed" / "movie_catalog.json"

app = FastAPI(title="Film Atlası")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/movies")
def movie_catalog():
    if not CATALOG_PATH.exists():
        raise HTTPException(status_code=503, detail="Film kataloğu hazır değil.")

    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@app.get("/", response_class=FileResponse)
def homepage():
    return FileResponse(WEB_DIR / "index.html")
