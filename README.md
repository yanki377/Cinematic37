# Cinematic 37

A movie recommendation portfolio project built with Python, NumPy, FastAPI, and vanilla HTML/CSS/JavaScript. Select and rate 3–10 movies to get a personal recommendation list without creating an account.

## Current features

- Biased matrix factorization combined with a popularity ranking through weighted reciprocal rank fusion.
- New-user preferences fitted against fixed movie factors using ridge regression; shared model parameters remain unchanged.
- Search across the 8,586 movies supported by the saved model.
- Turkish movie metadata and posters from TMDB, with local caching and fallback metadata.
- Responsive interface, movie details, and session-based film selections.
- Saved model artifacts and evaluation reports included for reproducibility.

## Run locally

Developed with Python 3.12.6. In the project root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` and set `TMDB_READ_TOKEN` to your TMDB API Read Access Token, without the `Bearer ` prefix. Keep this file private. If `.env` already exists, keep the existing file instead of copying over it.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```

Open http://127.0.0.1:8000/. API documentation is available at http://127.0.0.1:8000/docs.

The included model artifacts allow the app to start without retraining. `python prepare_catalog.py` refreshes the initial poster catalog and requires a valid TMDB token. Additional metadata is fetched on demand. Without a token, existing cached metadata and local MovieLens title search remain available.

## Evaluation and limitations

The MovieLens data was split globally by timestamp into training, validation, and test sets. Hyperparameters were selected before the final test; the final model was fitted on training plus validation data (90,752 ratings).

| Model | Test Recall@10 | Test NDCG@10 | Evaluated users |
| --- | ---: | ---: | ---: |
| Popularity | 0.0329 | 0.0889 | 25 |
| Popularity + matrix factorization | 0.0355 | 0.1007 | 25 |

These results cover warm users with eligible positive test targets, not all 74 test users. Unavailable positive movie targets remain in the metric denominators. Paired bootstrap confidence intervals for both improvements include zero, so the higher averages do not establish a reliable improvement. See `reports/final_test/` for full results.

The website's new-user onboarding is a separate scenario and has not yet received a dedicated cold-start evaluation. TMDB scores shown on movie cards are source ratings out of 10, not personalized match probabilities. Group recommendations are not part of this version.

## Project structure

```text
backend/                FastAPI endpoints and movie metadata service
recommender/            Training, ranking, metrics, and new-user inference
web/                    Current website and Cinematic 37 branding
artifacts/final_model/  Saved model, mappings, history, and ranking
data/raw/               MovieLens Small dataset and original usage terms
data/processed/         Initial catalog; runtime metadata cache is ignored
reports/                Exploratory figures and evaluation outputs
```

`main.py` contains the original experiments, `evaluate_final.py` refits and evaluates the final model, and `predict.py` demonstrates saved-model inference. The root `app.py`, `.streamlit/`, and `assets/` retain the earlier Streamlit prototype. The current website starts through `backend.app:app`.

## Data and attribution

MovieLens Small is provided by GroupLens. Its original usage conditions and citation are included in `data/raw/ml-latest-small/README.txt`; those conditions also apply to redistributed data and transformations. Citation: F. Maxwell Harper and Joseph A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015. https://doi.org/10.1145/2827872

Movie information and posters are provided by TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB. TMDB imagery is not licensed as this project's source code.
