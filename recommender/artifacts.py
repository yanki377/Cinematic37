import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)

    with np.load(
        artifact_dir / "mf_model.npz",
        allow_pickle=False,
    ) as saved:
        model = {
            name: saved[name].copy()
            for name in saved.files
        }

    model["global_mean"] = float(model["global_mean"])

    mappings = json.loads(
        (artifact_dir / "mappings.json").read_text(encoding="utf-8")
    )

    metadata = json.loads(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )

    user_to_idx = {
        int(user_id): int(idx)
        for user_id, idx in mappings["user_to_idx"].items()
    }

    movie_to_idx = {
        int(movie_id): int(idx)
        for movie_id, idx in mappings["movie_to_idx"].items()
    }

    if len(user_to_idx) != model["user_factors"].shape[0]:
        raise ValueError("Kullanıcı eşlemeleri modelle uyuşmuyor.")

    if len(movie_to_idx) != model["movie_factors"].shape[0]:
        raise ValueError("Film eşlemeleri modelle uyuşmuyor.")

    return {
        "model": model,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "ranking": pd.read_csv(
            artifact_dir / "popularity_ranking.csv"
        ),
        "history": pd.read_csv(
            artifact_dir / "seen_interactions.csv"
        ),
        "hybrid_settings": metadata["hybrid_settings"],
    }