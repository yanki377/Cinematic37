def combine_rankings(
    popularity_ranked,
    personal_ranked,
    popularity_weight=0.8,
    rank_constant=60,
    n=10,
):
    if not 0 <= popularity_weight <= 1:
        raise ValueError("Popülerlik ağırlığı 0–1 arasında olmalı.")

    if rank_constant <= 0 or n < 1:
        raise ValueError("Sıralama sabiti ve öneri sayısı pozitif olmalı.")

    if (
        not popularity_ranked["movieId"].is_unique
        or not personal_ranked["movieId"].is_unique
    ):
        raise ValueError("Sıralamalarda tekrarlanan film bulunamaz.")

    if set(popularity_ranked["movieId"]) != set(personal_ranked["movieId"]):
        raise ValueError("İki sıralama aynı aday filmleri içermeli.")

    combined = popularity_ranked.copy()

    combined["popularity_rank"] = range(1, len(combined) + 1)

    personal_positions = dict(zip(
        personal_ranked["movieId"],
        range(1, len(personal_ranked) + 1),
    ))

    combined["personal_rank"] = combined["movieId"].map(
        personal_positions
    )

    combined["hybrid_score"] = (
        popularity_weight
        / (rank_constant + combined["popularity_rank"])
        + (1 - popularity_weight)
        / (rank_constant + combined["personal_rank"])
    )

    return (
        combined.sort_values(
            ["hybrid_score", "popularity_rank", "movieId"],
            ascending=[False, True, True],
        )
        .head(n)
        .copy()
    )