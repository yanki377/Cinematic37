RANDOM_SEED = 42

MF_INITIALIZATION = {
    "n_factors": 20,
    "seed": RANDOM_SEED,
}

MF_TRAINING = {
    "epochs": 10,
    "learning_rate": 0.01,
    "regularization": 0.05,
    "seed": RANDOM_SEED,
}

HYBRID_SETTINGS = {
    "popularity_weight": 0.8,
    "rank_constant": 60,
}