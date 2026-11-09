from pathlib import Path

import pandas as pd


from recommender.metrics import paired_bootstrap_ci


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "reports" / "final_test"

user_metrics = pd.read_csv(REPORT_DIR / "user_metrics.csv")

baseline_rows = user_metrics[
    user_metrics["model"] == "popularity"
]

hybrid_rows = user_metrics[
    user_metrics["model"] == "hybrid_pop_mf"
]

paired = baseline_rows.merge(
    hybrid_rows,
    on=["userId", "relevant_count", "cold_relevant_count"],
    suffixes=("_baseline", "_hybrid"),
    validate="one_to_one",
)

assert set(paired["userId"]) == set(baseline_rows["userId"])
assert set(paired["userId"]) == set(hybrid_rows["userId"])

report_rows = []

for metric in ["recall_at_10", "ndcg_at_10"]:
    result = paired_bootstrap_ci(
        baseline_scores=paired[f"{metric}_baseline"],
        candidate_scores=paired[f"{metric}_hybrid"],
    )

    report_rows.append({
        "metric": metric,
        "evaluated_users": len(paired),
        **result,
    })

bootstrap_report = pd.DataFrame(report_rows)

print("\n--- SON TEST: HİBRİT EKSİ POPÜLERLİK ---")
print(bootstrap_report.round(5).to_string(index=False))

bootstrap_report.to_csv(
    REPORT_DIR / "bootstrap_comparison.csv",
    index=False,
)

