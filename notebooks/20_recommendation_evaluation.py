"""
20 - Recommendation Evaluation: Precision@k, Recall@k, MRR@k

Measures the quality of skill gap recommendations using ranking-based metrics:
- Precision@k: % of top-k skills we recommend that are actually in the gap
- Recall@k: % of employee's actual gaps captured in top-k recommendations
- MRR@k: Position of first correct recommendation (1/rank)
"""
import pandas as pd
import sys
sys.path.insert(0, "..")

from app.services.recommendation_evaluation_service import (
    evaluate_recommendations,
    get_recommendation_quality_report,
    parse_skill_gap,
)

# Load data
gap_df = pd.read_csv("../data/processed/employee_skill_gaps.csv")
org_gap = pd.read_csv("../docs/organization_skill_gap.csv")

print("\n" + "="*80)
print("RECOMMENDATION EVALUATION: PRECISION@K, RECALL@K, MRR@K")
print("="*80)

# Evaluate at multiple k values
print("\n1. DETAILED METRICS (sample of 20 employees)")
print("-" * 80)
results_df, summary = evaluate_recommendations(gap_df, org_gap, k_values=[1, 3, 5])

# Show sample
display_cols = ["EmployeeNumber", "JobRole", "num_skill_gaps",
                "precision@1", "recall@1", "mrr@1",
                "precision@3", "recall@3", "mrr@3",
                "precision@5", "recall@5", "mrr@5"]
print(results_df[display_cols].head(20).to_string(index=False))

# Summary statistics
print("\n2. AGGREGATE METRICS ACROSS ORGANIZATION")
print("-" * 80)
for metric, value in summary.items():
    print(f"  {metric:25s}: {value}")

# Detailed analysis by k
print("\n3. METRIC BREAKDOWN BY K")
print("-" * 80)
print(f"{'k':>5} {'Precision@k':>12} {'Recall@k':>12} {'MRR@k':>12}")
print("-" * 80)
for k in [1, 3, 5]:
    prec = summary.get(f"mean_precision@{k}", 0)
    rec = summary.get(f"mean_recall@{k}", 0)
    mrr_val = summary.get(f"mean_mrr@{k}", 0)
    print(f"{k:>5} {prec:>12.3f} {rec:>12.3f} {mrr_val:>12.3f}")

# Examples: Best and worst performers
print("\n4. EXAMPLE: BEST PERFORMERS (Top Recall@3)")
print("-" * 80)
best = results_df.nlargest(5, "recall@3")[["EmployeeNumber", "JobRole", "num_skill_gaps", 
                                            "precision@3", "recall@3", "mrr@3"]]
print(best.to_string(index=False))

print("\n5. EXAMPLE: WORST PERFORMERS (Lowest Recall@3)")
print("-" * 80)
worst = results_df.nsmallest(5, "recall@3")[["EmployeeNumber", "JobRole", "num_skill_gaps",
                                              "precision@3", "recall@3", "mrr@3"]]
print(worst.to_string(index=False))

# Interpretation guide
print("\n6. INTERPRETATION GUIDE")
print("-" * 80)
print("""
  PRECISION@K: Accuracy of recommendations
    └─ 1.0: All top-k recommendations are relevant
    └─ 0.5: 50% of top-k recommendations are relevant
    └─ 0.0: No relevant recommendations in top-k

  RECALL@K: Coverage of actual gaps
    └─ 1.0: We capture all employee's skill gaps in top-k
    └─ 0.5: We capture 50% of employee's gaps in top-k
    └─ 0.0: We capture no gaps in top-k

  MRR@K: Position of first correct recommendation
    └─ 1.0: First item is correct (rank 1)
    └─ 0.5: Second item is correct (rank 2)
    └─ 0.33: Third item is correct (rank 3)
    └─ 0.0: No correct item in top-k
""")

# Save results
results_df.to_csv("../docs/recommendation_evaluation.csv", index=False)
print("\n✓ Saved detailed results -> docs/recommendation_evaluation.csv")

print("\n" + "="*80)
