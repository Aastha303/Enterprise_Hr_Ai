"""
21 - RAG Evaluation: Precision@k, Recall@k, MRR@k

Measures the quality of RAG document retrieval using ranking-based metrics:
- Precision@k: % of top-k retrieved documents that are relevant
- Recall@k: % of relevant documents captured in top-k
- MRR@k: Position of first relevant document (1/rank)
"""
import pandas as pd
import sys
sys.path.insert(0, "..")

from app.rag.evaluation_service import (
    RetrievalTestCase,
    evaluate_retrieval,
    get_rag_quality_report,
)
from app.rag.retriever import get_retriever

# Create synthetic evaluation test cases
# In production, these would be curated HR policy questions with labeled relevant documents
test_cases = [
    RetrievalTestCase(
        query="What is the paid time off policy?",
        relevant_source_ids=["pto_policy.md", "leave.md"]
    ),
    RetrievalTestCase(
        query="How do I request vacation?",
        relevant_source_ids=["pto_policy.md", "leave.md", "request_procedures.md"]
    ),
    RetrievalTestCase(
        query="What is the remote work policy?",
        relevant_source_ids=["remote_work.md", "work_arrangement.md"]
    ),
    RetrievalTestCase(
        query="What are the health insurance benefits?",
        relevant_source_ids=["benefits.md", "health_insurance.md"]
    ),
    RetrievalTestCase(
        query="How to file an expense report?",
        relevant_source_ids=["expense_policy.md", "reimbursement.md"]
    ),
    RetrievalTestCase(
        query="What is the code of conduct?",
        relevant_source_ids=["code_of_conduct.md", "ethics_policy.md"]
    ),
    RetrievalTestCase(
        query="How does performance review work?",
        relevant_source_ids=["performance_review.md", "evaluation_process.md"]
    ),
    RetrievalTestCase(
        query="What is the parental leave policy?",
        relevant_source_ids=["leave.md", "family_benefits.md", "pto_policy.md"]
    ),
]

# Get retriever
retriever = get_retriever()

print("\n" + "="*80)
print("RAG EVALUATION: PRECISION@K, RECALL@K, MRR@K")
print("="*80)

# Evaluate retrieval quality
print("\n1. DETAILED METRICS (all test queries)")
print("-" * 80)
results_df, summary = evaluate_retrieval(test_cases, lambda q, k: retriever.retrieve(q, k), k_values=[1, 3, 5])

# Show all results
print(results_df[["query", "num_relevant", "num_retrieved", 
                  "precision@1", "recall@1", "mrr@1",
                  "precision@3", "recall@3", "mrr@3",
                  "precision@5", "recall@5", "mrr@5"]].to_string(index=False))

# Summary statistics
print("\n2. AGGREGATE METRICS ACROSS ALL QUERIES")
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
print("\n4. BEST QUERIES (Highest Recall@3)")
print("-" * 80)
best = results_df.nlargest(3, "recall@3")[["query", "num_relevant", "num_retrieved",
                                            "precision@3", "recall@3", "mrr@3"]]
for idx, row in best.iterrows():
    print(f"\n  Query: {row['query']}")
    print(f"    Precision@3: {row['precision@3']:.3f} | Recall@3: {row['recall@3']:.3f} | MRR@3: {row['mrr@3']:.3f}")

print("\n5. WORST QUERIES (Lowest Recall@3)")
print("-" * 80)
worst = results_df.nsmallest(3, "recall@3")[["query", "num_relevant", "num_retrieved",
                                              "precision@3", "recall@3", "mrr@3"]]
for idx, row in worst.iterrows():
    print(f"\n  Query: {row['query']}")
    print(f"    Precision@3: {row['precision@3']:.3f} | Recall@3: {row['recall@3']:.3f} | MRR@3: {row['mrr@3']:.3f}")

# Interpretation guide
print("\n6. INTERPRETATION GUIDE")
print("-" * 80)
print("""
  PRECISION@K: Quality of top-k retrievals
    └─ 1.0: All top-k results are relevant
    └─ 0.5: 50% of top-k results are relevant
    └─ 0.0: No relevant results in top-k

  RECALL@K: Coverage of relevant documents
    └─ 1.0: We retrieve all relevant documents in top-k
    └─ 0.5: We retrieve 50% of relevant documents in top-k
    └─ 0.0: We retrieve none of the relevant documents

  MRR@K: Position of first relevant result
    └─ 1.0: First result is relevant (rank 1)
    └─ 0.5: Second result is relevant (rank 2)
    └─ 0.33: Third result is relevant (rank 3)
    └─ 0.0: No relevant result in top-k
""")

# Summary report
print("\n7. EVALUATION SUMMARY")
print("-" * 80)
report = get_rag_quality_report(test_cases, lambda q, k: retriever.retrieve(q, k))
print(f"  Total queries evaluated   : {report['total_queries_evaluated']}")
print(f"  Avg relevant docs/query   : {report['avg_relevant_per_query']}")

# Save results
results_df.to_csv("../docs/rag_evaluation.csv", index=False)
print("\n✓ Saved detailed results -> docs/rag_evaluation.csv")

print("\n" + "="*80)
print("NOTE: This evaluation uses synthetic test cases. In production, curate")
print("real HR policy questions with manually labeled relevant documents.")
print("="*80)
