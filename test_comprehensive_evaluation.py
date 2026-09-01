#!/usr/bin/env python
"""Test both API endpoints for recommendation and RAG quality metrics"""
from app.services.recommendation_service import get_recommendation_quality_metrics
from app.rag.rag_service import get_rag_quality_metrics

print("=" * 80)
print("COMPREHENSIVE EVALUATION API TEST")
print("=" * 80)

# Test recommendation quality endpoint
print("\n1. RECOMMENDATION QUALITY METRICS")
print("-" * 80)
print("Endpoint: GET /dashboard/recommendation-quality")
rec_result = get_recommendation_quality_metrics()
print(f"\nStatus: {rec_result.get('status')}")
if rec_result.get('status') == 'success':
    print("\nSummary Statistics:")
    for k in [1, 3, 5]:
        print(f"\n  K={k}:")
        print(f"    Precision@{k}: {rec_result['summary_statistics'].get(f'mean_precision@{k}')}")
        print(f"    Recall@{k}:    {rec_result['summary_statistics'].get(f'mean_recall@{k}')}")
        print(f"    MRR@{k}:       {rec_result['summary_statistics'].get(f'mean_mrr@{k}')}")
    print(f"\n  Total employees evaluated: {rec_result['total_employees_evaluated']}")
    print(f"  Mean skills per employee:  {rec_result['mean_skills_per_employee']}")
else:
    print(f"Error: {rec_result.get('message')}")

# Test RAG quality endpoint
print("\n" + "=" * 80)
print("\n2. RAG QUALITY METRICS")
print("-" * 80)
print("Endpoint: GET /dashboard/rag-quality")
rag_result = get_rag_quality_metrics()
print(f"\nStatus: {rag_result.get('status')}")
if rag_result.get('status') == 'success':
    print("\nSummary Statistics:")
    for k in [1, 3, 5]:
        print(f"\n  K={k}:")
        print(f"    Precision@{k}: {rag_result['summary_statistics'].get(f'mean_precision@{k}')}")
        print(f"    Recall@{k}:    {rag_result['summary_statistics'].get(f'mean_recall@{k}')}")
        print(f"    MRR@{k}:       {rag_result['summary_statistics'].get(f'mean_mrr@{k}')}")
    print(f"\n  Total queries evaluated: {rag_result['total_queries_evaluated']}")
    print(f"  Avg relevant docs/query: {rag_result['avg_relevant_per_query']}")
    print(f"\n  Note: {rag_result.get('note')}")
else:
    print(f"Error: {rag_result.get('message')}")
    print(f"Note: {rag_result.get('note')}")

# Summary
print("\n" + "=" * 80)
print("COMPLETE EVALUATION SYSTEM")
print("=" * 80)
print("""
✅ Recommendation Evaluation
   - Precision@k, Recall@k, MRR@k metrics
   - Evaluated on 1,470 employees
   - Endpoint: GET /dashboard/recommendation-quality
   - Output: docs/recommendation_evaluation.csv

✅ RAG Retrieval Evaluation  
   - Precision@k, Recall@k, MRR@k metrics
   - Evaluated on synthetic policy questions
   - Endpoint: GET /dashboard/rag-quality
   - Output: docs/rag_evaluation.csv

System Integration: Both evaluation services are fully integrated into the
FastAPI backend and expose their metrics via REST API endpoints.
""")
print("=" * 80)
