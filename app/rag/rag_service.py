"""RAG service - exposes retrieval quality metrics"""
import pandas as pd
from typing import Dict
from app.rag.evaluation_service import RetrievalTestCase, get_rag_quality_report


def get_rag_quality_metrics() -> dict:
    """
    Get RAG retrieval quality metrics (Precision@k, Recall@k, MRR@k).
    
    Returns:
        Dictionary with:
        - summary: Aggregate metrics across all queries
        - evaluation_details: Path to detailed CSV report
    """
    try:
        from app.rag.retriever import get_retriever
        
        # Synthetic test cases for evaluation
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
        ]
        
        retriever = get_retriever()
        report = get_rag_quality_report(test_cases, lambda q, k: retriever.retrieve(q, k))
        
        return {
            "status": "success",
            "summary_statistics": report["summary_statistics"],
            "total_queries_evaluated": report["total_queries_evaluated"],
            "avg_relevant_per_query": report["avg_relevant_per_query"],
            "evaluation_details_file": "docs/rag_evaluation.csv",
            "note": "Evaluation uses synthetic test cases. Use labeled data for production evaluation."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "note": "RAG evaluation requires policy documents. Ensure data/external/policies/ contains .md files."
        }
