"""
RAG Evaluation Service - Ranking Metrics for Retrieval Quality

Evaluates the quality of RAG retrieval using ranking-based metrics:
- Precision@k: % of top-k retrieved chunks that are relevant
- Recall@k: % of relevant chunks captured in top-k
- MRR@k: Mean Reciprocal Rank (position of first relevant chunk)
"""
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalTestCase:
    """Single RAG retrieval evaluation test case"""
    query: str
    relevant_source_ids: List[str]  # Document IDs that contain answer to this query


def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int = 3) -> float:
    """
    Precision@k: What proportion of top-k retrieved documents are relevant?
    
    Args:
        retrieved_sources: Ranked list of retrieved document source IDs (highest relevance first)
        relevant_sources: Ground truth list of relevant document IDs for this query
        k: Evaluation cutoff (top-k documents)
    
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved_sources[:k]
    if len(top_k) == 0:
        return 0.0
    
    relevant_set = set(relevant_sources)
    hits = sum(1 for source in top_k if source in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int = 3) -> float:
    """
    Recall@k: What proportion of relevant documents appear in the top-k results?
    
    Args:
        retrieved_sources: Ranked list of retrieved document source IDs
        relevant_sources: Ground truth list of relevant document IDs for this query
        k: Evaluation cutoff (top-k documents)
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if len(relevant_sources) == 0:
        return 1.0  # Perfect recall if no relevant docs expected
    
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    hits = sum(1 for source in top_k if source in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_sources: List[str], relevant_sources: List[str], k: int = 5) -> float:
    """
    Mean Reciprocal Rank: Inverse of the rank position of the first relevant document.
    
    - MRR = 1.0 if first document is relevant
    - MRR = 0.5 if second document is relevant
    - MRR = 0.33 if third document is relevant
    - MRR = 0.0 if no relevant document in top-k
    
    Args:
        retrieved_sources: Ranked list of retrieved document sources
        relevant_sources: Ground truth list of relevant document IDs for this query
        k: Maximum rank position to consider
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    if len(relevant_sources) == 0:
        return 1.0
    
    top_k = retrieved_sources[:k]
    relevant_set = set(relevant_sources)
    
    for i, source in enumerate(top_k, start=1):
        if source in relevant_set:
            return 1.0 / i
    
    return 0.0


def evaluate_retrieval(
    test_cases: List[RetrievalTestCase],
    retrieval_fn,
    k_values: List[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate RAG retrieval quality across multiple test cases.
    
    Args:
        test_cases: List of RetrievalTestCase objects with queries and relevant sources
        retrieval_fn: Function that takes query string and k, returns list of (source_id, chunk) tuples
        k_values: List of k values to evaluate (default: [1, 3, 5])
    
    Returns:
        Tuple of (detailed metrics DataFrame, summary statistics dict)
    """
    if k_values is None:
        k_values = [1, 3, 5]
    
    results = []
    
    for test_case in test_cases:
        # Retrieve chunks
        retrieved_chunks = retrieval_fn(test_case.query, k=max(k_values))
        retrieved_sources = [chunk.source for chunk in retrieved_chunks]
        
        metrics = {
            "query": test_case.query,
            "num_relevant": len(test_case.relevant_source_ids),
            "num_retrieved": len(retrieved_sources),
        }
        
        # Calculate metrics at each k
        for k in k_values:
            p_at_k = precision_at_k(retrieved_sources, test_case.relevant_source_ids, k)
            r_at_k = recall_at_k(retrieved_sources, test_case.relevant_source_ids, k)
            mrr_k = mrr(retrieved_sources, test_case.relevant_source_ids, k)
            
            metrics[f"precision@{k}"] = round(p_at_k, 3)
            metrics[f"recall@{k}"] = round(r_at_k, 3)
            metrics[f"mrr@{k}"] = round(mrr_k, 3)
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    summary = {}
    for k in k_values:
        summary[f"mean_precision@{k}"] = round(results_df[f"precision@{k}"].mean(), 3)
        summary[f"mean_recall@{k}"] = round(results_df[f"recall@{k}"].mean(), 3)
        summary[f"mean_mrr@{k}"] = round(results_df[f"mrr@{k}"].mean(), 3)
    
    return results_df, summary


def get_rag_quality_report(
    test_cases: List[RetrievalTestCase],
    retrieval_fn
) -> Dict:
    """
    Generate RAG retrieval quality report with Precision@k, Recall@k, MRR@k metrics.
    
    Args:
        test_cases: Evaluation test cases
        retrieval_fn: Retrieval function
    
    Returns:
        Dictionary with evaluation metrics and summary statistics
    """
    results_df, summary = evaluate_retrieval(test_cases, retrieval_fn, k_values=[1, 3, 5])
    
    report = {
        "summary_statistics": summary,
        "total_queries_evaluated": len(results_df),
        "avg_relevant_per_query": round(results_df["num_relevant"].mean(), 1),
        "detailed_results": results_df,
    }
    
    return report
