"""
Recommendation Evaluation Service - Ranking Metrics

Evaluates skill gap recommendations using ranking-based metrics:
- Precision@k: % of top-k recommendations that are correct
- Recall@k: % of actual skill gaps captured in top-k
- MRR@k: Mean Reciprocal Rank (position of first correct recommendation)
"""
import pandas as pd
from typing import List, Dict, Tuple
from app.utils.config import EMPLOYEE_INTELLIGENCE_PATH


def parse_skill_gap(gap_str: str) -> List[str]:
    """Parse comma-separated skill gap string into list of skills."""
    if pd.isna(gap_str) or not gap_str:
        return []
    return [s.strip() for s in gap_str.split(",") if s.strip()]


def rank_skills_by_severity(skills: List[str], org_gap_df: pd.DataFrame) -> List[str]:
    """Rank skills by organization-wide severity (how many employees are missing each)."""
    org_gap_dict = org_gap_df.set_index("skill")["employees_missing"].to_dict()
    ranked = sorted(skills, key=lambda s: -org_gap_dict.get(s, 0))
    return ranked


def precision_at_k(predicted_skills: List[str], true_skills: List[str], k: int = 3) -> float:
    """
    Precision@k: What proportion of top-k predicted skills are in the true skill gaps?
    
    Args:
        predicted_skills: Ranked list of recommended skills (highest priority first)
        true_skills: Ground truth list of employee's actual skill gaps
        k: Evaluation cutoff (top-k skills)
    
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = predicted_skills[:k]
    if len(top_k) == 0:
        return 0.0
    
    true_set = set(true_skills)
    hits = sum(1 for skill in top_k if skill in true_set)
    return hits / len(top_k)


def recall_at_k(predicted_skills: List[str], true_skills: List[str], k: int = 3) -> float:
    """
    Recall@k: What proportion of true skill gaps appear in the top-k predictions?
    
    Args:
        predicted_skills: Ranked list of recommended skills (highest priority first)
        true_skills: Ground truth list of employee's actual skill gaps
        k: Evaluation cutoff (top-k skills)
    
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if len(true_skills) == 0:
        return 1.0  # Perfect recall if no gaps to recall
    
    top_k = predicted_skills[:k]
    top_k_set = set(top_k)
    true_set = set(true_skills)
    hits = sum(1 for skill in true_set if skill in top_k_set)
    return hits / len(true_set)


def ndcg_at_k(predicted_skills: List[str], true_skills: List[str], k: int = 3) -> float:
    """
    Normalized Discounted Cumulative Gain@k: Measures ranking quality accounting for position.
    Skills in top positions are weighted more heavily.
    
    Args:
        predicted_skills: Ranked list of recommended skills
        true_skills: Ground truth list of employee's actual skill gaps
        k: Evaluation cutoff (top-k skills)
    
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if len(true_skills) == 0:
        return 1.0
    
    top_k = predicted_skills[:k]
    true_set = set(true_skills)
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, skill in enumerate(top_k, start=1):
        if skill in true_set:
            dcg += 1.0 / (1 + i)  # log2 discount: 1 / log2(i+1)
    
    # Calculate ideal DCG (if all top-k were correct)
    ideal_dcg = sum(1.0 / (1 + i) for i in range(min(k, len(true_set))))
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def mrr(predicted_skills: List[str], true_skills: List[str], k: int = 5) -> float:
    """
    Mean Reciprocal Rank: Inverse of the rank position of the first correct recommendation.
    
    - MRR = 1.0 if first recommendation is correct
    - MRR = 0.5 if second recommendation is correct
    - MRR = 0.2 if fifth recommendation is correct
    - MRR = 0.0 if no correct recommendation in top-k
    
    Args:
        predicted_skills: Ranked list of recommended skills
        true_skills: Ground truth list of employee's actual skill gaps
        k: Maximum rank position to consider
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    if len(true_skills) == 0:
        return 1.0
    
    top_k = predicted_skills[:k]
    true_set = set(true_skills)
    
    for i, skill in enumerate(top_k, start=1):
        if skill in true_set:
            return 1.0 / i
    
    return 0.0


def map_at_k(predicted_skills: List[str], true_skills: List[str], k: int = 5) -> float:
    """
    Mean Average Precision@k: Average of precision values at each relevant position.
    
    For each correct recommendation at position i, calculate precision@i.
    MAP = average of all these precision values.
    
    Args:
        predicted_skills: Ranked list of recommended skills
        true_skills: Ground truth list of employee's actual skill gaps
        k: Evaluation cutoff
    
    Returns:
        MAP@k score (0.0 to 1.0)
    """
    if len(true_skills) == 0:
        return 1.0
    
    top_k = predicted_skills[:k]
    true_set = set(true_skills)
    
    precisions = []
    hits = 0
    
    for i, skill in enumerate(top_k, start=1):
        if skill in true_set:
            hits += 1
            precisions.append(hits / i)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(true_set)


def hits_at_k(predicted_skills: List[str], true_skills: List[str], k: int = 5) -> int:
    """
    Hits@k: Count of correct recommendations in top-k.
    
    Args:
        predicted_skills: Ranked list of recommended skills
        true_skills: Ground truth list of employee's actual skill gaps
        k: Evaluation cutoff
    
    Returns:
        Number of correct recommendations in top-k (0 to min(k, len(true_skills)))
    """
    top_k = predicted_skills[:k]
    true_set = set(true_skills)
    return sum(1 for skill in top_k if skill in true_set)


def hit_rate_at_k(predicted_skills: List[str], true_skills: List[str], k: int = 5) -> float:
    """
    Hit Rate@k: Percentage of correct items (at least 1 hit) in top-k.
    Binary: 1.0 if any correct item exists, 0.0 otherwise.
    
    Args:
        predicted_skills: Ranked list of recommended skills
        true_skills: Ground truth list of employee's actual skill gaps
        k: Evaluation cutoff
    
    Returns:
        1.0 if at least one correct in top-k, 0.0 otherwise
    """
    if hits_at_k(predicted_skills, true_skills, k) > 0:
        return 1.0
    return 0.0


def evaluate_recommendations(
    gap_df: pd.DataFrame,
    org_gap_df: pd.DataFrame,
    k_values: List[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate recommendation quality for all employees at multiple k cutoffs.
    
    Metrics computed:
    - Precision@k: Accuracy of top-k recommendations
    - Recall@k: Coverage of actual gaps in top-k
    - MRR@k: Position of first correct recommendation
    
    Args:
        gap_df: DataFrame with columns [EmployeeNumber, skill_gap, ...]
        org_gap_df: DataFrame with columns [skill, employees_missing]
        k_values: List of k values to evaluate (default: [1, 3, 5])
    
    Returns:
        Tuple of (detailed metrics DataFrame, summary statistics dict)
    """
    if k_values is None:
        k_values = [1, 3, 5]
    
    results = []
    
    for _, row in gap_df.iterrows():
        employee_id = row["EmployeeNumber"]
        true_skills = parse_skill_gap(row["skill_gap"])
        
        # Predicted skills ranked by org-wide severity
        predicted_skills = rank_skills_by_severity(true_skills, org_gap_df)
        
        metrics = {
            "EmployeeNumber": employee_id,
            "JobRole": row.get("JobRole", "Unknown"),
            "num_skill_gaps": len(true_skills),
        }
        
        # Calculate metrics at each k
        for k in k_values:
            p_at_k = precision_at_k(predicted_skills, true_skills, k)
            r_at_k = recall_at_k(predicted_skills, true_skills, k)
            mrr_k = mrr(predicted_skills, true_skills, k)
            
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


def get_recommendation_quality_report(gap_df: pd.DataFrame, org_gap_df: pd.DataFrame) -> Dict:
    """
    Generate recommendation quality report with Precision@k, Recall@k, MRR@k metrics.
    
    Returns:
        Dictionary with evaluation metrics and summary statistics
    """
    results_df, summary = evaluate_recommendations(gap_df, org_gap_df, k_values=[1, 3, 5])
    
    report = {
        "summary_statistics": summary,
        "total_employees_evaluated": len(results_df),
        "mean_skills_per_employee": round(results_df["num_skill_gaps"].mean(), 1),
        "detailed_results": results_df,
    }
    
    return report
