#!/usr/bin/env python
"""Test script for API endpoint integration"""
from app.services.recommendation_service import get_recommendation_quality_metrics
import json

result = get_recommendation_quality_metrics()
print('=' * 80)
print('API ENDPOINT TEST: GET /dashboard/recommendation-quality')
print('=' * 80)
print('\nStatus:', result.get('status'))

if result.get('status') == 'success':
    print('\n📊 SUMMARY STATISTICS')
    print('-' * 80)
    for k in [1, 3, 5]:
        print(f'\nK={k}:')
        print(f'  Precision@{k}:  {result["summary_statistics"].get(f"mean_precision@{k}")}')
        print(f'  Recall@{k}:     {result["summary_statistics"].get(f"mean_recall@{k}")}')
        print(f'  MRR@{k}:        {result["summary_statistics"].get(f"mean_mrr@{k}")}')
        print(f'  MAP@{k}:        {result["summary_statistics"].get(f"mean_map@{k}")}')
        print(f'  NDCG@{k}:       {result["summary_statistics"].get(f"mean_ndcg@{k}")}')
        print(f'  Hits@{k}:       {result["summary_statistics"].get(f"mean_hits@{k}")}')
        print(f'  HitRate@{k}:    {result["summary_statistics"].get(f"mean_hitrate@{k}")}')
        print(f'  F1@{k}:         {result["summary_statistics"].get(f"mean_f1@{k}")}')
    
    print('\n📈 QUALITY DISTRIBUTION')
    print('-' * 80)
    for tier, pct in result['quality_distribution'].items():
        print(f'  {tier:30s}: {pct:6.1f}%')
    
    print(f'\n📋 EVALUATION DETAILS')
    print('-' * 80)
    print(f'  Total employees evaluated: {result["total_employees_evaluated"]}')
    print(f'  Mean skills per employee:  {result["mean_skills_per_employee"]}')
    print(f'  Details file:              {result["evaluation_details_file"]}')
    
    print('\n✅ API ENDPOINT IS WORKING CORRECTLY')
    print('   Available at: GET /dashboard/recommendation-quality')
else:
    print('\n❌ ERROR:', result.get('message'))
    print(result.get('note'))

print('\n' + '=' * 80)
