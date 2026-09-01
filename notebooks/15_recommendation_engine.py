"""
15 - Upskilling Recommendation Engine (v1)
Deliberately dumb: plain rules mapping a missing skill to a recommendation,
prioritized by how many employees org-wide are missing that skill (so the
top recommendation targets the highest-severity gap first).

v2 (later): swap the rule lookup for a sentence-transformer + cosine
similarity match between missing skill and course descriptions, so
"MLOps" can match a course called "Deploying and Monitoring ML Systems"
even with no literal word overlap. Not needed for the MVP.
"""
import pandas as pd

gap_df = pd.read_csv("../data/processed/employee_skill_gaps.csv")
gap_df["skill_gap"] = gap_df["skill_gap"].fillna("")
org_gap = pd.read_csv("../docs/organization_skill_gap.csv").set_index("skill")

# Simple rule table: skill -> recommended action. Falls back to a generic
# "Learn {skill}" template for anything not explicitly mapped.
COURSE_MAP = {
    "Microsoft Excel": "Excel for Business Analysis (internal LMS)",
    "Microsoft PowerPoint": "Presentation Design Fundamentals",
    "Microsoft Outlook": "Effective Email & Calendar Management",
    "Microsoft Office software": "Microsoft Office Essentials",
    "Critical Thinking": "Structured Problem Solving workshop",
    "Active Listening": "Active Listening for Managers",
    "Reading Comprehension": "Business Writing & Comprehension",
    "Monitoring": "Performance Monitoring & KPIs",
    "Writing": "Business Writing Essentials",
    "Speaking": "Public Speaking & Presentation Skills",
    "Science": "Domain Science Fundamentals (role-specific)",
    "Active Learning": "Learning How to Learn",
    "Salesforce software": "Salesforce Administrator Certification",
    "Learning Strategies": "Instructional Design Basics",
    "Applicant tracking software": "ATS Systems for Recruiters",
}


def recommend(gap_str: str) -> str:
    if not gap_str:
        return "No action needed - no skill gap detected"
    skills = [s.strip() for s in gap_str.split(",") if s.strip()]
    # prioritize by organisation-wide severity (highest-impact gap first)
    skills_ranked = sorted(skills, key=lambda s: -org_gap["employees_missing"].get(s, 0))
    top_skill = skills_ranked[0]
    course = COURSE_MAP.get(top_skill, f"Learn {top_skill}")
    return f"Priority: {top_skill} -> {course}"


gap_df["recommendation"] = gap_df["skill_gap"].apply(recommend)

print("=== Example recommendations ===")
print(gap_df[["EmployeeNumber", "JobRole", "skill_gap", "recommendation"]].head(10).to_string(index=False))

gap_df[["EmployeeNumber", "recommendation"]].to_csv("../data/processed/employee_recommendations.csv", index=False)
print("\nSaved -> data/processed/employee_recommendations.csv")
