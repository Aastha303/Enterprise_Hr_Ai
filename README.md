# Enterprise HR AI — Workforce Intelligence & Upskilling Platform

An AI-powered HR platform built to help HR teams answer three practical questions:

1. **Which employees are at risk of leaving?**
2. **Where are the organisation's skill gaps?**
3. **What should each employee learn next to close those gaps?**

The project combines a real machine-learning pipeline for **employee attrition prediction**, data processing for **workforce and skill intelligence**, and a web application that brings everything together in one dashboard.

The project was built step-by-step — starting with the data pipeline, then ML modelling, workforce intelligence, and finally the application and agent layer.

---

## What the project does

The platform provides HR teams with:

* Employee attrition risk predictions
* Department-level attrition insights
* Organisation-wide skill-gap analysis
* Personalised upskilling recommendations
* Career-path readiness analysis
* HR policy Q&A
* An HR assistant that can route questions to different agents
* What-if policy simulation
* Financial cost exposure analysis
* Employee-level drill-downs

The goal is not just to predict problems, but to connect those predictions with **skills, learning recommendations, career growth, and HR decision-making**.

---

## Project Structure

```text
enterprise_hr_ai/
│
├── data/
│   ├── raw/                       # Original source CSVs
│   ├── processed/                 # Cleaned and derived datasets
│   ├── external/
│   │   ├── policies/              # Sample HR policy documents
│   │   └── career_paths.json      # Career progression reference
│   └── predictions/               # Prediction logs
│
├── notebooks/
│                                 # Data + ML pipeline scripts
│
├── models/
│   └── v1/                       # Versioned ML model and metadata
│
├── app/
│   ├── api/                      # FastAPI endpoints
│   ├── services/                 # Business logic
│   ├── ml/                       # Model loading and prediction
│   ├── rag/                      # Policy retrieval and generation
│   ├── agents/                   # HR agent tools and orchestration
│   └── validation/               # Pydantic validation schemas
│
├── frontend/
│   └── streamlit_app.py          # HR dashboard
│
├── tests/                        # Automated tests
├── docs/                         # Analysis and documentation
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ML Pipeline

The project follows a data-first approach.

The pipeline processes the source HR data, creates the required features, trains and evaluates different models, and generates the datasets used by the application.

The notebooks are numbered so they can be executed in sequence:

```text
01 → 02 → .....
```

Each step generates or updates files inside `data/processed/`, `docs/`, or `models/`.

### Run the pipeline

```bash
pip install -r requirements.txt

cd notebooks

for f in *.py; do
    python3 "$f"
done
```

> Run the notebooks in numerical order because later steps depend on outputs created by earlier steps.

---

## Attrition Prediction

The attrition model predicts whether an employee is likely to leave the organisation.

Several models were evaluated, including:

* Logistic Regression
* Random Forest
* XGBoost

The final model is **Logistic Regression**, which was selected because it provided better recall for employees who actually left while still maintaining reasonable overall performance.

### Model performance

**ROC-AUC: 0.797**

SHAP is also used to explain individual predictions and understand which features are contributing to an employee's risk score.

Model information is stored in:

```text
models/v1/
```

with metadata available in:

```text
models/v1/metadata.json
```

---

## Workforce Intelligence

The platform goes beyond attrition prediction and looks at the organisation's workforce from a skills perspective.

The system uses occupation-to-skill reference data based on **O*NET** to identify the skills expected for different roles.

It then compares those requirements with the available employee skill information to identify gaps.

The dashboard can show:

* Required skills
* Available skills
* Missing skills
* Skill-gap severity
* Department-level gaps
* Recommended learning areas

---

## Upskilling & Career Paths

For each employee, the system can identify:

```text
Current Role
      ↓
Required Skills
      ↓
Current Skills
      ↓
Skill Gap
      ↓
Recommended Learning
      ↓
Potential Next Role
```

Career readiness is calculated using the employee's current skills compared with the skills required for the next role.

The career-path reference is stored in:

```text
data/external/career_paths.json
```

---

## HR Assistant & Agent Layer

The project also includes a small HR assistant.

Instead of sending every question to one large system, the assistant identifies the type of request and routes it to the appropriate agent.

Current agent areas include:

* Policy
* Workforce
* Upskilling
* Career
* Recruitment

The flow is roughly:

```text
User Question
      ↓
Intent Detection
      ↓
Permission Check
      ↓
Relevant HR Tool
      ↓
Result
```

The agent layer is intentionally lightweight. It uses a hand-built orchestrator rather than LangGraph, keeping the MVP simple while following a similar tool → permission → result architecture.

The system also includes role-based permission checks before certain tools can be used.

---

## Policy Q&A

The HR policy assistant uses a simple RAG-style pipeline.

```text
Policy Documents
      ↓
TF-IDF Retrieval
      ↓
Relevant Policy Excerpt
      ↓
Answer
```

The system can work without an external API key.

By default, it uses an **extractive approach**, returning the most relevant policy information directly.

If an Anthropic API key is provided, the system can generate more natural, synthesised answers.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then start the API:

```bash
uvicorn app.main:app --reload
```

The policy documents included in this repository are **sample documents** and are not real company policies.

---

## Dashboard

The frontend is built with Streamlit and contains six main sections:

### 1. Executive Dashboard

Provides a high-level view of the organisation, including:

* Employee metrics
* Attrition risk
* Engagement
* Department-level risk
* Organisation skill gaps

### 2. Skill Gap & Upskilling

Shows:

* Missing skills
* Skill severity
* Employee-level gaps
* Upskilling recommendations

### 3. What-If Policy Simulator

Allows HR users to explore how different policy assumptions could affect workforce outcomes.

### 4. Financial Cost Exposure

Provides an estimate of potential financial exposure associated with workforce attrition.

### 5. Employee Drill-Down

Allows HR users to inspect an individual employee's intelligence record.

### 6. HR Assistant Chat

Provides a conversational interface for asking HR-related questions.

The dashboard also supports global filters so that HR users can explore the data interactively.

---

## Running the Application

### Start the FastAPI backend

Open a terminal and run:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the Streamlit dashboard

Open another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

---

## API Endpoints

| Method | Endpoint                             | Purpose                           |
| ------ | ------------------------------------ | --------------------------------- |
| POST   | `/predict/attrition`                 | Predict attrition for an employee |
| GET    | `/dashboard/summary`                 | Overall HR metrics                |
| GET    | `/dashboard/attrition-by-department` | Department-level attrition        |
| GET    | `/dashboard/skill-gaps`              | Organisation skill gaps           |
| GET    | `/dashboard/recommendations`         | Upskilling recommendations        |
| GET    | `/dashboard/financial-exposure`      | Financial exposure estimates      |
| GET    | `/employees/{id}`                    | Employee intelligence             |
| GET    | `/employees/{id}/raw`                | Raw employee information          |
| GET    | `/skills/{id}/gap`                   | Employee skill gap                |
| GET    | `/career/{id}/path`                  | Career path and readiness         |
| POST   | `/policy/ask`                        | Ask an HR policy question         |
| POST   | `/agent/chat`                        | Chat with the HR agent            |

---

## Testing

The project includes a pytest test suite.

Run:

```bash
pytest tests/ -v
```

The repository currently contains **23 tests** covering key parts of the application.

---

## Data Notes

Some important limitations are intentionally documented instead of being hidden.

### Employee names

The original HR datasets contain employee numbers but not actual employee names.

Therefore:

```text
data/processed/employee_names_SYNTHETIC.csv
```

contains synthetic names used only to make the dashboard easier to use.

These names are not real employee information.

### Employee skills

The source datasets also don't contain a reliable record of each employee's current skills.

Therefore:

```text
data/processed/employee_skills_SYNTHETIC.csv
```

is a deterministic simulated dataset based on the employee's mapped occupation and its associated skills.

It is clearly labelled as synthetic.

### Occupation mapping

The mapping between `JobRole` and O*NET occupations is an approximation using fuzzy/manual matching.

It is not a production-grade semantic matching system.

---

## What's Real vs. What's Approximate

### Built from actual project data and modelling

* Attrition prediction model
* Model comparison
* SHAP explanations
* Workforce analysis
* O*NET occupation/skill reference data
* TF-IDF policy retrieval
* Career readiness calculations
* Tool permission checks
* Workforce and dashboard APIs

### Approximation

* Job role → O*NET occupation mapping
* Employee skill assignments

### Synthetic / placeholder data

* Employee names
* Employee current skills
* Sample HR policy documents
* Career-path reference data

Being explicit about these limitations is intentional — the project is designed to show what is genuinely learned from data versus what is currently simulated for the MVP.

---

## Data Relationships

The project contains multiple HR datasets, but they are not all from the same employee population.

In particular:

```text
employee_attrition.csv
```

and

```text
hr_performance_engagement.csv
```

represent separate synthetic employee populations.

They are therefore **not directly joined**.

The reasoning behind the data relationships is documented in:

```text
docs/data_relationships.md
```

---

## Current Limitations / Future Work

The following features are intentionally left for a later phase:

* MLflow experiment tracking
* Automated model retraining
* Automated drift-triggered retraining
* Real semantic embedding model for skill matching
* LangGraph-based agent orchestration
* CI/CD pipeline
* Production-grade Docker configuration
* Production data sources
* Real employee skill data
* Real HR policy documents

The project already includes a **drift check** in:

```text
notebooks/18_drift_check.py
```

but the current system does not automatically retrain the model when drift is detected.

---

## Docker

Docker configuration is included as a starting point:

```bash
docker-compose up --build
```

However, the current Docker setup is intended as a development scaffold and should not be treated as a production deployment configuration without additional hardening.

---

## Why this project?

Most HR dashboards stop at displaying employee information.

This project tries to connect the complete decision-making flow:

```text
Employee Data
      ↓
Attrition Prediction
      ↓
Workforce Intelligence
      ↓
Skill Gap Detection
      ↓
Upskilling Recommendation
      ↓
Career Growth
      ↓
HR Decision Support
```

The idea is to move from **"What is happening?"** to **"Why is it happening?"** and finally to **"What can HR do about it?"**

---

## Tech Stack

**Backend**

* Python
* FastAPI
* Pydantic

**Machine Learning**

* Scikit-learn
* Logistic Regression
* Random Forest
* XGBoost
* SHAP

**Data**

* Pandas
* NumPy
* O*NET reference data

**RAG / AI**

* TF-IDF
* Anthropic API (optional)
* Custom HR agent/orchestrator

**Frontend**

* Streamlit

**Testing**

* Pytest

**Deployment**

* Docker
* Docker Compose

---

## Project Status

**Current stage: MVP / Development**

The core data pipeline, ML model, workforce intelligence layer, APIs, dashboard, policy assistant, and HR agent layer are implemented.

The next phase would focus on making the system production-ready through better semantic skill matching, automated ML lifecycle management, stronger agent orchestration, CI/CD, and real organisational data.
