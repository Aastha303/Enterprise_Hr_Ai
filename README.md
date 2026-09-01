# Enterprise HR AI — Workforce Intelligence & Upskilling Platform

An HR system that looks at employee data and answers three questions: who is
at risk of leaving, where the organisation's skill gaps are, and what each
employee should learn next to close them. Built as a real ML model
(attrition), honest data-engineering (skills/engagement), and a small web app
on top.

Built following the day-by-day plan in `Build Notes` — data first, then
modelling, then workforce intelligence, then the application layer.

> **Fixed in this pass (previously broken/missing when this zip was exported):**
> - `data/external/policies/*.md` and `data/external/career_paths.json` were
>   entirely missing from the export, so the policy chatbot had zero documents
>   to search and always fell back to the same "nothing found" message —
>   recreated all 5 policy docs + the career paths reference.
> - `data/processed/employee_intelligence.csv` (the table nearly every
>   endpoint reads from) was 0 bytes — reran the notebook that builds it.
> - The source HR dataset never had employee names, only `EmployeeNumber` —
>   added `data/processed/employee_names_SYNTHETIC.csv` (clearly labeled
>   synthetic, matching the project's own naming convention) and wired
>   `EmployeeName` through every service, tool, and API response.
> - A routing bug sent "what skills is this employee missing?" to the wrong
>   agent because of word order in the intent regex — fixed in
>   `app/agents/orchestrator.py`.
> - Rebuilt `frontend/streamlit_app.py` as a 6-tab dashboard (Executive
>   Dashboard, Skill Gap & Upskilling, What-If Policy Simulator, Financial
>   Cost Exposure, Employee Drill-Down, HR Assistant Chat) with global,
>   real-time filters and two new backend endpoints
>   (`/dashboard/financial-exposure`, `/employees/{id}/raw`) to power the
>   simulator and cost-exposure views.

## Project structure

```
enterprise_hr_ai/
├── data/
│   ├── raw/                 <- 5 source CSVs, untouched
│   ├── processed/           <- cleaned + derived tables
│   ├── external/
│   │   ├── policies/        <- 5 SAMPLE HR policy docs (placeholder, not real)
│   │   └── career_paths.json
│   └── predictions/         <- prediction_log.csv, written at request time
├── notebooks/                <- 18 numbered pipeline scripts, run in order
├── models/v1/                <- versioned model + metadata.json
├── app/
│   ├── api/                 <- FastAPI routers (attrition, dashboard, skills, policy, career, agent)
│   ├── services/             <- business logic
│   ├── ml/                   <- model loading + prediction
│   ├── rag/                  <- policy retrieval (TF-IDF) + generation (LLM or extractive fallback)
│   ├── agents/                <- tool registry, governance layer, orchestrator
│   └── validation/            <- pydantic input schemas
├── frontend/                  <- Streamlit dashboard + HR assistant chat
├── tests/                     <- pytest suite (23 tests)
├── docs/                       <- data profile, relationships, SHAP, model comparison, skill heatmap
├── Dockerfile, docker-compose.yml   <- scaffold only, see note below
└── requirements.txt
```

## Run the pipeline (already run once — outputs are checked in)

```bash
pip install -r requirements.txt
cd notebooks
for f in *.py; do python3 "$f"; done
```

Each script re-derives the corresponding file(s) in `data/processed/`,
`docs/`, or `models/`. Order matters (run in numeric order 01→18) — later
scripts depend on earlier outputs.

## Run the app

```bash
# Terminal 1 - API
uvicorn app.main:app --reload
# docs at http://127.0.0.1:8000/docs

# Terminal 2 - dashboard
streamlit run frontend/streamlit_app.py
```

Or with Docker (scaffold, see caveats below):
```bash
docker-compose up --build
```

### Enabling LLM-generated policy answers

By default the Policy Q&A chatbot and Policy Agent work with **no API key**
— they retrieve the most relevant policy excerpt via TF-IDF and return it
directly ("extractive" mode). To get fluent, synthesized answers instead,
set an Anthropic API key before starting the API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

## Run the tests

```bash
pytest tests/ -v
```

## What's real vs. what's a placeholder

- **Real, learned from data:** attrition model (Logistic Regression, ROC-AUC
  0.797, chosen over Random Forest/XGBoost for higher recall on the "Left"
  class — see `docs/model_comparison.csv`), SHAP explanations, the
  occupation ↔ skills reference data (O*NET, joins cleanly on
  `occupation_code`), the TF-IDF policy retrieval, career-path readiness
  math, and the tool permission/governance layer (a real role check, not a
  demo — see `app/agents/tools.py`).
- **An honest approximation:** `JobRole` → O*NET occupation mapping
  (`data/processed/role_occupation_map.csv`) — fuzzy/manual string matching,
  not the deck's taxonomy+embeddings approach (a real embedding model needs
  a model-hub download this sandbox can't reach; TF-IDF is the
  dependency-light stand-in used for policy retrieval instead).
- **A labelled placeholder, not real data:**
  `data/processed/employee_skills_SYNTHETIC.csv` — none of the 5 source
  files record what skills an employee currently has, so this is a
  deterministic simulated subset of each employee's mapped occupation's top
  skills. Also `data/external/policies/*.md` — 5 sample HR policies, not a
  real company's documents. Also `data/external/career_paths.json` — a
  small illustrative set of role progressions, not an org chart.
- **Deliberately not joined:** `employee_attrition.csv` and
  `hr_performance_engagement.csv` are two unrelated synthetic employee
  populations — see `docs/data_relationships.md`.
- **Agent layer is a hand-rolled stand-in for LangGraph:** same contract
  (intent → tool call → permission check → result), implemented with
  keyword-based intent detection instead of an LLM router or graph
  framework, to avoid a heavy dependency at this MVP scale. The
  `recruitment_agent` is a thin stub (returns salary data only) since no
  ATS/recruitment dataset was provided.

## Key documents

- `docs/data_relationships.md` — the full join map and why one tempting join was rejected
- `docs/model_comparison.csv` — Logistic Regression vs Random Forest vs XGBoost
- `docs/organization_skill_gap.csv` — org-wide missing-skill counts by severity
- `docs/skill_heatmap.csv` — required vs. available vs. gap, per skill (leadership view)
- `models/v1/metadata.json` — active model version, metrics, training date

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict/attrition` | Attrition prediction for a single employee record |
| GET | `/dashboard/summary` | Headline org metrics |
| GET | `/dashboard/attrition-by-department` | Attrition risk rolled up by department |
| GET | `/dashboard/skill-gaps` | Org-wide missing-skill counts + severity |
| GET | `/dashboard/recommendations` | All employees' upskilling recommendations |
| GET | `/employees/{id}` | Full intelligence record for one employee |
| GET | `/skills/{id}/gap` | Skill gap for one employee |
| GET | `/career/{id}/path` | Career-path next step + readiness % |
| POST | `/policy/ask` | RAG policy Q&A |
| POST | `/agent/chat` | Routes a message through the agent orchestrator (policy/workforce/upskilling/career/recruitment agents), permission-checked by `caller_role` |

## Still deferred (not built — genuinely later-phase, per the build notes)

MLflow experiment tracking, automated retraining triggers (the drift
*check* itself is real — see `notebooks/18_drift_check.py` — but nothing
retrains automatically on it yet), real embedding-model-based semantic
skill matching, a real LangGraph orchestrator, CI/CD, and a hardened
production Docker setup (the current Dockerfile/compose file is a working
scaffold, not something to point at the internet as-is).
