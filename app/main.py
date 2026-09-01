"""
FastAPI backend for the Enterprise HR AI - Workforce Intelligence & Upskilling Platform.

Run with:
    uvicorn app.main:app --reload --app-dir .
(from the enterprise_hr_ai/ project root)
"""
from fastapi import FastAPI
from app.api import attrition, dashboard, skills, policy, agent, career
from app.utils.logger import logger

app = FastAPI(
    title="Enterprise HR AI - Workforce Intelligence Platform",
    description="Predicts attrition, tracks engagement, finds skill gaps, recommends upskilling, "
                "answers HR policy questions (RAG), and routes requests through specialized agents.",
    version="1.1.0",
)

app.include_router(attrition.router)
app.include_router(dashboard.router)
app.include_router(skills.router)
app.include_router(policy.router)
app.include_router(agent.router)
app.include_router(career.router)


@app.on_event("startup")
def on_startup():
    logger.info("Application starting up")
    logger.info("Dataset and model paths configured")


@app.get("/")
def root():
    return {"status": "ok", "service": "Enterprise HR AI Platform", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}
