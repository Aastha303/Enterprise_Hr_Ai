"""
Generation half of RAG. If ANTHROPIC_API_KEY is set in the environment, the
retrieved chunks are handed to Claude to produce a grounded, synthesized
answer. If no key is set, this degrades to an EXTRACTIVE answer (the
retrieved text itself, verbatim, with its source) rather than failing -
policy Q&A stays usable without any API key configured, just less fluent.

Per the architecture in the deck: RAG handles knowledge retrieval only -
it never predicts attrition, calculates skill gaps, or takes automated
action. Those stay in their own services/agents.
"""
import os
from app.rag.retriever import get_retriever
from app.utils.logger import logger

MODEL = "claude-sonnet-5"


def _has_llm() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _generate_with_llm(question: str, context: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    prompt = (
        "Answer the HR policy question using ONLY the provided policy excerpts. "
        "If the excerpts don't contain the answer, say so plainly - do not guess.\n\n"
        f"Policy excerpts:\n{context}\n\nQuestion: {question}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer_policy_question(question: str) -> dict:
    retriever = get_retriever()

    # If the question reduces to zero meaningful search terms (e.g. "what's the
    # policy?" - "policy" itself never appears in the body text, only in doc
    # titles we deliberately don't index), don't report "no match" - that's
    # misleading. Show what's actually available instead.
    query_vec = retriever.vectorizer.transform([question]) if retriever.chunks else None
    if query_vec is not None and query_vec.nnz == 0:
        topics = sorted(set(c.source.replace(".md", "").replace("_", " ").title() for c in retriever.chunks))
        return {
            "answer": "That's too general for me to search on directly - I don't have a word to match against. "
                      "Try asking about one of these specific topics instead:\n- "
                      + "\n- ".join(topics),
            "sources": [],
            "mode": "clarification_needed",
        }

    # For benefits queries, retrieve more chunks to ensure comprehensive coverage
    question_lower = question.lower()
    if any(word in question_lower for word in ["benefit", "insurance", "401", "wellness", "match", "coverage", "disability"]):
        chunks = retriever.retrieve(question, k=12)  # Get more for better benefits coverage
    else:
        chunks = retriever.retrieve(question, k=5)

    if not chunks:
        return {
            "answer": "I couldn't find anything in the current policy documents that answers this. "
                      "This platform only has 5 sample policies loaded (parental leave, remote work, "
                      "PTO, benefits, travel/expense) - try rephrasing, or this may genuinely be outside scope.",
            "sources": [],
            "mode": "no_match",
        }

    # For benefits queries, prioritize benefits.md chunks and filter out unrelated content
    if any(word in question_lower for word in ["benefit", "insurance", "401", "wellness", "match", "coverage", "disability"]):
        # Keep benefits-related chunks only
        benefits_chunks = [c for c in chunks if c.source == "benefits.md"]
        other_chunks = [c for c in chunks if c.source in ("parental_leave.md",)]  # Allow parental leave as it mentions benefits
        chunks = benefits_chunks + other_chunks

    context = "\n---\n".join(f"[{c.source}] {c.text}" for c in chunks)
    sources = sorted(set(c.source for c in chunks))

    if _has_llm():
        try:
            answer = _generate_with_llm(question, context)
            return {"answer": answer, "sources": sources, "mode": "llm_grounded"}
        except Exception as e:
            logger.info(f"LLM generation failed, falling back to extractive answer: {e}")

    # Extractive fallback: no API key configured (or the call failed) - return the
    # retrieved text itself rather than a synthesized answer.
    extractive = " ".join(c.text for c in chunks)
    return {
        "answer": extractive,
        "sources": sources,
        "mode": "extractive_fallback",
    }
