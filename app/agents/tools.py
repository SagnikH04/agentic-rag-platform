import re

from app.models.schemas import Citation
from app.retrieval.hybrid import hybrid_search

INJECTION_PATTERNS = (
    r"\bignore\s+(all\s+)?(previous|prior)\s+(instructions|rules)\b",
    r"\b(system|developer)\s+prompt\b",
    r"\breveal\s+(your\s+)?instructions\b",
    r"\bdisregard\s+(the\s+)?(policy|rules|guardrails)\b",
    r"\byou\s+are\s+now\b",
)


async def retrieve_documents(question: str, top_k: int, filters: dict) -> list[Citation]:
    return await hybrid_search(question.strip(), top_k, filters)


def verify_citations(answer: str, citations: list[Citation]) -> float:
    if not citations:
        return 0.0
    terms = {
        term
        for term in re.findall(r"\b[\w-]+\b", answer.lower())
        if len(term) > 2 and term not in {"source", "sources", "grounded", "answer"}
    }
    if not terms:
        return 0.0
    coverage = []
    for citation in citations:
        evidence_terms = set(re.findall(r"\b[\w-]+\b", citation.text.lower()))
        coverage.append(len(terms & evidence_terms) / len(terms))
    return min(1.0, max(coverage) * 0.7 + sum(coverage) / len(coverage) * 0.3)


def detect_prompt_injection(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS)


def sentence_support(answer: str, citations: list[Citation]) -> float:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if sentence.strip()]
    if not sentences:
        return 0.0
    citation_terms = [set(re.findall(r"\b[\w-]+\b", citation.text.lower())) for citation in citations]
    supported = 0
    for sentence in sentences:
        terms = {term for term in re.findall(r"\b[\w-]+\b", sentence.lower()) if len(term) > 2}
        if terms and any(len(terms & evidence) / len(terms) >= 0.25 for evidence in citation_terms):
            supported += 1
    return supported / len(sentences)
