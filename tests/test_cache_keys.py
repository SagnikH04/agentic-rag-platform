from app.core.cache import answer_key, retrieval_key, stable_hash
from app.core.config import settings


def test_stable_hash_is_order_independent():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_retrieval_cache_key_changes_with_workspace_revision():
    first = retrieval_key("workspace-a", 1, "What is cached?", {"type": "pdf"}, 8)
    second = retrieval_key("workspace-a", 2, "What is cached?", {"type": "pdf"}, 8)
    assert first != second


def test_answer_cache_key_is_workspace_scoped():
    first = answer_key("workspace-a", 1, "history", "Question?")
    second = answer_key("workspace-b", 1, "history", "Question?")
    assert first != second


def test_answer_cache_key_changes_with_groq_model(monkeypatch):
    monkeypatch.setattr(settings, "groq_model", "qwen/qwen3.8-27b")
    first = answer_key("workspace-a", 1, "history", "Question?")
    monkeypatch.setattr(settings, "groq_model", "llama-3.3-70b-versatile")
    second = answer_key("workspace-a", 1, "history", "Question?")
    assert first != second


def test_answer_cache_key_scopes_filters_and_top_k():
    base = answer_key("workspace", 1, "history", "question", {"department": "legal"}, 3)
    assert base != answer_key("workspace", 1, "history", "question", {"department": "sales"}, 3)
    assert base != answer_key("workspace", 1, "history", "question", {"department": "legal"}, 8)
