import asyncio

from app.agents.orchestrator import AgentOrchestrator


class FakeRedis:
    async def get(self, key):
        return None

    async def set(self, *args, **kwargs):
        return True

    def lock(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def aclose(self):
        return None


class FakeMemory:
    async def recent(self, session_id, workspace_id):
        return []


def test_missing_groq_key_returns_clear_response(monkeypatch):
    orchestrator = AgentOrchestrator()
    orchestrator.client = None
    orchestrator.memory = FakeMemory()
    async def fake_redis_client():
        return FakeRedis()

    monkeypatch.setattr("app.agents.orchestrator.redis_client", fake_redis_client)

    response = asyncio.run(orchestrator.answer("What is indexed?", "session", {}, 3, "workspace"))

    assert response.status == "insufficient_evidence"
    assert "GROQ_API_KEY" in response.answer
    assert response.safety_flags == ["groq_api_key_missing"]
