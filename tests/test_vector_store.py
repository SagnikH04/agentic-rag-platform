import asyncio

from app.retrieval import vector_store


class FakeQueryResponse:
    points = ["first", "second"]


class FakeClient:
    def __init__(self):
        self.arguments = None

    async def query_points(self, **kwargs):
        self.arguments = kwargs
        return FakeQueryResponse()


async def _completed():
    return None


def test_search_vectors_uses_qdrant_query_points(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(vector_store, "client", client)
    monkeypatch.setattr(vector_store, "ensure_collection", _completed)

    points = asyncio.run(vector_store.search_vectors([0.1] * 384, 3, workspace_id="workspace-a"))

    assert points == ["first", "second"]
    assert client.arguments["query"] == [0.1] * 384
    assert client.arguments["limit"] == 3
